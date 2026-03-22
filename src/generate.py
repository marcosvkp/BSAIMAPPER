"""
generate.py — Pipeline de Geração V6

Fluxo:
  1. TimingNet  → seleciona frames com nota (picos de onset ponderados por IA)
  2. PlaceNet   → atribui mão/col/layer para cada nota selecionada
  3. AngleNet   → atribui ângulo por mão separadamente
  4. ViewNet    → avalia setores e re-corrige problemáticos (opcional)
  5. FlowFixer  → validação física final (100% determinístico)

Uso:
  python generate.py                            # modo interativo
  python generate.py --url URL --diffs 3,4,5
  python generate.py --url URL --diffs 4 --stars 6.5
"""

import os
import sys
import json
import shutil
import argparse
from collections import deque

import torch
import numpy as np

from note_filter import filter_notes, fix_orphan_dots, report as filter_report
from note_constraints import apply_constraints
from models import (
    get_timing_model, get_place_model, get_angle_model, get_view_model,
    ANGLE_HIST, VIEW_WIN, NUM_CUTS, PLACE_WIN, CTX_FEATS, MEL_BINS,
)
from audio_processor import extract_features, detect_bpm, add_silence, analyze_energy
from flow_fixer import FlowFixer
from youtube_downloader import download_from_youtube

# ─────────────────────────────────────────────────────────────────
# Constantes
# ─────────────────────────────────────────────────────────────────

MODELS_DIR = "models"
DIFFICULTY_MAP = {1: "Easy", 2: "Normal", 3: "Hard", 4: "Expert", 5: "ExpertPlus"}
DIFFICULTY_PARAMS = {
    "Easy":       {"njs": 10, "offset":  0.0},
    "Normal":     {"njs": 12, "offset":  0.0},
    "Hard":       {"njs": 14, "offset": -0.2},
    "Expert":     {"njs": 16, "offset": -0.4},
    "ExpertPlus": {"njs": 18, "offset": -0.7},
}

TEMP_PLACE = 0.85   # temperatura para hand/col/layer
TEMP_ANGLE = 0.85   # temperatura para cut direction
PLACE_PAD  = PLACE_WIN // 2   # = 3


def _sample(logits, temp=1.0):
    probs = torch.softmax(logits.squeeze() / temp, dim=-1)
    return torch.multinomial(probs, 1).item()


def _note(beat, hand, col, layer, cut):
    return {
        '_time':         float(beat),
        '_lineIndex':    int(col),
        '_lineLayer':    int(layer),
        '_type':         int(hand),
        '_cutDirection': int(cut),
    }


# ─────────────────────────────────────────────────────────────────
# Estado do AngleNet — por mão
# ─────────────────────────────────────────────────────────────────

class HandState:
    """
    Mantém o histórico das últimas ANGLE_HIST notas de UMA mão.
    Estado completamente separado por mão — essa é a garantia de paridade correta.
    """

    def __init__(self):
        self.cut_h   = deque([NUM_CUTS - 1] * ANGLE_HIST, maxlen=ANGLE_HIST)  # pad=DOT
        self.col_h   = deque([4]            * ANGLE_HIST, maxlen=ANGLE_HIST)  # pad=4
        self.layer_h = deque([3]            * ANGLE_HIST, maxlen=ANGLE_HIST)  # pad=3
        self.first   = True
        self.prev_beat = 0.0

    def predict(self, model, col, layer, beat, stars, device):
        if self.first:
            self.first = False
            cut = 1 if layer == 2 else 0   # heurística de abertura
            self._update(cut, col, layer, beat)
            return cut

        beat_gap = min((beat - self.prev_beat) / 8.0, 1.0)

        ch = torch.tensor(list(self.cut_h),   dtype=torch.long).unsqueeze(0).to(device)
        oh = torch.tensor(list(self.col_h),   dtype=torch.long).unsqueeze(0).to(device)
        ah = torch.tensor(list(self.layer_h), dtype=torch.long).unsqueeze(0).to(device)
        pos = torch.tensor([[col/3.0, layer/2.0]], dtype=torch.float32).to(device)
        gap = torch.tensor([[beat_gap]],           dtype=torch.float32).to(device)
        st  = torch.tensor([[stars]],              dtype=torch.float32).to(device)

        with torch.no_grad():
            logits = model(ch, oh, ah, pos, gap, st)
            cut = _sample(logits, TEMP_ANGLE)

        self._update(cut, col, layer, beat)
        return cut

    def _update(self, cut, col, layer, beat):
        self.cut_h.append(cut)
        self.col_h.append(col)
        self.layer_h.append(layer)
        self.prev_beat = beat


# ─────────────────────────────────────────────────────────────────
# Fase 1 — TimingNet
# ─────────────────────────────────────────────────────────────────

# Subdivisions do grid por dificuldade (em frações de beat)
# Easy=1/2, Normal=1/4, Hard=1/4, Expert=1/8, ExpertPlus=1/16
_GRID_SUBDIV = {
    "Easy":       0.500,
    "Normal":     0.250,
    "Hard":       0.250,
    "Expert":     0.250,
    "ExpertPlus": 0.250,
}

# NPS alvo por dificuldade — âncoras fixas, não dependem do modelo
_TARGET_NPS = {
    "Easy":       2.0,
    "Normal":     3.0,
    "Hard":       4.5,
    "Expert":     6.5,
    "ExpertPlus": 9.0,
}


def _phase1_timing(timing_model, ctx_feats, energy, bpm, stars, device,
                   diff_name="Expert"):
    """
    Seleciona frames para notas com dois critérios separados:

    1. GRID DE BEAT — só posições válidas no grid rítmico da música.
       Cada dificuldade tem uma subdivisão máxima (Ex: Expert = 1/8 beat).
       Isso garante que as notas caem em posições rítmicas reais da música,
       não em frames aleatórios entre beats.

    2. SCORE DO TIMING MODEL × ENERGIA — dentro das posições válidas do grid,
       seleciona as que têm maior probabilidade de nota segundo o modelo.
       O NPS alvo é fixo por dificuldade — não depende de ai_conf.

    Por que grid ao invés de cooldown em frames?
       Cooldown em frames (ex: 5 frames) permite notas em qualquer momento
       dentro do beat — resulta em padrões "fora do ritmo" que são impossíveis
       de ler. O grid força todas as notas a cair em subdivisões musicais reais.
    """
    T             = ctx_feats.shape[0]
    frame_dur     = 512 / 22050          # ~0.02322 s/frame
    secs_per_beat = 60.0 / bpm
    duration      = T * frame_dur

    ctx_t   = torch.tensor(ctx_feats, dtype=torch.float32).unsqueeze(0).to(device)
    stars_t = torch.tensor([[stars]], dtype=torch.float32).to(device)

    timing_model.eval()
    with torch.no_grad():
        probs = torch.sigmoid(timing_model(ctx_t, stars_t)).squeeze().cpu().numpy()

    # ── Passo 1: monta grid de posições válidas ───────────────────
    subdiv    = _GRID_SUBDIV.get(diff_name, 0.125)          # beats
    grid_sec  = subdiv * secs_per_beat                      # segundos
    grid_frames = max(1, int(grid_sec / frame_dur))         # frames

    # Agrupa frames em células do grid e pega o frame de maior score em cada célula
    grid_candidates = {}  # cell_idx → (score, frame_idx)
    for i in range(1, T - 1):
        if i * frame_dur < 2.0:          # ignora primeiros 2s (silêncio adicionado)
            continue
        cell = i // grid_frames
        score = float(probs[i]) * (0.7 + 0.3 * float(energy[i]))
        if cell not in grid_candidates or score > grid_candidates[cell][0]:
            grid_candidates[cell] = (score, i)

    # ── Passo 2: seleciona as células com maior score ─────────────
    # Usa NPS alvo fixo — não varia com ai_conf para evitar instabilidade
    target_nps = _TARGET_NPS.get(diff_name, 6.0)
    target_n   = int(max(60, duration * target_nps))

    # Threshold adaptativo: pega o top-N por score
    sorted_cells = sorted(grid_candidates.values(), key=lambda x: -x[0])
    selected_frames = sorted([idx for _, idx in sorted_cells[:target_n]])

    # ── Passo 3: garante gap mínimo entre frames selecionados ─────
    # Mesmo dentro do grid, dois picos adjacentes podem estar a 1 subdivisão.
    # Para dificuldades menores, exige gap de pelo menos 2 subdivisões.
    min_cells_gap = 2 if subdiv >= 0.25 else 1
    min_frame_gap = grid_frames * min_cells_gap
    filtered = []
    for idx in selected_frames:
        if not filtered or (idx - filtered[-1]) >= min_frame_gap:
            filtered.append(idx)

    print(f"      TimingNet : grid={subdiv:.4f}bt | alvo={target_nps:.1f} NPS "
          f"→ {len(filtered)} frames ({len(filtered)/duration:.2f} NPS real)")
    return filtered


# ─────────────────────────────────────────────────────────────────
# Fase 2 — PlaceNet
# ─────────────────────────────────────────────────────────────────

def _next_frame(frames: list, current: int):
    """Retorna o próximo frame após current na lista ordenada, ou None."""
    import bisect
    pos = bisect.bisect_right(frames, current)
    return frames[pos] if pos < len(frames) else None



def _phase2_place(place_model, selected_frames, mel_spec, ctx_feats,
                  bpm, stars, device):
    """
    Para cada frame selecionado, consulta o PlaceNet com uma janela
    de ±PLACE_PAD frames de mel+ctx e decide hand/col/layer/double.
    Retorna lista de DOT notes.
    """
    T   = mel_spec.shape[0]
    secs_per_beat = 60.0 / bpm
    frame_dur     = 512 / 22050
    stars_t = torch.tensor([[stars]], dtype=torch.float32).to(device)

    place_model.eval()
    notes     = []
    last_beat = -999.0

    with torch.no_grad():
        for idx in selected_frames:
            if idx * frame_dur < 2.0:
                continue

            # Janela de ±PLACE_PAD frames (com clamp nas bordas)
            lo = max(0, idx - PLACE_PAD)
            hi = min(T, idx + PLACE_PAD + 1)
            mel_w = mel_spec[lo:hi]
            ctx_w = ctx_feats[lo:hi]

            # Pad para PLACE_WIN se necessário
            if mel_w.shape[0] < PLACE_WIN:
                pad = PLACE_WIN - mel_w.shape[0]
                mel_w = np.pad(mel_w, ((0, pad), (0, 0)))
                ctx_w = np.pad(ctx_w, ((0, pad), (0, 0)))

            mel_t = torch.tensor(mel_w, dtype=torch.float32).unsqueeze(0).to(device)
            ctx_t = torch.tensor(ctx_w, dtype=torch.float32).unsqueeze(0).to(device)

            out   = place_model(mel_t, ctx_t, stars_t)
            hand  = _sample(out['hand'],  TEMP_PLACE)
            col   = _sample(out['col'],   TEMP_PLACE)
            layer = _sample(out['layer'], TEMP_PLACE)
            dbl   = _sample(out['is_double'], 1.0)

            beat_time = round((idx * frame_dur / secs_per_beat) * 16) / 16.0
            if beat_time <= last_beat:
                continue

            notes.append(_note(beat_time, hand, col, layer, 8))  # DOT
            last_beat = beat_time

            # Double só se há gap suficiente para a próxima nota selecionada
            next_frame_idx = _next_frame(selected_frames, idx)
            next_beat = round((next_frame_idx * frame_dur / secs_per_beat) * 16) / 16.0                 if next_frame_idx is not None else beat_time + 1.0
            gap_to_next = next_beat - beat_time

            if dbl == 1 and gap_to_next >= 0.25:  # só double se próxima nota >= 1/4 beat
                dbl_beat = round((beat_time + 1/16) * 16) / 16.0
                dbl_hand = 1 - hand
                dbl_col  = _sample(out['col'],   TEMP_PLACE)
                dbl_lay  = _sample(out['layer'], TEMP_PLACE)
                notes.append(_note(dbl_beat, dbl_hand, dbl_col, dbl_lay, 8))
                last_beat = dbl_beat

    print(f"      PlaceNet  : {len(notes)} notas (todas DOT)")
    return notes


# ─────────────────────────────────────────────────────────────────
# Fase 3 — AngleNet
# ─────────────────────────────────────────────────────────────────

def _phase3_angle(angle_model, notes, stars, device):
    """
    Percorre as notas em ordem cronológica.
    Estado separado por mão — HandState rastreia o histórico de cada mão.
    """
    if not notes or angle_model is None:
        return notes

    angle_model.eval()
    left  = HandState()
    right = HandState()
    result = [dict(n) for n in sorted(notes, key=lambda n: n['_time'])]

    for note in result:
        state = left if note['_type'] == 0 else right
        note['_cutDirection'] = state.predict(
            angle_model,
            note['_lineIndex'], note['_lineLayer'],
            note['_time'], stars, device,
        )

    dot_remaining = sum(1 for n in result if n['_cutDirection'] == 8)
    print(f"      AngleNet  : ângulos definidos ({dot_remaining} DOTs restantes)")
    return result


# ─────────────────────────────────────────────────────────────────
# Fase 4 — ViewNet (opcional)
# ─────────────────────────────────────────────────────────────────

def _phase4_view(view_model, angle_model, notes, stars, device, max_passes=3):
    """
    Sliding window sobre o mapa. Janelas com quality < 0.5 têm suas
    notas problemáticas re-corrigidas pelo AngleNet com temperatura menor.
    """
    if not notes or view_model is None:
        return notes

    view_model.eval()
    stars_t = torch.tensor([[stars]], dtype=torch.float32).to(device)

    for pass_num in range(max_passes):
        problems = 0
        N = len(notes)
        step = VIEW_WIN // 2

        for i in range(0, N, step):
            j  = min(i + VIEW_WIN, N)
            w  = notes[i:j]
            W  = len(w)

            feat = np.zeros((VIEW_WIN, 5), dtype=np.float32)
            for k, n in enumerate(w):
                feat[k] = [
                    n['_type'],
                    n['_lineIndex'] / 3.0,
                    n['_lineLayer'] / 2.0,
                    n['_cutDirection'] / 8.0,
                    0.5,
                ]

            feat_t = torch.tensor(feat, dtype=torch.float32).unsqueeze(0).to(device)
            with torch.no_grad():
                out     = view_model(feat_t, stars_t)
                quality = torch.sigmoid(out['quality']).item()
                mask    = torch.sigmoid(out['problem_mask']).squeeze().cpu().numpy()

            if quality < 0.5:
                problems += 1
                # Re-cria estados temporários para a correção localizada
                left  = HandState()
                right = HandState()
                for k in range(W):
                    if mask[k] > 0.5:
                        n     = notes[i + k]
                        state = left if n['_type'] == 0 else right
                        # Temperatura mais baixa = escolha mais conservadora
                        old_temp = TEMP_ANGLE
                        import models as _m
                        new_cut = state.predict(
                            angle_model,
                            n['_lineIndex'], n['_lineLayer'],
                            n['_time'], stars, device,
                        )
                        n['_cutDirection'] = new_cut

        print(f"      ViewNet   : passe {pass_num+1}/{max_passes}  "
              f"(setores problemáticos: {problems})")
        if problems == 0:
            break

    return notes


# ─────────────────────────────────────────────────────────────────
# Pipeline principal
# ─────────────────────────────────────────────────────────────────

def generate_difficulty(timing_model, place_model, angle_model, view_model,
                        mel_spec, ctx_feats, energy, bpm, stars, diff_name, device):
    print(f"   → {diff_name} ({stars:.2f}★)")

    duration_sec = mel_spec.shape[0] * (512 / 22050)

    selected  = _phase1_timing(timing_model, ctx_feats, energy, bpm, stars, device, diff_name)
    dot_notes = _phase2_place(place_model, selected, mel_spec, ctx_feats,
                              bpm, stars, device)

    # ── NoteFilter: limpa densidade ANTES do AngleNet ─────────────
    # O AngleNet recebe apenas notas que são fisicamente possíveis de jogar.
    # Isso evita que o histórico de paridade seja poluído por notas fantasma.
    dot_notes = filter_notes(dot_notes, bpm, diff_name, duration_sec)
    filter_report(dot_notes, bpm, duration_sec, diff_name)

    # ── Constraints: posição correta por mão, resolve layer 2 + UP ─
    dot_notes = apply_constraints(dot_notes)

    angled  = _phase3_angle(angle_model, dot_notes, stars, device)

    # ── Fix de DOTs órfãos: notas que escaparam sem ângulo ────────
    angled  = fix_orphan_dots(angled)

    refined = _phase4_view(view_model, angle_model, angled, stars, device)

    # FlowFixer — validação física final
    all_obj = FlowFixer.fix(refined, bpm)
    notes   = [o for o in all_obj if o['_type'] != FlowFixer.BOMB]
    bombs   = [o for o in all_obj if o['_type'] == FlowFixer.BOMB]

    notes.sort(key=lambda x: x['_time'])
    bombs = list({tuple(sorted(d.items())): d for d in bombs}.values())
    bombs.sort(key=lambda x: x['_time'])

    print(f"      Notas finais: {len(notes)}")
    return notes, bombs


# ─────────────────────────────────────────────────────────────────
# I/O
# ─────────────────────────────────────────────────────────────────

def _save_diff(notes, bombs, folder, filename):
    with open(os.path.join(folder, filename), 'w', encoding='utf-8') as f:
        json.dump({
            "_version":   "2.2.0",
            "_notes":     sorted(notes + bombs, key=lambda x: x['_time']),
            "_obstacles": [], "_events": [],
        }, f)


def _create_info(song_name, bpm, difficulties):
    diffs = []
    for name in difficulties:
        p = DIFFICULTY_PARAMS[name]
        diffs.append({
            "_difficulty":              name,
            "_beatmapFilename":         f"{name}.dat",
            "_noteJumpMovementSpeed":   p["njs"],
            "_noteJumpStartBeatOffset": p["offset"],
        })
    return {
        "_version": "2.1.0", "_songName": song_name,
        "_songSubName": "AI Generated", "_songAuthorName": "Artist",
        "_levelAuthorName": "BSAIMapper V6",
        "_beatsPerMinute": float(bpm),
        "_songFilename": "song.egg", "_coverImageFilename": "cover.png",
        "_environmentName": "DefaultEnvironment", "_songTimeOffset": 0,
        "_difficultyBeatmapSets": [{
            "_beatmapCharacteristicName": "Standard",
            "_difficultyBeatmaps": diffs,
        }],
    }


def _load_models(device):
    print("  Modelos:")
    tm = get_timing_model().to(device)
    pm = get_place_model().to(device)
    am = get_angle_model().to(device)
    vm = get_view_model().to(device)

    def load(model, name, required=True):
        path = os.path.join(MODELS_DIR, f"{name}_best.pth")
        if os.path.exists(path):
            model.load_state_dict(torch.load(path, map_location=device, weights_only=True))
            model.eval()
            print(f"    ✅ {name}_best.pth")
            return True
        msg = f"❌ {name}_best.pth — treine primeiro." if required \
              else f"⚠  {name}_best.pth — não encontrado (etapa opcional ignorada)."
        print(f"    {msg}")
        return False

    ok  = load(tm, "timing_net")
    ok &= load(pm, "place_net")
    ok &= load(am, "angle_net")
    view_ok = load(vm, "view_net", required=False)
    return tm, pm, am, (vm if view_ok else None), ok


# ─────────────────────────────────────────────────────────────────
# Entry point
# ─────────────────────────────────────────────────────────────────

def main():
    p = argparse.ArgumentParser(
        description="Gera mapas Beat Saber com IA a partir de uma URL do YouTube.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Exemplos:
  python generate.py
      Modo interativo — pergunta URL, dificuldades e estrelas.

  python generate.py --url https://youtu.be/xxx --diffs 3,4,5 --stars 4.5,6.5,8.5
      Gera Hard/Expert/ExpertPlus com estrelas definidas.

  python generate.py --url https://youtu.be/xxx --diffs 4
      Gera só Expert com estrelas padrão (6.0★).

Dificuldades: 1=Easy  2=Normal  3=Hard  4=Expert  5=ExpertPlus
        """,
    )
    p.add_argument("--url",   type=str)
    p.add_argument("--diffs", type=str)
    p.add_argument("--stars", type=str)
    args = p.parse_args()

    print("=" * 60)
    print("  BS AI MAPPER V6 — Gerador de Mapas")
    print("=" * 60)

    url = args.url or input("\n  URL da música (YouTube): ").strip()
    if not url:
        sys.exit("  URL inválida.")

    raw_diffs = args.diffs
    if not raw_diffs:
        print("\n  Dificuldades: 1=Easy  2=Normal  3=Hard  4=Expert  5=ExpertPlus")
        raw_diffs = input("  Escolha (ex: 3,4,5): ").strip()

    selected = []
    for x in raw_diffs.split(','):
        try:
            v = int(x.strip())
            if v in DIFFICULTY_MAP:
                selected.append(DIFFICULTY_MAP[v])
        except ValueError:
            pass
    if not selected:
        print("  Inválido — usando Expert.")
        selected = ["Expert"]
    selected.sort(key=lambda x: list(DIFFICULTY_MAP.values()).index(x))

    defaults = {"Easy": 2.0, "Normal": 3.0, "Hard": 4.5,
                "Expert": 6.0, "ExpertPlus": 8.0}
    difficulties = {}

    if args.stars:
        try:
            vals = [float(x.strip()) for x in args.stars.split(',')]
            for i, d in enumerate(selected):
                difficulties[d] = vals[i] if i < len(vals) else vals[-1]
        except ValueError:
            pass

    if not difficulties:
        print("\n  Estrelas (Enter = padrão):")
        for d in selected:
            df = defaults.get(d, 5.0)
            while True:
                raw = input(f"    {d} [{df}★]: ").strip()
                if not raw:
                    difficulties[d] = df; break
                try:
                    v = float(raw)
                    if v > 0:
                        difficulties[d] = v; break
                except ValueError:
                    pass

    print("\n[1/4] Baixando áudio...")
    mp3, cover = download_from_youtube(url, "data/temp")
    if not mp3:
        sys.exit("  Falha no download.")

    song_name  = os.path.splitext(os.path.basename(mp3))[0]
    out_folder = os.path.join("output", song_name.replace(" ", "_"))
    if os.path.exists(out_folder):
        shutil.rmtree(out_folder)
    os.makedirs(out_folder)

    print("\n[2/4] Analisando áudio...")
    bpm = round(detect_bpm(mp3))
    print(f"  BPM: {bpm}")

    audio_out = os.path.join(out_folder, "song.egg")
    add_silence(mp3, audio_out)
    if cover and os.path.exists(cover):
        shutil.copy(cover, os.path.join(out_folder, "cover.png"))

    mel_spec, ctx_feats, sr, hop = extract_features(audio_out, bpm)
    energy = analyze_energy(audio_out)
    if len(energy) < mel_spec.shape[0]:
        energy = np.pad(energy, (0, mel_spec.shape[0] - len(energy)))

    print("\n[3/4] Carregando modelos...")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"  Dispositivo: {device}")
    tm, pm, am, vm, ok = _load_models(device)
    if not ok:
        sys.exit("  Modelos necessários não encontrados.")

    print("\n[4/4] Gerando mapas...")
    for diff_name, stars in difficulties.items():
        notes, bombs = generate_difficulty(
            tm, pm, am, vm,
            mel_spec, ctx_feats, energy,
            bpm, stars, diff_name, device,
        )
        _save_diff(notes, bombs, out_folder, f"{diff_name}.dat")

    info = _create_info(song_name, bpm, difficulties)
    with open(os.path.join(out_folder, "Info.dat"), 'w', encoding='utf-8') as f:
        json.dump(info, f, indent=2)

    zip_path = os.path.join("output", song_name.replace(" ", "_"))
    if os.path.exists(zip_path + ".zip"):
        os.remove(zip_path + ".zip")
    shutil.make_archive(zip_path, 'zip', out_folder)

    try:
        shutil.rmtree("data/temp")
    except Exception:
        pass

    print(f"\n  ✅ {zip_path}.zip")


if __name__ == "__main__":
    main()
