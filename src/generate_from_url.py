import os
import json
import shutil
import torch
import numpy as np
from collections import deque
from youtube_downloader import download_from_youtube
from models_optimized import get_timing_model, get_note_model, get_flow_model, NOTE_HISTORY, FLOW_CONTEXT
from flow_fixer import FlowFixer
from audio_processor import extract_features, detect_bpm, add_silence, analyze_energy

DIFFICULTY_MAP = {1: "Easy", 2: "Normal", 3: "Hard", 4: "Expert", 5: "ExpertPlus"}
DIFFICULTY_PARAMS = {
    "Easy":       {"njs": 10, "offset":  0.0},
    "Normal":     {"njs": 12, "offset":  0.0},
    "Hard":       {"njs": 14, "offset": -0.2},
    "Expert":     {"njs": 16, "offset": -0.4},
    "ExpertPlus": {"njs": 18, "offset": -0.7},
}

_PADDING_NOTE = np.array([0, 1, 0, 8], dtype=np.float32)


def _build_note_dict(beat_time, hand, col, layer, cut):
    return {
        '_time':         float(beat_time),
        '_lineIndex':    int(col),
        '_lineLayer':    int(layer),
        '_type':         int(hand),
        '_cutDirection': int(cut),
    }


def _encode_history(history_deque):
    arr = np.zeros(NOTE_HISTORY * 4, dtype=np.float32)
    for i, note_vec in enumerate(history_deque):
        arr[i*4:(i+1)*4] = note_vec
    return arr


def _apply_flow_net(flow_model, notes, features, target_stars, device,
                    num_frames, frame_dur, bpm):
    """
    Passa o mapa gerado pelo NoteNet pelo FlowNet para refinamento.

    Para cada nota, monta uma janela de contexto de FLOW_CONTEXT*2+1 notas,
    consulta o FlowNet e aplica as correções sugeridas.

    Só aplica correção quando o modelo tem alta confiança (threshold > 0.75)
    para não sobre-corrigir notas que já estão certas.
    """
    if not notes or flow_model is None:
        return notes

    CORRECTION_THRESHOLD = 0.75  # confiança mínima para aplicar correção
    window_size = FLOW_CONTEXT * 2 + 1
    N = len(notes)

    corrected = [dict(n) for n in notes]  # copia para não mudar originais
    seconds_per_beat = 60.0 / bpm

    corrections = {'hand': 0, 'cut': 0, 'col': 0}

    flow_model.eval()
    with torch.no_grad():
        for i in range(N):
            # ── Monta janela de contexto ──────────────────────────
            window_flat = np.zeros(window_size * 4, dtype=np.float32)
            for w in range(window_size):
                note_idx = i - FLOW_CONTEXT + w
                if 0 <= note_idx < N:
                    n = corrected[note_idx]
                    window_flat[w*4]   = float(n['_type'])
                    window_flat[w*4+1] = float(n['_lineIndex'])
                    window_flat[w*4+2] = float(n['_lineLayer'])
                    window_flat[w*4+3] = float(n['_cutDirection'])
                else:
                    window_flat[w*4:(w+1)*4] = _PADDING_NOTE

            # ── Áudio local ────────────────────────────────────────
            beat_time = corrected[i]['_time']
            time_sec  = beat_time * seconds_per_beat
            frame_idx = min(int(time_sec / frame_dur), num_frames - 1)
            audio_local = features[frame_idx]

            # ── Inferência ─────────────────────────────────────────
            ctx_t   = torch.tensor(window_flat,  dtype=torch.float32).unsqueeze(0).to(device)
            audio_t = torch.tensor(audio_local,  dtype=torch.float32).unsqueeze(0).to(device)
            stars_t = torch.tensor([[target_stars]], dtype=torch.float32).to(device)

            out = flow_model(ctx_t, audio_t, stars_t)

            def confidence(logits):
                probs = torch.softmax(logits.squeeze(), dim=-1)
                return probs[1].item()  # prob de "precisa corrigir"

            def argmax(logits):
                return torch.argmax(logits.squeeze(), dim=-1).item()

            # ── Aplica correções com threshold de confiança ────────
            if confidence(out['hand_ok']) > CORRECTION_THRESHOLD:
                corrected[i]['_type'] = argmax(out['new_hand'])
                corrections['hand'] += 1

            if confidence(out['cut_ok']) > CORRECTION_THRESHOLD:
                corrected[i]['_cutDirection'] = argmax(out['new_cut'])
                corrections['cut'] += 1

            if confidence(out['col_ok']) > CORRECTION_THRESHOLD:
                corrected[i]['_lineIndex'] = argmax(out['new_col'])
                corrections['col'] += 1

    total_corrections = sum(corrections.values())
    if total_corrections > 0:
        print(f"      FlowNet: {total_corrections} correções "
              f"(mão={corrections['hand']} cut={corrections['cut']} col={corrections['col']})")

    return corrected


def generate_difficulty(timing_model, note_model, flow_model,
                        features, energy_profile,
                        bpm, sr, hop_length, difficulty_name, target_stars):
    print(f"   -> Gerando: {difficulty_name} ({target_stars:.2f}★)...")

    device     = next(timing_model.parameters()).device
    frame_dur  = hop_length / sr
    num_frames = len(features)

    # ── FASE 1: TimingNet ─────────────────────────────────────────
    inputs  = torch.tensor(features, dtype=torch.float32).unsqueeze(0).to(device)
    stars_t = torch.tensor([[target_stars]], dtype=torch.float32).to(device)

    with torch.no_grad():
        beat_probs = torch.sigmoid(
            timing_model(inputs, stars_t)
        ).squeeze().cpu().numpy()

    ai_conf    = float(np.mean(beat_probs))
    base_nps   = 1.2 + ai_conf * 6.5
    star_boost = 1.0 + np.clip((target_stars - 5.0) * 0.08, -0.2, 0.4)
    target_nps = base_nps * star_boost
    duration   = num_frames * frame_dur
    target_n   = int(max(80, duration * target_nps))

    print(f"      TimingNet: {target_nps:.2f} NPS → ~{target_n} notas alvo")

    min_frac  = 8.0 if target_stars > 7 else 4.0
    cooldown  = int((60.0 / bpm / min_frac) / frame_dur)
    MIN_START = 2.0

    candidates = []
    for i in range(1, num_frames - 1):
        if i * frame_dur < MIN_START:
            continue
        if beat_probs[i] > beat_probs[i-1] and beat_probs[i] > beat_probs[i+1]:
            score = beat_probs[i] * (0.8 + 0.4 * energy_profile[i])
            candidates.append((score, i))

    candidates.sort(reverse=True)
    selected = []
    for score, idx in candidates:
        if len(selected) >= target_n:
            break
        if all(abs(idx - s) >= cooldown for s in selected):
            selected.append(idx)
    selected.sort()
    print(f"      Frames selecionados: {len(selected)}")

    # ── FASE 2: NoteNet ───────────────────────────────────────────
    note_model.eval()
    history   = deque([_PADDING_NOTE.copy() for _ in range(NOTE_HISTORY)], maxlen=NOTE_HISTORY)
    raw_notes = []
    last_beat = -999.0

    with torch.no_grad():
        for idx in selected:
            audio_local = torch.tensor(
                features[idx], dtype=torch.float32
            ).unsqueeze(0).unsqueeze(0).to(device)

            history_vec = torch.tensor(
                _encode_history(history), dtype=torch.float32
            ).unsqueeze(0).unsqueeze(0).to(device)

            out = note_model(audio_local, history_vec, stars_t)

            def sample(logits, temperature=0.8):
                logits = logits.squeeze() / temperature
                probs  = torch.softmax(logits, dim=-1)
                return torch.multinomial(probs, 1).item()

            hand   = sample(out['hand'])
            col    = sample(out['col'])
            layer  = sample(out['layer'])
            cut    = sample(out['cut'])
            double = sample(out['double'], temperature=1.0)

            current_beat = (idx * frame_dur) / (60.0 / bpm)
            beat_time    = round(current_beat * 16) / 16.0

            if beat_time <= last_beat:
                continue

            note = _build_note_dict(beat_time, hand, col, layer, cut)
            raw_notes.append(note)
            history.append(np.array([hand, col, layer, cut], dtype=np.float32))
            last_beat = beat_time

            if double == 1:
                double_beat  = round((beat_time + 1/16) * 16) / 16.0
                double_hand  = 1 - hand
                double_col   = sample(out['double_col'])
                double_layer = sample(out['double_layer'])
                double_cut   = sample(out['double_cut'])
                note2 = _build_note_dict(double_beat, double_hand,
                                          double_col, double_layer, double_cut)
                raw_notes.append(note2)
                history.append(np.array([double_hand, double_col,
                                          double_layer, double_cut], dtype=np.float32))
                last_beat = double_beat

    print(f"      Notas brutas (NoteNet): {len(raw_notes)}")

    # ── FASE 3: FlowNet refina o mapa ─────────────────────────────
    refined_notes = _apply_flow_net(
        flow_model, raw_notes, features, target_stars,
        device, num_frames, frame_dur, bpm
    )

    # ── FASE 4: FlowFixer valida fisicamente ──────────────────────
    all_obj = FlowFixer.fix(refined_notes, bpm)
    notes   = [o for o in all_obj if o['_type'] != FlowFixer.BOMB]
    bombs   = [o for o in all_obj if o['_type'] == FlowFixer.BOMB]

    notes.sort(key=lambda x: x['_time'])
    bombs = [dict(t) for t in {tuple(d.items()) for d in bombs}]
    bombs.sort(key=lambda x: x['_time'])

    print(f"      Notas finais: {len(notes)}")
    return notes, bombs


def create_info_dat(song_name, bpm, audio_filename, cover_filename, difficulties_data):
    beatmap_sets = []
    for diff_name in difficulties_data:
        p = DIFFICULTY_PARAMS[diff_name]
        beatmap_sets.append({
            "_difficulty":              diff_name,
            "_beatmapFilename":         f"{diff_name}.dat",
            "_noteJumpMovementSpeed":   p["njs"],
            "_noteJumpStartBeatOffset": p["offset"],
        })
    return {
        "_version":              "2.1.0",
        "_songName":             song_name,
        "_songSubName":          "AI Generated",
        "_songAuthorName":       "Artist",
        "_levelAuthorName":      "BSIAMapperV3",
        "_beatsPerMinute":       float(bpm),
        "_songFilename":         audio_filename,
        "_coverImageFilename":   cover_filename,
        "_environmentName":      "DefaultEnvironment",
        "_songTimeOffset":       0,
        "_difficultyBeatmapSets": [{
            "_beatmapCharacteristicName": "Standard",
            "_difficultyBeatmaps": beatmap_sets,
        }],
    }


def save_difficulty_dat(notes, bombs, folder, filename):
    data = {
        "_version":   "2.2.0",
        "_notes":     sorted(notes + bombs, key=lambda x: x['_time']),
        "_obstacles": [],
        "_events":    [],
    }
    with open(os.path.join(folder, filename), 'w') as f:
        json.dump(data, f)


def main():
    print("=" * 60)
    print("   BEAT SABER AI MAPPER V3 — TimingNet + NoteNet + FlowNet")
    print("=" * 60)

    url = input("\nURL da música (YouTube): ").strip()
    if not url:
        print("URL inválida.")
        return

    print("\n[1/5] Baixando áudio...")
    mp3_path, cover_path = download_from_youtube(url, output_folder="data/temp_download")
    if not mp3_path:
        print("Falha no download.")
        return

    song_name = os.path.splitext(os.path.basename(mp3_path))[0]

    print("\n[2/5] Configuração de dificuldades")
    print("Dificuldades (1=Easy 2=Normal 3=Hard 4=Expert 5=ExpertPlus)")
    choices = input("Exemplo: 3,4,5 → ").strip()

    selected_diffs = []
    try:
        for p in choices.split(','):
            v = int(p.strip())
            if v in DIFFICULTY_MAP:
                selected_diffs.append(DIFFICULTY_MAP[v])
    except Exception:
        pass
    if not selected_diffs:
        print("Seleção inválida. Usando Expert.")
        selected_diffs = ["Expert"]

    order = list(DIFFICULTY_MAP.values())
    selected_diffs.sort(key=lambda x: order.index(x))

    difficulties = {}
    print("\n[3/5] Estrelas por dificuldade:")
    for diff in selected_diffs:
        while True:
            try:
                s = float(input(f"  {diff}: ").strip())
                if s > 0:
                    difficulties[diff] = s
                    break
            except ValueError:
                pass

    output_folder = os.path.join("output", song_name.replace(" ", "_"))
    if os.path.exists(output_folder):
        shutil.rmtree(output_folder)
    os.makedirs(output_folder)

    print("\n[4/5] Analisando áudio...")
    bpm = round(detect_bpm(mp3_path))
    print(f"BPM detectado: {bpm}")

    audio_out = os.path.join(output_folder, "song.egg")
    add_silence(mp3_path, audio_out)
    if cover_path and os.path.exists(cover_path):
        shutil.copy(cover_path, os.path.join(output_folder, "cover.png"))

    features, sr, hop_length = extract_features(audio_out, bpm)
    energy = analyze_energy(audio_out, hop_length=hop_length, sr=sr)
    if len(energy) < len(features):
        energy = np.pad(energy, (0, len(features) - len(energy)))

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Dispositivo: {device}")

    timing_model = get_timing_model().to(device)
    note_model   = get_note_model().to(device)
    flow_model   = get_flow_model().to(device)

    def load_model(model, path, required=True):
        if os.path.exists(path):
            model.load_state_dict(torch.load(path, map_location=device))
            model.eval()
            print(f"  ✅ Carregado: {path}")
            return True
        else:
            if required:
                print(f"  ❌ Não encontrado: {path} — treine primeiro.")
            else:
                print(f"  ⚠️  FlowNet não encontrado ({path}) — gerando sem refinamento.")
            return False

    ok  = load_model(timing_model, "models/timing_net_best.pth")
    ok &= load_model(note_model,   "models/note_net_best.pth")
    # FlowNet é opcional — se não existir, gera sem refinamento
    flow_ok = load_model(flow_model, "models/flow_net_best.pth", required=False)
    if not flow_ok:
        flow_model = None

    if not ok:
        return

    print("\n[5/5] Gerando mapas...")
    for diff_name, stars in difficulties.items():
        notes, bombs = generate_difficulty(
            timing_model, note_model, flow_model,
            features, energy, bpm, sr, hop_length, diff_name, stars
        )
        save_difficulty_dat(notes, bombs, output_folder, f"{diff_name}.dat")

    info = create_info_dat(song_name, bpm, "song.egg", "cover.png", difficulties)
    with open(os.path.join(output_folder, "Info.dat"), 'w') as f:
        json.dump(info, f, indent=2)

    zip_out = os.path.join("output", song_name.replace(" ", "_"))
    if os.path.exists(zip_out + ".zip"):
        os.remove(zip_out + ".zip")
    shutil.make_archive(zip_out, 'zip', output_folder)

    try:
        shutil.rmtree("data/temp_download")
    except Exception:
        pass

    print(f"\n✅ Mapa gerado: output/{song_name.replace(' ', '_')}.zip")


if __name__ == "__main__":
    main()