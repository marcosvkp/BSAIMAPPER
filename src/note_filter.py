"""
note_filter.py — Filtro de Densidade e Espaçamento V1

Roda APÓS o PlaceNet e ANTES do AngleNet.
É código determinístico puro — sem modelos, sem aleatoriedade.

Problemas que resolve:
  1. NPS absurdo: muitas notas em poucos beats
  2. Clusters: várias notas em <1/8 beat (impossível de jogar)
  3. Double spam: doubles em sequência sem respiro
  4. DOTs orphans: notas que escaparam do AngleNet sem ângulo (fix pós-AngleNet)

Regras por dificuldade (baseadas em mapas ranqueados reais):

  Dificuldade   Max NPS    Min gap (beats)   Max doubles seguidos
  Easy          2.5        1/2               1
  Normal        3.5        1/4               1
  Hard          5.0        1/8               2
  Expert        7.0        1/16              3
  ExpertPlus    sem limite  1/16             sem limite

Min gap é o intervalo MÍNIMO entre duas notas QUALQUER (mesma ou mão diferente).
O gap entre duas notas da MESMA MÃO é sempre >= 2× o min gap.
"""

from __future__ import annotations
from typing import List, Dict

# ─────────────────────────────────────────────────────────────────
# Perfis por dificuldade
# ─────────────────────────────────────────────────────────────────

_PROFILES = {
    #               max_nps  min_gap  same_hand_mult  max_consec_doubles
    "Easy":        (2.5,     0.500,   2.0,            1),
    "Normal":      (3.5,     0.250,   2.0,            1),
    "Hard":        (5.0,     0.125,   2.0,            2),
    "Expert":      (7.0,     0.125,   1.5,            3),
    "ExpertPlus":  (12.0,    0.125,   1.5,            99),
}

_DEFAULT_PROFILE = (7.0, 0.0625, 1.5, 3)


def _profile(diff_name: str):
    return _PROFILES.get(diff_name, _DEFAULT_PROFILE)


# ─────────────────────────────────────────────────────────────────
# Filtro principal
# ─────────────────────────────────────────────────────────────────

def filter_notes(notes: List[Dict], bpm: float, diff_name: str,
                 duration_sec: float | None = None) -> List[Dict]:
    """
    Filtra e limpa a lista de notas aplicando regras de densidade e espaçamento.

    Args:
        notes        : lista de dicts com _time, _type, _lineIndex, _lineLayer, _cutDirection
        bpm          : BPM da música
        diff_name    : nome da dificuldade ("Easy", "Normal", etc.)
        duration_sec : duração da música em segundos (usado para calcular NPS real)

    Returns:
        Lista filtrada, ordenada por _time.
    """
    if not notes:
        return []

    max_nps, min_gap, same_mult, max_dbl = _profile(diff_name)
    notes = sorted(notes, key=lambda n: n['_time'])

    # ── Passo 1: Remove notas com gap mínimo global violado ───────
    # Garante que nenhuma nota chega a menos de min_gap beats da anterior
    notes = _enforce_min_gap(notes, min_gap)

    # ── Passo 2: Garante gap mínimo por mão ───────────────────────
    # A mesma mão não pode aparecer a menos de same_mult × min_gap beats
    notes = _enforce_same_hand_gap(notes, min_gap * same_mult)

    # ── Passo 3: Limita doubles consecutivos ─────────────────────
    # Evita spam de doubles que cansa o jogador
    notes = _limit_consecutive_doubles(notes, max_dbl)

    # ── Passo 4: Limita NPS global ────────────────────────────────
    if duration_sec and duration_sec > 0:
        notes = _cap_nps(notes, max_nps, bpm, duration_sec)

    total_removed = 0  # já contabilizado internamente
    return notes


# ─────────────────────────────────────────────────────────────────
# Regras individuais
# ─────────────────────────────────────────────────────────────────

def _enforce_min_gap(notes: List[Dict], min_gap: float) -> List[Dict]:
    """Remove notas que chegam a menos de min_gap beats da nota anterior."""
    if not notes:
        return []
    result = [notes[0]]
    removed = 0
    for note in notes[1:]:
        gap = note['_time'] - result[-1]['_time']
        if gap >= min_gap - 1e-6:
            result.append(note)
        else:
            removed += 1
    if removed:
        print(f"      [NoteFilter] min_gap: removidas {removed} notas ({min_gap:.4f} beats)")
    return result


def _enforce_same_hand_gap(notes: List[Dict], min_gap: float) -> List[Dict]:
    """
    Garante que cada mão não aparece duas vezes em menos de min_gap beats.
    Ao encontrar violação, remove a nota mais recente da sequência.
    """
    last_beat = {0: -999.0, 1: -999.0}
    result = []
    removed = 0
    for note in notes:
        hand = note['_type']
        if hand not in (0, 1):
            result.append(note)
            continue
        gap = note['_time'] - last_beat[hand]
        if gap >= min_gap - 1e-6:
            result.append(note)
            last_beat[hand] = note['_time']
        else:
            removed += 1
    if removed:
        print(f"      [NoteFilter] same_hand_gap: removidas {removed} notas")
    return result


def _limit_consecutive_doubles(notes: List[Dict], max_consec: int) -> List[Dict]:
    """
    Limita doubles consecutivos. Um 'double' é quando duas notas têm o mesmo
    _time (ou diferença <= 1/16 beat). Após max_consec doubles seguidos,
    remove o excedente até o próximo momento sem double.
    """
    if max_consec >= 99:
        return notes

    DOUBLE_THRESH = 0.0625  # 1/16 beat

    result = []
    i = 0
    consec_doubles = 0
    removed = 0

    while i < len(notes):
        # Detecta se esta posição forma um double com a próxima
        if (i + 1 < len(notes) and
                abs(notes[i+1]['_time'] - notes[i]['_time']) <= DOUBLE_THRESH):
            if consec_doubles < max_consec:
                result.append(notes[i])
                result.append(notes[i+1])
                consec_doubles += 1
            else:
                removed += 2
            i += 2
        else:
            result.append(notes[i])
            consec_doubles = 0
            i += 1

    if removed:
        print(f"      [NoteFilter] double_spam: removidos {removed} doubles consecutivos")
    return result


def _cap_nps(notes: List[Dict], max_nps: float,
             bpm: float, duration_sec: float) -> List[Dict]:
    """
    Se o NPS real ultrapassar max_nps, remove notas com score de prioridade menor.
    Prioriza: notas com maior gap para as vizinhas (mais "no beat"),
    eliminando as que estão entre outras duas muito próximas.
    """
    current_nps = len(notes) / duration_sec
    if current_nps <= max_nps:
        return notes

    target_n = int(max_nps * duration_sec)
    to_remove = len(notes) - target_n

    if to_remove <= 0:
        return notes

    # Calcula "importância" de cada nota: maior gap para vizinhas = mais importante
    scores = []
    for i, note in enumerate(notes):
        prev_gap = note['_time'] - notes[i-1]['_time'] if i > 0 else 999.0
        next_gap = notes[i+1]['_time'] - note['_time'] if i < len(notes)-1 else 999.0
        scores.append(min(prev_gap, next_gap))

    # Remove as notas com menor score (mais apertadas entre vizinhas)
    indexed = sorted(enumerate(scores), key=lambda x: x[1])
    remove_set = set(idx for idx, _ in indexed[:to_remove])
    result = [n for i, n in enumerate(notes) if i not in remove_set]

    print(f"      [NoteFilter] NPS cap: {current_nps:.1f} → {len(result)/duration_sec:.1f} NPS "
          f"(removidas {to_remove} notas)")
    return result


# ─────────────────────────────────────────────────────────────────
# Fix de DOTs órfãos (pós-AngleNet)
# ─────────────────────────────────────────────────────────────────

_VERT_COMP = {
    0: 'up', 1: 'down', 4: 'up', 5: 'up', 6: 'down', 7: 'down',
}
_AFTER_UP   = [7, 6, 1]   # DOWN_RIGHT, DOWN_LEFT, DOWN
_AFTER_DOWN = [4, 5, 0]   # UP_LEFT, UP_RIGHT, UP
_FALLBACK   = [7, 4, 1, 0]


def fix_orphan_dots(notes: List[Dict]) -> List[Dict]:
    """
    Substitui DOTs (cut=8) que ainda restaram após o AngleNet por uma
    direção compatível com a nota anterior da mesma mão.

    Chamado APÓS o AngleNet, como último recurso antes do FlowFixer.
    """
    last_cut = {0: None, 1: None}
    fixed = 0

    for note in sorted(notes, key=lambda n: n['_time']):
        hand = note['_type']
        if hand not in (0, 1):
            continue

        cut = note['_cutDirection']

        if cut == 8:  # DOT — precisa de ângulo
            prev = last_cut[hand]
            prev_vert = _VERT_COMP.get(prev) if prev is not None else None

            if prev_vert == 'up':
                note['_cutDirection'] = _AFTER_UP[0]    # DOWN_RIGHT
            elif prev_vert == 'down':
                note['_cutDirection'] = _AFTER_DOWN[0]  # UP_LEFT
            else:
                note['_cutDirection'] = _FALLBACK[0]    # DOWN_RIGHT (padrão)
            fixed += 1
            last_cut[hand] = note['_cutDirection']
        else:
            last_cut[hand] = cut

    if fixed:
        print(f"      [NoteFilter] orphan_dots: corrigidos {fixed} DOTs sem ângulo")
    return notes


# ─────────────────────────────────────────────────────────────────
# Relatório
# ─────────────────────────────────────────────────────────────────

def report(notes: List[Dict], bpm: float, duration_sec: float,
           diff_name: str = ""):
    """Imprime estatísticas da lista de notas."""
    if not notes:
        print("      [NoteFilter] 0 notas")
        return

    nps  = len(notes) / max(duration_sec, 0.1)
    gaps = []
    for i in range(1, len(notes)):
        gaps.append(notes[i]['_time'] - notes[i-1]['_time'])
    min_gap = min(gaps) if gaps else 0.0
    avg_gap = sum(gaps) / len(gaps) if gaps else 0.0

    hand_counts = {0: 0, 1: 0}
    for n in notes:
        if n['_type'] in hand_counts:
            hand_counts[n['_type']] += 1
    dots = sum(1 for n in notes if n['_cutDirection'] == 8)

    label = f" [{diff_name}]" if diff_name else ""
    print(f"      [NoteFilter]{label} {len(notes)} notas | "
          f"{nps:.2f} NPS | "
          f"gap min={min_gap:.3f} avg={avg_gap:.3f} beats | "
          f"L={hand_counts[0]} R={hand_counts[1]} | "
          f"DOTs={dots}")
