"""
data_loader.py — Carregamento e encoding de dados V5

Gera os arrays NumPy que alimentam os três modelos:

  _mel.npy    : (num_frames, 64)  espectrograma mel por frame
  _ctx.npy    : (num_frames, 8)   features de contexto por frame
  _timing.npy : (num_frames,)     1.0 se há nota no frame (suavizado)
  _notes.npy  : (N, 8)            sequência de notas
                  [hand, col, layer, cut, beat_norm, beat_gap, col_norm, layer_norm]
  _stars.npy  : (1,)              estrelas ScoreSaber

Separando mel e ctx em arquivos distintos:
  - Permite carregar apenas ctx quando PatternModel não precisa do mel completo
  - mmap eficiente: mel é 8x maior que ctx
"""

import os
import numpy as np
from audio_processor import extract_features
from parser.loader import load_specific_difficulty
from parser.enums import NoteColor

# ─────────────────────────────────────────────────────────────────
# Encoding
# ─────────────────────────────────────────────────────────────────

# Nota de padding — usada ao construir histórico no início de uma sequência
# hand=0, col=1, layer=0, cut=8(DOT)
PADDING_NOTE = np.array([0, 1, 0, 8, 0.0, 0.5, 1/3, 0.0], dtype=np.float32)


def encode_note(note_obj) -> np.ndarray | None:
    """
    Converte um objeto Note do parser em vetor de 8 floats:
      [hand, col, layer, cut, beat_norm, beat_gap, col_norm, layer_norm]

    beat_norm e beat_gap são preenchidos por create_dataset_entry.
    Bombas retornam None — não são incluídas no dataset de notas.
    """
    if note_obj.c == NoteColor.BOMB:
        return None

    hand  = 0 if note_obj.c == NoteColor.RED else 1
    col   = int(np.clip(note_obj.x, 0, 3))
    layer = int(np.clip(note_obj.y, 0, 2))
    cut   = int(note_obj.d) if int(note_obj.d) <= 8 else 8

    return np.array([
        hand, col, layer, cut,
        0.0,        # beat_norm  — preenchido depois
        0.5,        # beat_gap   — preenchido depois
        col / 3.0,  # col_norm
        layer / 2.0 # layer_norm
    ], dtype=np.float32)


def create_dataset_entry(map_folder: str, difficulty_filename: str,
                         difficulty_name: str, stars: float):
    """
    Cria uma entrada de dataset para os três modelos a partir de um mapa.

    Retorna:
        (mel_spec, ctx_feats, timing_targets, note_sequence, stars_val)
        ou None em caso de erro.

    note_sequence shape: (N, 8)
        col 0: hand
        col 1: col
        col 2: layer
        col 3: cut
        col 4: beat_norm     — beat / total_beats  (posição relativa na música)
        col 5: beat_gap      — beats desde nota anterior (normalizado por BPM)
        col 6: col_norm      — col / 3.0
        col 7: layer_norm    — layer / 2.0
    """
    load_result = load_specific_difficulty(map_folder, difficulty_filename)
    if load_result is None:
        return None

    beatmap, bpm, audio_path = load_result
    mel_spec, ctx_feats, sr, hop_length = extract_features(audio_path, bpm)
    if mel_spec is None:
        return None

    num_frames     = mel_spec.shape[0]
    frame_dur      = hop_length / sr
    secs_per_beat  = 60.0 / float(bpm)

    # Ordena notas (sem bombas) por tempo
    real_notes = sorted(
        [n for n in beatmap.notes if n.c != NoteColor.BOMB],
        key=lambda n: n.b,
    )

    if len(real_notes) < 8:
        return None

    total_beats = real_notes[-1].b + 1.0

    # ── Timing targets ────────────────────────────────────────────
    timing_targets = np.zeros(num_frames, dtype=np.float32)
    for note in real_notes:
        fidx = int((note.b * secs_per_beat) / frame_dur)
        if fidx < num_frames:
            timing_targets[fidx] = 1.0
            # Suavização gaussiana leve ±1 frame
            if fidx + 1 < num_frames:
                timing_targets[fidx + 1] = max(timing_targets[fidx + 1], 0.3)
            if fidx - 1 >= 0:
                timing_targets[fidx - 1] = max(timing_targets[fidx - 1], 0.3)

    # ── Note sequence ─────────────────────────────────────────────
    note_list = []
    prev_beat  = 0.0
    for note in real_notes:
        vec = encode_note(note)
        if vec is None:
            continue
        beat_norm = float(note.b) / max(total_beats, 1.0)
        beat_gap  = min(float(note.b - prev_beat) / max(secs_per_beat, 0.01), 8.0) / 8.0
        vec[4] = beat_norm
        vec[5] = beat_gap
        note_list.append(vec)
        prev_beat = float(note.b)

    if len(note_list) < 8:
        return None

    note_sequence = np.stack(note_list, axis=0)  # (N, 8)

    return mel_spec, ctx_feats, timing_targets, note_sequence, float(stars)
