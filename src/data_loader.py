import os
import numpy as np
from audio_processor import extract_features
from parser.loader import load_specific_difficulty
from parser.enums import NoteColor

# ─────────────────────────────────────────────────────────────────
# Encoding de notas para o NoteNet
#
# Cada nota é representada por 5 valores inteiros:
#   has_note  : 0 ou 1
#   hand      : 0=esquerda, 1=direita
#   col       : 0-3 (lineIndex)
#   layer     : 0-2 (lineLayer)
#   cut       : 0-8 (cutDirection)
#
# O histórico das últimas NOTE_HISTORY notas é concatenado com as
# features de áudio no input do NoteNet.
# ─────────────────────────────────────────────────────────────────

NOTE_HISTORY  = 8   # quantas notas anteriores o NoteNet enxerga
NOTE_FEATURES = 5   # (has_note, hand, col, layer, cut)

# Nota de padding (usada para preencher o histórico no início)
# has_note=0 sinaliza "sem nota" — o modelo aprende a ignorar padding
PADDING_NOTE = np.array([0, 0, 1, 0, 8], dtype=np.float32)  # hand=L, col=1, layer=0, cut=DOT


def encode_note(note_obj):
    """
    Converte um objeto Note do parser para vetor de 5 floats.
    Bombas são codificadas como has_note=0 (ignoradas pelo NoteNet).
    """
    if note_obj.c == NoteColor.BOMB:
        return None  # bombas não são treinadas no NoteNet

    hand  = 0 if note_obj.c == NoteColor.RED else 1
    col   = int(np.clip(note_obj.x, 0, 3))
    layer = int(np.clip(note_obj.y, 0, 2))
    cut   = int(note_obj.d) if int(note_obj.d) <= 8 else 8

    return np.array([1, hand, col, layer, cut], dtype=np.float32)


def create_dataset_entry(map_folder: str, difficulty_filename: str,
                         difficulty_name: str, stars: float):
    """
    Cria uma entrada de dataset para os dois modelos:

    TimingNet targets:
        timing_targets : ndarray (num_frames,)  — 1.0 se há nota no frame, 0.0 caso contrário
                         Suavizado com gaussiana leve para treino mais estável

    NoteNet targets:
        note_sequence  : ndarray (N, 5)  — sequência ordenada de notas (has_note=1)
                         N = número de notas no mapa
                         Cada linha: [hand, col, layer, cut, beat_time_normalizado]
                         beat_time_normalizado = beat / total_beats (0→1)

    Returns:
        (audio_features, timing_targets, note_sequence, stars) ou None
    """
    load_result = load_specific_difficulty(map_folder, difficulty_filename)
    if load_result is None:
        return None

    beatmap, bpm, audio_path = load_result
    features, sr, hop_length = extract_features(audio_path, bpm)
    if features is None:
        return None

    num_frames    = features.shape[0]
    frame_duration = hop_length / sr
    seconds_per_beat = 60.0 / float(bpm)

    # ── Timing targets (para TimingNet) ──────────────────────────
    timing_targets = np.zeros(num_frames, dtype=np.float32)

    # Ordena todas as notas (sem bombas) por tempo
    real_notes = sorted(
        [n for n in beatmap.notes if n.c != NoteColor.BOMB],
        key=lambda n: n.b
    )

    if not real_notes:
        return None

    total_beats = real_notes[-1].b + 1.0  # normalização de tempo

    for note in real_notes:
        time_sec   = note.b * seconds_per_beat
        frame_idx  = int(time_sec / frame_duration)
        if frame_idx < num_frames:
            timing_targets[frame_idx] = 1.0
            # Suavização gaussiana leve: frames adjacentes recebem peso menor
            # Isso evita que o modelo precise acertar o frame exato
            if frame_idx + 1 < num_frames:
                timing_targets[frame_idx + 1] = max(timing_targets[frame_idx + 1], 0.3)
            if frame_idx - 1 >= 0:
                timing_targets[frame_idx - 1] = max(timing_targets[frame_idx - 1], 0.3)

    # ── Note sequence (para NoteNet) ─────────────────────────────
    # Cada nota: [hand, col, layer, cut, beat_norm]
    # beat_norm permite ao modelo saber a posição temporal relativa
    note_sequence_list = []
    for note in real_notes:
        encoded = encode_note(note)
        if encoded is None:
            continue
        beat_norm = float(note.b) / max(total_beats, 1.0)
        # Adiciona beat_norm como 6ª coluna (usado internamente, não é target)
        full = np.append(encoded, beat_norm).astype(np.float32)  # (6,)
        note_sequence_list.append(full)

    if len(note_sequence_list) < 4:
        return None

    note_sequence = np.stack(note_sequence_list, axis=0)  # (N, 6)

    return features, timing_targets, note_sequence, float(stars)