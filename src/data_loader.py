import json
import os
import numpy as np
from audio_processor import extract_features
# Importa as novas funções do loader
from parser.loader import load_specific_difficulty
from parser.enums import NoteColor

def create_dataset_entry(map_folder: str, difficulty_filename: str, difficulty_name: str, stars: float):
    """
    Cria uma entrada de dataset contendo features, targets e o nível de estrelas,
    para uma dificuldade específica.

    Args:
        map_folder: O caminho para a pasta do mapa.
        difficulty_filename: O nome do arquivo .dat da dificuldade específica.
        difficulty_name: O nome da dificuldade (ex: "ExpertPlus").
        stars: O nível de estrelas para esta dificuldade.

    Returns:
        Uma tupla (features, target_placement, vertical_distribution, stars) ou None se falhar.
    """
    # Usa o novo loader para obter o Beatmap padronizado para a dificuldade específica
    load_result = load_specific_difficulty(map_folder, difficulty_filename)
    if load_result is None:
        return None
    
    beatmap, bpm, audio_path = load_result

    features, sr, hop_length = extract_features(audio_path, bpm)
    if features is None: return None

    num_frames = features.shape[0]
    target_placement = np.zeros((num_frames, 12), dtype=np.float32)
    
    seconds_per_beat = 60.0 / float(bpm)
    frame_duration = hop_length / sr
    
    # Combina notas e bombas para o target de colocação
    all_map_objects = sorted(beatmap.notes + beatmap.bombs, key=lambda x: x.b)
    
    vertical_distribution = {0: 0, 1: 0, 2: 0}

    for obj in all_map_objects:
        beat_time = obj.b
        line = obj.x
        layer = obj.y
        
        # Ignora bombas para a distribuição vertical, mas inclui no target_placement
        if obj.c != NoteColor.BOMB:
            if layer in vertical_distribution:
                vertical_distribution[layer] += 1
            
        idx_pos = (layer * 4) + line
        time_sec = beat_time * seconds_per_beat
        frame_idx = int(time_sec / frame_duration)
        
        if frame_idx < num_frames:
            target_placement[frame_idx, idx_pos] = 1.0
            if frame_idx + 1 < num_frames: 
                target_placement[frame_idx+1, idx_pos] = max(target_placement[frame_idx+1, idx_pos], 0.2)
            if frame_idx - 1 >= 0: 
                target_placement[frame_idx-1, idx_pos] = max(target_placement[frame_idx-1, idx_pos], 0.2)

    return features, target_placement, vertical_distribution, stars
