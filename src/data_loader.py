import json
import os
import numpy as np
from audio_processor import extract_features

def load_beat_map(map_folder):
    """
    Carrega os dados do mapa e informações da música, suportando diferentes versões de formato.
    """
    info_path = os.path.join(map_folder, "Info.dat")
    if not os.path.exists(info_path):
        info_path = os.path.join(map_folder, "info.dat")
        if not os.path.exists(info_path): return None, None, None

    with open(info_path, 'r', encoding='utf-8') as f: info = json.load(f)

    # Compatibilidade com diferentes versões do Info.dat
    version = info.get('version', info.get('_version', '2.0.0'))
    is_v4 = version.startswith('4') or 'audio' in info

    if is_v4:
        bpm = info.get('audio', {}).get('bpm')
        song_filename = info.get('audio', {}).get('songFilename')
    else:
        bpm = info.get('_beatsPerMinute')
        song_filename = info.get('_songFilename')

    if not bpm or not song_filename: return None, None, None

    # Encontra o arquivo de dificuldade Expert+ ou Expert
    difficulty_filename = None
    if is_v4:
        diffs = info.get('difficultyBeatmaps', [])
        # Prioriza ExpertPlus
        for d in diffs: 
            if d.get('difficulty') == 'ExpertPlus': difficulty_filename = d.get('beatmapDataFilename'); break
        # Fallback para Expert
        if not difficulty_filename:
            for d in diffs: 
                if d.get('difficulty') == 'Expert': difficulty_filename = d.get('beatmapDataFilename'); break
    else: # Versões antigas
        sets = info.get('_difficultyBeatmapSets', [])
        for s in sets:
            diffs = s.get('_difficultyBeatmaps', [])
            for d in diffs: 
                if d.get('_difficulty') == 'ExpertPlus': difficulty_filename = d.get('_beatmapFilename'); break
            if difficulty_filename: break
            if not difficulty_filename:
                for d in diffs: 
                    if d.get('_difficulty') == 'Expert': difficulty_filename = d.get('_beatmapFilename'); break
            if difficulty_filename: break
    
    if not difficulty_filename: return None, None, None
    diff_path = os.path.join(map_folder, difficulty_filename)
    if not os.path.exists(diff_path): return None, None, None
        
    with open(diff_path, 'r', encoding='utf-8') as f: map_data = json.load(f)
    audio_path = os.path.join(map_folder, song_filename)
    
    return map_data, audio_path, bpm

def create_dataset_entry(map_folder):
    """
    Cria uma entrada de dataset a partir de uma pasta de mapa.
    Extrai features do áudio e cria arrays de target para posição, e agora, direção de corte.
    """
    map_data, audio_path, bpm = load_beat_map(map_folder)
    if not map_data or not audio_path or not os.path.exists(audio_path): return None

    features, sr, hop_length = extract_features(audio_path, bpm)
    if features is None: return None

    num_frames = features.shape[0]
    # Target para posição da nota (grid 4x3)
    target_placement = np.zeros((num_frames, 12), dtype=np.float32)
    # Target para direção de corte (0-8)
    target_cut_direction = np.full((num_frames,), -1, dtype=np.int64) # -1 = Sem nota

    seconds_per_beat = 60.0 / float(bpm)
    frame_duration = hop_length / sr
    
    notes = map_data.get('colorNotes', map_data.get('_notes', []))
    
    vertical_distribution = {0: 0, 1: 0, 2: 0}

    for note in notes:
        # Compatibilidade com nomes de chaves de diferentes versões
        beat_time = note.get('b', note.get('_time'))
        line = note.get('x', note.get('_lineIndex'))
        layer = note.get('y', note.get('_lineLayer'))
        cut_direction = note.get('d', note.get('_cutDirection', 8)) # Padrão 8 (any) se não especificado

        if beat_time is None or line is None or layer is None: continue
        if not (0 <= line <= 3 and 0 <= layer <= 2 and 0 <= cut_direction <= 8): continue
            
        if layer in vertical_distribution:
            vertical_distribution[layer] += 1
            
        idx_pos = (layer * 4) + line
        time_sec = beat_time * seconds_per_beat
        frame_idx = int(time_sec / frame_duration)
        
        if frame_idx < num_frames:
            # Define o target de posição
            target_placement[frame_idx, idx_pos] = 1.0
            
            # Define o target de direção de corte
            target_cut_direction[frame_idx] = cut_direction
            
            # Adiciona um "blur" suave nos frames adjacentes para ajudar o modelo
            if frame_idx + 1 < num_frames: 
                target_placement[frame_idx+1, idx_pos] = max(target_placement[frame_idx+1, idx_pos], 0.2)
            if frame_idx - 1 >= 0: 
                target_placement[frame_idx-1, idx_pos] = max(target_placement[frame_idx-1, idx_pos], 0.2)

    # Retorna as features e os dois tipos de targets
    return features, target_placement, target_cut_direction, vertical_distribution
