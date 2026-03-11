import json
import os
import numpy as np
from audio_processor import extract_features

def load_beat_map(map_folder):
    """
    Carrega os dados do mapa, o caminho do áudio, o BPM e o nome da dificuldade.
    Prioriza ExpertPlus, mas faz fallback para outras dificuldades ranqueadas se não encontrada.
    """
    info_path = os.path.join(map_folder, "Info.dat")
    if not os.path.exists(info_path):
        info_path = os.path.join(map_folder, "info.dat")
        if not os.path.exists(info_path): return None, None, None, None

    with open(info_path, 'r', encoding='utf-8') as f: info = json.load(f)

    bpm = info.get('_beatsPerMinute')
    song_filename = info.get('_songFilename')

    if not bpm or not song_filename: return None, None, None, None

    difficulty_filename = None
    difficulty_name = None
    
    # Lista de dificuldades em ordem de preferência
    difficulty_priority = ["ExpertPlus", "Expert", "Hard", "Normal", "Easy"]
    
    beatmap_sets = info.get('_difficultyBeatmapSets', [])
    for s in beatmap_sets:
        if s.get('_beatmapCharacteristicName') == 'Standard':
            diffs = s.get('_difficultyBeatmaps', [])
            
            # Cria um dicionário de dificuldades disponíveis para fácil acesso
            available_diffs = {d.get('_difficulty'): d.get('_beatmapFilename') for d in diffs}
            
            # Itera na ordem de prioridade
            for diff_level in difficulty_priority:
                if diff_level in available_diffs:
                    difficulty_name = diff_level
                    difficulty_filename = available_diffs[diff_level]
                    break # Para ao encontrar a de maior prioridade
            
            if difficulty_filename:
                break # Para o loop de sets se já encontramos uma dificuldade

    if not difficulty_filename: return None, None, None, None
    
    diff_path = os.path.join(map_folder, difficulty_filename)
    if not os.path.exists(diff_path): return None, None, None, None
        
    with open(diff_path, 'r', encoding='utf-8') as f: map_data = json.load(f)
    audio_path = os.path.join(map_folder, song_filename)
    
    return map_data, audio_path, bpm, difficulty_name

def create_dataset_entry(map_folder):
    """
    Cria uma entrada de dataset contendo features, targets e o nível de estrelas.
    """
    map_data, audio_path, bpm, difficulty_name = load_beat_map(map_folder)
    if not all([map_data, audio_path, os.path.exists(audio_path), difficulty_name]):
        return None

    # --- Carregar dados de estrelas ---
    scoresaber_path = os.path.join(map_folder, 'scoresaber.json')
    stars = None
    if os.path.exists(scoresaber_path):
        with open(scoresaber_path, 'r') as f:
            scoresaber_data = json.load(f)
        
        # Busca as estrelas para a dificuldade específica
        if difficulty_name in scoresaber_data:
            stars = scoresaber_data[difficulty_name].get('stars')

    # Se não houver estrelas para esta dificuldade, a entrada é inválida para o treinamento
    if stars is None:
        return None
    # --- Fim da carga de estrelas ---

    features, sr, hop_length = extract_features(audio_path, bpm)
    if features is None: return None

    num_frames = features.shape[0]
    target_placement = np.zeros((num_frames, 12), dtype=np.float32)
    
    seconds_per_beat = 60.0 / float(bpm)
    frame_duration = hop_length / sr
    
    notes = map_data.get('_notes', [])
    
    vertical_distribution = {0: 0, 1: 0, 2: 0}

    for note in notes:
        beat_time = note.get('_time')
        line = note.get('_lineIndex')
        layer = note.get('_lineLayer')
        
        if beat_time is None or line is None or layer is None: continue
        if line < 0 or line > 3 or layer < 0 or layer > 2: continue
            
        if layer in vertical_distribution:
            vertical_distribution[layer] += 1
            
        idx_pos = (layer * 4) + line
        time_sec = beat_time * seconds_per_beat
        frame_idx = int(time_sec / frame_duration)
        
        if frame_idx < num_frames:
            # Target mais nítido: Apenas 3 frames
            target_placement[frame_idx, idx_pos] = 1.0
            if frame_idx + 1 < num_frames: 
                target_placement[frame_idx+1, idx_pos] = max(target_placement[frame_idx+1, idx_pos], 0.2)
            if frame_idx - 1 >= 0: 
                target_placement[frame_idx-1, idx_pos] = max(target_placement[frame_idx-1, idx_pos], 0.2)

    # Retorna as features, os alvos, a distribuição e o novo dado: estrelas
    return features, target_placement, vertical_distribution, stars
