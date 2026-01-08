import json
import os
import numpy as np
from audio_processor import extract_features

def load_beat_map(map_folder):
    info_path = os.path.join(map_folder, "Info.dat")
    if not os.path.exists(info_path):
        info_path = os.path.join(map_folder, "info.dat")
        if not os.path.exists(info_path): return None, None, None

    with open(info_path, 'r', encoding='utf-8') as f: info = json.load(f)

    version = info.get('version', info.get('_version', '2.0.0'))
    is_v4 = version.startswith('4') or 'audio' in info

    if is_v4:
        bpm = info.get('audio', {}).get('bpm')
        song_filename = info.get('audio', {}).get('songFilename')
    else:
        bpm = info.get('_beatsPerMinute')
        song_filename = info.get('_songFilename')

    if not bpm or not song_filename: return None, None, None

    difficulty_filename = None
    if is_v4:
        diffs = info.get('difficultyBeatmaps', [])
        for d in diffs: 
            if d.get('difficulty') == 'ExpertPlus': difficulty_filename = d.get('beatmapDataFilename'); break
        if not difficulty_filename:
            for d in diffs: 
                if d.get('difficulty') == 'Expert': difficulty_filename = d.get('beatmapDataFilename'); break
    else:
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
    map_data, audio_path, bpm = load_beat_map(map_folder)
    if not map_data or not audio_path or not os.path.exists(audio_path): return None

    features, sr, hop_length = extract_features(audio_path, bpm)
    if features is None: return None

    num_frames = features.shape[0]
    target_placement = np.zeros((num_frames, 12), dtype=np.float32)
    
    seconds_per_beat = 60.0 / float(bpm)
    frame_duration = hop_length / sr
    
    notes = map_data.get('colorNotes', map_data.get('_notes', []))
    
    vertical_distribution = {0: 0, 1: 0, 2: 0}

    for note in notes:
        beat_time = note.get('b', note.get('_time'))
        line = note.get('x', note.get('_lineIndex'))
        layer = note.get('y', note.get('_lineLayer'))
        
        if beat_time is None or line is None or layer is None: continue
        if line < 0 or line > 3 or layer < 0 or layer > 2: continue
            
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

    return features, target_placement, vertical_distribution
