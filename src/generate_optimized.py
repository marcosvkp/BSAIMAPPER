import torch
import numpy as np
import json
import os
import shutil
from models_optimized import get_model
from pattern_manager import PatternManager
from flow_fixer import FlowFixer
from audio_processor import extract_features, detect_bpm, add_silence

def zip_folder(folder_path, output_path):
    if os.path.exists(output_path + ".zip"):
        os.remove(output_path + ".zip")
    shutil.make_archive(output_path, 'zip', folder_path)
    print(f"Mapa compactado: {output_path}.zip")

def generate_map_optimized(audio_path, output_folder):
    if not os.path.exists(output_folder): os.makedirs(output_folder)
    print(f"Processando: {audio_path}")
    
    bpm = detect_bpm(audio_path)
    processed_audio = "song.egg"
    add_silence(audio_path, os.path.join(output_folder, processed_audio))
    
    features, sr, hop_length = extract_features(os.path.join(output_folder, processed_audio), bpm)
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = get_model().to(device)
    
    if os.path.exists("models/director_net.pth"):
        model.load_state_dict(torch.load("models/director_net.pth", map_location=device, weights_only=True))
    else:
        print("Modelo DirectorNet não encontrado! Treine primeiro.")
        return

    model.eval()
    
    inputs = torch.tensor(features, dtype=torch.float32).unsqueeze(0).to(device)
    with torch.no_grad():
        p_beat, p_comp, p_vert = model(inputs)
        beat_probs = torch.sigmoid(p_beat).squeeze().cpu().numpy()
        comp_classes = torch.argmax(p_comp, dim=2).squeeze().cpu().numpy()
        vert_classes = torch.argmax(p_vert, dim=2).squeeze().cpu().numpy()
        
    pattern_manager = PatternManager()
    raw_notes = []
    
    frame_dur = hop_length / sr
    sec_per_beat = 60 / bpm
    
    threshold = 0.5
    cooldown = int(0.1 / frame_dur)
    last_frame = -cooldown
    
    # --- REGRA DE OURO: TEMPO MÍNIMO DE INÍCIO ---
    MIN_START_TIME = 3.0 # Segundos
    # ---------------------------------------------

    print("Gerando com Director AI...")
    
    for i in range(len(beat_probs)):
        # Verifica tempo mínimo antes de qualquer coisa
        time_sec = i * frame_dur
        if time_sec < MIN_START_TIME:
            continue

        if beat_probs[i] > threshold and (i - last_frame) > cooldown:
            if i > 0 and i < len(beat_probs)-1:
                if beat_probs[i] > beat_probs[i-1] and beat_probs[i] > beat_probs[i+1]:
                    
                    beat_time = round((time_sec / sec_per_beat) * 8) / 8
                    
                    comp_idx = comp_classes[i]
                    vert_idx = vert_classes[i]
                    intensity = beat_probs[i]
                    gap = (i - last_frame) * frame_dur
                    
                    meta = pattern_manager.get_pattern(intensity, comp_idx, vert_idx, gap)
                    new_notes = pattern_manager.apply_pattern(meta, beat_time, bpm)
                    
                    raw_notes.extend(new_notes)
                    last_frame = i
    
    print("Aplicando FlowFixer (Simulação de Paridade e Resets)...")
    all_objects = FlowFixer.fix(raw_notes, bpm)
    
    final_notes = [obj for obj in all_objects if obj['_type'] != 3]
    final_bombs = [obj for obj in all_objects if obj['_type'] == 3]
                    
    save_beatmap(final_notes, final_bombs, bpm, output_folder, processed_audio)
    zip_folder(output_folder, os.path.join("output", os.path.basename(output_folder)))

def save_beatmap(notes, bombs, bpm, folder, audio_name):
    notes.sort(key=lambda x: x['_time'])
    bombs.sort(key=lambda x: x['_time'])
    
    # Remove duplicatas de bombas (caso o FlowFixer gere bombas sobrepostas para as duas mãos)
    unique_bombs = []
    seen_bombs = set()
    for b in bombs:
        # Chave única: tempo + linha + layer
        key = (round(b['_time'], 3), b['_lineIndex'], b['_lineLayer'])
        if key not in seen_bombs:
            seen_bombs.add(key)
            unique_bombs.append(b)

    all_notes_v2 = notes + unique_bombs
    all_notes_v2.sort(key=lambda x: x['_time'])

    diff = {
        "_version": "2.0.0",
        "_notes": all_notes_v2,
        "_events": [],
        "_obstacles": [],
        "_customData": {
            "_time": notes[-1]['_time'] + 4 if notes else 0,
            "_BPMChanges": [],
            "_bookmarks": []
        }
    }
    
    info = {
        "_version": "2.1.0",
        "_songName": "AI Director Map",
        "_songSubName": "FlowFixer V5",
        "_songAuthorName": "BSIAMapper",
        "_levelAuthorName": "AI",
        "_beatsPerMinute": float(bpm),
        "_songFilename": audio_name,
        "_coverImageFilename": "cover.png",
        "_difficultyBeatmapSets": [{
            "_beatmapCharacteristicName": "Standard",
            "_difficultyBeatmaps": [{
                "_difficulty": "ExpertPlus",
                "_beatmapFilename": "ExpertPlus.dat",
                "_noteJumpMovementSpeed": 18,
                "_noteJumpStartBeatOffset": -0.784
            }]
        }]
    }
    
    with open(os.path.join(folder, "ExpertPlus.dat"), 'w') as f: json.dump(diff, f)
    with open(os.path.join(folder, "Info.dat"), 'w') as f: json.dump(info, f, indent=2)

if __name__ == "__main__":
    generate_map_optimized("musica.mp3", "output/DirectorMap")
