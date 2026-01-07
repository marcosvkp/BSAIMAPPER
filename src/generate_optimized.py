import torch
import numpy as np
import json
import os
import shutil
from models_optimized import get_model
from pattern_manager import PatternManager
from flow_fixer import FlowFixer
from audio_processor import extract_features, detect_bpm, add_silence, analyze_energy

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
    full_audio_path = os.path.join(output_folder, processed_audio)
    add_silence(audio_path, full_audio_path)
    
    features, sr, hop_length = extract_features(full_audio_path, bpm)
    
    # --- Análise de Energia (NOVO) ---
    print("Analisando perfil de energia da música...")
    energy_profile = analyze_energy(full_audio_path, hop_length=hop_length, sr=sr)
    
    # Ajuste de tamanho caso haja pequena discrepância de frames
    if len(energy_profile) > len(features):
        energy_profile = energy_profile[:len(features)]
    elif len(energy_profile) < len(features):
        energy_profile = np.pad(energy_profile, (0, len(features) - len(energy_profile)))
    
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
    
    base_threshold = 0.5
    base_cooldown = int(0.1 / frame_dur)
    last_frame = -base_cooldown
    
    # Controle de ocupação temporal (para bursts)
    occupied_until_beat = 0.0
    
    MIN_START_TIME = 3.0 

    print("Gerando com Director AI (Energy Aware + Burst Fill)...")
    
    for i in range(len(beat_probs)):
        time_sec = i * frame_dur
        current_beat = time_sec / sec_per_beat
        
        if time_sec < MIN_START_TIME:
            continue
            
        # Se o tempo atual já está ocupado por um burst anterior, pula
        if current_beat < occupied_until_beat:
            continue
            
        current_energy = energy_profile[i]
        
        # --- Threshold Dinâmico ---
        dynamic_threshold = base_threshold + (0.5 - current_energy) * 0.6
        dynamic_threshold = np.clip(dynamic_threshold, 0.15, 0.85)
        
        # --- Cooldown Dinâmico ---
        current_cooldown = base_cooldown
        if current_energy > 0.75:
            current_cooldown = int(base_cooldown * 0.6) 

        if beat_probs[i] > dynamic_threshold and (i - last_frame) > current_cooldown:
            # Peak picking local simples
            if i > 0 and i < len(beat_probs)-1:
                if beat_probs[i] > beat_probs[i-1] and beat_probs[i] > beat_probs[i+1]:
                    
                    beat_time = round(current_beat * 8) / 8
                    
                    comp_idx = comp_classes[i]
                    vert_idx = vert_classes[i]
                    intensity = beat_probs[i]
                    gap = (i - last_frame) * frame_dur
                    
                    meta = pattern_manager.get_pattern(intensity, comp_idx, vert_idx, gap, energy_level=current_energy)
                    
                    if meta: 
                        new_notes = pattern_manager.apply_pattern(meta, beat_time, bpm)
                        raw_notes.extend(new_notes)
                        last_frame = i
                        
                        # Se gerou um burst, atualiza o tempo ocupado
                        if meta['type'] == 'burst_fill':
                            # Burst ocupa 1 beat inteiro (4 notas de 1/4)
                            occupied_until_beat = beat_time + 1.0
                        elif meta['type'] == 'super_stream':
                            # Super Stream ocupa 2 beats inteiros (8 notas de 1/4)
                            occupied_until_beat = beat_time + 2.0
    
    print("Aplicando FlowFixer (Simulação de Paridade e Resets)...")
    all_objects = FlowFixer.fix(raw_notes, bpm)
    
    final_notes = [obj for obj in all_objects if obj['_type'] != 3]
    final_bombs = [obj for obj in all_objects if obj['_type'] == 3]
                    
    save_beatmap(final_notes, final_bombs, bpm, output_folder, processed_audio)
    zip_folder(output_folder, os.path.join("output", os.path.basename(output_folder)))

def save_beatmap(notes, bombs, bpm, folder, audio_name):
    notes.sort(key=lambda x: x['_time'])
    bombs.sort(key=lambda x: x['_time'])
    
    unique_bombs = []
    seen_bombs = set()
    for b in bombs:
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
        "_songSubName": "Energy Aware V6",
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
