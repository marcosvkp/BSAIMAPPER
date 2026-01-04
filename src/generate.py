import os
import json
import torch
import numpy as np
from audio_processor import detect_bpm, add_silence, extract_features
from model import get_model

def generate_map(audio_path, output_folder):
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    filename = os.path.basename(audio_path)
    print(f"Processando áudio: {filename}")
    
    raw_bpm = detect_bpm(audio_path)
    bpm = round(raw_bpm)
    print(f"BPM Detectado: {raw_bpm:.2f} -> Ajustado para: {bpm}")

    processed_audio_path = os.path.join(output_folder, "song.ogg")
    add_silence(audio_path, processed_audio_path, silence_duration_ms=3000)

    print("Extraindo features...")
    features, sr, hop_length = extract_features(processed_audio_path, bpm)
    
    if features is None: return

    print("Gerando notas com IA V4 (Sampling + Cooldown)...")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    model = get_model().to(device)
    model_path = "models/beat_saber_model_v4.pth"
    
    notes = []
    if os.path.exists(model_path):
        try:
            model.load_state_dict(torch.load(model_path, map_location=device, weights_only=True))
            model.eval()
            
            input_tensor = torch.tensor(features, dtype=torch.float32).unsqueeze(0).to(device)
            
            with torch.no_grad():
                logits = model(input_tensor)
                
            notes = post_process_notes_v4(logits, bpm, sr, hop_length)
            print(f"Geradas {len(notes)} notas.")

        except Exception as e:
            print(f"Erro ao usar o modelo: {e}")
            import traceback
            traceback.print_exc()
    else:
        print(f"Modelo '{model_path}' não encontrado! Treine o modelo primeiro.")

    create_info_file(output_folder, bpm, song_filename="song.ogg")
    create_difficulty_file(output_folder, notes, bpm)
    print(f"Mapa gerado em {output_folder}")

def apply_temperature(logits, temperature):
    return logits / temperature

def post_process_notes_v4(logits, bpm, sr, hop_length, temperature=1.05, threshold=0.15, cooldown_frames=4):
    notes = []
    seconds_per_beat = 60.0 / bpm
    frame_duration = hop_length / sr
    
    logits = apply_temperature(logits.squeeze(0), temperature)
    probs = torch.sigmoid(logits).cpu().numpy()

    cooldown_grid = np.zeros(12, dtype=int)
    last_cut_direction = {0: 0, 1: 1}

    for t in range(1, probs.shape[0] - 1):
        current_probs = probs[t].copy()
        for i in range(12):
            if cooldown_grid[i] > 0:
                current_probs[i] *= 0.1
                cooldown_grid[i] -= 1

        if np.max(current_probs) < threshold:
            continue

        is_peak = (current_probs > probs[t-1]) & (current_probs > probs[t+1])
        peak_indices = np.where(is_peak & (current_probs > threshold))[0]
        
        if len(peak_indices) == 0:
            continue
            
        peak_probs = current_probs[peak_indices]
        peak_probs /= (peak_probs.sum() + 1e-6)
        
        chosen_index = np.random.choice(peak_indices, p=peak_probs)
        
        time_sec = t * frame_duration
        beat_raw = time_sec / seconds_per_beat
        beat_quantized = round(beat_raw * 8) / 8.0
        
        if any(abs(n['_time'] - beat_quantized) < 0.1 for n in notes):
            continue

        line = chosen_index % 4
        layer = chosen_index // 4
        
        hand = 1 if (line >= 2) else 0
        
        cut_direction = last_cut_direction[hand]
        last_cut_direction[hand] = 1 - cut_direction

        note = {
            "_time": float(beat_quantized),
            "_lineIndex": int(line),
            "_lineLayer": int(layer),
            "_type": hand,
            "_cutDirection": int(cut_direction)
        }
        notes.append(note)
        
        cooldown_grid[chosen_index] = cooldown_frames
        for i in range(4):
            cooldown_grid[layer * 4 + i] = max(cooldown_grid[layer * 4 + i], cooldown_frames // 2)

    notes.sort(key=lambda x: x['_time'])
    return notes


def create_info_file(folder, bpm, song_filename="song.ogg"):
    info = {
        "_version": "2.0.0",
        "_songName": "AI Generated Map",
        "_songSubName": "",
        "_songAuthorName": "BSIAMapper",
        "_levelAuthorName": "Auto",
        "_beatsPerMinute": float(bpm),
        "_songTimeOffset": 0,
        "_shuffle": 0,
        "_shufflePeriod": 0.5,
        "_previewStartTime": 12,
        "_previewDuration": 10,
        "_songFilename": song_filename,
        "_coverImageFilename": "cover.jpg",
        "_environmentName": "DefaultEnvironment",
        "_allDirectionsEnvironmentName": "GlassDesertEnvironment",
        "_difficultyBeatmapSets": [
            {
                "_beatmapCharacteristicName": "Standard",
                "_difficultyBeatmaps": [
                    {
                        "_difficulty": "ExpertPlus",
                        "_difficultyRank": 9,
                        "_beatmapFilename": "ExpertPlus.dat",
                        "_noteJumpMovementSpeed": 18,
                        "_noteJumpStartBeatOffset": -0.1
                    }
                ]
            }
        ]
    }
    with open(os.path.join(folder, "Info.dat"), 'w') as f:
        json.dump(info, f, indent=2)

def create_difficulty_file(folder, notes, bpm):
    final_beat = max((n['_time'] for n in notes), default=0)
    
    end_buffer_beats = 2.0 * (bpm / 60.0)

    diff = {
        "_version": "2.0.0",
        "_events": [],
        "_notes": notes,
        "_obstacles": [],
        "customData": {
            "_time": final_beat + end_buffer_beats
        }
    }
    with open(os.path.join(folder, "ExpertPlus.dat"), 'w') as f:
        json.dump(diff, f, indent=2)

if __name__ == "__main__":
    generate_map("musica.mp3", "output/MeuMapaGerado")
