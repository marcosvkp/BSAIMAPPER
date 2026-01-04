import os
import json
import torch
import numpy as np
import shutil
from audio_processor import detect_bpm, add_silence, extract_features
from model import get_model

def zip_folder(folder_path, output_path):
    """Cria um arquivo ZIP a partir de uma pasta."""
    shutil.make_archive(output_path, 'zip', folder_path)
    print(f"Mapa compactado em {output_path}.zip")

def get_next_cut_direction(last_direction):
    """
    Gera uma direção de corte fluida, priorizando cardinais e evitando ângulos excessivos.
    Direções: 0:Cima, 1:Baixo, 2:Esq, 3:Dir, 4:CimaEsq, 5:CimaDir, 6:BaixoEsq, 7:BaixoDir, 8:Dot
    """
    # Mapeia direções opostas para forçar o flow
    opposites = {0: 1, 1: 0, 2: 3, 3: 2, 4: 7, 5: 6, 6: 5, 7: 4}
    
    # Probabilidades: Cardinais (70%), Diagonais (30%)
    # Após um corte para cima (0), as próximas opções são: Baixo (1), Baixo-Esq (6), Baixo-Dir (7)
    flow_rules = {
        0: ([1], [6, 7]),       # Cima -> Prioriza Baixo
        1: ([0], [4, 5]),       # Baixo -> Prioriza Cima
        2: ([3], [5, 7]),       # Esq -> Prioriza Dir
        3: ([2], [4, 6]),       # Dir -> Prioriza Esq
        4: ([7], [1, 3]),       # CimaEsq -> Prioriza BaixoDir
        5: ([6], [1, 2]),       # CimaDir -> Prioriza BaixoEsq
        6: ([5], [0, 3]),       # BaixoEsq -> Prioriza CimaDir
        7: ([4], [0, 2]),       # BaixoDir -> Prioriza CimaEsq
        8: ([0, 1, 2, 3], [])  # Dot -> Qualquer cardinal
    }
    
    cardinals, diagonals = flow_rules.get(last_direction, ([0, 1], []))
    
    # Se não houver opções, use um padrão seguro
    if not cardinals and not diagonals:
        return np.random.choice([0, 1])

    # Pesa as probabilidades
    choices = cardinals + diagonals
    if len(diagonals) == 0:
        probs = [1.0 / len(choices)] * len(choices)
    else:
        prob_cardinal = 0.7 / len(cardinals) if cardinals else 0
        prob_diagonal = 0.3 / len(diagonals) if diagonals else 0
        probs = ([prob_cardinal] * len(cardinals)) + ([prob_diagonal] * len(diagonals))
        
    # Normaliza probabilidades para somarem 1
    probs = np.array(probs) / sum(probs)
    
    return np.random.choice(choices, p=probs)

def post_process_notes_v6(logits, bpm, sr, hop_length, temperature=1.05, threshold=0.14, cooldown_frames=3):
    notes = []
    seconds_per_beat = 60.0 / bpm
    frame_duration = hop_length / sr
    min_hand_gap = (60.0 / bpm) / 2.1 # Cooldown por mão (1/2 de um beat, com uma pequena margem)

    logits = logits.squeeze(0) / temperature
    probs = torch.sigmoid(logits).cpu().numpy()

    cooldown_grid = np.zeros(12, dtype=int)
    last_cut_direction = {0: 1, 1: 1} # Mão Esq (0), Mão Dir (1)
    last_note_time = {-1: -1, 0: -1, 1: -1} # Tempo da última nota geral, da mão esq, e da mão dir

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
        
        if len(peak_indices) == 0: continue
            
        peak_probs = current_probs[peak_indices]
        peak_probs /= (peak_probs.sum() + 1e-6)
        
        chosen_index = np.random.choice(peak_indices, p=peak_probs)
        
        time_sec = t * frame_duration
        
        # REGRA: COOLDOWN POR MÃO
        hand = 1 if (chosen_index % 4 >= 2) else 0
        if time_sec - last_note_time[hand] < min_hand_gap:
            continue
        
        beat_quantized = round(time_sec / seconds_per_beat * 8) / 8.0
        
        # Evita notas duplas no mesmo tempo quantizado
        if abs(beat_quantized - last_note_time[-1]) < 0.01:
            continue

        line = chosen_index % 4
        layer = chosen_index // 4
        
        cut_direction = get_next_cut_direction(last_cut_direction[hand])
        
        note = {
            "_time": float(beat_quantized),
            "_lineIndex": int(line),
            "_lineLayer": int(layer),
            "_type": hand,
            "_cutDirection": int(cut_direction)
        }
        notes.append(note)
        
        last_cut_direction[hand] = cut_direction
        last_note_time[hand] = time_sec
        last_note_time[-1] = beat_quantized
        
        cooldown_grid[chosen_index] = cooldown_frames
        for i in range(4):
            cooldown_grid[layer * 4 + i] = max(cooldown_grid[layer * 4 + i], cooldown_frames // 2)

    notes.sort(key=lambda x: x['_time'])
    return notes

def generate_map(audio_path, output_folder):
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    filename = os.path.basename(audio_path)
    print(f"Processando áudio: {filename} para a pasta {output_folder}")
    
    raw_bpm = detect_bpm(audio_path)
    bpm = round(raw_bpm)
    print(f"BPM Detectado: {raw_bpm:.2f} -> Ajustado para: {bpm}")

    processed_audio_path = os.path.join(output_folder, "song.ogg")
    add_silence(audio_path, processed_audio_path, silence_duration_ms=3000)

    print("Extraindo features...")
    features, sr, hop_length = extract_features(processed_audio_path, bpm)
    
    if features is None: return

    print("Gerando notas com IA V6 (Flow Control + Zip)...")
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
                
            notes = post_process_notes_v6(logits, bpm, sr, hop_length)
            print(f"Geradas {len(notes)} notas.")

        except Exception as e:
            print(f"Erro ao usar o modelo: {e}")
            import traceback
            traceback.print_exc()
    else:
        print(f"Modelo '{model_path}' não encontrado! Treine o modelo primeiro.")

    create_info_file(output_folder, bpm, song_filename="song.ogg")
    create_difficulty_file(output_folder, notes, bpm)
    
    # Gera o ZIP
    zip_output_path = os.path.join("output", os.path.basename(output_folder))
    zip_folder(output_folder, zip_output_path)

    print(f"Mapa gerado e compactado com sucesso em {output_folder}\n")

def create_info_file(folder, bpm, song_filename="song.ogg"):
    info = {
        "_version": "2.0.0",
        "_songName": "AI Generated Map",
        "_songSubName": "",
        "_songAuthorName": "BSIAMapper",
        "_levelAuthorName": "AutoV6",
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
    NUM_MAPS_TO_GENERATE = 3
    BASE_AUDIO_FILE = "musica.mp3"
    BASE_OUTPUT_NAME = "MeuMapaGerado"

    print(f"Iniciando geração de {NUM_MAPS_TO_GENERATE} mapas diferentes...")
    
    for i in range(NUM_MAPS_TO_GENERATE):
        # Garante que cada execução tenha um seed aleatório diferente
        np.random.seed()
        output_folder = f"output/{BASE_OUTPUT_NAME}_{i+1}"
        generate_map(BASE_AUDIO_FILE, output_folder)
