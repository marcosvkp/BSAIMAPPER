import os
import json
import torch
import numpy as np
from audio_processor import detect_bpm, add_silence, extract_features
from model import get_model

def get_next_cut_direction(last_direction, hand):
    """
    Gera uma direção de corte fluida com base na direção anterior.
    Direções: 0:Cima, 1:Baixo, 2:Esq, 3:Dir, 4:CimaEsq, 5:CimaDir, 6:BaixoEsq, 7:BaixoDir
    """
    # Regras de flow: para cada direção, quais são as próximas válidas?
    flow_rules = {
        0: [1, 6, 7],       # Após Cima -> Baixo, BaixoEsq, BaixoDir
        1: [0, 4, 5],       # Após Baixo -> Cima, CimaEsq, CimaDir
        2: [3, 5, 7],       # Após Esq -> Dir, CimaDir, BaixoDir
        3: [2, 4, 6],       # Após Dir -> Esq, CimaEsq, BaixoEsq
        4: [1, 3, 7],       # Após CimaEsq -> Baixo, Dir, BaixoDir
        5: [1, 2, 6],       # Após CimaDir -> Baixo, Esq, BaixoEsq
        6: [0, 3, 5],       # Após BaixoEsq -> Cima, Dir, CimaDir
        7: [0, 2, 4],       # Após BaixoDir -> Cima, Esq, CimaEsq
        8: [0, 1, 2, 3]     # Após um Dot Note, qualquer direção cardinal é válida
    }
    
    valid_next = flow_rules.get(last_direction, [0, 1]) # Padrão seguro
    return np.random.choice(valid_next)

def post_process_notes_v5(logits, bpm, sr, hop_length, temperature=1.05, threshold=0.13, cooldown_frames=3):
    notes = []
    seconds_per_beat = 60.0 / bpm
    frame_duration = hop_length / sr
    
    logits = logits.squeeze(0) / temperature
    probs = torch.sigmoid(logits).cpu().numpy()

    cooldown_grid = np.zeros(12, dtype=int)
    # Inicia com um corte para baixo, uma escolha segura e comum.
    last_cut_direction = {0: 1, 1: 1} # Mão Esq (0), Mão Dir (1)

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
        
        # Cooldown de tempo um pouco mais permissivo para aumentar a densidade
        if any(abs(n['_time'] - beat_quantized) < 0.08 for n in notes):
            continue

        line = chosen_index % 4
        layer = chosen_index // 4
        hand = 1 if (line >= 2) else 0
        
        # Usa a nova lógica de flow para determinar a direção do corte
        cut_direction = get_next_cut_direction(last_cut_direction[hand], hand)
        
        note = {
            "_time": float(beat_quantized),
            "_lineIndex": int(line),
            "_lineLayer": int(layer),
            "_type": hand,
            "_cutDirection": int(cut_direction)
        }
        notes.append(note)
        
        # Atualiza a última direção de corte para esta mão
        last_cut_direction[hand] = cut_direction
        
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

    print("Gerando notas com IA V5 (Flow Rules + Densidade)...")
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
                
            notes = post_process_notes_v5(logits, bpm, sr, hop_length)
            print(f"Geradas {len(notes)} notas.")

        except Exception as e:
            print(f"Erro ao usar o modelo: {e}")
            import traceback
            traceback.print_exc()
    else:
        print(f"Modelo '{model_path}' não encontrado! Treine o modelo primeiro.")

    create_info_file(output_folder, bpm, song_filename="song.ogg")
    create_difficulty_file(output_folder, notes, bpm)
    print(f"Mapa gerado com sucesso em {output_folder}\n")

def create_info_file(folder, bpm, song_filename="song.ogg"):
    # (Função mantida igual)
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
    # (Função mantida igual)
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
        output_folder = f"output/{BASE_OUTPUT_NAME}_{i+1}"
        generate_map(BASE_AUDIO_FILE, output_folder)
