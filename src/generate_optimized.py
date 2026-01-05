import torch
import numpy as np
import json
import os
import shutil
from models_optimized import get_beat_model
from pattern_manager import PatternManager
from audio_processor import extract_features, detect_bpm, add_silence

def zip_folder(folder_path, output_path):
    """Cria um arquivo ZIP a partir de uma pasta."""
    if os.path.exists(output_path + ".zip"):
        os.remove(output_path + ".zip")
    shutil.make_archive(output_path, 'zip', folder_path)
    print(f"Mapa compactado com sucesso em: {output_path}.zip")

def generate_map_optimized(audio_path, output_folder):
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)
        
    print(f"Processando: {audio_path}")
    
    # 1. Processamento de Áudio
    bpm = detect_bpm(audio_path)
    # Usando .egg que é o padrão da comunidade (basicamente um ogg renomeado)
    processed_audio_name = "song.egg"
    processed_audio_path = os.path.join(output_folder, processed_audio_name)
    add_silence(audio_path, processed_audio_path)
    
    features, sr, hop_length = extract_features(processed_audio_path, bpm)
    
    # 2. Inferência (Beat Detection)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = get_beat_model().to(device)
    
    model_path = "models/beat_net_optimized.pth"
    if not os.path.exists(model_path):
        print(f"ERRO: Modelo não encontrado em {model_path}. Execute src/train_optimized.py primeiro.")
        return

    model.load_state_dict(torch.load(model_path, map_location=device, weights_only=True))
    model.eval()
    
    # Processar em chunks para economizar memória
    inputs = torch.tensor(features, dtype=torch.float32).unsqueeze(0).to(device)
    with torch.no_grad():
        logits = model(inputs)
        probs = torch.sigmoid(logits).squeeze().cpu().numpy()
        
    # 3. Geração de Notas (Pattern Matching)
    pattern_manager = PatternManager()
    notes = []
    
    frame_duration = hop_length / sr
    seconds_per_beat = 60 / bpm
    
    # Peak Picking simples
    threshold = 0.5
    cooldown = int(0.1 / frame_duration) # 100ms cooldown
    last_note_frame = -cooldown
    
    print("Gerando padrões...")
    for i in range(len(probs)):
        if probs[i] > threshold and (i - last_note_frame) > cooldown:
            # É um pico local?
            if i > 0 and i < len(probs)-1:
                if probs[i] > probs[i-1] and probs[i] > probs[i+1]:
                    # BEAT DETECTADO!
                    time_sec = i * frame_duration
                    beat_time = time_sec / seconds_per_beat
                    
                    # Quantização para o beat mais próximo (1/8)
                    quantized_beat = round(beat_time * 8) / 8
                    
                    # Decidir padrão
                    intensity = probs[i] # Probabilidade como proxy de intensidade
                    time_gap = (i - last_note_frame) * frame_duration
                    
                    # Usando o nome correto do método
                    pattern = pattern_manager.get_pattern_for_intensity(intensity, time_gap)
                    new_notes = pattern_manager.apply_pattern(pattern, quantized_beat, bpm)
                    
                    notes.extend(new_notes)
                    last_note_frame = i
                    
    # 4. Salvar
    save_beatmap(notes, bpm, output_folder, processed_audio_name)
    print(f"Mapa gerado com {len(notes)} notas em {output_folder}")
    
    # 5. Zipar
    zip_output_path = os.path.join("output", os.path.basename(output_folder))
    zip_folder(output_folder, zip_output_path)

def save_beatmap(notes, bpm, folder, audio_filename):
    # Ordenar notas por tempo
    notes.sort(key=lambda x: x['_time'])
    
    # Nome do arquivo de dificuldade
    diff_filename = "ExpertPlus.dat"
    
    diff_data = {
        "_version": "2.0.0",
        "_notes": notes,
        "_events": [],
        "_obstacles": [],
        "_customData": {
            "_time": notes[-1]['_time'] + 4 if notes else 0,
            "_BPMChanges": [],
            "_bookmarks": []
        }
    }
    
    # Info.dat completo conforme solicitado
    info_data = {
        "_version": "2.1.0",
        "_songName": "AI Generated Track",
        "_songSubName": "Optimized BSIAMapper",
        "_songAuthorName": "Unknown Artist",
        "_levelAuthorName": "BSIAMapper AI",
        "_beatsPerMinute": float(bpm),
        "_songTimeOffset": 0.0,
        "_shuffle": 0.0,
        "_shufflePeriod": 0.5,
        "_previewStartTime": 12.0,
        "_previewDuration": 10.0,
        "_songFilename": audio_filename,
        "_coverImageFilename": "cover.png",
        "_environmentName": "DefaultEnvironment",
        "_allDirectionsEnvironmentName": "GlassDesertEnvironment",
        "_environmentNames": [],
        "_colorSchemes": [],
        "_customData": {
            "_contributors": [],
            "_editors": {
                "_lastEditedBy": "BSIAMapper",
                "BSIAMapper": {
                    "version": "1.0.0"
                }
            }
        },
        "_difficultyBeatmapSets": [
            {
                "_beatmapCharacteristicName": "Standard",
                "_difficultyBeatmaps": [
                    {
                        "_difficulty": "ExpertPlus",
                        "_difficultyRank": 9,
                        "_beatmapFilename": diff_filename,
                        "_noteJumpMovementSpeed": 18,
                        "_noteJumpStartBeatOffset": 0.0,
                        "_beatmapColorSchemeIdx": 0,
                        "_environmentNameIdx": 0,
                        "_customData": {
                            "_difficultyLabel": "AI Expert+"
                        }
                    }
                ]
            }
        ]
    }
    
    with open(os.path.join(folder, diff_filename), 'w') as f:
        json.dump(diff_data, f, indent=2)
        
    with open(os.path.join(folder, "Info.dat"), 'w') as f:
        json.dump(info_data, f, indent=2)

if __name__ == "__main__":
    # Exemplo de uso
    # Certifique-se de ter um arquivo 'musica.mp3' na raiz ou ajuste o caminho
    audio_file = "musica.mp3"
    if os.path.exists(audio_file):
        generate_map_optimized(audio_file, "output/OptimizedMap_Final")
    else:
        print(f"Arquivo de áudio '{audio_file}' não encontrado. Por favor, coloque um arquivo mp3 na raiz do projeto.")
