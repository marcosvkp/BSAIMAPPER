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
    # REGRA 1: BPM Inteiro (ou .5 em casos raros, mas vamos forçar inteiro para estabilidade)
    bpm = round(raw_bpm)
    print(f"BPM Detectado: {raw_bpm:.2f} -> Ajustado para: {bpm}")

    processed_audio_path = os.path.join(output_folder, "song.ogg")
    add_silence(audio_path, processed_audio_path, silence_duration_ms=3000)

    print("Extraindo features...")
    features, sr, hop_length = extract_features(processed_audio_path, bpm)
    
    if features is None: return

    print("Gerando notas com IA V3...")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    model = get_model().to(device)
    model_path = "models/beat_saber_model.pth"
    
    notes = []
    if os.path.exists(model_path):
        try:
            model.load_state_dict(torch.load(model_path, map_location=device, weights_only=True))
            model.eval()
            
            input_tensor = torch.tensor(features, dtype=torch.float32).unsqueeze(0).to(device)
            
            with torch.no_grad():
                logits = model(input_tensor)
                probs = torch.sigmoid(logits)
                probs = probs.squeeze(0).cpu().numpy()
                
            notes = post_process_notes(probs, bpm, sr, hop_length)
            print(f"Geradas {len(notes)} notas.")

        except Exception as e:
            print(f"Erro ao usar o modelo: {e}")
            import traceback
            traceback.print_exc()
    else:
        print("Modelo não encontrado!")

    create_info_file(output_folder, bpm, song_filename="song.ogg")
    create_difficulty_file(output_folder, notes)
    print(f"Mapa gerado em {output_folder}")

def post_process_notes(probs, bpm, sr, hop_length):
    """
    Transforma probabilidades em notas aplicando regras de jogabilidade (Flow).
    """
    notes = []
    seconds_per_beat = 60.0 / bpm
    frame_duration = hop_length / sr
    
    # Peak Picking Parameters
    threshold = 0.4
    
    # Cooldowns separados por mão (em segundos)
    # Esquerda: Linhas 0, 1
    # Direita: Linhas 2, 3
    last_time_left = -1.0
    last_time_right = -1.0
    min_gap = 0.15 # ~1/4 beat a 100 BPM. Limita spam insano.
    
    # Iterar frame a frame
    for t in range(probs.shape[0]):
        # Verifica se é um pico local
        if t < 1 or t >= probs.shape[0] - 1: continue
        
        # Para cada posição
        for i in range(12):
            val = probs[t, i]
            
            # É pico?
            if val > threshold and val > probs[t-1, i] and val > probs[t+1, i]:
                
                # Quantização
                time_sec = t * frame_duration
                beat_raw = time_sec / seconds_per_beat
                beat_quantized = round(beat_raw * 4) / 4.0 # Snap to 1/4
                
                # Determinar mão
                line = i % 4
                layer = i // 4
                is_right_hand = (line >= 2)
                
                # Checar cooldown da mão específica
                if is_right_hand:
                    if time_sec - last_time_right < min_gap: continue
                else:
                    if time_sec - last_time_left < min_gap: continue
                
                # Evitar duplicatas exatas (mesmo tempo, mesma linha/layer)
                duplicate = False
                for n in notes:
                    if abs(n['_time'] - beat_quantized) < 0.01 and n['_lineIndex'] == line and n['_lineLayer'] == layer:
                        duplicate = True
                        break
                if duplicate: continue

                # Atualizar cooldown
                if is_right_hand: last_time_right = time_sec
                else: last_time_left = time_sec
                
                # Cor: Esquerda = Vermelho (0), Direita = Azul (1)
                # No Beat Saber padrão: 0=Esq(Red), 1=Dir(Blue)
                note_type = 1 if is_right_hand else 0
                
                note = {
                    "_time": float(beat_quantized),
                    "_lineIndex": int(line),
                    "_lineLayer": int(layer),
                    "_type": int(note_type),
                    "_cutDirection": 8 # Dot
                }
                notes.append(note)
    
    notes.sort(key=lambda x: x['_time'])
    return notes

def create_info_file(folder, bpm, song_filename="song.ogg"):
    info = {
        "_version": "2.0.0",
        "_songName": "AI Generated",
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
                        "_noteJumpMovementSpeed": 16,
                        "_noteJumpStartBeatOffset": 0
                    }
                ]
            }
        ]
    }
    with open(os.path.join(folder, "Info.dat"), 'w') as f:
        json.dump(info, f, indent=2)

def create_difficulty_file(folder, notes):
    diff = {
        "_version": "2.0.0",
        "_events": [],
        "_notes": notes,
        "_obstacles": [],
    }
    with open(os.path.join(folder, "ExpertPlus.dat"), 'w') as f:
        json.dump(diff, f, indent=2)

if __name__ == "__main__":
    generate_map("musica.mp3", "output/MeuMapa2")
