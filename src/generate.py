import os
import json
import torch
import numpy as np
import shutil
from audio_processor import detect_bpm, add_silence, extract_features
from model import get_model

# --- PERFIS DE GERAÇÃO DE MAPAS (STYLE PROFILES) ---
# Altere a variável TARGET_STYLE no final do arquivo para escolher um estilo.
STYLE_PROFILES = {
    "Standard": {
        "temperature": 1.1,
        "threshold": 0.15,
        "hand_gap_divisor": 2.0,  # Cooldown por mão (maior = mais lento)
        "diagonal_probability": 0.3, # Probabilidade de usar ângulos diagonais
        "author_name": "AutoV8_Standard"
    },
    "ACC": {
        "temperature": 1.0,       # Mais previsível
        "threshold": 0.20,        # Menos notas, mais deliberadas
        "hand_gap_divisor": 1.5,  # Padrões mais lentos e claros
        "diagonal_probability": 0.05, # Quase nenhum ângulo
        "author_name": "AutoV8_ACC"
    },
    "Speed": {
        "temperature": 1.05,
        "threshold": 0.12,        # Muitas notas
        "hand_gap_divisor": 2.8,  # Permite streams muito rápidos
        "diagonal_probability": 0.2, # Ângulos simples para flow rápido
        "author_name": "AutoV8_Speed"
    },
    "Tech": {
        "temperature": 1.3,       # Padrões mais inesperados e complexos
        "threshold": 0.16,        # Densidade moderada
        "hand_gap_divisor": 1.8,  # Ritmo complexo, não necessariamente spam
        "diagonal_probability": 0.5, # Muitos ângulos e padrões estranhos
        "author_name": "AutoV8_Tech"
    },
    # Crie seus próprios híbridos! Ex: SpeedTech
    "SpeedTech": {
        "temperature": 1.2,       # Um pouco do caos do Tech
        "threshold": 0.13,        # Densidade do Speed
        "hand_gap_divisor": 2.5,  # Rápido, mas não no máximo
        "diagonal_probability": 0.4, # Ângulos frequentes
        "author_name": "AutoV8_SpeedTech"
    }
}
# ----------------------------------------------------

def zip_folder(folder_path, output_path):
    """Cria um arquivo ZIP a partir de uma pasta."""
    shutil.make_archive(output_path, 'zip', folder_path)
    print(f"Mapa compactado em {output_path}.zip")

def get_next_cut_direction_v2(last_direction, hand, line, layer, diagonal_probability):
    # Regras de fluxo básicas (similar ao anterior, mas podemos refinar)
    flow_rules = {
        0: ([1], [6, 7]), 1: ([0], [4, 5]), 2: ([3], [5, 7]), 3: ([2], [4, 6]),
        4: ([7], [1, 3]), 5: ([6], [1, 2]), 6: ([5], [0, 3]), 7: ([4], [0, 2]),
        8: ([0, 1, 2, 3, 4, 5, 6, 7], []) # Any direction
    }
    
    # Incentivar cortes para fora com base na mão e posição
    # Mão esquerda (0): prefere cortes para a esquerda (2, 4, 6)
    # Mão direita (1): prefere cortes para a direita (3, 5, 7)
    if hand == 0: # Mão esquerda
        if line <= 1: # Colunas da esquerda
            preferred_cuts = [2, 6, 4] 
        else: # Colunas da direita
            preferred_cuts = [0, 1]
    else: # Mão direita
        if line >= 2: # Colunas da direita
            preferred_cuts = [3, 7, 5]
        else: # Colunas da esquerda
            preferred_cuts = [0, 1]

    # Combina com as regras de fluxo
    cardinals, diagonals = flow_rules.get(last_direction, ([0, 1, 2, 3], [4, 5, 6, 7]))
    
    possible_choices = [c for c in cardinals if c in preferred_cuts] + \
                       [d for d in diagonals if d in preferred_cuts]
    
    # Se não houver opções preferidas, usa as regras de fluxo normais
    if not possible_choices:
        choices = cardinals + diagonals
        if not choices: return np.random.choice([0,1]) # Fallback
        
        if not diagonals or not cardinals:
            probs = [1.0 / len(choices)] * len(choices)
        else:
            prob_cardinal = (1 - diagonal_probability) / len(cardinals)
            prob_diagonal = diagonal_probability / len(diagonals)
            probs = ([prob_cardinal] * len(cardinals)) + ([prob_diagonal] * len(diagonals))
        
        return np.random.choice(choices, p=np.array(probs) / sum(probs))

    # Dá um peso maior para as escolhas preferidas
    return np.random.choice(possible_choices)

def post_process_notes_v8(logits, bpm, sr, hop_length, style_params):
    # Desempacota os parâmetros do perfil de estilo
    temperature = style_params["temperature"]
    threshold = style_params["threshold"]
    hand_gap_divisor = style_params["hand_gap_divisor"]
    diagonal_probability = style_params["diagonal_probability"]
    
    notes = []
    seconds_per_beat = 60.0 / bpm
    frame_duration = hop_length / sr
    min_hand_gap = (60.0 / bpm) / hand_gap_divisor
    cooldown_frames = 4  # Aumentado para mais espaço

    logits = logits.squeeze(0) / temperature
    probs = torch.sigmoid(logits).cpu().numpy()

    # Grid de cooldown por mão [mão][posição]
    cooldown_grid = np.zeros((2, 12), dtype=int) 
    last_cut_direction = {0: 1, 1: 0} # Começar com direções opostas
    last_note_time = {-1: -1, 0: -1, 1: -1}
    last_note_pos = {0: None, 1: None}
    
    # 0 para esquerda, 1 para direita. Começa com a mão direita para a primeira nota.
    parity_tracker = 1 

    for t in range(1, probs.shape[0] - 1):
        current_probs = probs[t].copy()

        # Aplicar cooldown específico da mão
        for hand in range(2):
            for i in range(12):
                if cooldown_grid[hand][i] > 0:
                    # Reduz a probabilidade se a *mesma mão* usou essa posição recentemente
                    current_probs[i] *= (0.1 / (hand + 1)) # Penalidade maior para a mão da nota
                    cooldown_grid[hand][i] -= 1

        if np.max(current_probs) < threshold: continue

        is_peak = (current_probs > probs[t-1]) & (current_probs > probs[t+1])
        peak_indices = np.where(is_peak & (current_probs > threshold))[0]
        
        if len(peak_indices) == 0: continue
            
        # Ordenar picos pela probabilidade
        sorted_peaks = sorted(peak_indices, key=lambda i: current_probs[i], reverse=True)

        chosen_note = None
        for index in sorted_peaks:
            time_sec = t * frame_duration
            
            # Tenta usar a mão da paridade atual
            hand = parity_tracker

            # Se a mão da paridade está em cooldown, tenta a outra mão
            if time_sec - last_note_time[hand] < min_hand_gap:
                hand = 1 - hand # Troca para a outra mão
                # Se a outra mão também está em cooldown, pula este pico
                if time_sec - last_note_time[hand] < min_hand_gap:
                    continue

            beat_quantized = round(time_sec / seconds_per_beat * 8) / 8.0
            if abs(beat_quantized - last_note_time[-1]) < 0.01: continue

            line = index % 4
            layer = index // 4

            # --- Lógica Anti-Vision Block ---
            other_hand = 1 - hand
            if last_note_pos[other_hand] is not None:
                last_line, last_layer = last_note_pos[other_hand]
                # Se a última nota da outra mão estava na mesma linha e a nova está na frente
                if last_line == line and layer > last_layer and (time_sec - last_note_time[other_hand]) < (seconds_per_beat * 2):
                     continue # Pula, pois pode ser um vision block

            # Se todas as verificações passaram, esta é a nossa nota
            chosen_note = {
                "index": index,
                "time_sec": time_sec,
                "beat_quantized": beat_quantized,
                "hand": hand,
                "line": line,
                "layer": layer
            }
            break # Sai do loop de picos

        if chosen_note is None:
            continue

        # Desempacota a nota escolhida
        index = chosen_note["index"]
        time_sec = chosen_note["time_sec"]
        beat_quantized = chosen_note["beat_quantized"]
        hand = chosen_note["hand"]
        line = chosen_note["line"]
        layer = chosen_note["layer"]

        # --- Lógica de Direção de Corte Melhorada ---
        cut_direction = get_next_cut_direction_v2(last_cut_direction[hand], hand, line, layer, diagonal_probability)
        
        note = {"_time": float(beat_quantized), "_lineIndex": int(line), "_lineLayer": int(layer), "_type": hand, "_cutDirection": int(cut_direction)}
        notes.append(note)
        
        # Atualiza o estado
        last_cut_direction[hand] = cut_direction
        last_note_time[hand] = time_sec
        last_note_time[-1] = beat_quantized
        last_note_pos[hand] = (line, layer)
        
        # Alterna a paridade para a próxima nota
        parity_tracker = 1 - hand

        # Aplica cooldown
        cooldown_grid[hand][index] = cooldown_frames
        # Cooldown geral na mesma linha para evitar sobreposição imediata
        for i in range(4):
            cooldown_grid[hand][layer * 4 + i] = max(cooldown_grid[hand][layer * 4 + i], cooldown_frames // 2)

    notes.sort(key=lambda x: x['_time'])
    return notes

def generate_map(audio_path, output_folder, style_params):
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    filename = os.path.basename(audio_path)
    print(f"Processando áudio: {filename} para a pasta {output_folder} (Estilo: {style_params['author_name']})")
    
    raw_bpm = detect_bpm(audio_path)
    bpm = round(raw_bpm)
    print(f"BPM Detectado: {raw_bpm:.2f} -> Ajustado para: {bpm}")

    processed_audio_path = os.path.join(output_folder, "song.ogg")
    add_silence(audio_path, processed_audio_path, silence_duration_ms=3000)

    print("Extraindo features...")
    features, sr, hop_length = extract_features(processed_audio_path, bpm)
    
    if features is None: return

    print(f"Gerando notas com IA V8 ({style_params['author_name']})...")
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
                
            notes = post_process_notes_v8(logits, bpm, sr, hop_length, style_params)
            print(f"Geradas {len(notes)} notas.")

        except Exception as e:
            print(f"Erro ao usar o modelo: {e}")
            import traceback
            traceback.print_exc()
    else:
        print(f"Modelo '{model_path}' não encontrado! Treine o modelo primeiro.")

    create_info_file(output_folder, bpm, style_params, song_filename="song.ogg")
    create_difficulty_file(output_folder, notes, bpm)
    
    zip_output_path = os.path.join("output", os.path.basename(output_folder))
    zip_folder(output_folder, zip_output_path)

    print(f"Mapa gerado e compactado com sucesso em {output_folder}\n")

def create_info_file(folder, bpm, style_params, song_filename="song.ogg"):
    info = {
        "_version": "2.0.0",
        "_songName": f"AI Map ({style_params['author_name']})",
        "_songSubName": "",
        "_songAuthorName": "BSIAMapper",
        "_levelAuthorName": style_params['author_name'],
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
        "customData": {"_time": final_beat + end_buffer_beats}
    }
    with open(os.path.join(folder, "ExpertPlus.dat"), 'w') as f:
        json.dump(diff, f, indent=2)

if __name__ == "__main__":
    # --- CONTROLE PRINCIPAL ---
    # 1. Escolha o estilo do mapa que você quer gerar.
    # Opções: "Standard", "ACC", "Speed", "Tech", "SpeedTech"
    TARGET_STYLE = "Speed"
    
    # 2. Defina quantos mapas diferentes (seeds) você quer gerar desse estilo.
    NUM_MAPS_TO_GENERATE = 3
    # --------------------------

    BASE_AUDIO_FILE = "musica.mp3"
    BASE_OUTPUT_NAME = f"MeuMapa_{TARGET_STYLE}"
    
    try:
        style_params = STYLE_PROFILES[TARGET_STYLE]
    except KeyError:
        print(f"Erro: Estilo '{TARGET_STYLE}' não encontrado! Usando 'Standard' como padrão.")
        TARGET_STYLE = "Standard"
        style_params = STYLE_PROFILES[TARGET_STYLE]

    print(f"Iniciando geração de {NUM_MAPS_TO_GENERATE} mapas no estilo '{TARGET_STYLE}'...")
    
    for i in range(NUM_MAPS_TO_GENERATE):
        np.random.seed()
        output_folder = f"output/{BASE_OUTPUT_NAME}_{i+1}"
        generate_map(BASE_AUDIO_FILE, output_folder, style_params)
