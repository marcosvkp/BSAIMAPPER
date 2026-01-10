import os
import json
import shutil
import torch
import numpy as np
import copy
import random
import math
from youtube_downloader import download_from_youtube
from models_optimized import get_model, get_angle_model
from pattern_manager import PatternManager
from flow_fixer import FlowFixer
from audio_processor import extract_features, detect_bpm, add_silence, analyze_energy

# Mapeamento de IDs para Nomes de Dificuldade
DIFFICULTY_MAP = {
    1: "Easy",
    2: "Normal",
    3: "Hard",
    4: "Expert",
    5: "ExpertPlus"
}

# Parâmetros de Geração por Dificuldade (Valores Padrão)
# max_nps: Limite suave de notas por segundo
DIFFICULTY_PARAMS = {
    "Easy":       {"base_threshold": 0.35, "cooldown_mod": 2.5, "njs": 10, "offset": 0.0,    "max_nps": 2.5},
    "Normal":     {"base_threshold": 0.30, "cooldown_mod": 2.0, "njs": 12, "offset": 0.0,    "max_nps": 4.0},
    "Hard":       {"base_threshold": 0.25, "cooldown_mod": 1.5, "njs": 14, "offset": -0.2,   "max_nps": 5.5},
    "Expert":     {"base_threshold": 0.20, "cooldown_mod": 1.2, "njs": 16, "offset": -0.4,   "max_nps": 7.0},
    "ExpertPlus": {"base_threshold": 0.12, "cooldown_mod": 1.0, "njs": 18, "offset": -0.784, "max_nps": 9.0}
}

def zip_folder(folder_path, output_path):
    if os.path.exists(output_path + ".zip"):
        os.remove(output_path + ".zip")
    shutil.make_archive(output_path, 'zip', folder_path)
    print(f"Pacote final criado: {output_path}.zip")

def fix_vision_blocks(notes):
    """
    Corrige notas que bloqueiam a visão (Vision Blocks).
    Se duas notas de mãos opostas estão muito próximas no tempo e ocupam a mesma posição (X, Y),
    move a nota mais recente para uma camada diferente.
    """
    if not notes: return notes
    
    # Ordena por tempo
    notes.sort(key=lambda x: x['_time'])
    
    # Janela de tempo para considerar Vision Block (em beats)
    # 0.1 beats é bem curto, mas suficiente para bloquear a visão se estiver na frente
    VB_WINDOW = 1.25
    
    count_fixed = 0
    
    for i in range(len(notes)):
        current = notes[i]
        current_time = current['_time']
        current_line = current['_lineIndex']
        current_layer = current['_lineLayer']
        current_type = current['_type']
        
        # Olha para trás para encontrar conflitos
        for j in range(i - 1, -1, -1):
            prev = notes[j]
            dt = current_time - prev['_time']
            
            if dt > VB_WINDOW:
                break # Sai do loop se já passou da janela
            
            # Se for a mesma mão, não é vision block (é stream ou stack)
            if prev['_type'] == current_type:
                continue
                
            # Se estiverem na mesma posição X e Y
            if prev['_lineIndex'] == current_line and prev['_lineLayer'] == current_layer:
                # CONFLITO DETECTADO!
                # Estratégia: Mover a nota atual (current) para outra layer.
                
                # Tenta mover para cima primeiro
                if current_layer < 2:
                    current['_lineLayer'] += 1
                # Se já estiver no topo, move para baixo
                elif current_layer == 2:
                    current['_lineLayer'] = 1 # Move para o meio
                
                # Atualiza as variáveis locais para refletir a mudança (caso haja outro conflito)
                current_layer = current['_lineLayer']
                count_fixed += 1
                
                # Se ainda colidir (ex: tinha outra nota no meio), poderia tentar de novo,
                # mas uma mudança geralmente resolve o bloqueio direto.
                break
                
    if count_fixed > 0:
        print(f"      [Fix] Vision Blocks corrigidos: {count_fixed}")
        
    return notes

def evaluate_flow_quality(notes, other_hand_notes):
    """
    Avalia Flow + Colisões.
    """
    score = 0
    if not notes: return 0, 0
    
    notes.sort(key=lambda x: x['_time'])
    
    UP_GROUP = [0, 4, 5]
    DOWN_GROUP = [1, 6, 7]
    LEFT_GROUP = [2, 4, 6]
    RIGHT_GROUP = [3, 5, 7]
    
    last_cut = -1
    last_time = -10
    resets_count = 0
    collisions_count = 0
    
    for n in notes:
        cut = n['_cutDirection']
        time = n['_time']
        line = n['_lineIndex']
        
        # --- 1. Flow Check ---
        if cut != 8:
            dt = time - last_time
            is_reset = False
            if dt < 0.5:
                if (cut in UP_GROUP and last_cut in UP_GROUP): is_reset = True
                elif (cut in DOWN_GROUP and last_cut in DOWN_GROUP): is_reset = True
                elif (cut in LEFT_GROUP and last_cut in LEFT_GROUP): is_reset = True
                elif (cut in RIGHT_GROUP and last_cut in RIGHT_GROUP): is_reset = True
                
                if is_reset:
                    score -= 10.0
                    resets_count += 1
                else:
                    score += 2.0
            else:
                if is_reset: score -= 1.0
                else: score += 1.0
            last_cut = cut
            last_time = time
            
        # --- 2. Collision Check (Nova Lógica) ---
        # Procura nota da outra mão no mesmo tempo
        for other in other_hand_notes:
            dist = abs(other['_time'] - time)
            if dist > 0.1: continue # Longe no tempo
            
            other_line = other['_lineIndex']
            other_cut = other.get('_cutDirection', 8)
            
            # Colisão de Espaço: Mesma linha/layer (Impossível fisicamente)
            if line == other_line and n['_lineLayer'] == other['_lineLayer']:
                score -= 50.0
                collisions_count += 1
                continue
                
            # Colisão de Cruzamento (Hand Clap)
            # Ex: Esquerda na Col 1 cortando pra Direita (3)
            #     Direita na Col 2 cortando pra Esquerda (2)
            # Isso vai bater os sabres.

            # Se as mãos estão próximas (distância de 1 coluna)
            if abs(line - other_line) <= 1:
                # Se cortes convergem horizontalmente
                if cut in RIGHT_GROUP and other_cut in LEFT_GROUP: # -> <-
                    score -= 20.0
                    collisions_count += 1
                # Se cortes convergem verticalmente (menos grave mas perigoso)
                # Ex: Cima e Baixo na mesma coluna vertical
                if line == other_line:
                     score -= 20.0
                     collisions_count += 1

    return score, resets_count + collisions_count

def apply_angle_net(notes, angle_model, device, num_attempts=10, enforce_flow=False):
    if not notes: return notes
    
    print(f"      Aplicando AngleNet (Collision Aware - Best of {num_attempts})...")
    if enforce_flow:
        print("      [Hybrid Mode] Respeitando flow sugerido pelo FlowFixer.")
    
    left_notes = [n for n in notes if n['_type'] == 0]
    right_notes = [n for n in notes if n['_type'] == 1]
    
    def process_hand_autoregressive(hand_notes, other_hand_notes, hand_idx, hand_name):
        if not hand_notes: return []
        
        hand_notes.sort(key=lambda x: x['_time'])
        seq_len = len(hand_notes)
        
        # Prepara inputs estáticos
        pos_indices = []
        time_diffs = []
        hand_indices = []
        other_pos_indices = []
        other_time_diffs = []
        
        prev_time = hand_notes[0]['_time']
        
        for n in hand_notes:
            pos = min(11, max(0, (n['_lineLayer'] * 4) + n['_lineIndex']))
            pos_indices.append(pos)
            hand_indices.append(hand_idx)
            dt = n['_time'] - prev_time
            dt = min(4.0, max(0.0, dt))
            time_diffs.append(dt)
            prev_time = n['_time']
            
            # Busca outra mão
            best_dist = 100.0
            best_other = None
            for on in other_hand_notes:
                d = abs(on['_time'] - n['_time'])
                if d < best_dist:
                    best_dist = d
                    best_other = on
            
            if best_other and best_dist < 0.5:
                op = min(11, max(0, (best_other['_lineLayer'] * 4) + best_other['_lineIndex']))
                other_pos_indices.append(op)
                other_time_diffs.append(best_dist)
            else:
                other_pos_indices.append(12)
                other_time_diffs.append(1.0)
            
        pos_t = torch.tensor(pos_indices, dtype=torch.long).unsqueeze(0).to(device)
        hand_t = torch.tensor(hand_indices, dtype=torch.long).unsqueeze(0).to(device)
        dt_t = torch.tensor(time_diffs, dtype=torch.float32).unsqueeze(0).unsqueeze(-1).to(device)
        opos_t = torch.tensor(other_pos_indices, dtype=torch.long).unsqueeze(0).to(device)
        odt_t = torch.tensor(other_time_diffs, dtype=torch.float32).unsqueeze(0).unsqueeze(-1).to(device)
        
        best_seq = []
        best_score = -float('inf')
        
        print(f"         [{hand_name}] Iniciando {num_attempts} tentativas...")
        
        for attempt in range(num_attempts):
            current_prev_angle = 9
            generated_angles = []
            prev_angles_t = torch.full((1, seq_len), 9, dtype=torch.long).to(device)
            
            for t in range(seq_len):
                if t > 0: prev_angles_t[:, t] = current_prev_angle
                
                with torch.no_grad():
                    logits = angle_model(
                        pos_t[:, :t+1], hand_t[:, :t+1], dt_t[:, :t+1], 
                        prev_angles_t[:, :t+1], opos_t[:, :t+1], odt_t[:, :t+1]
                    )
                    last_logit = logits[:, -1, :]
                
                if enforce_flow:
                    suggested_cut = hand_notes[t]['_cutDirection']
                    mask = torch.full_like(last_logit, -float('inf'))
                    
                    allowed = []
                    if suggested_cut in [0, 4, 5]: allowed = [0, 4, 5]
                    elif suggested_cut in [1, 6, 7]: allowed = [1, 6, 7]
                    elif suggested_cut == 2: allowed = [2]
                    elif suggested_cut == 3: allowed = [3]
                    else: allowed = [0, 1, 2, 3, 4, 5, 6, 7, 8]
                    
                    mask[0, allowed] = 0
                    last_logit = last_logit + mask

                temperature = 0.7
                probs = torch.softmax(last_logit / temperature, dim=1)
                pred = torch.multinomial(probs, 1).item()
                
                generated_angles.append(pred)
                current_prev_angle = pred
            
            candidate_notes = copy.deepcopy(hand_notes)
            for i, n in enumerate(candidate_notes):
                n['_cutDirection'] = generated_angles[i]
            
            score, issues = evaluate_flow_quality(candidate_notes, other_hand_notes)

            # print(f"            Tentativa {attempt+1}: Score {score:.1f} | Issues: {issues}")

            if score > best_score:
                best_score = score
                best_seq = candidate_notes
        
        print(f"         [{hand_name}] Melhor Score: {best_score:.1f}")
        return best_seq

    # Processa Esquerda (olhando para Direita original)
    new_left = process_hand_autoregressive(left_notes, right_notes, 0, "Left Hand")

    # Processa Direita (olhando para Esquerda NOVA - já refinada!)
    # Isso é importante: A mão direita deve reagir ao que a esquerda JÁ DECIDIU fazer.
    new_right = process_hand_autoregressive(right_notes, new_left, 1, "Right Hand")
    
    return new_left + new_right

def print_note_distribution(notes, bpm):
    """
    Imprime a distribuição de notas em intervalos de 20 segundos.
    """
    if not notes:
        print("      [Debug] Nenhuma nota gerada.")
        return

    sec_per_beat = 60 / bpm
    # Encontrar o tempo da última nota em segundos
    last_note_beat = max(n['_time'] for n in notes)
    total_seconds = last_note_beat * sec_per_beat
    
    interval = 20 # segundos
    num_buckets = math.ceil(total_seconds / interval)
    
    buckets = [0] * (num_buckets + 1)
    
    for n in notes:
        time_sec = n['_time'] * sec_per_beat
        bucket_idx = int(time_sec // interval)
        if bucket_idx < len(buckets):
            buckets[bucket_idx] += 1
            
    print("\n      [DEBUG] Distribuição de Notas (Intervalos de 20s):")
    for i, count in enumerate(buckets):
        if count == 0 and i > num_buckets: continue
        start_t = i * interval
        end_t = (i + 1) * interval
        
        # Formatação mm:ss
        start_str = f"{int(start_t//60)}:{int(start_t%60):02d}"
        end_str = f"{int(end_t//60)}:{int(end_t%60):02d}"
        
        print(f"      - {start_str} a {end_str}: {count} notas")
    print("")

def generate_difficulty(model, angle_model, features, energy_profile, bpm, sr, hop_length, difficulty_name, difficulty_params, flow_mode, flow_intensity=1.0):
    print(f"   -> Gerando dificuldade: {difficulty_name}")
    
    base_threshold = difficulty_params["base_threshold"]
    cooldown_modifier = difficulty_params["cooldown_mod"]
    max_nps = difficulty_params.get("max_nps", 10.0)
    
    print(f"      [Config] Max NPS Alvo: {max_nps}")
    
    pattern_manager = PatternManager(difficulty=difficulty_name)
    device = next(model.parameters()).device
    
    frame_dur = hop_length / sr
    sec_per_beat = 60 / bpm
    total_frames = features.shape[0]
    total_duration_sec = total_frames * frame_dur
    
    current_mem = [9] * 8 
    current_grid_pos = 6
    
    raw_notes = []
    last_frame = -100
    occupied_until_beat = 0.0
    
    # Definição de Hot Start e End
    MIN_START_TIME = 3.0 
    MAX_END_TIME = total_duration_sec - 2.0
    
    BLOCK_SIZE = 400 
    STEP = 100 
    
    all_probs = np.zeros(total_frames)
    
    for start_idx in range(0, total_frames, STEP):
        end_idx = min(start_idx + BLOCK_SIZE, total_frames)
        if end_idx - start_idx < 10: break
        
        feat_block = features[start_idx:end_idx]
        block_len = feat_block.shape[0]
        
        grid_tensor = torch.full((1, block_len), current_grid_pos, dtype=torch.long).to(device)
        mem_tensor = torch.tensor(current_mem, dtype=torch.long).unsqueeze(0).repeat(1, block_len, 1).to(device)
        
        input_tensor = torch.tensor(feat_block, dtype=torch.float32).unsqueeze(0).to(device)
        
        with torch.no_grad():
            p_beat, p_comp, p_vert, p_angle = model(input_tensor, grid_tensor, mem_tensor)
            beat_probs = torch.sigmoid(p_beat).squeeze().cpu().numpy()
            comp_classes = torch.argmax(p_comp, dim=2).squeeze().cpu().numpy()
            vert_classes = torch.argmax(p_vert, dim=2).squeeze().cpu().numpy()
            angle_classes = torch.argmax(p_angle, dim=2).squeeze().cpu().numpy()

        valid_len = min(STEP, block_len)
        
        for i in range(valid_len):
            global_idx = start_idx + i
            if global_idx >= total_frames: break
            
            prob = beat_probs[i]
            all_probs[global_idx] = prob
            
            time_sec = global_idx * frame_dur
            current_beat = time_sec / sec_per_beat
            
            # --- REGRAS DE TEMPO (Start / End) ---
            if time_sec < MIN_START_TIME: continue
            if time_sec > MAX_END_TIME: continue
            
            if current_beat < occupied_until_beat: continue
            
            # --- VERIFICAÇÃO DE NPS (Densidade) ---
            # Verifica quantas notas existem no último 1.0 segundo
            # Janela de tempo em beats = 1.0 / sec_per_beat
            window_beats = 1.0 / sec_per_beat
            
            # Otimização: Olhar apenas as últimas 20 notas para não percorrer a lista toda
            recent_count = 0
            check_limit = 20
            start_check = max(0, len(raw_notes) - check_limit)
            
            for n_idx in range(len(raw_notes) - 1, start_check - 1, -1):
                if raw_notes[n_idx]['_time'] > (current_beat - window_beats):
                    recent_count += 1
                else:
                    break # Como está ordenado, se passou do tempo, paramos
            
            if recent_count >= max_nps:
                continue # Pula geração se excedeu NPS
            
            current_energy = energy_profile[global_idx]
            
            energy_influence = 0.6 if difficulty_name not in ["Easy", "Normal"] else 0.3
            dynamic_threshold = base_threshold + (0.5 - current_energy) * energy_influence
            dynamic_threshold = np.clip(dynamic_threshold, 0.05, 0.95)
            
            current_cooldown = int(0.1 / frame_dur) * cooldown_modifier
            if current_energy > 0.75: current_cooldown = int(current_cooldown * 0.6)
            
            if prob > dynamic_threshold and (global_idx - last_frame) > current_cooldown:
                is_peak = True
                if i > 0 and i < len(beat_probs)-1:
                    if prob < beat_probs[i-1] or prob < beat_probs[i+1]:
                        is_peak = False
                
                if is_peak:
                    beat_time = round(current_beat * 8) / 8
                    comp_idx = comp_classes[i]
                    vert_idx = vert_classes[i]
                    angle_idx = angle_classes[i]

                    if difficulty_name == "ExpertPlus" and comp_idx < 2 and current_energy > 0.6:
                        comp_idx += 1

                    meta = pattern_manager.get_pattern(prob, comp_idx, vert_idx, angle_idx, (global_idx - last_frame) * frame_dur, energy_level=current_energy)

                    if meta:
                        new_notes = pattern_manager.apply_pattern(meta, beat_time, bpm)
                        if new_notes:
                            raw_notes.extend(new_notes)
                            last_frame = global_idx

                            last_note = new_notes[-1]
                            cut_dir = last_note.get('_cutDirection', 8)
                            line = last_note.get('_lineIndex', 0)
                            layer = last_note.get('_lineLayer', 0)

                            current_grid_pos = (layer * 4) + line
                            current_mem.pop(0)
                            current_mem.append(cut_dir)

                            if meta['type'] == 'burst_fill':
                                occupied_until_beat = beat_time + 1.0
                            elif meta['type'] == 'super_stream':
                                occupied_until_beat = beat_time + 2.0

    print(f"      [Debug] Max Prob: {np.max(all_probs):.4f}")
    print(f"      Notas geradas (Raw): {len(raw_notes)}")
    
    if len(raw_notes) < 10:
        print("      AVISO: Poucas notas geradas.")

    final_notes = []
    final_bombs = []

    if flow_mode == 'ai':
        if angle_model:
            final_notes = apply_angle_net(raw_notes, angle_model, device, num_attempts=10)
        else:
            print("      AVISO: AngleNet não carregada. Usando Raw.")
            final_notes = raw_notes
            
    elif flow_mode == 'fixer':
        if flow_intensity <= 0.01:
            final_notes = raw_notes
        elif flow_intensity >= 0.99:
            all_objects = FlowFixer.fix(copy.deepcopy(raw_notes), bpm)
            final_notes = [obj for obj in all_objects if obj['_type'] != 3]
            final_bombs = [obj for obj in all_objects if obj['_type'] == 3]
        else:
            print(f"      Aplicando FlowFixer Híbrido ({flow_intensity})")
            raw_backup = copy.deepcopy(raw_notes)
            fixed_objects = FlowFixer.fix(raw_backup, bpm)
            fixed_notes_map = {}
            fixed_bombs_list = []
            
            for obj in fixed_objects:
                if obj['_type'] == 3: fixed_bombs_list.append(obj)
                else:
                    key = (round(obj['_time'], 3))
                    if key not in fixed_notes_map: fixed_notes_map[key] = []
                    fixed_notes_map[key].append(obj)

            for raw_note in raw_notes:
                t_key = round(raw_note['_time'], 3)
                if random.random() < flow_intensity:
                    if t_key in fixed_notes_map and fixed_notes_map[t_key]:
                        final_notes.append(fixed_notes_map[t_key].pop(0))
                    else:
                        final_notes.append(raw_note)
                else:
                    final_notes.append(raw_note)
            
            for bomb in fixed_bombs_list:
                if random.random() < flow_intensity:
                    final_bombs.append(bomb)
    
    elif flow_mode == 'hybrid':
        print("      [Hybrid] 1. Aplicando FlowFixer para corrigir resets...")
        fixed_objects = FlowFixer.fix(copy.deepcopy(raw_notes), bpm)
        fixed_notes = [obj for obj in fixed_objects if obj['_type'] != 3]
        fixed_bombs = [obj for obj in fixed_objects if obj['_type'] == 3]
        
        if angle_model:
            print("      [Hybrid] 2. Aplicando AngleNet com restrições de flow...")
            final_notes = apply_angle_net(fixed_notes, angle_model, device, num_attempts=10, enforce_flow=True)
        else:
            print("      AVISO: AngleNet não carregada. Usando resultado do FlowFixer.")
            final_notes = fixed_notes
            
        final_bombs = fixed_bombs

    else:
        final_notes = raw_notes

    # --- CORREÇÃO DE VISION BLOCK ---
    final_notes = fix_vision_blocks(final_notes)

    final_notes.sort(key=lambda x: x['_time'])
    final_bombs.sort(key=lambda x: x['_time'])
    
    # --- DEBUG DE DISTRIBUIÇÃO ---
    print_note_distribution(final_notes, bpm)
    
    unique_bombs = []
    seen_bombs = set()
    for b in final_bombs:
        key = (round(b['_time'], 3), b['_lineIndex'], b['_lineLayer'])
        if key not in seen_bombs:
            seen_bombs.add(key)
            unique_bombs.append(b)
            
    return final_notes, unique_bombs

def create_info_dat(song_name, bpm, audio_filename, cover_filename, custom_difficulty_params):
    beatmap_sets = []
    for diff_name, params in custom_difficulty_params.items():
        beatmap_sets.append({
            "_difficulty": diff_name,
            "_beatmapFilename": f"{diff_name}.dat",
            "_noteJumpMovementSpeed": params["njs"],
            "_noteJumpStartBeatOffset": params["offset"]
        })
        
    info = {
        "_version": "2.1.0",
        "_songName": song_name,
        "_songSubName": "AI Generated",
        "_songAuthorName": "Artist",
        "_levelAuthorName": "BSIAMapper",
        "_beatsPerMinute": float(bpm),
        "_songFilename": audio_filename,
        "_coverImageFilename": cover_filename,
        "_environmentName": "DefaultEnvironment",
        "_songTimeOffset": 0,
        "_difficultyBeatmapSets": [{
            "_beatmapCharacteristicName": "Standard",
            "_difficultyBeatmaps": beatmap_sets
        }]
    }
    return info

def save_difficulty_dat(notes, bombs, folder, filename):
    all_objects = notes + bombs
    all_objects.sort(key=lambda x: x['_time'])
    
    data = {
        "_version": "2.0.0",
        "_notes": all_objects,
        "_events": [],
        "_obstacles": [],
        "_customData": {
            "_time": notes[-1]['_time'] + 4 if notes else 0,
            "_BPMChanges": [],
            "_bookmarks": []
        }
    }
    
    with open(os.path.join(folder, filename), 'w') as f:
        json.dump(data, f)

def main():
    print("==================================================")
    print("   BEAT SABER AI MAPPER - PIPELINE COMPLETA")
    print("==================================================")
    
    url = input("Digite a URL da música (YouTube): ").strip()
    if not url:
        print("URL inválida.")
        return

    print("\n[1/5] Baixando e processando áudio...")
    mp3_path, cover_path = download_from_youtube(url, output_folder="data/temp_download")
    
    if not mp3_path:
        print("Falha no download.")
        return
        
    song_name = os.path.splitext(os.path.basename(mp3_path))[0]
    
    print("\n[2/5] Analisando BPM...")
    bpm = detect_bpm(mp3_path)
    print(f"BPM Detectado: {bpm}")
    
    if bpm < 100:
        print(f"AVISO: BPM detectado ({bpm}) parece baixo para Nightcore/Pop.")
        print("Deseja dobrar o BPM para melhorar a sincronia? (Recomendado: Sim)")
        choice = input("Dobrar BPM? (S/n): ").strip().lower()
        if choice != 'n':
            bpm *= 2
            print(f"Novo BPM: {bpm}")
    
    print("\n[3/5] Configuração de Dificuldades")
    print("Quais dificuldades deseja gerar?")
    print("   1 = Easy, 2 = Normal, 3 = Hard, 4 = Expert, 5 = ExpertPlus")
    
    choices = input("Exemplo: 1,3,5 -> ").strip()
    selected_diffs = []
    
    try:
        parts = choices.split(',')
        for p in parts:
            val = int(p.strip())
            if val in DIFFICULTY_MAP:
                selected_diffs.append(DIFFICULTY_MAP[val])
    except:
        print("Entrada inválida. Gerando apenas Expert (4).")
        selected_diffs = ["Expert"]
        
    if not selected_diffs:
        selected_diffs = ["Expert"]
        
    order = ["Easy", "Normal", "Hard", "Expert", "ExpertPlus"]
    selected_diffs.sort(key=lambda x: order.index(x))
    
    custom_difficulty_params = {}
    print("\n[4/5] Ajuste Fino da Dificuldade")
    for diff_name in selected_diffs:
        print(f"\n   Configurando '{diff_name}':")
        
        # 1. Multiplicador de Densidade (Threshold)
        while True:
            try:
                multiplier_str = input(f"      - Multiplicador de Notas (1.0 = Padrão): ").strip()
                if not multiplier_str:
                    multiplier = 1.0
                else:
                    multiplier = float(multiplier_str)
                
                if multiplier <= 0:
                    print("         O multiplicador deve ser positivo.")
                    continue
                break
            except ValueError:
                print("         Entrada inválida.")

        # 2. Max NPS
        default_nps = DIFFICULTY_PARAMS[diff_name]["max_nps"]
        while True:
            try:
                nps_str = input(f"      - Max NPS (Padrão: {default_nps}): ").strip()
                if not nps_str:
                    user_nps = default_nps
                else:
                    user_nps = float(nps_str)
                
                if user_nps <= 0:
                    print("         NPS deve ser positivo.")
                    continue
                break
            except ValueError:
                print("         Entrada inválida.")

        new_params = DIFFICULTY_PARAMS[diff_name].copy()
        new_params['base_threshold'] /= multiplier
        new_params['cooldown_mod'] /= multiplier
        new_params['max_nps'] = user_nps
        
        custom_difficulty_params[diff_name] = new_params

    # --- ESCOLHA DO MÉTODO DE FLOW ---
    print("\n[DEBUG] Configuração do Flow (Pós-Processamento)")
    print("Escolha o método para definir os ângulos das notas:")
    print("   1 = AngleNet (IA Treinada - Imita Humanos)")
    print("   2 = FlowFixer (Regras Matemáticas - Seguro)")
    print("   3 = Raw (Sem correção - Apenas o que a DirectorNet gerou)")
    print("   4 = Hybrid (FlowFixer -> AngleNet) [NOVO]")

    flow_choice = input("Opção [2]: ").strip()
    flow_mode = 'fixer'
    flow_intensity = 1.0
    
    if flow_choice == '1':
        flow_mode = 'ai'
        print("   -> Usando AngleNet (IA) com Refinamento Iterativo.")
    elif flow_choice == '3':
        flow_mode = 'raw'
        print("   -> Usando Raw (Sem correção).")
    elif flow_choice == '4':
        flow_mode = 'hybrid'
        print("   -> Usando Hybrid (FlowFixer define flow, AngleNet define ângulos).")
    else:
        flow_mode = 'fixer'
        print("   -> Usando FlowFixer (Regras).")
        print("      Intensidade do FlowFixer (0.1 a 1.0) [1.0]:")
        try:
            intensity_str = input("      Intensidade: ").strip()
            if intensity_str:
                flow_intensity = float(intensity_str)
        except:
            pass

    output_folder = os.path.join("output", song_name)
    if os.path.exists(output_folder):
        shutil.rmtree(output_folder)
    os.makedirs(output_folder)
    
    final_audio_name = "song.egg"
    final_cover_name = "cover.png"
    
    full_audio_path = os.path.join(output_folder, final_audio_name)
    add_silence(mp3_path, full_audio_path)
    
    if cover_path and os.path.exists(cover_path):
        shutil.copy(cover_path, os.path.join(output_folder, final_cover_name))
    else:
        from PIL import Image
        img = Image.new('RGB', (256, 256), color = (73, 109, 137))
        img.save(os.path.join(output_folder, final_cover_name))

    print("      Extraindo features e energia...")
    features, sr, hop_length = extract_features(full_audio_path, bpm)
    energy_profile = analyze_energy(full_audio_path, hop_length=hop_length, sr=sr)
    
    if len(energy_profile) > len(features):
        energy_profile = energy_profile[:len(features)]
    elif len(energy_profile) < len(features):
        energy_profile = np.pad(energy_profile, (0, len(features) - len(energy_profile)))

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    model = get_model().to(device)
    model_path = "models/director_net_best.pth"
    if not os.path.exists(model_path): model_path = "models/director_net.pth"
    if os.path.exists(model_path):
        print(f"      Carregando DirectorNet: {model_path}")
        model.load_state_dict(torch.load(model_path, map_location=device, weights_only=True))
        model.eval()
    else:
        print("ERRO: DirectorNet não encontrada.")
        return

    angle_model = None
    if flow_mode in ['ai', 'hybrid']:
        angle_model = get_angle_model().to(device)
        angle_path = "models/angle_net.pth"
        if os.path.exists(angle_path):
            print(f"      Carregando AngleNet: {angle_path}")
            angle_model.load_state_dict(torch.load(angle_path, map_location=device, weights_only=True))
            angle_model.eval()
        else:
            print("      AVISO: AngleNet não encontrada! Treine com 'python src/train_angle.py'.")
            if flow_mode == 'ai':
                print("      Voltando para FlowFixer.")
                flow_mode = 'fixer'

    print("\n[5/5] Gerando mapas...")
    
    for diff_name, diff_params in custom_difficulty_params.items():
        notes, bombs = generate_difficulty(model, angle_model, features, energy_profile, bpm, sr, hop_length, diff_name, diff_params, flow_mode, flow_intensity)
        save_difficulty_dat(notes, bombs, output_folder, f"{diff_name}.dat")
        
    info_content = create_info_dat(song_name, bpm, final_audio_name, final_cover_name, custom_difficulty_params)
    with open(os.path.join(output_folder, "Info.dat"), 'w') as f:
        json.dump(info_content, f, indent=2)
        
    print("\n[6/6] Finalizando...")
    zip_folder(output_folder, os.path.join("output", song_name))
    
    try:
        shutil.rmtree("data/temp_download")
    except:
        pass
        
    print("\nSUCESSO! Mapa gerado em 'output/'")

if __name__ == "__main__":
    main()
