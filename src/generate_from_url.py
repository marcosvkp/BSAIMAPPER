import os
import shutil
import torch
import numpy as np
from youtube_downloader import download_from_youtube
from models_optimized import get_model, DirectorNet # Importar V1 para fallback
from pattern_manager import PatternManager
from flow_fixer import FlowFixer
from audio_processor import extract_features, detect_bpm, add_silence, analyze_energy
from utils import zip_folder, create_info_dat, save_difficulty_dat

# Mapeamento de IDs para Nomes de Dificuldade
DIFFICULTY_MAP = {
    1: "Easy",
    2: "Normal",
    3: "Hard",
    4: "Expert",
    5: "ExpertPlus"
}

# Parâmetros de Geração por Dificuldade (Heurísticas)
DIFFICULTY_PARAMS = {
    "Easy":       {"base_threshold": 0.35, "cooldown_mod": 2.0, "njs": 10, "offset": 0.0},
    "Normal":     {"base_threshold": 0.30, "cooldown_mod": 1.5, "njs": 12, "offset": 0.0},
    "Hard":       {"base_threshold": 0.25, "cooldown_mod": 1.2, "njs": 14, "offset": -0.2},
    "Expert":     {"base_threshold": 0.20, "cooldown_mod": 1.0, "njs": 16, "offset": -0.4},
    "ExpertPlus": {"base_threshold": 0.15, "cooldown_mod": 0.8, "njs": 18, "offset": -0.784}
}

def generate_difficulty(model, features, energy_profile, bpm, sr, hop_length, difficulty_name, intensity_factor=1.0, is_v2_model=False):
    """
    Gera as notas para uma dificuldade específica usando o modelo carregado.
    """
    print(f"   -> Gerando dificuldade: {difficulty_name} (Intensidade: {intensity_factor:.2f})...")
    
    params = DIFFICULTY_PARAMS[difficulty_name]
    base_threshold = params["base_threshold"]
    cooldown_modifier = params["cooldown_mod"]
    
    pattern_manager = PatternManager(difficulty=difficulty_name)
    
    device = next(model.parameters()).device
    inputs = torch.tensor(features, dtype=torch.float32).unsqueeze(0).to(device)
    
    with torch.no_grad():
        outputs = model(inputs)
        if is_v2_model:
            p_beat, p_comp, p_vert, p_cut = outputs
            cut_classes = torch.argmax(p_cut, dim=2).squeeze().cpu().numpy()
        else:
            p_beat, p_comp, p_vert = outputs
            cut_classes = None # Sem previsão de corte para o modelo V1

        beat_probs = torch.sigmoid(p_beat).squeeze().cpu().numpy()
        comp_classes = torch.argmax(p_comp, dim=2).squeeze().cpu().numpy()
        vert_classes = torch.argmax(p_vert, dim=2).squeeze().cpu().numpy()

    raw_notes = []
    frame_dur = hop_length / sr
    sec_per_beat = 60 / bpm
    
    base_cooldown = int(0.1 / frame_dur) * cooldown_modifier
    last_frame = -base_cooldown
    occupied_until_beat = 0.0
    MIN_START_TIME = 3.0 
    
    for i in range(len(beat_probs)):
        time_sec = i * frame_dur
        current_beat = time_sec / sec_per_beat
        
        if time_sec < MIN_START_TIME: continue
        if current_beat < occupied_until_beat: continue
            
        current_energy = energy_profile[i]
        
        energy_influence = 0.6
        if difficulty_name in ["Easy", "Normal"]:
            energy_influence = 0.3

        dynamic_threshold = base_threshold + (0.5 - current_energy) * energy_influence
        dynamic_threshold /= intensity_factor
        dynamic_threshold = np.clip(dynamic_threshold, 0.10, 0.95)
        
        current_cooldown = base_cooldown
        if current_energy > 0.75:
            current_cooldown = int(base_cooldown * 0.6)

        if beat_probs[i] > dynamic_threshold and (i - last_frame) > current_cooldown:
            if i > 0 and i < len(beat_probs)-1:
                if beat_probs[i] > beat_probs[i-1] and beat_probs[i] > beat_probs[i+1]:
                    
                    beat_time = round(current_beat * 8) / 8
                    
                    # Extrai a previsão de corte se disponível
                    predicted_cut = cut_classes[i] if cut_classes is not None else None
                    
                    meta = pattern_manager.get_pattern(
                        intensity=beat_probs[i],
                        complexity_idx=comp_classes[i],
                        vertical_idx=vert_classes[i],
                        predicted_cut_direction=predicted_cut, # Passa a previsão para o PatternManager
                        time_gap=(i - last_frame) * frame_dur,
                        energy_level=current_energy
                    )
                    
                    if meta: 
                        new_notes = pattern_manager.apply_pattern(meta, beat_time, bpm)
                        raw_notes.extend(new_notes)
                        last_frame = i
                        
                        if meta.get('type') in ['burst_fill', 'super_stream']:
                            occupied_until_beat = beat_time + (1.0 if meta['type'] == 'burst_fill' else 2.0)

    print(f"      Notas geradas (bruto): {len(raw_notes)}")
    
    if len(raw_notes) == 0:
        print(f"      AVISO: Nenhuma nota gerada para {difficulty_name}. Verifique os thresholds ou aumente a intensidade.")

    all_objects = FlowFixer.fix(raw_notes, bpm)
    final_notes = [obj for obj in all_objects if obj['_type'] != 3]
    final_bombs = [obj for obj in all_objects if obj['_type'] == 3]
    
    final_notes.sort(key=lambda x: x['_time'])
    final_bombs.sort(key=lambda x: x['_time'])
    
    unique_bombs = []
    seen_bombs = set()
    for b in final_bombs:
        key = (round(b['_time'], 3), b['_lineIndex'], b['_lineLayer'])
        if key not in seen_bombs:
            seen_bombs.add(key)
            unique_bombs.append(b)
            
    return final_notes, unique_bombs

def main():
    print("==================================================")
    print("   BEAT SABER AI MAPPER - PIPELINE COMPLETA V2")
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
    
    print("\n[2/5] Configuração de Dificuldades")
    print("Quais dificuldades deseja gerar? (1=E, 2=N, 3=H, 4=X, 5=X+)")
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
    
    print(f"Dificuldades selecionadas: {', '.join(selected_diffs)}")
    
    intensity_factor = 1.0
    try:
        intensity_str = input("Fator de Intensidade (ex: 1.0=padrão, 1.5=mais notas): ").strip()
        if intensity_str: intensity_factor = float(intensity_str)
    except ValueError:
        print("Entrada inválida. Usando intensidade padrão (1.0).")
        intensity_factor = 1.0

    output_folder = os.path.join("output", song_name)
    if os.path.exists(output_folder): shutil.rmtree(output_folder)
    os.makedirs(output_folder)
    
    final_audio_name = "song.egg"
    final_cover_name = "cover.png"
    
    print("\n[3/5] Analisando áudio e extraindo features...")
    bpm = detect_bpm(mp3_path)
    print(f"BPM Detectado: {bpm}")
    
    full_audio_path = os.path.join(output_folder, final_audio_name)
    add_silence(mp3_path, full_audio_path)
    
    if cover_path and os.path.exists(cover_path):
        shutil.copy(cover_path, os.path.join(output_folder, final_cover_name))
    else:
        from PIL import Image
        img = Image.new('RGB', (256, 256), color = (73, 109, 137))
        img.save(os.path.join(output_folder, final_cover_name))

    features, sr, hop_length = extract_features(full_audio_path, bpm)
    energy_profile = analyze_energy(full_audio_path, hop_length=hop_length, sr=sr)
    
    if len(energy_profile) > len(features):
        energy_profile = energy_profile[:len(features)]
    elif len(energy_profile) < len(features):
        energy_profile = np.pad(energy_profile, (0, len(features) - len(energy_profile)))

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = get_model().to(device)
    model_path = "models/director_net_v2.pth"
    is_v2_model = False
    
    if os.path.exists(model_path):
        print(f"Carregando modelo V2: '{model_path}'")
        model.load_state_dict(torch.load(model_path, map_location=device, weights_only=True))
        is_v2_model = True
    else:
        print(f"AVISO: Modelo V2 '{model_path}' não encontrado.")
        old_model_path = "models/director_net.pth"
        if os.path.exists(old_model_path):
            print(f"Usando modelo V1 de fallback: '{old_model_path}'")
            model = DirectorNet().to(device)
            model.load_state_dict(torch.load(old_model_path, map_location=device, weights_only=True))
        else:
            print("ERRO: Nenhum modelo treinado encontrado.")
            return
    model.eval()

    print("\n[4/5] Gerando mapas...")
    
    for diff in selected_diffs:
        notes, bombs = generate_difficulty(model, features, energy_profile, bpm, sr, hop_length, diff, intensity_factor, is_v2_model)
        save_difficulty_dat(notes, bombs, output_folder, f"{diff}.dat")
        
    info_content = create_info_dat(song_name, bpm, final_audio_name, final_cover_name, selected_diffs, DIFFICULTY_PARAMS)
    with open(os.path.join(output_folder, "Info.dat"), 'w') as f:
        import json
        json.dump(info_content, f, indent=2)
        
    print("\n[5/5] Finalizando...")
    zip_folder(output_folder, os.path.join("output", song_name))
    
    try:
        shutil.rmtree("data/temp_download")
    except:
        pass
        
    print("\nSUCESSO! Mapa gerado em 'output/'")

if __name__ == "__main__":
    main()
