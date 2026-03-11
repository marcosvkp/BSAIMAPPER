import os
import json
import shutil
import torch
import numpy as np
from youtube_downloader import download_from_youtube
from models_optimized import get_model
from pattern_manager import PatternManager
from flow_fixer import FlowFixer
from audio_processor import extract_features, detect_bpm, add_silence, analyze_energy

# Mapeamento de IDs para Nomes de Dificuldade
DIFFICULTY_MAP = {1: "Easy", 2: "Normal", 3: "Hard", 4: "Expert", 5: "ExpertPlus"}

# Parâmetros base por dificuldade. O threshold agora será modificado dinamicamente pelas estrelas.
DIFFICULTY_PARAMS = {
    "Easy":       {"njs": 10, "offset": 0.0},
    "Normal":     {"njs": 12, "offset": 0.0},
    "Hard":       {"njs": 14, "offset": -0.2},
    "Expert":     {"njs": 16, "offset": -0.4},
    "ExpertPlus": {"njs": 18, "offset": -0.7}
}

def zip_folder(folder_path, output_path):
    if os.path.exists(output_path + ".zip"): os.remove(output_path + ".zip")
    shutil.make_archive(output_path, 'zip', folder_path)
    print(f"Pacote final criado: {output_path}.zip")

def generate_difficulty(model, features, energy_profile, bpm, sr, hop_length, difficulty_name, target_stars):
    """
    Gera as notas para uma dificuldade específica, agora condicionado pelo target_stars.
    """
    print(f"   -> Gerando: {difficulty_name} ({target_stars:.2f} estrelas)...")
    
    device = next(model.parameters()).device
    inputs = torch.tensor(features, dtype=torch.float32).unsqueeze(0).to(device)
    stars_tensor = torch.tensor([[target_stars]], dtype=torch.float32).to(device)
    
    with torch.no_grad():
        # O modelo agora recebe as estrelas como entrada
        p_beat, p_comp, p_vert = model(inputs, stars_tensor)
        beat_probs = torch.sigmoid(p_beat).squeeze().cpu().numpy()
        comp_classes = torch.argmax(p_comp, dim=2).squeeze().cpu().numpy()
        vert_classes = torch.argmax(p_vert, dim=2).squeeze().cpu().numpy()

    # --- Lógica de Densidade de Notas (NPS) baseada em Estrelas ---
    # Mapeia estrelas para um fator de densidade. Ex: 1 estrela = 0.5, 10 estrelas = 1.5
    # Isso ajusta o quão "agressivamente" o modelo colocará notas.
    nps_target_factor = np.interp(target_stars, [1, 13], [0.6, 1.8])
    base_threshold = 0.5 / nps_target_factor

    pattern_manager = PatternManager(difficulty=difficulty_name)
    raw_notes = []
    frame_dur = hop_length / sr
    sec_per_beat = 60 / bpm
    
    # Cooldown dinâmico baseado em estrelas
    cooldown_factor = np.interp(target_stars, [1, 13], [2.5, 0.7])
    base_cooldown = int((0.1 / frame_dur) * cooldown_factor)
    last_frame = -base_cooldown
    occupied_until_beat = 0.0
    MIN_START_TIME = 3.0 
    
    for i in range(len(beat_probs)):
        time_sec = i * frame_dur
        if time_sec < MIN_START_TIME: continue
        
        current_beat = time_sec / sec_per_beat
        if current_beat < occupied_until_beat: continue
            
        current_energy = energy_profile[i]
        
        # Threshold dinâmico ajustado pela energia da música
        dynamic_threshold = base_threshold + (0.5 - current_energy) * 0.4
        dynamic_threshold = np.clip(dynamic_threshold, 0.1, 0.9)
        
        current_cooldown = base_cooldown * (1.0 - (current_energy * 0.5)) # Menos cooldown em partes energéticas

        if beat_probs[i] > dynamic_threshold and (i - last_frame) > current_cooldown:
            if i > 0 and i < len(beat_probs)-1 and beat_probs[i] > beat_probs[i-1] and beat_probs[i] > beat_probs[i+1]:
                beat_time = round(current_beat * 8) / 8.0 # Arredonda para o 1/8 de beat mais próximo
                
                meta = pattern_manager.get_pattern(beat_probs[i], comp_classes[i], vert_classes[i], (i - last_frame) * frame_dur, energy_level=current_energy)
                
                if meta: 
                    new_notes = pattern_manager.apply_pattern(meta, beat_time, bpm)
                    raw_notes.extend(new_notes)
                    last_frame = i
                    
                    if meta['type'] in ['burst_fill', 'super_stream']:
                        occupied_until_beat = beat_time + (0.25 * len(new_notes))

    print(f"      Notas geradas (bruto): {len(raw_notes)}")
    if not raw_notes:
        print(f"      AVISO: Nenhuma nota gerada para {difficulty_name}. Tente um valor de estrelas mais alto ou verifique o modelo.")

    all_objects = FlowFixer.fix(raw_notes, bpm)
    final_notes = [obj for obj in all_objects if obj['_type'] != 3]
    final_bombs = [obj for obj in all_objects if obj['_type'] == 3]
    
    final_notes.sort(key=lambda x: x['_time'])
    unique_bombs = [dict(t) for t in {tuple(d.items()) for d in final_bombs}]
    unique_bombs.sort(key=lambda x: x['_time'])
            
    return final_notes, unique_bombs

def create_info_dat(song_name, bpm, audio_filename, cover_filename, difficulties_data):
    beatmap_sets = []
    for diff_name, data in difficulties_data.items():
        params = DIFFICULTY_PARAMS[diff_name]
        beatmap_sets.append({
            "_difficulty": diff_name,
            "_beatmapFilename": f"{diff_name}.dat",
            "_noteJumpMovementSpeed": params["njs"],
            "_noteJumpStartBeatOffset": params["offset"],
        })
        
    info = {
        "_version": "2.1.0",
        "_songName": song_name,
        "_songSubName": "AI Generated",
        "_songAuthorName": "Artist",
        "_levelAuthorName": "BSIAMapperV2",
        "_beatsPerMinute": float(bpm),
        "_songFilename": audio_filename,
        "_coverImageFilename": cover_filename,
        "_environmentName": "DefaultEnvironment",
        "_songTimeOffset": 0,
        "_difficultyBeatmapSets": [{"_beatmapCharacteristicName": "Standard", "_difficultyBeatmaps": beatmap_sets}]
    }
    return info

def save_difficulty_dat(notes, bombs, folder, filename):
    all_objects = sorted(notes + bombs, key=lambda x: x['_time'])
    data = {"_version": "2.2.0", "_notes": all_objects, "_obstacles": [], "_events": []}
    with open(os.path.join(folder, filename), 'w') as f: json.dump(data, f)

def main():
    print("\n   BEAT SABER AI MAPPER V2 - GERAÇÃO POR URL\n" + "="*50)
    
    url = input("Digite a URL da música (YouTube): ").strip()
    if not url: print("URL inválida."); return

    print("\n[1/5] Baixando e processando áudio...")
    mp3_path, cover_path = download_from_youtube(url, output_folder="data/temp_download")
    if not mp3_path: print("Falha no download."); return
    song_name = os.path.splitext(os.path.basename(mp3_path))[0]
    
    print("\n[2/5] Configuração de Dificuldades")
    print("Quais dificuldades deseja gerar? (1=Easy, 2=Normal, 3=Hard, 4=Expert, 5=ExpertPlus)")
    choices = input("Exemplo: 3,4,5 -> ").strip()
    
    selected_diffs = []
    try:
        for p in choices.split(','):
            val = int(p.strip())
            if val in DIFFICULTY_MAP: selected_diffs.append(DIFFICULTY_MAP[val])
    except: selected_diffs = []
    if not selected_diffs: print("Seleção inválida. Usando Expert (4) por padrão."); selected_diffs = ["Expert"]
    
    order = list(DIFFICULTY_MAP.values())
    selected_diffs.sort(key=lambda x: order.index(x))
    
    # --- Nova Etapa: Perguntar as Estrelas para cada dificuldade ---
    difficulties_with_stars = {}
    print("\n[3/5] Defina o nível de estrelas para cada dificuldade:")
    for diff_name in selected_diffs:
        while True:
            try:
                stars_input = float(input(f"  - {diff_name}: ").strip())
                if stars_input > 0:
                    difficulties_with_stars[diff_name] = {"stars": stars_input}
                    break
                else: print("  Por favor, insira um número positivo.")
            except ValueError: print("  Entrada inválida. Use um número (ex: 8.87).")

    output_folder = os.path.join("output", song_name.replace(" ", "_"))
    if os.path.exists(output_folder): shutil.rmtree(output_folder)
    os.makedirs(output_folder)
    
    final_audio_name, final_cover_name = "song.egg", "cover.png"
    
    print("\n[4/5] Analisando áudio e extraindo features...")
    raw_bpm = detect_bpm(mp3_path)
    bpm = round(raw_bpm) # Arredonda o BPM para inteiro
    print(f"BPM Detectado: {raw_bpm:.2f} -> Usando: {bpm}")
    
    full_audio_path = os.path.join(output_folder, final_audio_name)
    add_silence(mp3_path, full_audio_path)
    if cover_path and os.path.exists(cover_path): shutil.copy(cover_path, os.path.join(output_folder, final_cover_name))
    
    features, sr, hop_length = extract_features(full_audio_path, bpm)
    energy_profile = analyze_energy(full_audio_path, hop_length=hop_length, sr=sr)
    if len(energy_profile) != len(features): energy_profile = np.pad(energy_profile, (0, len(features) - len(energy_profile)))

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = get_model().to(device)
    model_path = "models/director_net_v2_stars.pth"
    if os.path.exists(model_path):
        model.load_state_dict(torch.load(model_path, map_location=device))
        model.eval()
    else: print(f"ERRO: Modelo '{model_path}' não encontrado. Treine o modelo primeiro."); return

    print("\n[5/5] Gerando mapas...")
    for diff_name, data in difficulties_with_stars.items():
        notes, bombs = generate_difficulty(model, features, energy_profile, bpm, sr, hop_length, diff_name, data['stars'])
        save_difficulty_dat(notes, bombs, output_folder, f"{diff_name}.dat")
        
    info_content = create_info_dat(song_name, bpm, final_audio_name, final_cover_name, difficulties_with_stars)
    with open(os.path.join(output_folder, "Info.dat"), 'w') as f: json.dump(info_content, f, indent=2)
        
    zip_folder(output_folder, os.path.join("output", song_name.replace(" ", "_")))
    try: shutil.rmtree("data/temp_download")
    except: pass
        
    print("\nSUCESSO! Mapa gerado em 'output/'")

if __name__ == "__main__":
    main()
