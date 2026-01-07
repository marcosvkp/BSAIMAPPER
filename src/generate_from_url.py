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
DIFFICULTY_MAP = {
    1: "Easy",
    2: "Normal",
    3: "Hard",
    4: "Expert",
    5: "ExpertPlus"
}

# Parâmetros de Geração por Dificuldade (Heurísticas)
# NPS Alvo (Notas por Segundo) e Threshold Base
DIFFICULTY_PARAMS = {
    "Easy":       {"base_threshold": 0.75, "cooldown_mod": 2.0, "njs": 10, "offset": 0.0},
    "Normal":     {"base_threshold": 0.65, "cooldown_mod": 1.5, "njs": 12, "offset": 0.0},
    "Hard":       {"base_threshold": 0.55, "cooldown_mod": 1.2, "njs": 14, "offset": -0.2},
    "Expert":     {"base_threshold": 0.45, "cooldown_mod": 1.0, "njs": 16, "offset": -0.4},
    "ExpertPlus": {"base_threshold": 0.35, "cooldown_mod": 0.8, "njs": 18, "offset": -0.784}
}

def zip_folder(folder_path, output_path):
    if os.path.exists(output_path + ".zip"):
        os.remove(output_path + ".zip")
    shutil.make_archive(output_path, 'zip', folder_path)
    print(f"Pacote final criado: {output_path}.zip")

def generate_difficulty(model, features, energy_profile, bpm, sr, hop_length, difficulty_name):
    """
    Gera as notas para uma dificuldade específica usando o modelo carregado.
    """
    print(f"   -> Gerando dificuldade: {difficulty_name}...")
    
    params = DIFFICULTY_PARAMS[difficulty_name]
    base_threshold = params["base_threshold"]
    cooldown_modifier = params["cooldown_mod"]
    
    # Instancia PatternManager com a dificuldade correta para filtrar padrões
    pattern_manager = PatternManager(difficulty=difficulty_name)
    
    # Inferência (já feita fora, mas precisamos processar os outputs)
    # Para economizar VRAM/CPU, passamos os tensores já processados ou rodamos aqui se for rápido.
    # Como o modelo é leve, vamos rodar a lógica de seleção aqui.
    
    device = next(model.parameters()).device
    inputs = torch.tensor(features, dtype=torch.float32).unsqueeze(0).to(device)
    
    with torch.no_grad():
        p_beat, p_comp, p_vert = model(inputs)
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
        
        # Threshold Dinâmico ajustado pela dificuldade
        # Easy/Normal: Menos sensível a energia baixa
        dynamic_threshold = base_threshold + (0.5 - current_energy) * 0.6
        dynamic_threshold = np.clip(dynamic_threshold, 0.15, 0.95)
        
        # Cooldown Dinâmico
        current_cooldown = base_cooldown
        if current_energy > 0.75:
            current_cooldown = int(base_cooldown * 0.6)

        if beat_probs[i] > dynamic_threshold and (i - last_frame) > current_cooldown:
            # Peak picking
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
                        
                        if meta['type'] == 'burst_fill':
                            occupied_until_beat = beat_time + 1.0
                        elif meta['type'] == 'super_stream':
                            occupied_until_beat = beat_time + 2.0

    print(f"      Notas geradas (bruto): {len(raw_notes)}")
    
    # FlowFixer
    all_objects = FlowFixer.fix(raw_notes, bpm)
    final_notes = [obj for obj in all_objects if obj['_type'] != 3]
    final_bombs = [obj for obj in all_objects if obj['_type'] == 3]
    
    # Ordenar
    final_notes.sort(key=lambda x: x['_time'])
    final_bombs.sort(key=lambda x: x['_time'])
    
    # Remover bombas duplicadas
    unique_bombs = []
    seen_bombs = set()
    for b in final_bombs:
        key = (round(b['_time'], 3), b['_lineIndex'], b['_lineLayer'])
        if key not in seen_bombs:
            seen_bombs.add(key)
            unique_bombs.append(b)
            
    return final_notes, unique_bombs

def create_info_dat(song_name, bpm, audio_filename, cover_filename, difficulties_data):
    """
    Gera o Info.dat contendo todas as dificuldades geradas.
    """
    diff_sets = []
    
    beatmap_sets = []
    
    for diff_name in difficulties_data:
        params = DIFFICULTY_PARAMS[diff_name]
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
    
    # 1. URL
    url = input("Digite a URL da música (YouTube): ").strip()
    if not url:
        print("URL inválida.")
        return

    # 2. Download
    print("\n[1/5] Baixando e processando áudio...")
    mp3_path, cover_path = download_from_youtube(url, output_folder="data/temp_download")
    
    if not mp3_path:
        print("Falha no download.")
        return
        
    song_name = os.path.splitext(os.path.basename(mp3_path))[0]
    
    # 3. Escolha de Dificuldades
    print("\n[2/5] Configuração de Dificuldades")
    print("Quais dificuldades deseja gerar?")
    print("   1 = Easy")
    print("   2 = Normal")
    print("   3 = Hard")
    print("   4 = Expert")
    print("   5 = ExpertPlus")
    
    choices = input("\nExemplo: 1,3,5 -> ").strip()
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
        
    # Ordenar por dificuldade (Easy -> ExpertPlus)
    order = ["Easy", "Normal", "Hard", "Expert", "ExpertPlus"]
    selected_diffs.sort(key=lambda x: order.index(x))
    
    print(f"Dificuldades selecionadas: {', '.join(selected_diffs)}")
    
    # 4. Preparação do Output
    output_folder = os.path.join("output", song_name)
    if os.path.exists(output_folder):
        shutil.rmtree(output_folder)
    os.makedirs(output_folder)
    
    # Mover/Copiar arquivos base
    final_audio_name = "song.egg"
    final_cover_name = "cover.png"
    
    # Processar áudio (silêncio)
    print("\n[3/5] Analisando áudio e extraindo features...")
    bpm = detect_bpm(mp3_path)
    print(f"BPM Detectado: {bpm}")
    
    full_audio_path = os.path.join(output_folder, final_audio_name)
    add_silence(mp3_path, full_audio_path)
    
    if cover_path and os.path.exists(cover_path):
        shutil.copy(cover_path, os.path.join(output_folder, final_cover_name))
    else:
        # Criar cover dummy se não existir
        from PIL import Image
        img = Image.new('RGB', (256, 256), color = (73, 109, 137))
        img.save(os.path.join(output_folder, final_cover_name))

    # Extrair Features
    features, sr, hop_length = extract_features(full_audio_path, bpm)
    energy_profile = analyze_energy(full_audio_path, hop_length=hop_length, sr=sr)
    
    # Ajuste de tamanho
    if len(energy_profile) > len(features):
        energy_profile = energy_profile[:len(features)]
    elif len(energy_profile) < len(features):
        energy_profile = np.pad(energy_profile, (0, len(features) - len(energy_profile)))

    # Carregar Modelo
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = get_model().to(device)
    if os.path.exists("models/director_net.pth"):
        model.load_state_dict(torch.load("models/director_net.pth", map_location=device, weights_only=True))
        model.eval()
    else:
        print("ERRO: Modelo 'models/director_net.pth' não encontrado.")
        return

    # 5. Geração Loop
    print("\n[4/5] Gerando mapas...")
    
    generated_data = {} # diff_name -> (notes, bombs)
    
    for diff in selected_diffs:
        notes, bombs = generate_difficulty(model, features, energy_profile, bpm, sr, hop_length, diff)
        generated_data[diff] = (notes, bombs)
        save_difficulty_dat(notes, bombs, output_folder, f"{diff}.dat")
        
    # Criar Info.dat
    info_content = create_info_dat(song_name, bpm, final_audio_name, final_cover_name, selected_diffs)
    with open(os.path.join(output_folder, "Info.dat"), 'w') as f:
        json.dump(info_content, f, indent=2)
        
    # 6. Empacotar
    print("\n[5/5] Finalizando...")
    zip_folder(output_folder, os.path.join("output", song_name))
    
    # Limpeza
    try:
        shutil.rmtree("data/temp_download")
    except:
        pass
        
    print("\nSUCESSO! Mapa gerado em 'output/'")

if __name__ == "__main__":
    main()
