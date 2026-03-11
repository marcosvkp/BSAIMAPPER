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
from difficulty_tuning import compute_target_nps, compute_min_beat_fraction, candidate_score
from generation_logging import GenerationLogger

DIFFICULTY_MAP = {1: "Easy", 2: "Normal", 3: "Hard", 4: "Expert", 5: "ExpertPlus"}
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

def generate_difficulty(model, features, energy_profile, bpm, sr, hop_length, difficulty_name, target_stars, logger=None):
    print(f"   -> Gerando: {difficulty_name} ({target_stars:.2f} estrelas)...")
    
    device = next(model.parameters()).device
    inputs = torch.tensor(features, dtype=torch.float32).unsqueeze(0).to(device)
    stars_tensor = torch.tensor([[target_stars]], dtype=torch.float32).to(device)
    
    with torch.no_grad():
        p_beat, p_comp, p_vert = model(inputs, stars_tensor)
        beat_probs = torch.sigmoid(p_beat).squeeze().cpu().numpy()
        comp_classes = torch.argmax(p_comp, dim=2).squeeze().cpu().numpy()
        vert_classes = torch.argmax(p_vert, dim=2).squeeze().cpu().numpy()

    # --- Lógica de Densidade de Notas (Target Density) ---
    # A densidade é guiada principalmente pela confiança média da IA,
    # com as estrelas atuando como ajuste suave (não uma regra bruta).
    ai_conf = float(np.mean(beat_probs))
    target_nps = compute_target_nps(ai_conf, target_stars)

    duration_seconds = len(beat_probs) * (hop_length / sr)
    target_total_notes = int(max(80, duration_seconds * target_nps))

    print(f"      Meta de Densidade AI-driven: {target_nps:.2f} NPS (~{target_total_notes} notas)")

    frame_dur = hop_length / sr
    # Cooldown mínimo absoluto (físico) para evitar sobreposição impossível
    # 60/BPM = seg/beat. /4 = 1/4 de beat.
    # Em mapas rápidos, permitimos até 1/8.
    min_beat_fraction = compute_min_beat_fraction(target_stars)
    min_cooldown_frames = int((60.0 / bpm / min_beat_fraction) / frame_dur)
    
    # 1. Encontrar TODOS os picos locais (candidatos a nota)
    # Não usamos threshold aqui. Se for um pico, é um candidato.
    candidates = []
    MIN_START_TIME = 2.0
    
    for i in range(1, len(beat_probs) - 1):
        time_sec = i * frame_dur
        if time_sec < MIN_START_TIME: continue
        
        prob = beat_probs[i]
        
        # É um pico local?
        if prob > beat_probs[i-1] and prob > beat_probs[i+1]:
            # Adiciona à lista de candidatos: (probabilidade, índice_frame)
            # Multiplicamos a probabilidade pela energia local para dar preferência a partes intensas
            score = candidate_score(prob, energy_profile[i])
            candidates.append((score, i))
            
    # 2. Ordenar candidatos pela confiança da IA (do maior para o menor)
    candidates.sort(key=lambda x: x[0], reverse=True)
    
    # 3. Selecionar os melhores picos respeitando o cooldown e a meta de notas
    selected_indices = []
    # Margem de segurança para o cooldown (evitar notas coladas demais)
    cooldown_margin = min_cooldown_frames
    
    for score, idx in candidates:
        if len(selected_indices) >= target_total_notes:
            break
            
        # Verifica conflito de cooldown com notas já selecionadas
        # (Uma implementação simples de verificação de proximidade)
        is_too_close = False
        
        # Otimização: verificar apenas numa janela próxima seria mais rápido, 
        # mas para <5000 notas isso é ok.
        for selected_idx in selected_indices:
            if abs(selected_idx - idx) < cooldown_margin:
                is_too_close = True
                break
        
        if not is_too_close:
            selected_indices.append(idx)
            
    # Recupera a ordem temporal
    selected_indices.sort()
    
    # 4. Gerar as notas finais usando o PatternManager
    pattern_manager = PatternManager(difficulty=difficulty_name)
    raw_notes = []
    last_frame = -999
    
    for idx in selected_indices:
        current_beat = (idx * frame_dur) / (60.0 / bpm)
        # Quantização para 1/16 de beat
        beat_time = round(current_beat * 16) / 16.0
        
        # Evita duplicatas exatas de tempo (caso o arredondamento gere colisão)
        if raw_notes and beat_time <= raw_notes[-1]['_time']:
            continue
            
        time_gap = (idx - last_frame) * frame_dur
        
        # O PatternManager constrói a nota baseada na "intenção" (classes) da IA naquele momento
        new_notes = pattern_manager.apply_pattern(
            time=beat_time,
            bpm=bpm,
            complexity_idx=comp_classes[idx],
            vertical_idx=vert_classes[idx],
            time_gap=time_gap,
            intensity=float(beat_probs[idx]),
            star_level=target_stars,
        )
        
        if new_notes:
            raw_notes.extend(new_notes)
            last_frame = idx
            
            # Se gerou um burst/stream, avança o "last_frame" virtualmente
            # para evitar sobrepor o stream com a próxima nota selecionada
            if len(new_notes) > 1:
                # Estimativa grosseira: cada nota extra consome um pouco de tempo
                extra_frames = int((len(new_notes) * 0.15) / frame_dur) 
                # Removemos futuros candidatos que colidiriam com este burst
                # (Isso é implícito pois já filtramos por cooldown simples antes, 
                # mas o burst ocupa mais espaço que uma nota simples)
                pass

    print(f"      Notas geradas (bruto): {len(raw_notes)}")
    if logger is not None:
        logger.log("difficulty_raw", difficulty=difficulty_name, stars=target_stars, selected_frames=len(selected_indices), raw_notes=len(raw_notes), target_nps=target_nps)
    
    # FlowFixer
    all_objects = FlowFixer.fix(raw_notes, bpm)
    final_notes = [obj for obj in all_objects if obj['_type'] != 3]
    final_bombs = [obj for obj in all_objects if obj['_type'] == 3]
    
    final_notes.sort(key=lambda x: x['_time'])
    unique_bombs = [dict(t) for t in {tuple(d.items()) for d in final_bombs}]
    unique_bombs.sort(key=lambda x: x['_time'])
    if logger is not None:
        logger.log("difficulty_final", difficulty=difficulty_name, stars=target_stars, notes=len(final_notes), bombs=len(unique_bombs))

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
    print("="*50 + "\n   BEAT SABER AI MAPPER V2 - GERAÇÃO POR URL\n" + "="*50)
    logger = GenerationLogger(session_name="generate_from_url")
    
    url = input("Digite a URL da música (YouTube): ").strip()
    logger.log("start", url=url)
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
    bpm = round(raw_bpm)
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
        notes, bombs = generate_difficulty(model, features, energy_profile, bpm, sr, hop_length, diff_name, data['stars'], logger=logger)
        save_difficulty_dat(notes, bombs, output_folder, f"{diff_name}.dat")
        
    info_content = create_info_dat(song_name, bpm, final_audio_name, final_cover_name, difficulties_with_stars)
    with open(os.path.join(output_folder, "Info.dat"), 'w') as f: json.dump(info_content, f, indent=2)
        
    zip_folder(output_folder, os.path.join("output", song_name.replace(" ", "_")))
    try: shutil.rmtree("data/temp_download")
    except: pass

    log_path, event_count = logger.summary()
    print(f"Log salvo em {log_path} ({event_count} eventos).")
    print("\nSUCESSO! Mapa gerado em 'output/'")

if __name__ == "__main__":
    main()
