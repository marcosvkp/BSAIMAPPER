import torch
import numpy as np
import json
import os
import shutil
from models_optimized import get_model
from pattern_manager import PatternManager
from flow_fixer import FlowFixer
from audio_processor import extract_features, detect_bpm, add_silence, analyze_energy
from difficulty_tuning import compute_target_nps, compute_min_beat_fraction, candidate_score
from generation_logging import GenerationLogger

def zip_folder(folder_path, output_path):
    if os.path.exists(output_path + ".zip"):
        os.remove(output_path + ".zip")
    shutil.make_archive(output_path, 'zip', folder_path)
    print(f"Mapa compactado: {output_path}.zip")

def _select_note_frames(beat_probs, energy_profile, bpm, frame_dur, target_stars):
    """Seleciona frames candidatos com base na confiança da IA e meta suave de densidade."""
    peaks = []
    for i in range(1, len(beat_probs) - 1):
        if beat_probs[i] > beat_probs[i - 1] and beat_probs[i] > beat_probs[i + 1]:
            score = beat_probs[i] * (0.7 + 0.6 * energy_profile[i])
            peaks.append((score, i))

    if not peaks:
        return []

    # Densidade alvo guiada pela IA + ajuste suave por estrelas (sem regra rígida)
    ai_conf = float(np.mean(beat_probs))
    base_nps = 1.2 + ai_conf * 6.5
    star_boost = 1.0 + np.clip((target_stars - 5.0) * 0.08, -0.2, 0.4)
    target_nps = base_nps * star_boost

    duration_seconds = len(beat_probs) * frame_dur
    target_total_notes = int(max(80, duration_seconds * target_nps))

    min_fraction = 8.0 if target_stars >= 7.0 else 4.0
    cooldown_frames = max(1, int((60.0 / bpm / min_fraction) / frame_dur))

    peaks.sort(key=lambda x: x[0], reverse=True)
    selected = []
    for score, idx in peaks:
        if len(selected) >= target_total_notes:
            break
        if any(abs(idx - j) < cooldown_frames for j in selected):
            continue
        selected.append(idx)

    selected.sort()
    return selected


def generate_map_optimized(audio_path, output_folder, target_stars=7.0):
    if not os.path.exists(output_folder): os.makedirs(output_folder)
    print(f"Processando: {audio_path}")
    logger = GenerationLogger(session_name="generate_optimized")
    logger.log("start", audio_path=audio_path, output_folder=output_folder, target_stars=target_stars)
    
    bpm = detect_bpm(audio_path)
    processed_audio = "song.egg"
    full_audio_path = os.path.join(output_folder, processed_audio)
    add_silence(audio_path, full_audio_path)
    
    features, sr, hop_length = extract_features(full_audio_path, bpm)
    
    # --- Análise de Energia (NOVO) ---
    print("Analisando perfil de energia da música...")
    energy_profile = analyze_energy(full_audio_path, hop_length=hop_length, sr=sr)
    
    # Ajuste de tamanho caso haja pequena discrepância de frames
    if len(energy_profile) > len(features):
        energy_profile = energy_profile[:len(features)]
    elif len(energy_profile) < len(features):
        energy_profile = np.pad(energy_profile, (0, len(features) - len(energy_profile)))
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = get_model().to(device)
    
    model_path = "models/director_net_v2_stars.pth"
    if os.path.exists(model_path):
        model.load_state_dict(torch.load(model_path, map_location=device))
    else:
        print(f"Modelo '{model_path}' não encontrado! Treine primeiro.")
        return

    model.eval()
    
    inputs = torch.tensor(features, dtype=torch.float32).unsqueeze(0).to(device)
    stars_tensor = torch.tensor([[target_stars]], dtype=torch.float32).to(device)
    with torch.no_grad():
        p_beat, p_comp, p_vert = model(inputs, stars_tensor)
        beat_probs = torch.sigmoid(p_beat).squeeze().cpu().numpy()
        comp_classes = torch.argmax(p_comp, dim=2).squeeze().cpu().numpy()
        vert_classes = torch.argmax(p_vert, dim=2).squeeze().cpu().numpy()
        
    pattern_manager = PatternManager()
    raw_notes = []

    frame_dur = hop_length / sr
    sec_per_beat = 60 / bpm

    selected_indices = _select_note_frames(
        beat_probs=beat_probs,
        energy_profile=energy_profile,
        bpm=bpm,
        frame_dur=frame_dur,
        target_stars=target_stars,
    )

    print(f"Gerando com Director AI (estrelas={target_stars:.2f})...")

    last_frame = -99999
    for idx in selected_indices:
        time_sec = idx * frame_dur
        if time_sec < 2.0:
            continue

        beat_time = round((time_sec / sec_per_beat) * 16) / 16.0
        comp_idx = int(comp_classes[idx])
        vert_idx = int(vert_classes[idx])
        intensity = float(beat_probs[idx])
        gap = (idx - last_frame) * frame_dur

        new_notes = pattern_manager.apply_pattern(
            time=beat_time,
            bpm=bpm,
            complexity_idx=comp_idx,
            vertical_idx=vert_idx,
            time_gap=gap,
            intensity=intensity,
            star_level=target_stars,
        )
        if new_notes:
            raw_notes.extend(new_notes)
            last_frame = idx

    print("Aplicando FlowFixer (Simulação de Paridade e Resets)...")
    all_objects = FlowFixer.fix(raw_notes, bpm)
    
    final_notes = [obj for obj in all_objects if obj['_type'] != 3]
    final_bombs = [obj for obj in all_objects if obj['_type'] == 3]
    logger.log("post_flow_fixer", notes=len(final_notes), bombs=len(final_bombs))
                    
    save_beatmap(final_notes, final_bombs, bpm, output_folder, processed_audio)
    zip_folder(output_folder, os.path.join("output", os.path.basename(output_folder)))
    log_path, event_count = logger.summary()
    print(f"Log salvo em {log_path} ({event_count} eventos).")

def save_beatmap(notes, bombs, bpm, folder, audio_name):
    notes.sort(key=lambda x: x['_time'])
    bombs.sort(key=lambda x: x['_time'])
    
    unique_bombs = []
    seen_bombs = set()
    for b in bombs:
        key = (round(b['_time'], 3), b['_lineIndex'], b['_lineLayer'])
        if key not in seen_bombs:
            seen_bombs.add(key)
            unique_bombs.append(b)

    all_notes_v2 = notes + unique_bombs
    all_notes_v2.sort(key=lambda x: x['_time'])

    diff = {
        "_version": "2.0.0",
        "_notes": all_notes_v2,
        "_events": [],
        "_obstacles": [],
        "_customData": {
            "_time": notes[-1]['_time'] + 4 if notes else 0,
            "_BPMChanges": [],
            "_bookmarks": []
        }
    }
    
    info = {
        "_version": "2.1.0",
        "_songName": "AI Director Map",
        "_songSubName": "Energy Aware V6",
        "_songAuthorName": "BSIAMapper",
        "_levelAuthorName": "AI",
        "_beatsPerMinute": float(bpm),
        "_songFilename": audio_name,
        "_coverImageFilename": "cover.png",
        "_difficultyBeatmapSets": [{
            "_beatmapCharacteristicName": "Standard",
            "_difficultyBeatmaps": [{
                "_difficulty": "ExpertPlus",
                "_beatmapFilename": "ExpertPlus.dat",
                "_noteJumpMovementSpeed": 18,
                "_noteJumpStartBeatOffset": -0.784
            }]
        }]
    }
    
    with open(os.path.join(folder, "ExpertPlus.dat"), 'w') as f: json.dump(diff, f)
    with open(os.path.join(folder, "Info.dat"), 'w') as f: json.dump(info, f, indent=2)

if __name__ == "__main__":
    generate_map_optimized("musica.mp3", "output/DirectorMap", target_stars=7.5)
