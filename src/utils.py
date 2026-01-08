import os
import shutil
import json

def zip_folder(folder_path, output_path):
    """Compacta uma pasta em um arquivo zip."""
    if os.path.exists(output_path + ".zip"):
        os.remove(output_path + ".zip")
    shutil.make_archive(output_path, 'zip', folder_path)
    print(f"Pacote final criado: {output_path}.zip")

def save_json_file(data, folder, filename):
    """Salva um dicionário como arquivo JSON."""
    with open(os.path.join(folder, filename), 'w') as f:
        json.dump(data, f, indent=2)

def create_info_dat(song_name, bpm, audio_filename, cover_filename, difficulties_data, difficulty_params):
    """
    Gera o Info.dat contendo todas as dificuldades geradas.
    """
    beatmap_sets = []
    
    for diff_name in difficulties_data:
        params = difficulty_params[diff_name]
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
    """Salva o arquivo de dificuldade (.dat)."""
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
    
    save_json_file(data, folder, filename)
