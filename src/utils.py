import os
import shutil
import json
import numpy as np

class NumpyEncoder(json.JSONEncoder):
    """
    Encoder JSON especial que converte tipos de dados do NumPy (como int64, float32)
    para tipos nativos do Python que o JSON consegue serializar.
    """
    def default(self, obj):
        if isinstance(obj, np.integer):
            return int(obj)
        if isinstance(obj, np.floating):
            return float(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return super(NumpyEncoder, self).default(obj)

def zip_folder(folder_path, output_path):
    """Compacta uma pasta em um arquivo zip."""
    if os.path.exists(output_path + ".zip"):
        os.remove(output_path + ".zip")
    shutil.make_archive(output_path, 'zip', folder_path)
    print(f"Pacote final criado: {output_path}.zip")

def save_json_file(data, folder, filename):
    """Salva um dicionário como arquivo JSON, usando o encoder customizado."""
    with open(os.path.join(folder, filename), 'w') as f:
        json.dump(data, f, indent=2, cls=NumpyEncoder)

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
    """
    Salva o arquivo de dificuldade (.dat) no formato compatível v2.0.0.
    Este formato usa uma lista única '_notes' para notas e bombas.
    """
    # Combina notas e bombas em uma única lista. O FlowFixer já atribui o '_type' correto.
    all_objects = notes + bombs
    
    # Ordena todos os objetos pelo tempo para garantir a ordem correta no arquivo do mapa.
    all_objects.sort(key=lambda x: x['_time'])
    
    # Define o tempo final para customData, se houver objetos.
    last_time = all_objects[-1]['_time'] if all_objects else 0

    # Estrutura de dados para o formato v2.0.0
    data = {
        "_version": "2.0.0",
        "_notes": all_objects, # Lista única para notas e bombas
        "_obstacles": [],
        "_events": [],
        "_customData": {
            "_time": last_time + 4,
            "_BPMChanges": [],
            "_bookmarks": []
        }
    }
    
    # Salva o arquivo JSON usando o encoder que lida com tipos NumPy.
    save_json_file(data, folder, filename)
