import shutil

import requests
import os
import zipfile
import io
import json
import time
from concurrent.futures import ThreadPoolExecutor

def get_scoresaber_info(map_hash, retries=3, delay=5):
    """
    Busca as informações de dificuldade (estrelas) da API do ScoreSaber para um determinado hash de mapa.
    """
    # Endpoint para obter todas as dificuldades de um hash
    difficulties_url = f"https://scoresaber.com/api/leaderboard/get-difficulties/{map_hash}"
    
    for attempt in range(retries):
        try:
            response = requests.get(difficulties_url, timeout=15)
            response.raise_for_status()
            difficulties_data = response.json()
            
            # Se não houver dados, o mapa não é ranqueado no ScoreSaber
            if not difficulties_data:
                return None

            map_difficulties = {}
            # Para cada dificuldade encontrada, busca as informações detalhadas (que contêm as estrelas)
            for diff_info in difficulties_data:
                leaderboard_id = diff_info.get('leaderboardId')
                if not leaderboard_id:
                    continue

                leaderboard_url = f"https://scoresaber.com/api/leaderboard/by-id/{leaderboard_id}/info"
                lb_response = requests.get(leaderboard_url, timeout=15)
                lb_response.raise_for_status()
                leaderboard_details = lb_response.json()

                # Extrai as informações relevantes
                difficulty_name = diff_info.get('difficultyRaw').split('_')[1] # Ex: "_ExpertPlus_SoloStandard" -> "ExpertPlus"
                stars = leaderboard_details.get('stars')
                
                if difficulty_name and stars is not None:
                    map_difficulties[difficulty_name] = {'stars': stars}

            return map_difficulties if map_difficulties else None

        except requests.exceptions.RequestException as e:
            print(f"  [ScoreSaber] Tentativa {attempt + 1} falhou para o hash {map_hash}: {e}")
            if attempt < retries - 1:
                time.sleep(delay)
            else:
                return None # Retorna None após todas as tentativas falharem

def download_map(doc, output_dir):
    map_id = doc['id']
    map_name = doc['name']
    map_hash = doc['versions'][0]['hash']
    
    extract_path = os.path.join(output_dir, map_id)

    # Verifica se já existe e se os dados do ScoreSaber já foram baixados
    if os.path.exists(extract_path) and os.path.exists(os.path.join(extract_path, 'scoresaber.json')):
        return f"Pulado (já existe com dados ScoreSaber): {map_name}"

    download_url = doc['versions'][0]['downloadURL']
    
    try:
        # 1. Baixar e extrair o mapa
        map_resp = requests.get(download_url, timeout=30)
        map_resp.raise_for_status()
        
        with zipfile.ZipFile(io.BytesIO(map_resp.content)) as z:
            z.extractall(extract_path)
        
        # 2. Buscar informações do ScoreSaber
        print(f"Buscando dados do ScoreSaber para: {map_name} ({map_hash})")
        scoresaber_data = get_scoresaber_info(map_hash)
        
        if scoresaber_data:
            # 3. Salvar as informações do ScoreSaber em um arquivo JSON
            scoresaber_json_path = os.path.join(extract_path, 'scoresaber.json')
            with open(scoresaber_json_path, 'w') as f:
                json.dump(scoresaber_data, f, indent=4)
            return f"Baixado e Info Salva: {map_name} ({map_id})"
        else:
            # Se não encontrar dados no ScoreSaber, remove a pasta para evitar mapas não ranqueados no dataset
            shutil.rmtree(extract_path)
            return f"Ignorado (não ranqueado no ScoreSaber): {map_name}"

    except Exception as e:
        # Limpa a pasta em caso de erro para não deixar dados incompletos
        if os.path.exists(extract_path):
            shutil.rmtree(extract_path)
        return f"Erro em {map_id}: {e}"

def download_ranked_maps(output_dir="data/raw_maps", limit=300):
    """
    Baixa mapas rankeados em paralelo.
    """
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    print(f"Buscando lista de {limit} mapas rankeados do BeatSaver...")
    
    maps_to_download = []
    page = 0
    while len(maps_to_download) < limit:
        url = f"https://api.beatsaver.com/search/text/{page}?sortOrder=Relevance&ranked=true"
        try:
            resp = requests.get(url)
            resp.raise_for_status()
            data = resp.json()
            docs = data.get('docs', [])
            if not docs:
                print("Não foram encontrados mais mapas na API do BeatSaver.")
                break
            
            maps_to_download.extend(docs)
            page += 1
            print(f"Coletados {len(maps_to_download)}/{limit} metadados de mapas...")
        except Exception as e:
            print(f"Erro ao contatar a API do BeatSaver: {e}")
            break
            
    maps_to_download = maps_to_download[:limit]
    print(f"\nIniciando download e processamento de {len(maps_to_download)} mapas...")

    # Reduzir o número de workers para evitar rate limiting das APIs
    with ThreadPoolExecutor(max_workers=5) as executor:
        futures = [executor.submit(download_map, doc, output_dir) for doc in maps_to_download]
        for future in futures:
            print(future.result())

if __name__ == "__main__":
    # Aumentar o limite para obter um dataset mais robusto
    download_ranked_maps(limit=1000)
