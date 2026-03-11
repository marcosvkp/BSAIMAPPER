import requests
import os
import zipfile
import io
import json
import time
from concurrent.futures import ThreadPoolExecutor

def download_map(doc, output_dir, hashes_dict):
    """
    Baixa e extrai um mapa do BeatSaver.
    Atualiza o dicionário de hashes com o hash da versão mais recente.
    """
    map_id = doc['id']
    map_name = doc['name']
    map_hash = doc['versions'][0]['hash']
    
    extract_path = os.path.join(output_dir, map_id)

    # Verifica se já existe e se o hash corresponde à versão mais recente
    if map_id in hashes_dict and hashes_dict[map_id] == map_hash and os.path.exists(extract_path):
        return f"Pulado (já existe e atualizado): {map_name} ({map_id})"

    download_url = doc['versions'][0]['downloadURL']
    
    try:
        # 1. Baixar e extrair o mapa
        map_resp = requests.get(download_url, timeout=30)
        map_resp.raise_for_status()
        
        with zipfile.ZipFile(io.BytesIO(map_resp.content)) as z:
            z.extractall(extract_path)
            
        # Atualiza o hash no dicionário
        hashes_dict[map_id] = map_hash
        return f"Baixado: {map_name} ({map_id})"

    except Exception as e:
        # Limpa a pasta em caso de erro para não deixar dados incompletos
        if os.path.exists(extract_path):
            import shutil
            shutil.rmtree(extract_path)
        return f"Erro em {map_id}: {e}"

def download_ranked_maps(output_dir="data/raw_maps", limit=300):
    """
    Baixa mapas rankeados em paralelo e mantém um registro de hashes.
    """
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
        
    hashes_file = os.path.join("data", "map_hashes.json")
    if os.path.exists(hashes_file):
        with open(hashes_file, 'r') as f:
            map_hashes = json.load(f)
    else:
        map_hashes = {}

    print(f"Buscando lista de {limit} mapas rankeados do BeatSaver...")
    
    maps_to_download = []
    page = 0
    while len(maps_to_download) < limit:
        url = f"https://api.beatsaver.com/search/text/{page}?sortOrder=Relevance&ranked=true&leaderboard=ScoreSaber"
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
        futures = [executor.submit(download_map, doc, output_dir, map_hashes) for doc in maps_to_download]
        for future in futures:
            print(future.result())
            
    # Salva o arquivo de hashes atualizado
    with open(hashes_file, 'w') as f:
        json.dump(map_hashes, f, indent=4)
    print(f"Arquivo de hashes atualizado em {hashes_file}")

if __name__ == "__main__":
    # Aumentar o limite para obter um dataset mais robusto
    download_ranked_maps(limit=1000)
