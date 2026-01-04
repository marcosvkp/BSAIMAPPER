import requests
import os
import zipfile
import io
from concurrent.futures import ThreadPoolExecutor

def download_map(doc, output_dir):
    map_id = doc['id']
    map_name = doc['name']
    
    # Verifica se já existe
    if os.path.exists(os.path.join(output_dir, map_id)):
        return f"Pulado (já existe): {map_name}"

    download_url = doc['versions'][0]['downloadURL']
    
    try:
        map_resp = requests.get(download_url, timeout=30)
        map_resp.raise_for_status()
        
        with zipfile.ZipFile(io.BytesIO(map_resp.content)) as z:
            extract_path = os.path.join(output_dir, map_id)
            z.extractall(extract_path)
            return f"Baixado: {map_name} ({map_id})"
    except Exception as e:
        return f"Erro em {map_id}: {e}"

def download_ranked_maps(output_dir="data/raw_maps", limit=100):
    """
    Baixa mapas rankeados em paralelo.
    """
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    print(f"Buscando lista de {limit} mapas rankeados...")
    # BeatSaver API page size is usually 20. We might need multiple requests if limit > 20.
    # Mas para simplificar, vamos pegar os 'sortOrder=Relevance' que a API entregar.
    # A API do BeatSaver permite search?q=&ranked=true
    
    maps_to_download = []
    page = 0
    while len(maps_to_download) < limit:
        url = f"https://api.beatsaver.com/search/text/{page}?sortOrder=Relevance&ranked=true"
        try:
            resp = requests.get(url)
            resp.raise_for_status()
            data = resp.json()
            docs = data.get('docs', [])
            if not docs: break
            
            maps_to_download.extend(docs)
            page += 1
        except Exception as e:
            print(f"Erro na API: {e}")
            break
            
    maps_to_download = maps_to_download[:limit]
    print(f"Iniciando download de {len(maps_to_download)} mapas...")

    with ThreadPoolExecutor(max_workers=10) as executor:
        futures = [executor.submit(download_map, doc, output_dir) for doc in maps_to_download]
        for future in futures:
            print(future.result())

if __name__ == "__main__":
    download_ranked_maps(limit=100)
