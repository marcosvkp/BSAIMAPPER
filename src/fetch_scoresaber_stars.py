import os
import json
import time
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
from scoresaber_api import get_scoresaber_info

class RateLimiter:
    """
    Controlador de taxa thread-safe.
    Garante que as chamadas à API respeitem um intervalo mínimo.
    """
    def __init__(self, requests_per_minute=250):
        self.interval = 60.0 / requests_per_minute
        self.lock = threading.Lock()
        self.last_call = 0

    def wait_for_slot(self):
        with self.lock:
            now = time.time()
            elapsed = now - self.last_call
            wait_time = self.interval - elapsed
            
            if wait_time > 0:
                time.sleep(wait_time)
            
            self.last_call = time.time()

def process_map(map_id, map_hash, raw_maps_dir, limiter):
    """
    Função worker para processar um único mapa.
    Retorna o status: 'skipped', 'success', 'failed', 'missing_folder'.
    """
    map_folder = os.path.join(raw_maps_dir, map_id)
    
    # Se a pasta do mapa não existir, não temos o que fazer
    if not os.path.exists(map_folder):
        return 'missing_folder'

    scoresaber_json_path = os.path.join(map_folder, 'scoresaber.json')

    # Verifica se já temos os dados
    if os.path.exists(scoresaber_json_path):
        try:
            with open(scoresaber_json_path, 'r') as f:
                data = json.load(f)
            if data: # JSON válido e não vazio
                return 'skipped'
        except json.JSONDecodeError:
            print(f"  [Worker] scoresaber.json corrompido em {map_id}. Refazendo...")
            try:
                os.remove(scoresaber_json_path)
            except: pass

    # --- Seção Crítica (Rate Limit) ---
    limiter.wait_for_slot()
    
    # Busca na API
    # print(f"  [API] Buscando estrelas para {map_id}...")
    stars_data = get_scoresaber_info(map_hash)

    if stars_data:
        try:
            with open(scoresaber_json_path, 'w') as f:
                json.dump(stars_data, f, indent=4)
            return 'success'
        except Exception as e:
            print(f"  [Erro] Falha ao salvar JSON para {map_id}: {e}")
            return 'failed'
    else:
        # print(f"  [Info] Sem dados no ScoreSaber para {map_id}.")
        return 'failed'

def fetch_stars(raw_maps_dir="data/raw_maps", hashes_file="data/map_hashes.json", max_workers=10):
    if not os.path.exists(hashes_file):
        print(f"Erro: Arquivo de hashes não encontrado em {hashes_file}. Execute downloader.py primeiro.")
        return

    with open(hashes_file, 'r') as f:
        map_hashes = json.load(f)

    total_maps = len(map_hashes)
    print(f"Iniciando busca de estrelas para {total_maps} mapas...")
    print(f"Usando {max_workers} threads com Rate Limit de ~250 req/min.")

    # Configura o Rate Limiter (250 req/min = 1 req a cada ~0.24s)
    # ScoreSaber permite ~400, mas vamos usar uma margem de segurança
    limiter = RateLimiter(requests_per_minute=180)

    stats = {
        'skipped': 0,
        'success': 0,
        'failed': 0,
        'missing_folder': 0
    }
    
    processed_count = 0

    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        # Submete todas as tarefas
        futures = {
            executor.submit(process_map, map_id, map_hash, raw_maps_dir, limiter): map_id 
            for map_id, map_hash in map_hashes.items()
        }

        # Processa os resultados conforme eles chegam
        for future in as_completed(futures):
            status = future.result()
            stats[status] += 1
            processed_count += 1
            
            # Log de progresso a cada 50 mapas ou quando ocorre um sucesso/falha (para não poluir com 'skipped')
            if status in ['success', 'failed'] or processed_count % 100 == 0:
                print(f"[{processed_count}/{total_maps}] Status: {status.upper()} | "
                      f"Sucessos: {stats['success']} | Falhas: {stats['failed']} | Pulados: {stats['skipped']}")

    print("\n" + "="*40)
    print("RESUMO DA BUSCA DE ESTRELAS")
    print("="*40)
    print(f"Total de Mapas: {total_maps}")
    print(f"Já existiam (Skipped): {stats['skipped']}")
    print(f"Atualizados (Success): {stats['success']}")
    print(f"Não encontrados/Erro (Failed): {stats['failed']}")
    print(f"Pastas ausentes: {stats['missing_folder']}")
    print("="*40)

if __name__ == "__main__":
    fetch_stars()
