import os
import numpy as np
from data_loader import create_dataset_entry
from concurrent.futures import ThreadPoolExecutor, as_completed
from collections import Counter

def process_map(map_folder, processed_dir):
    map_id = os.path.basename(map_folder)
    save_path_x = os.path.join(processed_dir, f"{map_id}_x.npy")
    save_path_y = os.path.join(processed_dir, f"{map_id}_y.npy")

    if os.path.exists(save_path_x) and os.path.exists(save_path_y):
        return None, map_id, "already_processed"

    try:
        data = create_dataset_entry(map_folder)
        if data is None:
            return None, map_id, "read_error"
            
        features, targets, vertical_dist = data
        
        np.save(save_path_x, features)
        np.save(save_path_y, targets)
        
        return vertical_dist, map_id, "success"
        
    except Exception as e:
        return None, map_id, f"fatal_error: {e}"

def preprocess_all(raw_dir="data/raw_maps", processed_dir="data/processed", max_workers=8):
    if not os.path.exists(processed_dir):
        os.makedirs(processed_dir)
        
    map_folders = [os.path.join(raw_dir, d) for d in os.listdir(raw_dir) if os.path.isdir(os.path.join(raw_dir, d))]
    
    print(f"Encontrados {len(map_folders)} mapas para processar.")
    
    total_vertical_distribution = Counter()
    processed_count = 0
    
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = [executor.submit(process_map, folder, processed_dir) for folder in map_folders]
        
        for future in as_completed(futures):
            dist, map_id, status = future.result()
            
            if status == "success":
                print(f"Processado: {map_id}")
                total_vertical_distribution.update(dist)
                processed_count += 1
            elif status == "read_error":
                print(f"Ignorando {map_id} (erro na leitura)")
            elif status.startswith("fatal_error"):
                print(f"Erro fatal em {map_id}: {status}")

    print(f"\nConcluído! {processed_count} novos mapas processados e salvos em {processed_dir}")

    # Análise da Distribuição Vertical
    total_notes = sum(total_vertical_distribution.values())
    if total_notes > 0:
        print("\nAnálise de Distribuição Vertical do Dataset:")
        print(f"Total de notas analisadas: {total_notes}")
        
        dist_percent = {row: (count / total_notes) * 100 for row, count in total_vertical_distribution.items()}
        
        print(f"  - Linha 0 (Baixo): {dist_percent.get(0, 0):.2f}%")
        print(f"  - Linha 1 (Meio):  {dist_percent.get(1, 0):.2f}%")
        print(f"  - Linha 2 (Cima):  {dist_percent.get(2, 0):.2f}%")
    else:
        print("\nNenhuma nota nova foi processada para analisar a distribuição.")

if __name__ == "__main__":
    # Ajuste max_workers conforme o número de núcleos da sua CPU para melhor performance
    preprocess_all(max_workers=os.cpu_count() or 4)
