import os
import numpy as np
from data_loader import create_dataset_entry
from concurrent.futures import ThreadPoolExecutor, as_completed
from collections import Counter

def calculate_metadata(targets, window=200):
    """
    Gera metadados (Complexidade e Verticalidade) baseados no contexto local do mapa.
    """
    num_frames = targets.shape[0]
    metadata = np.zeros((num_frames, 2), dtype=np.float32)
    
    has_note = np.any(targets > 0.1, axis=1)
    
    step = 10
    half_window = window // 2
    
    for i in range(0, num_frames, step):
        start = max(0, i - half_window)
        end = min(num_frames, i + half_window)
        
        chunk_notes = has_note[start:end]
        density = np.sum(chunk_notes) / len(chunk_notes) if len(chunk_notes) > 0 else 0
            
        if density > 0.15: comp_val = 2.0
        elif density > 0.05: comp_val = 1.0
        else: comp_val = 0.0
        
        chunk_targets = targets[start:end]
        l0 = np.sum(chunk_targets[:, 0:4])
        l1 = np.sum(chunk_targets[:, 4:8])
        l2 = np.sum(chunk_targets[:, 8:12])
        total = l0 + l1 + l2 + 1e-6
        
        avg_height = (0*l0 + 1*l1 + 2*l2) / total
        vert_val = round(avg_height)
        
        fill_end = min(num_frames, i + step)
        metadata[i:fill_end, 0] = comp_val
        metadata[i:fill_end, 1] = vert_val
        
    return metadata

def process_map(map_folder, processed_dir):
    map_id = os.path.basename(map_folder)
    save_path_x = os.path.join(processed_dir, f"{map_id}_x.npy")
    save_path_y = os.path.join(processed_dir, f"{map_id}_y.npy")
    save_path_meta = os.path.join(processed_dir, f"{map_id}_meta.npy")
    save_path_stars = os.path.join(processed_dir, f"{map_id}_stars.npy") # Novo caminho

    # Pula apenas se TODOS os arquivos de dados já existirem
    if all(os.path.exists(p) for p in [save_path_x, save_path_y, save_path_meta, save_path_stars]):
        return None, map_id, "already_processed"

    try:
        data = create_dataset_entry(map_folder)
        if data is None:
            return None, map_id, "read_error"
            
        # Desempacota o novo valor de estrelas
        features, targets, vertical_dist, stars = data
        
        # Calcula os metadados como antes
        metadata = calculate_metadata(targets)
        
        # Salva todos os arquivos de dados
        np.save(save_path_x, features)
        np.save(save_path_y, targets)
        np.save(save_path_meta, metadata)
        np.save(save_path_stars, np.array([stars])) # Salva as estrelas como um array numpy
        
        return vertical_dist, map_id, "success"
        
    except Exception as e:
        # Adiciona mais detalhes ao log de erro
        import traceback
        error_details = traceback.format_exc()
        return None, map_id, f"fatal_error: {e}\n{error_details}"

def preprocess_all(raw_dir="data/raw_maps", processed_dir="data/processed", max_workers=8):
    if not os.path.exists(processed_dir):
        os.makedirs(processed_dir)
        
    map_folders = [os.path.join(raw_dir, d) for d in os.listdir(raw_dir) if os.path.isdir(os.path.join(raw_dir, d))]
    
    print(f"Encontrados {len(map_folders)} mapas para processar.")
    print("Gerando features, targets, metadados e DADOS DE ESTRELAS...")
    
    total_vertical_distribution = Counter()
    processed_count = 0
    skipped_count = 0
    error_count = 0
    
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = [executor.submit(process_map, folder, processed_dir) for folder in map_folders]
        
        for future in as_completed(futures):
            dist, map_id, status = future.result()
            
            if status == "success":
                print(f"Processado: {map_id}")
                total_vertical_distribution.update(dist)
                processed_count += 1
            elif status == "already_processed":
                skipped_count += 1
            elif status == "read_error":
                # Silencioso para não poluir o log, mas pode ser ativado para debug
                # print(f"Ignorando {map_id} (erro na leitura ou dados faltando)")
                error_count += 1
            elif status.startswith("fatal_error"):
                print(f"Erro fatal em {map_id}: {status}")
                error_count += 1

    print(f"\nConcluído! {processed_count} mapas processados, {skipped_count} pulados, {error_count} com erro.")

if __name__ == "__main__":
    preprocess_all(max_workers=os.cpu_count() or 4)
