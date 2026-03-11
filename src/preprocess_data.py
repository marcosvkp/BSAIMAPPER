import os
import numpy as np
from data_loader import create_dataset_entry
from parser.loader import get_all_valid_difficulties # Importa a função para listar dificuldades
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

def process_map_difficulty(map_folder, diff_info, processed_dir):
    """
    Processa uma única dificuldade de um mapa.
    diff_info: (diff_name, diff_filename, stars)
    """
    map_id = os.path.basename(map_folder)
    diff_name, diff_filename, stars = diff_info
    
    # Nome do arquivo de saída inclui o ID do mapa, a dificuldade e as estrelas para unicidade
    # Ex: 1a2b3_ExpertPlus_x.npy
    base_name = f"{map_id}_{diff_name}"
    
    save_path_x = os.path.join(processed_dir, f"{base_name}_x.npy")
    save_path_y = os.path.join(processed_dir, f"{base_name}_y.npy")
    save_path_meta = os.path.join(processed_dir, f"{base_name}_meta.npy")
    save_path_stars = os.path.join(processed_dir, f"{base_name}_stars.npy")

    # Pula se TODOS os arquivos já existirem
    if all(os.path.exists(p) for p in [save_path_x, save_path_y, save_path_meta, save_path_stars]):
        return None, base_name, "already_processed"

    try:
        # Chama create_dataset_entry com a dificuldade específica
        data = create_dataset_entry(map_folder, diff_filename, diff_name, stars)
        
        if data is None:
            return None, base_name, "read_error"
            
        features, targets, vertical_dist, stars_val = data
        
        # Calcula metadados
        metadata = calculate_metadata(targets)
        
        # Salva os arquivos
        np.save(save_path_x, features)
        np.save(save_path_y, targets)
        np.save(save_path_meta, metadata)
        np.save(save_path_stars, np.array([stars_val]))
        
        return vertical_dist, base_name, "success"
        
    except Exception as e:
        import traceback
        error_details = traceback.format_exc()
        return None, base_name, f"fatal_error: {e}\n{error_details}"

def preprocess_all(raw_dir="data/raw_maps", processed_dir="data/processed", max_workers=8):
    if not os.path.exists(processed_dir):
        os.makedirs(processed_dir)
        
    map_folders = [os.path.join(raw_dir, d) for d in os.listdir(raw_dir) if os.path.isdir(os.path.join(raw_dir, d))]
    
    print(f"Encontradas {len(map_folders)} pastas de mapas.")
    print("Buscando todas as dificuldades válidas (com estrelas)...")
    
    # Lista de tarefas: (map_folder, diff_info)
    all_tasks = []
    
    # Pré-escaneamento para encontrar todas as dificuldades válidas
    for folder in map_folders:
        valid_diffs = get_all_valid_difficulties(folder)
        for diff_info in valid_diffs:
            all_tasks.append((folder, diff_info))
            
    print(f"Total de {len(all_tasks)} dificuldades válidas encontradas para processamento.")
    print("Gerando features, targets, metadados e DADOS DE ESTRELAS...")
    
    processed_count = 0
    skipped_count = 0
    error_count = 0
    
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = [executor.submit(process_map_difficulty, folder, diff_info, processed_dir) for folder, diff_info in all_tasks]
        
        for future in as_completed(futures):
            dist, base_name, status = future.result()
            
            if status == "success":
                print(f"Processado: {base_name}")
                processed_count += 1
            elif status == "already_processed":
                skipped_count += 1
            elif status == "read_error":
                print(f"  Falha ao ler dados para {base_name}")
                error_count += 1
            elif status.startswith("fatal_error"):
                print(f"Erro fatal em {base_name}: {status}")
                error_count += 1

    print(f"\nConcluído! {processed_count} dificuldades processadas, {skipped_count} puladas, {error_count} com erro.")

if __name__ == "__main__":
    preprocess_all(max_workers=os.cpu_count() or 4)
