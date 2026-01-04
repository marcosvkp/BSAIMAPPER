import os
import numpy as np
from data_loader import create_dataset_entry

def preprocess_all(raw_dir="data/raw_maps", processed_dir="data/processed"):
    if not os.path.exists(processed_dir):
        os.makedirs(processed_dir)
        
    map_folders = [os.path.join(raw_dir, d) for d in os.listdir(raw_dir) if os.path.isdir(os.path.join(raw_dir, d))]
    
    print(f"Encontrados {len(map_folders)} mapas para processar.")
    
    count = 0
    for folder in map_folders:
        map_id = os.path.basename(folder)
        save_path_x = os.path.join(processed_dir, f"{map_id}_x.npy")
        save_path_y = os.path.join(processed_dir, f"{map_id}_y.npy")
        
        # Se já existe, pula (útil se parar e continuar depois)
        if os.path.exists(save_path_x) and os.path.exists(save_path_y):
            continue
            
        try:
            print(f"Processando {map_id}...")
            data = create_dataset_entry(folder)
            
            if data is None:
                print(f"Ignorando {map_id} (erro na leitura)")
                continue
                
            features, targets = data
            
            # Salvar como numpy binary (muito rápido de ler)
            np.save(save_path_x, features)
            np.save(save_path_y, targets)
            count += 1
            
        except Exception as e:
            print(f"Erro fatal em {map_id}: {e}")
            
    print(f"Concluído! {count} mapas processados e salvos em {processed_dir}")

if __name__ == "__main__":
    preprocess_all()
