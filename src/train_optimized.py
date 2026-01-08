import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
import numpy as np
from models_optimized import get_model, DirectorNetV2 # Importar o modelo V2

# --- Parâmetros de Treino ---
BATCH_SIZE = 32 # Reduzido para acomodar modelo maior
SEQ_LEN = 256   # Aumentado para dar mais contexto ao modelo
EPOCHS = 30
LEARNING_RATE = 0.0005
PROCESSED_DIR = "data/processed"
MODEL_SAVE_PATH = "models/director_net_v2.pth"

class DirectorV2Dataset(Dataset):
    """
    Dataset para o DirectorNetV2. Carrega features, targets de beat, metadados
    e os novos targets de direção de corte.
    """
    def __init__(self, processed_dir, seq_len=SEQ_LEN):
        self.processed_dir = processed_dir
        self.files = [f for f in os.listdir(processed_dir) if f.endswith('_x.npy')]
        self.seq_len = seq_len
        
    def __len__(self):
        # Cada arquivo pode gerar múltiplas amostras
        return len(self.files) * 5
    
    def __getitem__(self, idx):
        file_idx = idx % len(self.files)
        base_filename = self.files[file_idx].replace('_x.npy', '')
        
        # Carrega todos os arquivos de dados necessários
        features = np.load(os.path.join(self.processed_dir, f"{base_filename}_x.npy"), mmap_mode='r')
        targets = np.load(os.path.join(self.processed_dir, f"{base_filename}_y.npy"), mmap_mode='r')
        metadata = np.load(os.path.join(self.processed_dir, f"{base_filename}_meta.npy"), mmap_mode='r')
        cut_dirs = np.load(os.path.join(self.processed_dir, f"{base_filename}_cut.npy"), mmap_mode='r')

        total_frames = features.shape[0]
        start = np.random.randint(0, max(1, total_frames - self.seq_len))
        end = start + self.seq_len
        
        # Recorta a sequência
        feat_crop = np.array(features[start:end])
        targ_crop = np.array(targets[start:end])
        meta_crop = np.array(metadata[start:end])
        cut_crop = np.array(cut_dirs[start:end])

        # Padding para garantir tamanho fixo da sequência
        if feat_crop.shape[0] < self.seq_len:
            pad_len = self.seq_len - feat_crop.shape[0]
            feat_crop = np.pad(feat_crop, ((0, pad_len), (0, 0)))
            targ_crop = np.pad(targ_crop, ((0, pad_len), (0, 0)))
            meta_crop = np.pad(meta_crop, ((0, pad_len), (0, 0)))
            cut_crop = np.pad(cut_crop, (0, pad_len), constant_values=-1)

        # --- Prepara os Tensors de Target ---
        # 1. Target de Beat (se existe nota ou não)
        t_beat = np.any(targ_crop > 0.1, axis=1).astype(np.float32).reshape(-1, 1)
        
        # 2. Target de Complexidade (do metadado pré-calculado)
        t_comp = meta_crop[:, 0].astype(np.longlong)
        
        # 3. Target de Verticalidade (do metadado pré-calculado)
        t_vert = meta_crop[:, 1].astype(np.longlong)
        
        # 4. Target de Direção de Corte
        t_cut = cut_crop.astype(np.longlong)

        return (torch.tensor(feat_crop, dtype=torch.float32), 
                torch.tensor(t_beat, dtype=torch.float32),
                torch.tensor(t_comp, dtype=torch.long),
                torch.tensor(t_vert, dtype=torch.long),
                torch.tensor(t_cut, dtype=torch.long))

def train():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Usando dispositivo: {device}")

    dataset = DirectorV2Dataset(PROCESSED_DIR, seq_len=SEQ_LEN)
    dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=4, pin_memory=True)
    
    model = get_model() # get_model() agora retorna DirectorNetV2
    model.to(device)
    
    optimizer = optim.AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=1e-5)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=EPOCHS * len(dataloader))
    
    # --- Funções de Perda ---
    # Pos_weight ajuda a lidar com o desbalanceamento (muito mais não-beats do que beats)
    crit_beat = nn.BCEWithLogitsLoss(pos_weight=torch.tensor([10.0]).to(device))
    # Ignore_index=-1 é CRUCIAL para a direção de corte, pois só calculamos a perda onde há uma nota.
    crit_class = nn.CrossEntropyLoss(ignore_index=-1) 
    
    print("Iniciando Treino do DirectorNetV2...")
    
    for epoch in range(EPOCHS):
        model.train()
        total_loss, total_loss_b, total_loss_c, total_loss_v, total_loss_d = 0, 0, 0, 0, 0
        
        for batch in dataloader:
            feats, t_beat, t_comp, t_vert, t_cut = [b.to(device) for b in batch]
            
            optimizer.zero_grad()
            
            # O modelo agora retorna 4 saídas
            p_beat, p_comp, p_vert, p_cut = model(feats)
            
            # --- Cálculo das Perdas Individuais ---
            loss_b = crit_beat(p_beat, t_beat)
            
            # Permutar para (Batch, Classes, SeqLen) como esperado pela CrossEntropyLoss
            loss_c = crit_class(p_comp.permute(0, 2, 1), t_comp)
            loss_v = crit_class(p_vert.permute(0, 2, 1), t_vert)
            loss_d = crit_class(p_cut.permute(0, 2, 1), t_cut)
            
            # --- Perda Total Ponderada ---
            # A perda de beat e direção são as mais importantes.
            loss = (1.5 * loss_b) + (0.5 * loss_c) + (0.5 * loss_v) + (1.0 * loss_d)
            
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0) # Previne gradientes explosivos
            optimizer.step()
            scheduler.step()
            
            total_loss += loss.item()
            total_loss_b += loss_b.item()
            total_loss_c += loss_c.item()
            total_loss_v += loss_v.item()
            total_loss_d += loss_d.item()
            
        avg_loss = total_loss / len(dataloader)
        avg_b = total_loss_b / len(dataloader)
        avg_c = total_loss_c / len(dataloader)
        avg_v = total_loss_v / len(dataloader)
        avg_d = total_loss_d / len(dataloader)
        
        print(f"Epoch {epoch+1}/{EPOCHS} | Loss Total: {avg_loss:.4f} "
              f"[B: {avg_b:.4f}, C: {avg_c:.4f}, V: {avg_v:.4f}, D: {avg_d:.4f}]")
        
    if not os.path.exists("models"):
        os.makedirs("models")
    torch.save(model.state_dict(), MODEL_SAVE_PATH)
    print(f"Modelo V2 salvo em '{MODEL_SAVE_PATH}'!")

if __name__ == "__main__":
    # Certifique-se de ter executado preprocess_data.py primeiro!
    train()
