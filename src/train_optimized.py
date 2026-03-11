import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
import numpy as np
from models_optimized import get_model

# Parâmetros de Treinamento
BATCH_SIZE = 64
SEQ_LEN = 250  # Aumentar um pouco a janela para dar mais contexto
EPOCHS = 30    # Mais épocas para um aprendizado mais fino
LEARNING_RATE = 0.0007
NUM_WORKERS = os.cpu_count() or 4

class DirectorDataset(Dataset):
    def __init__(self, processed_dir, seq_len):
        self.processed_dir = processed_dir
        # Garante que estamos pegando apenas a base dos nomes de arquivo
        self.files = list(set([f.replace('_x.npy', '') for f in os.listdir(processed_dir) if f.endswith('_x.npy')]))
        self.seq_len = seq_len
        
    def __len__(self):
        # Aumenta a quantidade de amostras por época para treinar mais robustamente
        return len(self.files) * 10
    
    def __getitem__(self, idx):
        file_base = self.files[idx % len(self.files)]
        
        # Caminhos para todos os arquivos de dados
        path_x = os.path.join(self.processed_dir, f"{file_base}_x.npy")
        path_y = os.path.join(self.processed_dir, f"{file_base}_y.npy")
        path_stars = os.path.join(self.processed_dir, f"{file_base}_stars.npy")

        # Carrega os dados usando mmap_mode para eficiência de memória
        features = np.load(path_x, mmap_mode='r')
        targets = np.load(path_y, mmap_mode='r')
        stars = np.load(path_stars) # Estrelas é um valor único, não precisa de mmap

        # Seleciona um trecho aleatório da música
        total_frames = features.shape[0]
        start = np.random.randint(0, max(1, total_frames - self.seq_len))
        end = start + self.seq_len
        
        feat_crop = np.array(features[start:end])
        targ_crop = np.array(targets[start:end])
        
        # --- Engenharia de Targets On-the-Fly ---
        beat_target = np.any(targ_crop > 0.1, axis=1).astype(np.float32).reshape(-1, 1)
        
        density = np.mean(beat_target)
        if density > 0.15: comp_val = 2
        elif density > 0.05: comp_val = 1
        else: comp_val = 0
        comp_target = np.full((self.seq_len,), comp_val, dtype=np.longlong)
        
        l0 = np.sum(targ_crop[:, 0:4]); l1 = np.sum(targ_crop[:, 4:8]); l2 = np.sum(targ_crop[:, 8:12])
        total = l0 + l1 + l2 + 1e-6
        avg_h = (0*l0 + 1*l1 + 2*l2) / total
        vert_val = int(round(avg_h))
        vert_target = np.full((self.seq_len,), vert_val, dtype=np.longlong)
        
        # Padding, se necessário
        pad_len = self.seq_len - feat_crop.shape[0]
        if pad_len > 0:
            feat_crop = np.pad(feat_crop, ((0, pad_len), (0, 0)))
            beat_target = np.pad(beat_target, ((0, pad_len), (0, 0)))
            # Usar -100 para ignorar no cálculo da loss de classificação
            comp_target = np.pad(comp_target, (0, pad_len), constant_values=-100)
            vert_target = np.pad(vert_target, (0, pad_len), constant_values=-100)

        return (torch.tensor(feat_crop, dtype=torch.float32), 
                torch.tensor(beat_target, dtype=torch.float32),
                torch.tensor(comp_target, dtype=torch.long),
                torch.tensor(vert_target, dtype=torch.long),
                torch.tensor([stars.item()], dtype=torch.float32)) # Passa as estrelas como um tensor

def train():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Usando dispositivo: {device}")

    dataset = DirectorDataset("data/processed", seq_len=SEQ_LEN)
    dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=NUM_WORKERS, persistent_workers=True)
    
    model = get_model().to(device)
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', patience=3, factor=0.5, verbose=True)
    
    # Funções de Perda (Loss)
    # Aumentar o peso positivo para a detecção de beat, pois é um evento mais raro
    crit_beat = nn.BCEWithLogitsLoss(pos_weight=torch.tensor([10.0]).to(device))
    # Ignora o padding (-100) no cálculo da perda de classificação
    crit_class = nn.CrossEntropyLoss(ignore_index=-100)
    
    print("Iniciando Treino do Modelo Diretor V2 (Condicional à Dificuldade)...")
    
    for epoch in range(EPOCHS):
        model.train()
        # Acumuladores para as perdas individuais
        epoch_loss, beat_loss_acc, comp_loss_acc, vert_loss_acc = 0, 0, 0, 0
        
        for i, batch in enumerate(dataloader):
            # Desempacota o novo tensor de estrelas
            feats, t_beat, t_comp, t_vert, t_stars = [b.to(device) for b in batch]
            
            optimizer.zero_grad()
            
            # Passa as estrelas para o modelo
            p_beat, p_comp, p_vert = model(feats, t_stars)
            
            # Reshape para CrossEntropy: (Batch, Classes, Seq) vs (Batch, Seq)
            p_comp = p_comp.permute(0, 2, 1) 
            p_vert = p_vert.permute(0, 2, 1)
            
            # Calcula as perdas individuais
            loss_b = crit_beat(p_beat, t_beat)
            loss_c = crit_class(p_comp, t_comp)
            loss_v = crit_class(p_vert, t_vert)
            
            # Soma ponderada das perdas. A detecção de beat é a mais importante.
            loss = loss_b * 1.5 + loss_c * 0.75 + loss_v * 0.75
            
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0) # Previne gradientes explosivos
            optimizer.step()
            
            # Acumula as perdas para o log
            epoch_loss += loss.item()
            beat_loss_acc += loss_b.item()
            comp_loss_acc += loss_c.item()
            vert_loss_acc += loss_v.item()

        # Calcula e exibe as médias das perdas no final da época
        avg_loss = epoch_loss / len(dataloader)
        avg_beat = beat_loss_acc / len(dataloader)
        avg_comp = comp_loss_acc / len(dataloader)
        avg_vert = vert_loss_acc / len(dataloader)
        
        scheduler.step(avg_loss)
        
        print(f"Epoch {epoch+1}/{EPOCHS} | Loss Total: {avg_loss:.4f} | "
              f"Beat: {avg_beat:.4f} | Comp: {avg_comp:.4f} | Vert: {avg_vert:.4f} | "
              f"LR: {optimizer.param_groups[0]['lr']:.6f}")
        
    if not os.path.exists("models"):
        os.makedirs("models")
    torch.save(model.state_dict(), "models/director_net_v2_stars.pth")
    print("Modelo V2 (com estrelas) salvo!")

if __name__ == "__main__":
    train()
