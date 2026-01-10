import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
import numpy as np
from models_optimized import get_model

BATCH_SIZE = 128
SEQ_LEN = 200
EPOCHS = 50 # Um pouco mais pois a tarefa é mais complexa
LEARNING_RATE = 0.0008

class DirectorDataset(Dataset):
    def __init__(self, processed_dir, seq_len=200):
        self.processed_dir = processed_dir
        self.files = [f for f in os.listdir(processed_dir) if f.endswith('_x.npy')]
        self.seq_len = seq_len
        
    def __len__(self):
        return len(self.files) * 5
    
    def __getitem__(self, idx):
        file_idx = idx % len(self.files)
        file_x = self.files[file_idx]
        file_y = file_x.replace('_x.npy', '_y.npy')
        
        # Carrega features e targets
        features = np.load(os.path.join(self.processed_dir, file_x), mmap_mode='r')
        targets = np.load(os.path.join(self.processed_dir, file_y), mmap_mode='r')
        
        total_frames = features.shape[0]
        start = np.random.randint(0, max(1, total_frames - self.seq_len))
        end = start + self.seq_len
        
        feat_crop = np.array(features[start:end])
        targ_crop = np.array(targets[start:end])
        
        # --- Engenharia de Targets On-the-Fly ---
        # 1. Beat (Binário)
        beat_target = np.any(targ_crop > 0.1, axis=1).astype(np.float32).reshape(-1, 1)
        
        # 2. Complexidade (Baseada na densidade da janela atual)
        density = np.mean(beat_target)
        if density > 0.15: comp_val = 2 # Tech/Stream
        elif density > 0.05: comp_val = 1 # Dance
        else: comp_val = 0 # Chill
        # Expande para o tamanho da sequencia (simplificação: mesma classe pra janela toda)
        comp_target = np.full((self.seq_len,), comp_val, dtype=np.longlong)
        
        # 3. Verticalidade (Média ponderada)
        l0 = np.sum(targ_crop[:, 0:4])
        l1 = np.sum(targ_crop[:, 4:8])
        l2 = np.sum(targ_crop[:, 8:12])
        total = l0 + l1 + l2 + 1e-6
        avg_h = (0*l0 + 1*l1 + 2*l2) / total
        vert_val = int(round(avg_h))
        vert_target = np.full((self.seq_len,), vert_val, dtype=np.longlong)
        
        # Padding
        if feat_crop.shape[0] < self.seq_len:
            pad = self.seq_len - feat_crop.shape[0]
            feat_crop = np.pad(feat_crop, ((0, pad), (0, 0)))
            beat_target = np.pad(beat_target, ((0, pad), (0, 0)))
            # Para classificação, padding ignora no loss (usualmente -100)
            comp_target = np.pad(comp_target, (0, pad), constant_values=0)
            vert_target = np.pad(vert_target, (0, pad), constant_values=0)

        return (torch.tensor(feat_crop, dtype=torch.float32), 
                torch.tensor(beat_target, dtype=torch.float32),
                torch.tensor(comp_target, dtype=torch.long),
                torch.tensor(vert_target, dtype=torch.long))

def train():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dataset = DirectorDataset("data/processed", seq_len=SEQ_LEN)
    dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=4)
    
    model = get_model().to(device)
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)
    
    # Losses
    crit_beat = nn.BCEWithLogitsLoss(pos_weight=torch.tensor([8.0]).to(device))
    crit_class = nn.CrossEntropyLoss()
    
    print("Iniciando Treino Multi-Head...")
    
    for epoch in range(EPOCHS):
        model.train()
        total_loss = 0
        
        for batch in dataloader:
            feats, t_beat, t_comp, t_vert = [b.to(device) for b in batch]
            
            optimizer.zero_grad()
            p_beat, p_comp, p_vert = model(feats)
            
            # Reshape para CrossEntropy: (Batch, Classes, Seq) vs (Batch, Seq)
            p_comp = p_comp.permute(0, 2, 1) 
            p_vert = p_vert.permute(0, 2, 1)
            
            loss_b = crit_beat(p_beat, t_beat)
            loss_c = crit_class(p_comp, t_comp)
            loss_v = crit_class(p_vert, t_vert)
            
            # Soma ponderada das perdas
            loss = loss_b + 0.5 * loss_c + 0.5 * loss_v
            
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
            
        print(f"Epoch {epoch+1}/{EPOCHS} | Loss: {total_loss/len(dataloader):.4f}")
        
    torch.save(model.state_dict(), "models/director_net.pth")
    print("Modelo salvo!")

if __name__ == "__main__":
    train()
