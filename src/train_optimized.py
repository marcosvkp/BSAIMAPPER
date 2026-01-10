import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
import numpy as np
from models_optimized import get_model
from torch.cuda.amp import GradScaler, autocast

BATCH_SIZE = 256
SEQ_LEN = 400
EPOCHS = 60 # Reduzido pois converge rápido com dropout
LEARNING_RATE = 0.0005
NOTE_MEM = 8

class DirectorDataset(Dataset):
    def __init__(self, processed_dir, seq_len=400, note_mem=8):
        self.processed_dir = processed_dir
        self.files = [f for f in os.listdir(processed_dir) if f.endswith('_x.npy')]
        self.seq_len = seq_len
        self.note_mem = note_mem
        
    def __len__(self):
        return len(self.files) * 5
    
    def __getitem__(self, idx):
        file_idx = idx % len(self.files)
        file_x = self.files[file_idx]
        file_y = file_x.replace('_x.npy', '_y.npy')
        file_meta = file_x.replace('_x.npy', '_meta.npy')
        
        features = np.load(os.path.join(self.processed_dir, file_x), mmap_mode='r')
        targets = np.load(os.path.join(self.processed_dir, file_y), mmap_mode='r')
        metadata = np.load(os.path.join(self.processed_dir, file_meta), mmap_mode='r')
        
        total_frames = features.shape[0]
        start = np.random.randint(0, max(1, total_frames - self.seq_len))
        end = start + self.seq_len
        
        feat_crop = np.array(features[start:end])
        targ_crop = np.array(targets[start:end])
        meta_crop = np.array(metadata[start:end])
        
        # --- Targets ---
        beat_target = np.any(targ_crop > 0.1, axis=1).astype(np.float32).reshape(-1, 1)
        comp_target = meta_crop[:, 0].astype(np.longlong)
        vert_target = meta_crop[:, 1].astype(np.longlong)
        
        # --- Grid Position Input ---
        grid_indices = np.argmax(targ_crop, axis=1)
        has_note = np.max(targ_crop, axis=1) > 0.1
        grid_pos = np.where(has_note, grid_indices, 6).astype(np.longlong)

        # --- Note Memory Input ---
        note_types = np.zeros_like(grid_indices)
        note_types[grid_indices < 4] = 1 # Down
        note_types[grid_indices >= 8] = 0 # Up
        note_types[(grid_indices >= 4) & (grid_indices < 8)] = 8 # Dot/Any
        note_types[~has_note] = 9
        
        note_mem_seq = np.zeros((self.seq_len, self.note_mem), dtype=np.longlong)
        note_mem_seq.fill(9)
        
        for i in range(self.seq_len):
            if i > 0:
                available = min(i, self.note_mem)
                past_notes = note_types[i-available:i]
                note_mem_seq[i, -available:] = past_notes
        
        # --- MEMORY DROPOUT (CRUCIAL) ---
        # Em 70% dos casos, zeramos a memória parcial ou totalmente
        # Isso força o modelo a olhar para o áudio
        if np.random.rand() < 0.7:
            # Opção 1: Apagar tudo (simula início da música ou erro)
            if np.random.rand() < 0.3:
                note_mem_seq.fill(9)
                grid_pos.fill(6)
            else:
                # Opção 2: Apagar pedaços aleatórios (ruído)
                mask = np.random.rand(self.seq_len) < 0.5
                note_mem_seq[mask] = 9
                grid_pos[mask] = 6

        # Padding
        if feat_crop.shape[0] < self.seq_len:
            pad = self.seq_len - feat_crop.shape[0]
            feat_crop = np.pad(feat_crop, ((0, pad), (0, 0)))
            beat_target = np.pad(beat_target, ((0, pad), (0, 0)))
            comp_target = np.pad(comp_target, (0, pad), constant_values=0)
            vert_target = np.pad(vert_target, (0, pad), constant_values=0)
            grid_pos = np.pad(grid_pos, (0, pad), constant_values=6)
            note_mem_seq = np.pad(note_mem_seq, ((0, pad), (0, 0)), constant_values=9)
            
            angle_target = np.zeros(self.seq_len, dtype=np.longlong)
            angle_target[vert_target == 2] = 0
            angle_target[vert_target == 0] = 1
            angle_target[vert_target == 1] = 8
            angle_target = np.pad(angle_target, (0, pad), constant_values=8)
        else:
            angle_target = np.zeros(self.seq_len, dtype=np.longlong)
            angle_target[vert_target == 2] = 0
            angle_target[vert_target == 0] = 1
            angle_target[vert_target == 1] = 8

        return (torch.tensor(feat_crop, dtype=torch.float32), 
                torch.tensor(grid_pos, dtype=torch.long),
                torch.tensor(note_mem_seq, dtype=torch.long),
                torch.tensor(beat_target, dtype=torch.float32),
                torch.tensor(comp_target, dtype=torch.long),
                torch.tensor(vert_target, dtype=torch.long),
                torch.tensor(angle_target, dtype=torch.long))

def train():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Treinando em: {device}")
    
    dataset = DirectorDataset("data/processed", seq_len=SEQ_LEN, note_mem=NOTE_MEM)
    dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=8, pin_memory=True)
    
    model = get_model().to(device)
    optimizer = optim.AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=1e-4)
    scaler = GradScaler()
    
    # Aumentei o peso positivo para garantir que ele não ignore os beats
    crit_beat = nn.BCEWithLogitsLoss(pos_weight=torch.tensor([15.0]).to(device))
    crit_class = nn.CrossEntropyLoss()
    
    best_loss = float('inf')
    
    print("Iniciando Treino Otimizado (Com Memory Dropout)...")
    
    for epoch in range(EPOCHS):
        model.train()
        total_loss = 0
        
        for batch in dataloader:
            feats, grid, mem, t_beat, t_comp, t_vert, t_angle = [b.to(device) for b in batch]
            
            optimizer.zero_grad()
            
            with autocast():
                p_beat, p_comp, p_vert, p_angle = model(feats, grid, mem)
                
                p_comp = p_comp.permute(0, 2, 1) 
                p_vert = p_vert.permute(0, 2, 1)
                p_angle = p_angle.permute(0, 2, 1)

                loss_b = crit_beat(p_beat, t_beat)
                loss_c = crit_class(p_comp, t_comp)
                loss_v = crit_class(p_vert, t_vert)
                loss_a = crit_class(p_angle, t_angle)

                loss = loss_b + 0.5 * loss_c + 0.5 * loss_v + 0.3 * loss_a
            
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            
            total_loss += loss.item()
            
        avg_loss = total_loss/len(dataloader)
        print(f"Epoch {epoch+1}/{EPOCHS} | Loss: {avg_loss:.4f}")
        
        if avg_loss < best_loss:
            best_loss = avg_loss
            torch.save(model.state_dict(), "models/director_net_best.pth")
            
    torch.save(model.state_dict(), "models/director_net_final.pth")
    print("Modelo salvo!")

if __name__ == "__main__":
    train()
