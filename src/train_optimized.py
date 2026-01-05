import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
import numpy as np
from models_optimized import get_beat_model

# --- Configurações Otimizadas ---
BATCH_SIZE = 64        # Aumentado para aproveitar a GPU
SEQ_LEN = 200          # Reduzido de 1000 para 200 (foco em contexto local)
EPOCHS = 20            # Reduzido drasticamente (modelo menor converge mais rápido)
LEARNING_RATE = 0.001
# --------------------------------

class OptimizedDataset(Dataset):
    def __init__(self, processed_dir, seq_len=200):
        self.processed_dir = processed_dir
        self.files = [f for f in os.listdir(processed_dir) if f.endswith('_x.npy')]
        self.seq_len = seq_len
        
    def __len__(self):
        return len(self.files) * 5 # Data augmentation "virtual" (amostra 5 trechos por arquivo)
    
    def __getitem__(self, idx):
        file_idx = idx % len(self.files)
        file_x = self.files[file_idx]
        file_y = file_x.replace('_x.npy', '_y.npy')
        
        # Carregamento preguiçoso (lazy loading) para economizar RAM
        features = np.load(os.path.join(self.processed_dir, file_x), mmap_mode='r')
        targets = np.load(os.path.join(self.processed_dir, file_y), mmap_mode='r')
        
        # Seleção inteligente de janela: foca onde tem notas
        total_frames = features.shape[0]
        if total_frames <= self.seq_len:
            start = 0
        else:
            # Tenta encontrar uma região com notas
            for _ in range(5):
                start = np.random.randint(0, total_frames - self.seq_len)
                # Verifica se tem notas nessa janela (soma dos targets > 0)
                # O target original tem shape (frames, 12). Vamos simplificar para (frames, 1) -> Tem nota ou não?
                window_target = targets[start:start+self.seq_len]
                if np.sum(window_target) > 0:
                    break
        
        end = start + self.seq_len
        
        feat_crop = np.array(features[start:end]) # Converte mmap para array real
        targ_crop = np.array(targets[start:end])
        
        # Simplificação do Target:
        # De (Seq, 12) para (Seq, 1). 1 se houver QUALQUER nota naquele frame.
        # Isso treina a rede para detectar "BEATS", não posições exatas.
        beat_target = np.any(targ_crop > 0.1, axis=1).astype(np.float32).reshape(-1, 1)
        
        # Padding se necessário
        if feat_crop.shape[0] < self.seq_len:
            pad_len = self.seq_len - feat_crop.shape[0]
            feat_crop = np.pad(feat_crop, ((0, pad_len), (0, 0)))
            beat_target = np.pad(beat_target, ((0, pad_len), (0, 0)))
            
        return torch.tensor(feat_crop, dtype=torch.float32), torch.tensor(beat_target, dtype=torch.float32)

def train_optimized():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Treinando em: {device}")
    
    dataset = OptimizedDataset("data/processed", seq_len=SEQ_LEN)
    dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=4, pin_memory=True)
    
    model = get_beat_model().to(device)
    
    # BCEWithLogitsLoss é mais estável numericamente que Sigmoid + BCELoss
    # pos_weight ajuda a lidar com o desbalanceamento (muito mais silêncio que notas)
    criterion = nn.BCEWithLogitsLoss(pos_weight=torch.tensor([10.0]).to(device))
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)
    
    print("Iniciando Treino Otimizado (Beat Detection Only)...")
    
    for epoch in range(EPOCHS):
        model.train()
        total_loss = 0
        
        for batch_idx, (data, target) in enumerate(dataloader):
            data, target = data.to(device), target.to(device)
            
            optimizer.zero_grad()
            output = model(data) # Output: (Batch, Seq, 1)
            
            loss = criterion(output, target)
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
            
            if batch_idx % 10 == 0:
                print(f"Epoch {epoch+1} | Batch {batch_idx} | Loss: {loss.item():.4f}")
                
        print(f"Epoch {epoch+1} Finalizada. Média Loss: {total_loss/len(dataloader):.4f}")
        
    torch.save(model.state_dict(), "models/beat_net_optimized.pth")
    print("Modelo salvo: models/beat_net_optimized.pth")

if __name__ == "__main__":
    train_optimized()
