import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import numpy as np
from model import get_model

class PreprocessedBeatSaberDataset(Dataset):
    def __init__(self, processed_dir, seq_len=1000):
        self.processed_dir = processed_dir
        self.files = [f for f in os.listdir(processed_dir) if f.endswith('_x.npy')]
        self.seq_len = seq_len
        
    def __len__(self):
        return len(self.files)
    
    def __getitem__(self, idx):
        file_x = self.files[idx]
        file_y = file_x.replace('_x.npy', '_y.npy')
        
        path_x = os.path.join(self.processed_dir, file_x)
        path_y = os.path.join(self.processed_dir, file_y)
        
        try:
            features = np.load(path_x)
            targets = np.load(path_y)
            
            # SAMPLING GUIADO POR EVENTOS
            # Encontrar onde tem notas (qualquer valor > 0)
            # targets shape: (frames, 12)
            has_note = np.any(targets > 0.1, axis=1)
            note_indices = np.where(has_note)[0]
            
            total_frames = features.shape[0]
            
            if len(note_indices) > 0 and total_frames > self.seq_len:
                # Escolhe uma nota aleatória para ser o centro (ou parte) da janela
                center_idx = np.random.choice(note_indices)
                
                # Define início aleatório, mas garantindo que a nota escolhida esteja dentro
                # Tenta colocar a nota no meio, mas com variação
                offset = np.random.randint(0, self.seq_len)
                start = max(0, center_idx - offset)
                
                # Ajusta se estourar o final
                if start + self.seq_len > total_frames:
                    start = max(0, total_frames - self.seq_len)
            else:
                # Se não tiver notas (mapa vazio?) ou for curto, pega do começo
                start = 0
                
            end = start + self.seq_len
            
            feat_crop = features[start:end]
            targ_crop = targets[start:end]
            
            # Padding se necessário (para mapas muito curtos)
            if feat_crop.shape[0] < self.seq_len:
                pad_len = self.seq_len - feat_crop.shape[0]
                feat_crop = np.concatenate([feat_crop, np.zeros((pad_len, 82))])
                targ_crop = np.concatenate([targ_crop, np.zeros((pad_len, 12))])
            
            return torch.tensor(feat_crop, dtype=torch.float32), torch.tensor(targ_crop, dtype=torch.float32)
            
        except Exception as e:
            print(f"Erro ao carregar {file_x}: {e}")
            return torch.zeros(self.seq_len, 82), torch.zeros(self.seq_len, 12)

def train():
    PROCESSED_DIR = "data/processed"
    BATCH_SIZE = 32
    EPOCHS = 100 # Menos épocas necessárias agora que cada batch é rico
    LEARNING_RATE = 0.0005
    
    if not os.path.exists(PROCESSED_DIR) or not os.listdir(PROCESSED_DIR):
        print("Dados processados não encontrados. Execute src/preprocess_data.py!")
        return

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Usando dispositivo: {device}")

    # seq_len menor (500 frames ~ 11s) para focar em padrões locais e aumentar a variedade do batch
    dataset = PreprocessedBeatSaberDataset(PROCESSED_DIR, seq_len=500)
    
    # Windows: num_workers=0 evita problemas de multiprocessing com CUDA às vezes. 
    # Se funcionar com >0, melhor.
    dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=4, persistent_workers=True)
    
    model = get_model().to(device)
    
    # Com sampling guiado, o desbalanceamento diminui drasticamente.
    # Podemos reduzir o peso de 10.0 para algo mais suave, como 4.0 ou 5.0
    pos_weight = torch.ones([12]).to(device) * 5.0 
    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', patience=5, factor=0.5)
    
    print("Iniciando treinamento V3 (Sampling Guiado + BatchNorm)...")
    model.train()
    
    for epoch in range(EPOCHS):
        total_loss = 0
        batches = 0
        for batch_idx, (data, target) in enumerate(dataloader):
            data, target = data.to(device), target.to(device)
            
            optimizer.zero_grad()
            output = model(data)
            
            loss = criterion(output, target)
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
            batches += 1
            
        avg_loss = total_loss/batches if batches > 0 else 0
        scheduler.step(avg_loss)
        print(f"Epoch {epoch+1}/{EPOCHS}, Loss: {avg_loss:.4f}, LR: {optimizer.param_groups[0]['lr']:.6f}")
        
    if not os.path.exists("models"):
        os.makedirs("models")
    torch.save(model.state_dict(), "models/beat_saber_model.pth")
    print("Modelo salvo!")

if __name__ == "__main__":
    train()
