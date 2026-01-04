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
            
            has_note = np.any(targets > 0.1, axis=1)
            note_indices = np.where(has_note)[0]
            
            total_frames = features.shape[0]
            
            if len(note_indices) > 0 and total_frames > self.seq_len:
                center_idx = np.random.choice(note_indices)
                offset = np.random.randint(0, self.seq_len)
                start = max(0, center_idx - offset)
                
                if start + self.seq_len > total_frames:
                    start = max(0, total_frames - self.seq_len)
            else:
                start = 0
                
            end = start + self.seq_len
            
            feat_crop = features[start:end]
            targ_crop = targets[start:end]
            
            if feat_crop.shape[0] < self.seq_len:
                pad_len = self.seq_len - feat_crop.shape[0]
                feat_crop = np.concatenate([feat_crop, np.zeros((pad_len, 82))])
                targ_crop = np.concatenate([targ_crop, np.zeros((pad_len, 12))])
            
            return torch.tensor(feat_crop, dtype=torch.float32), torch.tensor(targ_crop, dtype=torch.float32)
            
        except Exception as e:
            print(f"Erro ao carregar {file_x}: {e}")
            return torch.zeros(self.seq_len, 82), torch.zeros(self.seq_len, 12)

class VerticalDiversityLoss(nn.Module):
    def __init__(self, pos_weight, diversity_lambda=0.1, target_dist=None):
        super().__init__()
        self.bce_loss = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
        self.diversity_lambda = diversity_lambda
        # Distribuição alvo: um pouco mais de notas baixas e médias, menos nas de cima.
        if target_dist is None:
            self.target_dist = torch.tensor([0.4, 0.4, 0.2]) 
        else:
            self.target_dist = torch.tensor(target_dist)

    def forward(self, output, target):
        # 1. Perda de classificação principal
        main_loss = self.bce_loss(output, target)

        # 2. Perda de diversidade vertical
        # output shape: (batch, seq_len, 12)
        
        # Ativações sigmoid para interpretar como probabilidades
        probs = torch.sigmoid(output)
        
        # Soma as probabilidades nas 4 colunas para cada uma das 3 linhas
        # Linha 0 (baixa): canais 0-3
        # Linha 1 (meio):  canais 4-7
        # Linha 2 (cima):  canais 8-11
        row_probs_0 = probs[:, :, 0:4].sum(dim=2)
        row_probs_1 = probs[:, :, 4:8].sum(dim=2)
        row_probs_2 = probs[:, :, 8:12].sum(dim=2)
        
        # Média de ativação por linha em todo o batch
        avg_row_activation_0 = row_probs_0.mean()
        avg_row_activation_1 = row_probs_1.mean()
        avg_row_activation_2 = row_probs_2.mean()
        
        # Concatena em um tensor de distribuição
        current_dist = torch.stack([
            avg_row_activation_0, 
            avg_row_activation_1, 
            avg_row_activation_2
        ])
        
        # Normaliza a distribuição (soma para 1)
        current_dist = current_dist / (current_dist.sum() + 1e-6)
        
        # Usa KL Divergence para medir a "distância" da distribuição atual para a alvo
        # Adiciona um pequeno epsilon para evitar log(0)
        self.target_dist = self.target_dist.to(current_dist.device)
        diversity_loss = nn.functional.kl_div(
            (current_dist + 1e-6).log(), 
            self.target_dist, 
            reduction='batchmean'
        )
        
        # 3. Combina as perdas
        total_loss = main_loss + self.diversity_lambda * diversity_loss
        
        return total_loss

def train():
    PROCESSED_DIR = "data/processed"
    BATCH_SIZE = 32
    EPOCHS = 100
    LEARNING_RATE = 0.0005
    DIVERSITY_LAMBDA = 0.2 # Fator de regularização. Começar baixo e aumentar se necessário.
    
    if not os.path.exists(PROCESSED_DIR) or not os.listdir(PROCESSED_DIR):
        print("Dados processados não encontrados. Execute src/preprocess_data.py!")
        return

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Usando dispositivo: {device}")

    dataset = PreprocessedBeatSaberDataset(PROCESSED_DIR, seq_len=500)
    dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=4, persistent_workers=True)
    
    model = get_model().to(device)
    
    pos_weight = torch.ones([12]).to(device) * 5.0 
    # Usa a nova função de perda customizada
    criterion = VerticalDiversityLoss(pos_weight=pos_weight, diversity_lambda=DIVERSITY_LAMBDA)
    
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', patience=5, factor=0.5)
    
    print("Iniciando treinamento V4 (Loss com Diversidade Vertical)...")
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
    torch.save(model.state_dict(), "models/beat_saber_model_v4.pth")
    print("Modelo V4 (com diversidade) salvo!")

if __name__ == "__main__":
    train()
