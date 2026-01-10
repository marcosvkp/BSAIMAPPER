import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
import numpy as np
import json
from models_optimized import get_angle_model

BATCH_SIZE = 64
SEQ_LEN = 64 
EPOCHS = 50
LEARNING_RATE = 0.001

class AngleDataset(Dataset):
    def __init__(self, raw_maps_dir, seq_len=64):
        self.sequences = []
        self.seq_len = seq_len
        self._load_data(raw_maps_dir)
        
    def _load_data(self, raw_dir):
        print("Carregando sequências de notas para treino de Flow (Collision Aware)...")
        map_folders = [os.path.join(raw_dir, d) for d in os.listdir(raw_dir) if os.path.isdir(os.path.join(raw_dir, d))]
        
        for folder in map_folders:
            try:
                diff_file = None
                for f in os.listdir(folder):
                    if f.endswith(".dat") and "Expert" in f:
                        diff_file = os.path.join(folder, f)
                        break
                
                if not diff_file: continue
                
                with open(diff_file, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                    
                notes = data.get('_notes', data.get('colorNotes', []))
                if not notes: continue
                
                parsed_notes = []
                for n in notes:
                    t = n.get('_time', n.get('b'))
                    x = n.get('_lineIndex', n.get('x'))
                    y = n.get('_lineLayer', n.get('y'))
                    c = n.get('_cutDirection', n.get('d'))
                    type = n.get('_type', n.get('c')) 
                    
                    if t is None or x is None or y is None or c is None or type is None: continue
                    if type not in [0, 1]: continue 
                    
                    parsed_notes.append({'t': t, 'x': x, 'y': y, 'c': c, 'h': type})
                
                parsed_notes.sort(key=lambda k: k['t'])
                
                left_hand = [n for n in parsed_notes if n['h'] == 0]
                right_hand = [n for n in parsed_notes if n['h'] == 1]
                
                self._create_sequences(left_hand, right_hand)
                self._create_sequences(right_hand, left_hand)
                
            except Exception as e:
                continue
                
        print(f"Total de sequências de flow extraídas: {len(self.sequences)}")

    def _find_closest_note(self, target_time, other_notes):
        # Busca linear simples (pode ser otimizada, mas ok para treino offline)
        best_dist = 100.0
        best_note = None
        
        # Otimização: busca local se a lista estiver ordenada
        # Mas aqui vamos simplificar
        for n in other_notes:
            dist = abs(n['t'] - target_time)
            if dist < best_dist:
                best_dist = dist
                best_note = n
            # Se já passou muito, para
            if n['t'] > target_time + 2.0: break
            
        return best_note, best_dist

    def _create_sequences(self, hand_notes, other_hand_notes):
        if len(hand_notes) < self.seq_len: return
        
        for i in range(0, len(hand_notes) - self.seq_len, 8): 
            seq = hand_notes[i : i + self.seq_len]
            
            pos_indices = []
            hand_indices = []
            time_diffs = []
            prev_angles = [] 
            other_pos_indices = [] # Nova feature
            other_time_diffs = []  # Nova feature
            targets = []     
            
            prev_time = seq[0]['t']
            last_angle = 9 
            
            if i > 0:
                last_angle = hand_notes[i-1]['c']

            for n in seq:
                pos = min(11, max(0, (n['y'] * 4) + n['x']))
                pos_indices.append(pos)
                hand_indices.append(n['h'])
                
                dt = n['t'] - prev_time
                dt = min(4.0, max(0.0, dt))
                time_diffs.append(dt)
                
                prev_angles.append(last_angle)
                
                # --- Other Hand Logic ---
                other_n, dist = self._find_closest_note(n['t'], other_hand_notes)
                
                if other_n and dist < 0.5: # Se a outra mão está perto (< 0.5 beats)
                    o_pos = min(11, max(0, (other_n['y'] * 4) + other_n['x']))
                    other_pos_indices.append(o_pos)
                    other_time_diffs.append(dist)
                else:
                    other_pos_indices.append(12) # 12 = Longe/Nenhuma
                    other_time_diffs.append(1.0) # Valor alto
                
                targets.append(n['c'])
                
                last_angle = n['c']
                prev_time = n['t']
            
            self.sequences.append({
                'pos': np.array(pos_indices, dtype=np.longlong),
                'hand': np.array(hand_indices, dtype=np.longlong),
                'dt': np.array(time_diffs, dtype=np.float32),
                'prev_angle': np.array(prev_angles, dtype=np.longlong),
                'other_pos': np.array(other_pos_indices, dtype=np.longlong),
                'other_dt': np.array(other_time_diffs, dtype=np.float32),
                'target': np.array(targets, dtype=np.longlong)
            })

    def __len__(self):
        return len(self.sequences)

    def __getitem__(self, idx):
        item = self.sequences[idx]
        return (torch.tensor(item['pos']), 
                torch.tensor(item['hand']), 
                torch.tensor(item['dt']).unsqueeze(-1), 
                torch.tensor(item['prev_angle']),
                torch.tensor(item['other_pos']),
                torch.tensor(item['other_dt']).unsqueeze(-1),
                torch.tensor(item['target']))

def train_angle_net():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Treinando AngleNet (Collision-Aware) em: {device}")
    
    dataset = AngleDataset("data/raw_maps", seq_len=SEQ_LEN)
    if len(dataset) == 0:
        print("Nenhum mapa encontrado em data/raw_maps.")
        return

    dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True)
    
    model = get_angle_model().to(device)
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)
    criterion = nn.CrossEntropyLoss()
    
    print("Iniciando Treino de Flow...")
    
    for epoch in range(EPOCHS):
        model.train()
        total_loss = 0
        correct = 0
        total = 0
        
        for pos, hand, dt, prev_angle, other_pos, other_dt, target in dataloader:
            pos, hand, dt, prev_angle, other_pos, other_dt, target = \
                pos.to(device), hand.to(device), dt.to(device), prev_angle.to(device), \
                other_pos.to(device), other_dt.to(device), target.to(device)
            
            optimizer.zero_grad()
            logits = model(pos, hand, dt, prev_angle, other_pos, other_dt)
            
            loss = criterion(logits.view(-1, 9), target.view(-1))
            
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
            
            preds = torch.argmax(logits, dim=2)
            correct += (preds == target).sum().item()
            total += target.numel()
            
        acc = correct / total
        print(f"Epoch {epoch+1}/{EPOCHS} | Loss: {total_loss/len(dataloader):.4f} | Acc: {acc:.2%}")
        
    torch.save(model.state_dict(), "models/angle_net.pth")
    print("Modelo AngleNet salvo!")

if __name__ == "__main__":
    train_angle_net()
