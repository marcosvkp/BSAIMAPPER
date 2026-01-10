import torch
import torch.nn as nn

class DirectorNet(nn.Module):
    def __init__(self, input_size=84, grid_size=12, note_mem=8, hidden_size=192):
        super().__init__()
        self.conv1 = nn.Conv1d(input_size, 96, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm1d(96)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(0.25)
        self.conv2 = nn.Conv1d(96, 192, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm1d(192)
        
        # Embedding para grid (linha, camada) - 12 posições possíveis
        self.grid_emb = nn.Embedding(grid_size, 16)
        
        # Embedding para memória de notas - 10 tipos (0-8 direções + 9 vazio/pad)
        self.note_emb = nn.Embedding(10, 8) 
        
        # GRU Input: 192 (audio) + 16 (grid) + 8*note_mem (histórico notas)
        gru_input_size = 192 + 16 + 8 * note_mem
        self.gru = nn.GRU(gru_input_size, hidden_size, num_layers=2, batch_first=True, bidirectional=True)
        
        self.attn = nn.MultiheadAttention(hidden_size*2, num_heads=4, batch_first=True)
        
        # Heads
        self.fc_beat = nn.Linear(hidden_size*2, 1)
        self.fc_complexity = nn.Linear(hidden_size*2, 3)
        self.fc_vertical = nn.Linear(hidden_size*2, 3)
        self.fc_angle = nn.Linear(hidden_size*2, 9) # 0-8 (Cut directions)

    def forward(self, x, grid_pos, note_mem_seq):
        # x: (batch, seq, features)
        x = x.permute(0, 2, 1)
        x = self.dropout(self.relu(self.bn1(self.conv1(x))))
        x = self.relu(self.bn2(self.conv2(x)))
        x = x.permute(0, 2, 1)
        
        # Embeddings
        grid_e = self.grid_emb(grid_pos) 
        
        note_e = self.note_emb(note_mem_seq) 
        note_e = note_e.view(note_e.shape[0], note_e.shape[1], -1) 
        
        x = torch.cat([x, grid_e, note_e], dim=-1)
        
        feat, _ = self.gru(x)
        feat, _ = self.attn(feat, feat, feat)
        
        return self.fc_beat(feat), self.fc_complexity(feat), self.fc_vertical(feat), self.fc_angle(feat)

class AngleNet(nn.Module):
    """
    Modelo especialista em Flow (Refiner).
    Agora com consciência da OUTRA MÃO para evitar colisões.
    Input: [Pos, Hand, TimeDiff, PrevAngle, OtherHandPos, OtherHandTimeDiff]
    """
    def __init__(self, hidden_size=128):
        super().__init__()
        # Embeddings
        self.pos_emb = nn.Embedding(12, 8) # Posição Atual
        self.hand_emb = nn.Embedding(2, 4) # Mão Atual
        self.angle_emb = nn.Embedding(10, 8) # Ângulo Anterior
        
        # Outra Mão
        self.other_pos_emb = nn.Embedding(13, 8) # 0-11 posições + 12 (Nenhuma nota perto)
        
        # Input: 8+4+8+8 + 1(dt) + 1(other_dt) = 30 features
        self.lstm = nn.LSTM(30, hidden_size, num_layers=2, batch_first=True, bidirectional=True)
        
        self.fc = nn.Linear(hidden_size * 2, 9) # 9 classes (0-8)

    def forward(self, pos_idx, hand_idx, time_diff, prev_angle_idx, other_pos_idx, other_time_diff):
        
        p_e = self.pos_emb(pos_idx)
        h_e = self.hand_emb(hand_idx)
        a_e = self.angle_emb(prev_angle_idx)
        o_p_e = self.other_pos_emb(other_pos_idx)
        
        x = torch.cat([p_e, h_e, a_e, o_p_e, time_diff, other_time_diff], dim=-1)
        
        out, _ = self.lstm(x)
        logits = self.fc(out)
        return logits

def get_model():
    return DirectorNet()

def get_angle_model():
    return AngleNet()
