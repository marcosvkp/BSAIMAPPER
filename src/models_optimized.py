import torch
import torch.nn as nn

class DirectorNet(nn.Module):
    """
    Modelo 'Diretor': Não apenas detecta beats, mas dirige o estilo.
    """
    def __init__(self, input_size=82, hidden_size=128):
        super(DirectorNet, self).__init__()
        
        # --- Backbone (Compartilhado) ---
        # Extrai características musicais brutas
        self.conv1 = nn.Conv1d(input_size, 64, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm1d(64)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(0.2)
        
        self.conv2 = nn.Conv1d(64, 128, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm1d(128)
        
        self.gru = nn.GRU(128, hidden_size, num_layers=2, batch_first=True, bidirectional=True)
        
        # --- Heads (Especialistas) ---
        
        # Head 1: Beat Detection (Quando bater?)
        # Saída: Probabilidade (0-1)
        self.fc_beat = nn.Linear(hidden_size * 2, 1)
        
        # Head 2: Complexity Control (Qual padrão usar?)
        # Saída: 3 Classes (0: Chill, 1: Dance, 2: Tech/Stream)
        self.fc_complexity = nn.Linear(hidden_size * 2, 3)
        
        # Head 3: Vertical Bias (Onde focar?)
        # Saída: 3 Classes (0: Baixo, 1: Meio, 2: Cima)
        self.fc_vertical = nn.Linear(hidden_size * 2, 3)
        
    def forward(self, x):
        # Backbone
        x = x.permute(0, 2, 1)
        x = self.dropout(self.relu(self.bn1(self.conv1(x))))
        x = self.relu(self.bn2(self.conv2(x)))
        x = x.permute(0, 2, 1)
        
        feat, _ = self.gru(x)
        
        # Heads
        beat_logits = self.fc_beat(feat)
        complexity_logits = self.fc_complexity(feat)
        vertical_logits = self.fc_vertical(feat)
        
        return beat_logits, complexity_logits, vertical_logits

def get_model():
    return DirectorNet()
