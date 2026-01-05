import torch
import torch.nn as nn

class BeatNet(nn.Module):
    """
    Modelo OTIMIZADO para detecção de ritmo (Onsets).
    Foca apenas em 'QUANDO' colocar uma nota.
    Substitui o modelo pesado anterior por uma arquitetura CRNN (Conv + GRU) leve.
    """
    def __init__(self, input_size=82, hidden_size=128):
        super(BeatNet, self).__init__()
        
        # 1. Feature Extractor (CNN 1D)
        # Reduz a dimensionalidade temporal e extrai padrões locais
        self.conv1 = nn.Conv1d(input_size, 64, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm1d(64)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(0.2)
        
        self.conv2 = nn.Conv1d(64, 128, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm1d(128)
        
        # 2. Temporal Context (GRU)
        # GRU é ~25% mais rápido que LSTM e consome menos VRAM
        # Bidirecional para olhar passado e futuro
        self.gru = nn.GRU(128, hidden_size, num_layers=2, batch_first=True, bidirectional=True)
        
        # 3. Classifier
        # Saída binária: 1 = Tem nota, 0 = Não tem nota
        self.fc = nn.Linear(hidden_size * 2, 1) 
        
    def forward(self, x):
        # x: (batch, time, features) -> (batch, features, time) para CNN
        x = x.permute(0, 2, 1)
        
        x = self.dropout(self.relu(self.bn1(self.conv1(x))))
        x = self.relu(self.bn2(self.conv2(x)))
        
        # Volta para (batch, time, features) para RNN
        x = x.permute(0, 2, 1)
        
        out, _ = self.gru(x)
        
        logits = self.fc(out)
        return logits

class ComplexityNet(nn.Module):
    """
    Modelo opcional para classificar a 'intensidade' ou 'tipo de padrão' do trecho.
    Ex: 0 = Vazio, 1 = Single Notes, 2 = Stream/Burst
    """
    def __init__(self, input_size=82, hidden_size=64, num_classes=3):
        super(ComplexityNet, self).__init__()
        self.conv1 = nn.Conv1d(input_size, 32, kernel_size=5, padding=2)
        self.pool = nn.AdaptiveAvgPool1d(1) # Global Pooling
        self.fc = nn.Linear(32, num_classes)
        
    def forward(self, x):
        x = x.permute(0, 2, 1)
        x = torch.relu(self.conv1(x))
        x = self.pool(x).squeeze(-1)
        return self.fc(x)

def get_beat_model():
    return BeatNet()
