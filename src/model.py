import torch
import torch.nn as nn

class BeatSaberMapper(nn.Module):
    def __init__(self, input_size=82, hidden_size=512, output_size=12):
        super(BeatSaberMapper, self).__init__()
        
        # CNN 1D
        self.conv1 = nn.Conv1d(in_channels=input_size, out_channels=128, kernel_size=5, padding=2)
        # BatchNorm ajuda MUITO na convergência
        self.bn1 = nn.BatchNorm1d(128)
        self.relu = nn.ReLU()
        self.dropout_cnn = nn.Dropout(0.2)
        
        # LSTM
        self.lstm = nn.LSTM(128, hidden_size, num_layers=3, batch_first=True, bidirectional=True, dropout=0.3)
        
        # FC
        self.fc1 = nn.Linear(hidden_size * 2, 256)
        self.dropout_fc = nn.Dropout(0.3)
        self.fc2 = nn.Linear(256, output_size)
        
    def forward(self, x):
        # x: (batch, time, features) -> (batch, features, time)
        x = x.permute(0, 2, 1)
        
        x = self.conv1(x)
        x = self.bn1(x) # Normalização aplicada
        x = self.relu(x)
        x = self.dropout_cnn(x)
        
        # Volta para (batch, time, features)
        x = x.permute(0, 2, 1)
        
        out, _ = self.lstm(x)
        
        out = self.dropout_fc(self.fc1(out))
        out = self.relu(out)
        out = self.fc2(out)
        
        return out

def get_model():
    # Retorna o modelo na CPU. O loop de treino decide para onde mover.
    return BeatSaberMapper()
