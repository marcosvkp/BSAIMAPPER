import torch
import torch.nn as nn

class DirectorNet(nn.Module):
    """
    Modelo 'Diretor' V1: Arquitetura original com GRU bidirecional.
    Mantida para referência.
    """
    def __init__(self, input_size=82, hidden_size=128):
        super(DirectorNet, self).__init__()
        
        self.conv1 = nn.Conv1d(input_size, 64, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm1d(64)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(0.2)
        
        self.conv2 = nn.Conv1d(64, 128, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm1d(128)
        
        self.gru = nn.GRU(128, hidden_size, num_layers=2, batch_first=True, bidirectional=True)
        
        self.fc_beat = nn.Linear(hidden_size * 2, 1)
        self.fc_complexity = nn.Linear(hidden_size * 2, 3)
        self.fc_vertical = nn.Linear(hidden_size * 2, 3)
        
    def forward(self, x):
        x = x.permute(0, 2, 1)
        x = self.dropout(self.relu(self.bn1(self.conv1(x))))
        x = self.relu(self.bn2(self.conv2(x)))
        x = x.permute(0, 2, 1)
        
        feat, _ = self.gru(x)
        
        beat_logits = self.fc_beat(feat)
        complexity_logits = self.fc_complexity(feat)
        vertical_logits = self.fc_vertical(feat)
        
        return beat_logits, complexity_logits, vertical_logits

class DirectorNetV2(nn.Module):
    """
    Modelo 'Diretor' V2: Arquitetura aprimorada com mais profundidade, atenção e previsão de ângulo.
    - GRU mais profundo para capturar dependências temporais complexas.
    - Multi-head Attention para focar nas features mais relevantes da sequência.
    - Head de Direção de Corte para prever o ângulo da nota.
    """
    def __init__(self, input_size=82, hidden_size=128, num_gru_layers=3, num_attention_heads=4, num_cut_directions=9):
        super(DirectorNetV2, self).__init__()
        
        self.conv_backbone = nn.Sequential(
            nn.Conv1d(input_size, 64, kernel_size=3, padding=1),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Conv1d(64, hidden_size, kernel_size=3, padding=1),
            nn.BatchNorm1d(hidden_size),
            nn.ReLU()
        )
        
        self.gru = nn.GRU(hidden_size, hidden_size, num_layers=num_gru_layers, batch_first=True, bidirectional=True, dropout=0.2 if num_gru_layers > 1 else 0)
        
        gru_output_size = hidden_size * 2
        
        self.attention = nn.MultiheadAttention(embed_dim=gru_output_size, num_heads=num_attention_heads, batch_first=True)
        self.layer_norm1 = nn.LayerNorm(gru_output_size)
        
        self.feed_forward = nn.Sequential(
            nn.Linear(gru_output_size, gru_output_size * 2),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(gru_output_size * 2, gru_output_size)
        )
        self.layer_norm2 = nn.LayerNorm(gru_output_size)

        # Head 1: Beat Detection (Quando bater?)
        self.fc_beat = nn.Linear(gru_output_size, 1)
        
        # Head 2: Complexity Control (Qual padrão usar?)
        self.fc_complexity = nn.Linear(gru_output_size, 3)
        
        # Head 3: Vertical Bias (Onde focar?)
        self.fc_vertical = nn.Linear(gru_output_size, 3)
        
        # Head 4: Cut Direction (Qual ângulo usar?)
        self.fc_cut_direction = nn.Linear(gru_output_size, num_cut_directions)
        
    def forward(self, x):
        x = x.permute(0, 2, 1)
        x = self.conv_backbone(x)
        x = x.permute(0, 2, 1)
        
        gru_out, _ = self.gru(x)
        
        attn_output, _ = self.attention(gru_out, gru_out, gru_out)
        feat = self.layer_norm1(gru_out + attn_output)
        
        ff_output = self.feed_forward(feat)
        feat = self.layer_norm2(feat + ff_output)
        
        beat_logits = self.fc_beat(feat)
        complexity_logits = self.fc_complexity(feat)
        vertical_logits = self.fc_vertical(feat)
        cut_direction_logits = self.fc_cut_direction(feat)
        
        return beat_logits, complexity_logits, vertical_logits, cut_direction_logits

def get_model():
    """
    Retorna a instância do modelo V2.
    """
    return DirectorNetV2()
