import torch
import torch.nn as nn

class DirectorNet(nn.Module):
    """
    Modelo 'Diretor' V2: Condicionado pela dificuldade (estrelas) para gerar mapas mais coerentes.
    """
    def __init__(self, input_size=82, hidden_size=128, star_embedding_size=16):
        super(DirectorNet, self).__init__()
        
        # --- Camada de Embedding para as Estrelas ---
        # Transforma o número de estrelas (ex: 8.5) em um vetor denso que a rede pode entender.
        self.star_embed = nn.Linear(1, star_embedding_size)
        
        # --- Backbone (Compartilhado) ---
        # Extrai características musicais brutas
        self.conv1 = nn.Conv1d(input_size, 64, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm1d(64)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(0.2)
        
        self.conv2 = nn.Conv1d(64, 128, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm1d(128)
        
        # A entrada da GRU agora inclui as features do áudio + o embedding das estrelas
        gru_input_size = 128 + star_embedding_size
        self.gru = nn.GRU(gru_input_size, hidden_size, num_layers=2, batch_first=True, bidirectional=True, dropout=0.2)
        
        # --- Heads (Especialistas) ---
        # As cabeças de saída permanecem as mesmas, recebendo a saída da GRU
        
        # Head 1: Beat Detection (Quando bater?)
        self.fc_beat = nn.Linear(hidden_size * 2, 1)
        
        # Head 2: Complexity Control (Qual padrão usar?)
        self.fc_complexity = nn.Linear(hidden_size * 2, 3)
        
        # Head 3: Vertical Bias (Onde focar?)
        self.fc_vertical = nn.Linear(hidden_size * 2, 3)
        
    def forward(self, x, stars):
        # x: (batch, seq_len, features)
        # stars: (batch, 1)

        # 1. Processa o áudio com as CNNs
        x = x.permute(0, 2, 1) # (batch, features, seq_len)
        x = self.dropout(self.relu(self.bn1(self.conv1(x))))
        x = self.relu(self.bn2(self.conv2(x)))
        x = x.permute(0, 2, 1) # (batch, seq_len, 128)
        
        # 2. Processa as estrelas com a camada de embedding
        # stars_embedded: (batch, star_embedding_size)
        stars_embedded = self.relu(self.star_embed(stars))
        
        # 3. Expande o embedding de estrelas para corresponder ao comprimento da sequência
        # O unsqueeze adiciona uma dimensão de sequência: (batch, 1, star_embedding_size)
        # O expand repete os valores ao longo da dimensão da sequência
        seq_len = x.size(1)
        stars_expanded = stars_embedded.unsqueeze(1).expand(-1, seq_len, -1) # (batch, seq_len, star_embedding_size)
        
        # 4. Concatena as features do áudio com o embedding de estrelas
        combined_features = torch.cat((x, stars_expanded), dim=2) # (batch, seq_len, 128 + star_embedding_size)
        
        # 5. Passa as features combinadas pela GRU
        feat, _ = self.gru(combined_features)
        
        # 6. Passa a saída da GRU pelas cabeças de decisão
        beat_logits = self.fc_beat(feat)
        complexity_logits = self.fc_complexity(feat)
        vertical_logits = self.fc_vertical(feat)
        
        return beat_logits, complexity_logits, vertical_logits

def get_model():
    return DirectorNet()
