import torch
import torch.nn as nn

class DirectorNet(nn.Module):
    """
    Modelo 'Diretor' V2: Condicionado pela dificuldade (estrelas) para gerar mapas mais coerentes.

    V2.1 — input_size atualizado para 93 features (era 82).
    Veja audio_processor.py para o detalhamento completo das 93 features.
    """
    def __init__(self, input_size=93, hidden_size=128, star_embedding_size=16):
        super(DirectorNet, self).__init__()

        # --- Camada de Embedding para as Estrelas ---
        self.star_embed = nn.Linear(1, star_embedding_size)

        # --- Backbone (Compartilhado) ---
        self.conv1 = nn.Conv1d(input_size, 64, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm1d(64)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(0.2)

        self.conv2 = nn.Conv1d(64, 128, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm1d(128)

        # A entrada da GRU inclui features do áudio + embedding das estrelas
        gru_input_size = 128 + star_embedding_size
        self.gru = nn.GRU(gru_input_size, hidden_size, num_layers=2,
                          batch_first=True, bidirectional=True, dropout=0.2)

        # --- Heads (Especialistas) ---
        self.fc_beat       = nn.Linear(hidden_size * 2, 1)   # Quando bater?
        self.fc_complexity = nn.Linear(hidden_size * 2, 3)   # Qual padrão?
        self.fc_vertical   = nn.Linear(hidden_size * 2, 3)   # Onde focar?

    def forward(self, x, stars):
        # x: (batch, seq_len, features=93)
        # stars: (batch, 1)

        # 1. CNNs sobre o áudio
        x = x.permute(0, 2, 1)                          # (batch, features, seq_len)
        x = self.dropout(self.relu(self.bn1(self.conv1(x))))
        x = self.relu(self.bn2(self.conv2(x)))
        x = x.permute(0, 2, 1)                          # (batch, seq_len, 128)

        # 2. Embedding de estrelas
        stars_embedded = self.relu(self.star_embed(stars))  # (batch, star_embedding_size)

        # 3. Expande para toda a sequência
        seq_len = x.size(1)
        stars_expanded = stars_embedded.unsqueeze(1).expand(-1, seq_len, -1)

        # 4. Concatena
        combined_features = torch.cat((x, stars_expanded), dim=2)

        # 5. GRU
        feat, _ = self.gru(combined_features)

        # 6. Heads de decisão
        beat_logits       = self.fc_beat(feat)
        complexity_logits = self.fc_complexity(feat)
        vertical_logits   = self.fc_vertical(feat)

        return beat_logits, complexity_logits, vertical_logits


def get_model():
    return DirectorNet()