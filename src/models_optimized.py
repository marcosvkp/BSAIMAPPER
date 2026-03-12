import torch
import torch.nn as nn

AUDIO_FEATURES = 8
NOTE_HISTORY   = 8
NOTE_FEATURES  = 4  # hand, col, layer, cut

# Contexto do FlowNet: quantas notas antes e depois da nota atual ele enxerga
FLOW_CONTEXT = 4  # 4 antes + nota atual + 4 depois = janela de 9


class TimingNet(nn.Module):
    """
    Decide quando colocar notas baseado no áudio.
    Conv1d → GRU bidirecional → sigmoid head
    """

    def __init__(self, audio_features=AUDIO_FEATURES,
                 hidden_size=256, star_embed_size=16):
        super().__init__()

        self.star_embed = nn.Linear(1, star_embed_size)
        self.conv1 = nn.Conv1d(audio_features, 64, kernel_size=3, padding=1)
        self.bn1   = nn.BatchNorm1d(64)
        self.conv2 = nn.Conv1d(64, 128, kernel_size=3, padding=1)
        self.bn2   = nn.BatchNorm1d(128)
        self.relu    = nn.ReLU()
        self.dropout = nn.Dropout(0.2)

        gru_in = 128 + star_embed_size
        self.gru = nn.GRU(
            gru_in, hidden_size, num_layers=2,
            batch_first=True, bidirectional=True, dropout=0.2
        )
        self.fc_timing = nn.Linear(hidden_size * 2, 1)

    def forward(self, x, stars):
        x = x.permute(0, 2, 1)
        x = self.dropout(self.relu(self.bn1(self.conv1(x))))
        x = self.relu(self.bn2(self.conv2(x)))
        x = x.permute(0, 2, 1)
        stars_emb = self.relu(self.star_embed(stars))
        stars_exp = stars_emb.unsqueeze(1).expand(-1, x.size(1), -1)
        feat, _ = self.gru(torch.cat([x, stars_exp], dim=2))
        return self.fc_timing(feat)


class NoteNet(nn.Module):
    """
    Decide o quê gerar em cada posição com nota.
    history_proj + audio_proj → GRU 2 layers bidirecional → 8 heads
    """

    def __init__(self, audio_features=AUDIO_FEATURES,
                 note_history=NOTE_HISTORY, note_features=NOTE_FEATURES,
                 hidden_size=256, star_embed_size=16):
        super().__init__()

        self.note_history  = note_history
        self.note_features = note_features

        self.star_embed = nn.Linear(1, star_embed_size)
        self.relu    = nn.ReLU()
        self.dropout = nn.Dropout(0.2)

        history_in = note_history * note_features  # 32
        self.history_proj = nn.Sequential(
            nn.Linear(history_in, 64),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(64, 64),
            nn.ReLU(),
        )
        self.audio_proj = nn.Sequential(
            nn.Linear(audio_features, 32),
            nn.ReLU(),
        )

        gru_in = 64 + 32 + star_embed_size  # 112
        self.gru = nn.GRU(
            gru_in, hidden_size, num_layers=2,
            batch_first=True, bidirectional=True, dropout=0.2
        )

        feat_size = hidden_size * 2  # 512
        self.fc_hand         = nn.Linear(feat_size, 2)
        self.fc_col          = nn.Linear(feat_size, 4)
        self.fc_layer        = nn.Linear(feat_size, 3)
        self.fc_cut          = nn.Linear(feat_size, 9)
        self.fc_double       = nn.Linear(feat_size, 2)
        self.fc_double_col   = nn.Linear(feat_size, 4)
        self.fc_double_layer = nn.Linear(feat_size, 3)
        self.fc_double_cut   = nn.Linear(feat_size, 9)

    def forward(self, audio, history, stars):
        B, T, _ = audio.shape
        stars_emb = self.relu(self.star_embed(stars))
        stars_exp = stars_emb.unsqueeze(1).expand(-1, T, -1)
        hist_feat  = self.history_proj(history)
        audio_feat = self.audio_proj(audio)
        combined = self.dropout(torch.cat([hist_feat, audio_feat, stars_exp], dim=2))
        feat, _ = self.gru(combined)
        return {
            'hand':         self.fc_hand(feat),
            'col':          self.fc_col(feat),
            'layer':        self.fc_layer(feat),
            'cut':          self.fc_cut(feat),
            'double':       self.fc_double(feat),
            'double_col':   self.fc_double_col(feat),
            'double_layer': self.fc_double_layer(feat),
            'double_cut':   self.fc_double_cut(feat),
        }


class FlowNet(nn.Module):
    """
    Refinador de fluxo pós-geração (FASE 3).

    Recebe uma janela de notas ao redor da nota atual e decide se ela
    precisa ser corrigida — e como. Opera nota a nota sobre o mapa gerado.

    Input por nota:
      - Janela de contexto: (FLOW_CONTEXT*2 + 1) notas × 4 valores = 36
        [ nota_-4, nota_-3, ..., nota_atual, ..., nota_+4 ]
        Cada nota: (hand, col, layer, cut) normalizado
      - Áudio local do frame da nota atual: 8 features
      - Estrelas: 1 valor

    Output (para a nota central da janela):
      - hand_ok  : a mão está certa?         (2: manter/corrigir)
      - cut_ok   : a direção está certa?     (2: manter/corrigir)
      - col_ok   : a coluna está certa?      (2: manter/corrigir)
      - new_hand : qual mão sugerida         (2 classes)
      - new_cut  : qual direção sugerida     (9 classes)
      - new_col  : qual coluna sugerida      (4 classes)

    Treinado com perturbações artificiais:
      Dataset real → injeta erros aleatórios → modelo aprende a desfazer

    Arquitetura:
      MLP sobre a janela de contexto (sem GRU — cada nota é independente)
      Isso é intencional: o FlowNet não precisa de memória de longo prazo,
      só precisa entender o contexto local da nota.

    Parâmetros estimados: ~1.2M
    """

    def __init__(self, context=FLOW_CONTEXT, note_features=NOTE_FEATURES,
                 audio_features=AUDIO_FEATURES, hidden_size=256, star_embed_size=16):
        super().__init__()

        self.context       = context
        self.window_size   = context * 2 + 1          # 9 notas
        self.note_features = note_features

        self.star_embed = nn.Linear(1, star_embed_size)
        self.relu    = nn.ReLU()
        self.dropout = nn.Dropout(0.3)

        # Projeta a janela de contexto de notas
        # Input: 9 notas × 4 valores = 36
        context_in = self.window_size * note_features
        self.context_proj = nn.Sequential(
            nn.Linear(context_in, 128),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(128, 128),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(128, 64),
            nn.ReLU(),
        )

        # Projeta áudio local
        self.audio_proj = nn.Sequential(
            nn.Linear(audio_features, 32),
            nn.ReLU(),
        )

        # Fusão: 64 (contexto) + 32 (áudio) + 16 (stars) = 112
        fuse_in = 64 + 32 + star_embed_size
        self.fuse = nn.Sequential(
            nn.Linear(fuse_in, hidden_size),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
        )

        # Heads de decisão
        # "Está ok?" — se o modelo acha que está ok, não muda nada
        self.fc_hand_ok = nn.Linear(hidden_size, 2)   # manter / corrigir mão
        self.fc_cut_ok  = nn.Linear(hidden_size, 2)   # manter / corrigir cut
        self.fc_col_ok  = nn.Linear(hidden_size, 2)   # manter / corrigir coluna

        # "O que colocar?" — só usado se o head de ok prediz "corrigir"
        self.fc_new_hand = nn.Linear(hidden_size, 2)  # nova mão
        self.fc_new_cut  = nn.Linear(hidden_size, 9)  # novo cut
        self.fc_new_col  = nn.Linear(hidden_size, 4)  # nova coluna

    def forward(self, context_notes, audio, stars):
        """
        Args:
            context_notes : (B, window_size * note_features)  — janela flat
            audio         : (B, audio_features)               — áudio local
            stars         : (B, 1)

        Returns:
            dict com logits para cada head, shape (B, classes)
        """
        stars_emb    = self.relu(self.star_embed(stars))      # (B, 16)
        context_feat = self.context_proj(context_notes)       # (B, 64)
        audio_feat   = self.audio_proj(audio)                 # (B, 32)

        fused = self.fuse(
            torch.cat([context_feat, audio_feat, stars_emb], dim=1)  # (B, 112)
        )

        return {
            'hand_ok':  self.fc_hand_ok(fused),   # (B, 2)
            'cut_ok':   self.fc_cut_ok(fused),    # (B, 2)
            'col_ok':   self.fc_col_ok(fused),    # (B, 2)
            'new_hand': self.fc_new_hand(fused),  # (B, 2)
            'new_cut':  self.fc_new_cut(fused),   # (B, 9)
            'new_col':  self.fc_new_col(fused),   # (B, 4)
        }


def get_timing_model():
    model  = TimingNet()
    params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"TimingNet: {params:,} parâmetros")
    return model


def get_note_model():
    model  = NoteNet()
    params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"NoteNet:   {params:,} parâmetros")
    return model


def get_flow_model():
    model  = FlowNet()
    params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"FlowNet:   {params:,} parâmetros")
    return model