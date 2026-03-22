"""
models.py — Arquitetura V6

Quatro modelos com responsabilidades TOTALMENTE separadas:

  TimingNet  — "TEM NOTA AQUI?" (binário por frame)
  PlaceNet   — "ONDE COLOCAR?" (col, layer, mão) — só processa frames COM nota
  AngleNet   — "QUAL ÂNGULO?" (cut direction, por mão, com histórico)
  ViewNet    — "ESTÁ JOGÁVEL?" (avaliador por janelas de 32 notas)

Filosofia:
  TimingNet aprende RITMO. PlaceNet aprende ESPAÇO. AngleNet aprende FÍSICA.
  Misturar as três tarefas num único modelo faz o gradiente mais forte (timing)
  dominar e os outros heads colapsarem para saída constante (col=1, layer=0 todo frame).
"""

import torch
import torch.nn as nn

# ─────────────────────────────────────────────────────────────────
# Constantes
# ─────────────────────────────────────────────────────────────────

CTX_FEATS  = 8    # features de contexto temporal (onset, rms, beat_phase, ...)
MEL_BINS   = 64   # mel spectrogram bins

PLACE_WIN  = 7    # janela de ±3 frames ao redor da nota para o PlaceNet

ANGLE_HIST = 12   # últimos N cuts da mesma mão que o AngleNet vê
NUM_CUTS   = 9    # 0=UP 1=DOWN 2=LEFT 3=RIGHT 4=UL 5=UR 6=DL 7=DR 8=DOT
NUM_COLS   = 4
NUM_LAYERS = 3

VIEW_WIN   = 32   # notas por janela de avaliação do ViewNet
VIEW_FEATS = 5    # (hand, col/3, layer/2, cut/8, beat_gap)


# ─────────────────────────────────────────────────────────────────
# TimingNet
# ─────────────────────────────────────────────────────────────────

class TimingNet(nn.Module):
    """
    Detecta QUANDO colocar notas a partir das 8 features de contexto de áudio.

    Usa apenas ctx_feats (onset_strength, rms, beat_phase, etc.) — não usa mel_spec.
    As 8 features já capturam toda a informação de onset necessária para timing,
    e são 8x mais rápidas de processar que o mel de 64 bins.

    Input:
      - ctx_feats : (B, T, 8)   features de contexto temporal
      - stars     : (B, 1)      dificuldade alvo

    Output:
      - (B, T, 1)  logits — prob de nota por frame (sigmoid → 0/1)

    Arquitetura: Conv1d × 2 → GRU bidirecional → head
    ~1.8M parâmetros
    """

    def __init__(self, ctx=CTX_FEATS, hidden=256, star_emb=16):
        super().__init__()
        self.star_embed = nn.Linear(1, star_emb)
        self.conv1 = nn.Conv1d(ctx, 64,  kernel_size=3, padding=1)
        self.bn1   = nn.BatchNorm1d(64)
        self.conv2 = nn.Conv1d(64,  128, kernel_size=3, padding=1)
        self.bn2   = nn.BatchNorm1d(128)
        self.relu    = nn.ReLU()
        self.dropout = nn.Dropout(0.2)
        self.gru = nn.GRU(
            128 + star_emb, hidden,
            num_layers=2, batch_first=True,
            bidirectional=True, dropout=0.2,
        )
        self.fc = nn.Linear(hidden * 2, 1)

    def forward(self, ctx_feats, stars):
        """
        Args:
            ctx_feats : (B, T, 8)
            stars     : (B, 1)
        Returns:
            (B, T, 1) logits
        """
        x = ctx_feats.permute(0, 2, 1)
        x = self.dropout(self.relu(self.bn1(self.conv1(x))))
        x = self.relu(self.bn2(self.conv2(x)))
        x = x.permute(0, 2, 1)
        s = self.relu(self.star_embed(stars)).unsqueeze(1).expand(-1, x.size(1), -1)
        feat, _ = self.gru(torch.cat([x, s], dim=2))
        return self.fc(feat)


# ─────────────────────────────────────────────────────────────────
# PlaceNet
# ─────────────────────────────────────────────────────────────────

class PlaceNet(nn.Module):
    """
    Decide ONDE colocar cada nota: mão, coluna, camada, double.

    SEPARAÇÃO CRÍTICA: PlaceNet só vê frames que o TimingNet já selecionou.
    Nunca processa frames sem nota.

    Por que isso resolve o colapso de posição:
      O modelo antigo (PatternModel) recebia T=512 frames onde ~98% não tinham nota.
      O head `has_note` dominava o gradiente com pos_weight=10, e os outros heads
      (col, layer, hand) aprendiam a solução trivial: sempre prever a posição mais
      frequente no dataset (ex: col=1, layer=0). O modelo "aprendia" sem errar no
      has_note e sem nunca precisar acertar col/layer.

      PlaceNet recebe apenas N notas por sample (N << T). Cada forward é uma nota real.
      O modelo foca 100% em "dado este contexto de áudio, qual posição faz sentido?"

    Input por nota:
      - mel_win  : (B, PLACE_WIN, 64)  janela de mel ±3 frames ao redor da nota
      - ctx_win  : (B, PLACE_WIN, 8)   janela de ctx ±3 frames
      - stars    : (B, 1)

    Output por nota:
      - hand     : (B, 2)
      - col      : (B, 4)
      - layer    : (B, 3)
      - is_double: (B, 2)

    Arquitetura: GRU sobre janela de 7 frames → pooling central → 4 heads
    ~500k parâmetros
    """

    def __init__(self, mel=MEL_BINS, ctx=CTX_FEATS,
                 win=PLACE_WIN, hidden=128, star_emb=16):
        super().__init__()
        self.win = win
        self.star_embed = nn.Sequential(nn.Linear(1, star_emb), nn.ReLU())
        self.gru = nn.GRU(
            mel + ctx + star_emb, hidden,
            num_layers=2, batch_first=True,
            bidirectional=True, dropout=0.15,
        )
        feat = hidden * 2
        self.fc_hand   = nn.Linear(feat, 2)
        self.fc_col    = nn.Linear(feat, 4)
        self.fc_layer  = nn.Linear(feat, 3)
        self.fc_double = nn.Linear(feat, 2)

    def forward(self, mel_win, ctx_win, stars):
        """
        Args:
            mel_win : (B, PLACE_WIN, 64)
            ctx_win : (B, PLACE_WIN, 8)
            stars   : (B, 1)
        Returns:
            dict com logits (B, classes)
        """
        W = mel_win.size(1)
        s = self.star_embed(stars).unsqueeze(1).expand(-1, W, -1)
        x = torch.cat([mel_win, ctx_win, s], dim=2)
        feat, _ = self.gru(x)
        center = feat[:, W // 2, :]   # frame central = a nota em si
        return {
            'hand':      self.fc_hand(center),
            'col':       self.fc_col(center),
            'layer':     self.fc_layer(center),
            'is_double': self.fc_double(center),
        }


# ─────────────────────────────────────────────────────────────────
# AngleNet
# ─────────────────────────────────────────────────────────────────

class AngleNet(nn.Module):
    """
    Decide o ÂNGULO DE CORTE de cada nota, operando POR MÃO.

    Roda duas vezes por geração:
      1. Todas as notas da mão esquerda (hand=0)
      2. Todas as notas da mão direita (hand=1)

    Por que por mão?
      O fluxo UP→DOWN→UP é da MÃO ESQUERDA. A direita pode fazer DOWN→UP→DOWN
      simultaneamente. Misturar as duas no histórico destrói o aprendizado de paridade:
      o modelo veria L:UP → R:DOWN → L:UP e concluiria que "DOWN sempre precede UP",
      o que é incorreto — o DOWN era da outra mão.

    Input por nota:
      - cut_hist   : (B, 12) long — últimos 12 cuts desta mão (pad=8=DOT)
      - col_hist   : (B, 12) long — últimas 12 colunas (pad=4)
      - layer_hist : (B, 12) long — últimas 12 camadas (pad=3)
      - pos_now    : (B, 2)  float — [col/3, layer/2]
      - beat_gap   : (B, 1)  float — beats desde última nota desta mão (/8)
      - stars      : (B, 1)  float

    Output:
      - (B, 9) logits para direção de corte

    Arquitetura: Embeddings separados → GRU bidirecional → fusão com pos+gap → head
    ~800k parâmetros
    """

    def __init__(self, hist=ANGLE_HIST, star_emb=8, hidden=128):
        super().__init__()
        self.hist = hist
        self.cut_emb   = nn.Embedding(NUM_CUTS + 1,    16, padding_idx=NUM_CUTS)
        self.col_emb   = nn.Embedding(NUM_COLS + 1,     8, padding_idx=NUM_COLS)
        self.layer_emb = nn.Embedding(NUM_LAYERS + 1,   8, padding_idx=NUM_LAYERS)
        self.star_embed = nn.Sequential(nn.Linear(1, star_emb), nn.ReLU())
        self.gru = nn.GRU(
            32 + star_emb, hidden,
            num_layers=2, batch_first=True,
            bidirectional=True, dropout=0.15,
        )
        fuse = hidden * 2 + 3  # +2 pos_now +1 beat_gap
        self.head = nn.Sequential(
            nn.LayerNorm(fuse),
            nn.Linear(fuse, hidden),
            nn.GELU(),
            nn.Dropout(0.2),
            nn.Linear(hidden, NUM_CUTS),
        )

    def forward(self, cut_hist, col_hist, layer_hist, pos_now, beat_gap, stars):
        hist = torch.cat([
            self.cut_emb(cut_hist),
            self.col_emb(col_hist),
            self.layer_emb(layer_hist),
        ], dim=2)
        s = self.star_embed(stars).unsqueeze(1).expand(-1, self.hist, -1)
        _, h = self.gru(torch.cat([hist, s], dim=2))
        h_cat = torch.cat([h[-2], h[-1]], dim=1)
        ctx = torch.cat([h_cat, pos_now, beat_gap], dim=1)
        return self.head(ctx)


# ─────────────────────────────────────────────────────────────────
# ViewNet
# ─────────────────────────────────────────────────────────────────

class ViewNet(nn.Module):
    """
    Avalia a JOGABILIDADE de janelas de 32 notas.

    Simula o comportamento humano ao jogar: verifica SPS, resets forçados,
    crossovers, e identifica notas específicas que causam problemas.

    Input:
      - notes_win    : (B, 32, 5)  [hand, col/3, layer/2, cut/8, beat_gap]
      - stars        : (B, 1)

    Output:
      - quality      : (B, 1)   logit — setor jogável?
      - sps_pred     : (B, 1)   SPS previsto (normalizado por 10)
      - problem_mask : (B, 32)  logits — qual nota é problemática?

    Arquitetura: Transformer Encoder 2L/4H + pooling → heads
    ~600k parâmetros
    """

    def __init__(self, win=VIEW_WIN, feats=VIEW_FEATS,
                 d_model=64, nhead=4, n_layers=2, star_emb=8):
        super().__init__()
        self.win = win
        self.star_embed = nn.Sequential(nn.Linear(1, star_emb), nn.ReLU())
        self.note_proj  = nn.Sequential(
            nn.Linear(feats + star_emb, d_model),
            nn.LayerNorm(d_model),
            nn.ReLU(),
        )
        self.pos_enc = nn.Embedding(win, d_model)
        enc = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=nhead,
            dim_feedforward=d_model * 4,
            dropout=0.1, batch_first=True,
        )
        self.transformer = nn.TransformerEncoder(enc, num_layers=n_layers)
        self.fc_quality = nn.Linear(d_model, 1)
        self.fc_sps     = nn.Linear(d_model, 1)
        self.fc_mask    = nn.Linear(d_model, 1)

    def forward(self, notes_win, stars):
        B, W, _ = notes_win.shape
        s = self.star_embed(stars).unsqueeze(1).expand(-1, W, -1)
        x = self.note_proj(torch.cat([notes_win, s], dim=2))
        pos = torch.arange(W, device=x.device).unsqueeze(0).expand(B, -1)
        x = x + self.pos_enc(pos)
        x = self.transformer(x)
        pooled = x.mean(dim=1)
        return {
            'quality':      self.fc_quality(pooled),
            'sps_pred':     self.fc_sps(pooled),
            'problem_mask': self.fc_mask(x).squeeze(-1),
        }


# ─────────────────────────────────────────────────────────────────
# Fábrica
# ─────────────────────────────────────────────────────────────────

def _count(m):
    return sum(x.numel() for x in m.parameters() if x.requires_grad)

def get_timing_model():
    m = TimingNet()
    print(f"  TimingNet  : {_count(m):>10,} parâmetros")
    return m

def get_place_model():
    m = PlaceNet()
    print(f"  PlaceNet   : {_count(m):>10,} parâmetros")
    return m

def get_angle_model():
    m = AngleNet()
    print(f"  AngleNet   : {_count(m):>10,} parâmetros")
    return m

def get_view_model():
    m = ViewNet()
    print(f"  ViewNet    : {_count(m):>10,} parâmetros")
    return m
