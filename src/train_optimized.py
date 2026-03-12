import os
import math
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset, WeightedRandomSampler
import numpy as np
from collections import defaultdict
from models_optimized import get_model

# ─────────────────────────────────────────────────────────────────
# Parâmetros de Treinamento
# ─────────────────────────────────────────────────────────────────
BATCH_SIZE    = 256
SEQ_LEN       = 512   # +50 frames de contexto vs versão anterior
EPOCHS        = 60    # +20 épocas: loss ainda estava caindo no epoch 40
LEARNING_RATE = 0.0007
NUM_WORKERS   = os.cpu_count() or 4

# Balanceamento: faixas de estrelas e peso relativo de cada faixa.
# Faixas altas recebem peso maior para compensar sub-representação no dataset.
STAR_BINS = [
    (0.0,  4.0, 1.0),   # Fácil       — peso normal
    (4.0,  5.5, 1.2),   # Moderado    — leve boost
    (5.5,  7.0, 2.0),   # Difícil     — boost médio
    (7.0,  9.0, 3.0),   # Muito difícil — boost alto
    (9.0, 99.0, 4.0),   # Extremo     — boost máximo
]

# Peso da loss de diversidade de posição.
# 0.15 era pequeno demais — o modelo ignorava essa penalidade.
# 0.50 coloca Div na mesma ordem de grandeza que Comp e Vert (~0.5 cada),
# forçando o modelo a realmente aprender a variar padrões.
DIVERSITY_LOSS_WEIGHT = 0.50
# Janela de frames para verificar repetição de posição (complexity/vertical)
DIVERSITY_WINDOW = 8


# ─────────────────────────────────────────────────────────────────
# Dataset com leitura de estrelas antecipada para balanceamento
# ─────────────────────────────────────────────────────────────────

class DirectorDataset(Dataset):
    def __init__(self, processed_dir, seq_len):
        self.processed_dir = processed_dir
        self.seq_len = seq_len

        all_bases = list(set([
            f.replace('_x.npy', '')
            for f in os.listdir(processed_dir)
            if f.endswith('_x.npy')
        ]))

        # Carrega as estrelas de cada arquivo para construir os pesos de amostragem
        self.files = []
        self.star_values = []

        print("  Escaneando estrelas do dataset para balanceamento...")
        for base in all_bases:
            stars_path = os.path.join(processed_dir, f"{base}_stars.npy")
            if not os.path.exists(stars_path):
                continue
            try:
                stars_val = float(np.load(stars_path).item())
                self.files.append(base)
                self.star_values.append(stars_val)
            except Exception:
                continue

        self.star_values = np.array(self.star_values, dtype=np.float32)
        self._print_star_distribution()

    def _print_star_distribution(self):
        print("  Distribuição do dataset por faixa de estrelas:")
        for lo, hi, _ in STAR_BINS:
            count = int(np.sum((self.star_values >= lo) & (self.star_values < hi)))
            label = f"  {lo:.1f}★ – {hi:.1f}★" if hi < 90 else f"  {lo:.1f}★+"
            print(f"    {label}: {count} dificuldades")

    def get_sample_weights(self):
        """
        Retorna um peso por amostra para uso no WeightedRandomSampler.
        Amostras de faixas de estrelas altas recebem peso maior.
        """
        weights = np.ones(len(self.files), dtype=np.float32)
        for lo, hi, w in STAR_BINS:
            mask = (self.star_values >= lo) & (self.star_values < hi)
            weights[mask] = w
        return weights

    def __len__(self):
        return len(self.files) * 12

    def __getitem__(self, idx):
        file_base = self.files[idx % len(self.files)]

        path_x    = os.path.join(self.processed_dir, f"{file_base}_x.npy")
        path_y    = os.path.join(self.processed_dir, f"{file_base}_y.npy")
        path_meta = os.path.join(self.processed_dir, f"{file_base}_meta.npy")
        path_stars= os.path.join(self.processed_dir, f"{file_base}_stars.npy")

        features = np.load(path_x,    mmap_mode='r')
        targets  = np.load(path_y,    mmap_mode='r')
        metadata = np.load(path_meta, mmap_mode='r')
        stars    = np.load(path_stars)

        total_frames = features.shape[0]
        start = np.random.randint(0, max(1, total_frames - self.seq_len))
        end   = start + self.seq_len

        feat_crop = np.array(features[start:end])
        targ_crop = np.array(targets[start:end])
        comp_target = np.array(metadata[start:end, 0]).astype(np.int64)
        vert_target = np.array(metadata[start:end, 1]).astype(np.int64)
        beat_target = np.any(targ_crop > 0.1, axis=1).astype(np.float32).reshape(-1, 1)

        pad_len = self.seq_len - feat_crop.shape[0]
        if pad_len > 0:
            feat_crop   = np.pad(feat_crop,   ((0, pad_len), (0, 0)))
            beat_target = np.pad(beat_target, ((0, pad_len), (0, 0)))
            comp_target = np.pad(comp_target, (0, pad_len), constant_values=-100)
            vert_target = np.pad(vert_target, (0, pad_len), constant_values=-100)

        return (
            torch.tensor(feat_crop,   dtype=torch.float32),
            torch.tensor(beat_target, dtype=torch.float32),
            torch.tensor(comp_target, dtype=torch.long),
            torch.tensor(vert_target, dtype=torch.long),
            torch.tensor([stars.item()], dtype=torch.float32),
        )


# ─────────────────────────────────────────────────────────────────
# Loss de Diversidade de Posição
# ─────────────────────────────────────────────────────────────────

def diversity_loss(p_comp, p_vert, window=DIVERSITY_WINDOW):
    """
    Penaliza o modelo quando ele prevê a mesma classe de complexidade/vertical
    em muitos frames consecutivos dentro de uma janela.

    Intuição: mapas bons variam padrões. Se o modelo prevê "complexity=0" em
    todos os 8 frames seguidos, é sintoma de colapso para padrão único.

    Args:
        p_comp: logits de complexidade  (batch, 3, seq_len)  — após permute
        p_vert: logits de vertical      (batch, 3, seq_len)
        window: tamanho da janela de verificação

    Returns:
        Escalar de loss.
    """
    # Converte para probabilidades (softmax)
    prob_comp = torch.softmax(p_comp, dim=1)  # (B, 3, T)
    prob_vert = torch.softmax(p_vert, dim=1)

    batch, classes, seq = prob_comp.shape
    if seq < window:
        return torch.tensor(0.0, device=p_comp.device)

    # Calcula a probabilidade máxima (confiança) em cada frame
    max_comp = prob_comp.max(dim=1).values  # (B, T)
    max_vert = prob_vert.max(dim=1).values

    # Desliza uma janela e mede o quanto o modelo está "travado" na mesma classe
    # Penalidade = variância baixa dentro da janela (= padrão repetido)
    diversity_penalties = []
    for t in range(0, seq - window, window // 2):
        window_comp = max_comp[:, t:t + window]  # (B, window)
        window_vert = max_vert[:, t:t + window]

        # Entropia das predições dentro da janela — alta entropia = boa diversidade
        # Usamos std da classe prevista como proxy simples e eficiente
        std_comp = window_comp.std(dim=1).mean()  # escalar
        std_vert = window_vert.std(dim=1).mean()

        # Queremos std ALTA (diversidade). Penalizamos std BAIXA.
        penalty = torch.clamp(0.5 - std_comp, min=0) + torch.clamp(0.5 - std_vert, min=0)
        diversity_penalties.append(penalty)

    if not diversity_penalties:
        return torch.tensor(0.0, device=p_comp.device)

    return torch.stack(diversity_penalties).mean()


# ─────────────────────────────────────────────────────────────────
# Warmup + ReduceLROnPlateau combinados
# ─────────────────────────────────────────────────────────────────

class WarmupScheduler:
    """
    Aplica warmup linear por `warmup_epochs` épocas, depois delega ao scheduler base.
    Evita gradientes explosivos no início do treino com LR alto.
    """
    def __init__(self, optimizer, warmup_epochs, base_lr, scheduler):
        self.optimizer      = optimizer
        self.warmup_epochs  = warmup_epochs
        self.base_lr        = base_lr
        self.scheduler      = scheduler
        self.current_epoch  = 0

    def step(self, val_loss=None):
        self.current_epoch += 1
        if self.current_epoch <= self.warmup_epochs:
            lr = self.base_lr * (self.current_epoch / self.warmup_epochs)
            for pg in self.optimizer.param_groups:
                pg['lr'] = lr
        elif val_loss is not None:
            self.scheduler.step(val_loss)

    def get_lr(self):
        return self.optimizer.param_groups[0]['lr']


# ─────────────────────────────────────────────────────────────────
# Treino principal
# ─────────────────────────────────────────────────────────────────

def train():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Usando dispositivo: {device}\n")

    # ── Dataset ──────────────────────────────────────────────────
    dataset = DirectorDataset("data/processed", seq_len=SEQ_LEN)

    # WeightedRandomSampler: garante que faixas altas de estrelas apareçam
    # proporcionalmente durante o treino, mesmo sendo minoria no dataset.
    sample_weights = dataset.get_sample_weights()
    # Repete os pesos para len(dataset) (que é files * 12)
    all_weights = np.tile(sample_weights, 12)
    sampler = WeightedRandomSampler(
        weights=torch.tensor(all_weights, dtype=torch.float32),
        num_samples=len(dataset),
        replacement=True,
    )

    dataloader = DataLoader(
        dataset,
        batch_size=BATCH_SIZE,
        sampler=sampler,          # substitui shuffle=True
        num_workers=NUM_WORKERS,
        persistent_workers=True,
        pin_memory=(device.type == "cuda"),
    )

    print(f"Total de amostras por época: {len(dataset)} | Batches: {len(dataloader)}\n")

    # ── Modelo ───────────────────────────────────────────────────
    model = get_model().to(device)
    total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Parâmetros treináveis: {total_params:,}\n")

    # Retoma do melhor checkpoint se existir (continua de onde parou)
    checkpoint_path = "models/director_net_v2_stars_best.pth"
    if os.path.exists(checkpoint_path):
        model.load_state_dict(torch.load(checkpoint_path, map_location=device))
        print(f"✅ Retomando do checkpoint: {checkpoint_path}\n")

    # ── Otimizador & Scheduler ───────────────────────────────────
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE, weight_decay=1e-5)
    plateau_scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', patience=3, factor=0.5
    )
    scheduler = WarmupScheduler(
        optimizer,
        warmup_epochs=3,
        base_lr=LEARNING_RATE,
        scheduler=plateau_scheduler,
    )

    # ── Funções de Perda ─────────────────────────────────────────
    # pos_weight=10: penaliza falsos negativos na detecção de beat
    # (beats são raros no sinal, ~5-15% dos frames têm nota)
    crit_beat  = nn.BCEWithLogitsLoss(pos_weight=torch.tensor([10.0]).to(device))
    crit_class = nn.CrossEntropyLoss(ignore_index=-100)

    print("Iniciando Treino V3 (Balanceado por Estrelas + Diversity Loss)...\n")
    print(f"{'Epoch':>6} | {'Loss':>8} | {'Beat':>8} | {'Comp':>8} | {'Vert':>8} | {'Div':>8} | {'LR':>10}")
    print("-" * 75)

    best_loss = float('inf')
    os.makedirs("models", exist_ok=True)

    for epoch in range(EPOCHS):
        model.train()
        acc_loss = acc_beat = acc_comp = acc_vert = acc_div = 0.0

        for batch in dataloader:
            feats, t_beat, t_comp, t_vert, t_stars = [b.to(device) for b in batch]

            optimizer.zero_grad()

            p_beat, p_comp, p_vert = model(feats, t_stars)

            # Reshape para CrossEntropy: (Batch, Classes, Seq)
            p_comp_ce = p_comp.permute(0, 2, 1)
            p_vert_ce = p_vert.permute(0, 2, 1)

            # Perdas supervisionadas
            loss_b = crit_beat(p_beat, t_beat)
            loss_c = crit_class(p_comp_ce, t_comp)
            loss_v = crit_class(p_vert_ce, t_vert)

            # Loss de diversidade (regularização de padrão)
            loss_d = diversity_loss(p_comp_ce, p_vert_ce)

            # Soma ponderada — beat é o sinal mais importante
            loss = (loss_b * 1.5
                    + loss_c * 0.75
                    + loss_v * 0.75
                    + loss_d * DIVERSITY_LOSS_WEIGHT)

            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

            acc_loss += loss.item()
            acc_beat += loss_b.item()
            acc_comp += loss_c.item()
            acc_vert += loss_v.item()
            acc_div  += loss_d.item()

        n = len(dataloader)
        avg_loss = acc_loss / n
        avg_beat = acc_beat / n
        avg_comp = acc_comp / n
        avg_vert = acc_vert / n
        avg_div  = acc_div  / n

        scheduler.step(avg_loss)
        current_lr = scheduler.get_lr()

        print(f"{epoch+1:>6} | {avg_loss:>8.4f} | {avg_beat:>8.4f} | "
              f"{avg_comp:>8.4f} | {avg_vert:>8.4f} | {avg_div:>8.4f} | {current_lr:>10.6f}")

        # Salva o melhor modelo automaticamente
        if avg_loss < best_loss:
            best_loss = avg_loss
            torch.save(model.state_dict(), "models/director_net_v2_stars_best.pth")

    # Salva o modelo final da última época também
    torch.save(model.state_dict(), "models/director_net_v2_stars.pth")
    print(f"\nTreino concluído! Melhor loss: {best_loss:.4f}")
    print("Modelos salvos:")
    print("  models/director_net_v2_stars.pth       (última época)")
    print("  models/director_net_v2_stars_best.pth  (melhor época)")


if __name__ == "__main__":
    train()