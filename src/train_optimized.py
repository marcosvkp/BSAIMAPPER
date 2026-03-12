import os
import random
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset, WeightedRandomSampler
import numpy as np
from collections import defaultdict
from models_optimized import (
    get_timing_model, get_note_model, get_flow_model,
    NOTE_HISTORY, NOTE_FEATURES, FLOW_CONTEXT
)

# ─────────────────────────────────────────────────────────────────
# Parâmetros
# ─────────────────────────────────────────────────────────────────
PROCESSED_DIR  = "data/processed"
MODELS_DIR     = "models"
BATCH_SIZE     = 256
SEQ_LEN        = 512
EPOCHS_TIMING  = 60
EPOCHS_NOTE    = 80
EPOCHS_FLOW    = 50
LEARNING_RATE  = 0.0007
NUM_WORKERS    = 6

STAR_BINS = [
    (0.0,  4.0, 1.0),
    (4.0,  5.5, 1.2),
    (5.5,  7.0, 2.0),
    (7.0,  9.0, 3.0),
    (9.0, 99.0, 4.0),
]

# ─────────────────────────────────────────────────────────────────
# Dataset TimingNet  (inalterado)
# ─────────────────────────────────────────────────────────────────

class TimingDataset(Dataset):
    def __init__(self, processed_dir, seq_len):
        self.processed_dir = processed_dir
        self.seq_len       = seq_len
        bases = list(set([
            f.replace('_x.npy', '')
            for f in os.listdir(processed_dir)
            if f.endswith('_x.npy')
        ]))
        self.files, self.stars = [], []
        for base in bases:
            sp = os.path.join(processed_dir, f"{base}_stars.npy")
            tp = os.path.join(processed_dir, f"{base}_timing.npy")
            if not os.path.exists(sp) or not os.path.exists(tp):
                continue
            try:
                self.stars.append(float(np.load(sp).item()))
                self.files.append(base)
            except Exception:
                continue
        self.stars = np.array(self.stars, dtype=np.float32)
        print(f"  TimingDataset: {len(self.files)} dificuldades")

    def get_sample_weights(self):
        weights = np.ones(len(self.files), dtype=np.float32)
        for lo, hi, w in STAR_BINS:
            mask = (self.stars >= lo) & (self.stars < hi)
            weights[mask] = w
        return weights

    def __len__(self):
        return len(self.files) * 12

    def __getitem__(self, idx):
        base   = self.files[idx % len(self.files)]
        feats  = np.load(os.path.join(self.processed_dir, f"{base}_x.npy"),      mmap_mode='r')
        timing = np.load(os.path.join(self.processed_dir, f"{base}_timing.npy"), mmap_mode='r')
        stars  = np.load(os.path.join(self.processed_dir, f"{base}_stars.npy"))
        total  = feats.shape[0]
        start  = np.random.randint(0, max(1, total - self.seq_len))
        end    = start + self.seq_len
        feat_c   = np.array(feats[start:end])
        timing_c = np.array(timing[start:end]).reshape(-1, 1)
        pad = self.seq_len - feat_c.shape[0]
        if pad > 0:
            feat_c   = np.pad(feat_c,   ((0, pad), (0, 0)))
            timing_c = np.pad(timing_c, ((0, pad), (0, 0)))
        return (
            torch.tensor(feat_c,   dtype=torch.float32),
            torch.tensor(timing_c, dtype=torch.float32),
            torch.tensor([stars.item()], dtype=torch.float32),
        )


# ─────────────────────────────────────────────────────────────────
# Dataset NoteNet  (inalterado)
# ─────────────────────────────────────────────────────────────────

PADDING_NOTE = np.array([0, 1, 0, 8], dtype=np.float32)

class NoteDataset(Dataset):
    def __init__(self, processed_dir, seq_len, hop_length=512, sr=22050):
        self.processed_dir = processed_dir
        self.seq_len       = seq_len
        self.frame_dur     = hop_length / sr
        bases = list(set([
            f.replace('_x.npy', '')
            for f in os.listdir(processed_dir)
            if f.endswith('_x.npy')
        ]))
        self.files, self.stars = [], []
        for base in bases:
            sp  = os.path.join(processed_dir, f"{base}_stars.npy")
            np_ = os.path.join(processed_dir, f"{base}_notes.npy")
            xp  = os.path.join(processed_dir, f"{base}_x.npy")
            if not all(os.path.exists(p) for p in [sp, np_, xp]):
                continue
            try:
                notes = np.load(np_)
                if len(notes) < NOTE_HISTORY + 1:
                    continue
                self.stars.append(float(np.load(sp).item()))
                self.files.append(base)
            except Exception:
                continue
        self.stars = np.array(self.stars, dtype=np.float32)
        print(f"  NoteDataset:   {len(self.files)} dificuldades")

    def get_sample_weights(self):
        weights = np.ones(len(self.files), dtype=np.float32)
        for lo, hi, w in STAR_BINS:
            mask = (self.stars >= lo) & (self.stars < hi)
            weights[mask] = w
        return weights

    def __len__(self):
        return len(self.files) * 16

    def __getitem__(self, idx):
        base      = self.files[idx % len(self.files)]
        audio_all = np.load(os.path.join(self.processed_dir, f"{base}_x.npy"),     mmap_mode='r')
        notes_all = np.load(os.path.join(self.processed_dir, f"{base}_notes.npy"), mmap_mode='r')
        stars_val = float(np.load(os.path.join(self.processed_dir, f"{base}_stars.npy")).item())
        N = len(notes_all)
        start  = np.random.randint(0, max(1, N - self.seq_len))
        end    = min(start + self.seq_len, N)
        window = notes_all[start:end]
        W = len(window)
        audio_seq   = np.zeros((W, audio_all.shape[1]), dtype=np.float32)
        history_seq = np.zeros((W, NOTE_HISTORY * 4),   dtype=np.float32)
        t_hand = np.zeros(W, dtype=np.int64)
        t_col  = np.zeros(W, dtype=np.int64)
        t_layer= np.zeros(W, dtype=np.int64)
        t_cut  = np.zeros(W, dtype=np.int64)
        t_double=np.zeros(W, dtype=np.int64)
        num_frames = audio_all.shape[0]
        for i in range(W):
            note = window[i]
            beat_norm = float(note[5])
            frame_idx = min(int(beat_norm * num_frames), num_frames - 1)
            audio_seq[i] = audio_all[frame_idx]
            for h in range(NOTE_HISTORY):
                src_idx = (start + i - NOTE_HISTORY + h)
                if src_idx < 0:
                    history_seq[i, h*4:(h+1)*4] = PADDING_NOTE
                else:
                    prev = notes_all[src_idx]
                    history_seq[i, h*4:(h+1)*4] = prev[1:5]
            t_hand[i]  = int(note[1])
            t_col[i]   = int(note[2])
            t_layer[i] = int(note[3])
            t_cut[i]   = int(note[4])
            if i + 1 < W:
                t_double[i] = 1 if abs(float(window[i+1][5]) - float(note[5])) < 0.02 else 0
        pad = self.seq_len - W
        if pad > 0:
            audio_seq    = np.pad(audio_seq,    ((0, pad), (0, 0)))
            history_seq  = np.pad(history_seq,  ((0, pad), (0, 0)))
            t_hand       = np.pad(t_hand,   (0, pad), constant_values=-100)
            t_col        = np.pad(t_col,    (0, pad), constant_values=-100)
            t_layer      = np.pad(t_layer,  (0, pad), constant_values=-100)
            t_cut        = np.pad(t_cut,    (0, pad), constant_values=-100)
            t_double     = np.pad(t_double, (0, pad), constant_values=-100)
        return (
            torch.tensor(audio_seq,   dtype=torch.float32),
            torch.tensor(history_seq, dtype=torch.float32),
            torch.tensor([stars_val], dtype=torch.float32),
            torch.tensor(t_hand,   dtype=torch.long),
            torch.tensor(t_col,    dtype=torch.long),
            torch.tensor(t_layer,  dtype=torch.long),
            torch.tensor(t_cut,    dtype=torch.long),
            torch.tensor(t_double, dtype=torch.long),
        )


# ─────────────────────────────────────────────────────────────────
# Dataset FlowNet
#
# Gera exemplos de (nota perturbada → nota correta) a partir das
# notas reais do dataset, sem precisar re-processar nada.
#
# Estratégia de perturbação:
#   Para cada nota N:
#     - Com prob PERTURB_HAND: troca a mão (target: mão original)
#     - Com prob PERTURB_CUT:  troca o cut por um aleatório errado
#     - Com prob PERTURB_COL:  troca a coluna por uma errada
#   O modelo aprende a identificar e corrigir cada tipo de erro.
#
# Input:
#   - Janela de FLOW_CONTEXT*2+1 notas, onde a nota central pode
#     ter sido perturbada
#   - Áudio local do frame da nota central
#   - Estrelas
#
# Target:
#   - hand_ok  : 1 se mão foi perturbada (deve corrigir), 0 se ok
#   - cut_ok   : 1 se cut foi perturbado, 0 se ok
#   - col_ok   : 1 se col foi perturbada, 0 se ok
#   - new_hand : mão original (target quando hand_ok=1)
#   - new_cut  : cut original  (target quando cut_ok=1)
#   - new_col  : col original  (target quando col_ok=1)
# ─────────────────────────────────────────────────────────────────

PERTURB_HAND = 0.35
PERTURB_CUT  = 0.45
PERTURB_COL  = 0.25

# Nota de padding para bordas da janela
FLOW_PADDING = np.array([0, 1, 0, 8], dtype=np.float32)  # hand=L, col=1, layer=0, cut=DOT


class FlowDataset(Dataset):
    def __init__(self, processed_dir, hop_length=512, sr=22050):
        self.processed_dir = processed_dir
        self.frame_dur     = hop_length / sr
        self.window_size   = FLOW_CONTEXT * 2 + 1

        bases = list(set([
            f.replace('_x.npy', '')
            for f in os.listdir(processed_dir)
            if f.endswith('_x.npy')
        ]))

        self.files, self.stars = [], []
        for base in bases:
            sp  = os.path.join(processed_dir, f"{base}_stars.npy")
            np_ = os.path.join(processed_dir, f"{base}_notes.npy")
            xp  = os.path.join(processed_dir, f"{base}_x.npy")
            if not all(os.path.exists(p) for p in [sp, np_, xp]):
                continue
            try:
                notes = np.load(np_)
                # Precisa de ao menos uma janela completa
                if len(notes) < self.window_size:
                    continue
                self.stars.append(float(np.load(sp).item()))
                self.files.append(base)
            except Exception:
                continue

        self.stars = np.array(self.stars, dtype=np.float32)
        print(f"  FlowDataset:   {len(self.files)} dificuldades")

    def get_sample_weights(self):
        weights = np.ones(len(self.files), dtype=np.float32)
        for lo, hi, w in STAR_BINS:
            mask = (self.stars >= lo) & (self.stars < hi)
            weights[mask] = w
        return weights

    def __len__(self):
        return len(self.files) * 32  # mais amostras por arquivo = dataset maior

    def __getitem__(self, idx):
        base      = self.files[idx % len(self.files)]
        audio_all = np.load(os.path.join(self.processed_dir, f"{base}_x.npy"),     mmap_mode='r')
        notes_all = np.load(os.path.join(self.processed_dir, f"{base}_notes.npy"), mmap_mode='r')
        stars_val = float(np.load(os.path.join(self.processed_dir, f"{base}_stars.npy")).item())

        N = len(notes_all)
        # Seleciona uma nota central aleatória (garantindo que há contexto ao redor)
        center = np.random.randint(FLOW_CONTEXT, N - FLOW_CONTEXT)
        center_note = notes_all[center]  # (6,): has_note, hand, col, layer, cut, beat_norm

        # ── Monta a janela de contexto ────────────────────────────
        # 9 notas × 4 valores = 36 features
        window_flat = np.zeros(self.window_size * NOTE_FEATURES, dtype=np.float32)
        for w in range(self.window_size):
            note_idx = center - FLOW_CONTEXT + w
            if 0 <= note_idx < N:
                note = notes_all[note_idx]
                window_flat[w*NOTE_FEATURES:(w+1)*NOTE_FEATURES] = note[1:5]  # hand,col,layer,cut
            else:
                window_flat[w*NOTE_FEATURES:(w+1)*NOTE_FEATURES] = FLOW_PADDING

        # ── Áudio local da nota central ───────────────────────────
        num_frames = audio_all.shape[0]
        beat_norm  = float(center_note[5])
        frame_idx  = min(int(beat_norm * num_frames), num_frames - 1)
        audio_local = np.array(audio_all[frame_idx], dtype=np.float32)

        # ── Valores originais (targets) ───────────────────────────
        orig_hand  = int(center_note[1])
        orig_col   = int(center_note[2])
        orig_layer = int(center_note[3])
        orig_cut   = int(center_note[4])

        # ── Injeta perturbações na nota central da janela ─────────
        center_offset = FLOW_CONTEXT * NOTE_FEATURES  # offset da nota central no vetor flat

        t_hand_ok = 0  # 0 = está ok (não precisa corrigir)
        t_cut_ok  = 0
        t_col_ok  = 0

        # Perturba mão
        if random.random() < PERTURB_HAND:
            wrong_hand = 1 - orig_hand  # inverte
            window_flat[center_offset + 0] = float(wrong_hand)
            t_hand_ok = 1  # precisa corrigir

        # Perturba cut — escolhe uma direção diferente da original
        if random.random() < PERTURB_CUT:
            wrong_cut = random.choice([c for c in range(9) if c != orig_cut])
            window_flat[center_offset + 3] = float(wrong_cut)
            t_cut_ok = 1

        # Perturba coluna — escolhe uma coluna diferente da original
        if random.random() < PERTURB_COL:
            wrong_col = random.choice([c for c in range(4) if c != orig_col])
            window_flat[center_offset + 1] = float(wrong_col)
            t_col_ok = 1

        return (
            torch.tensor(window_flat,  dtype=torch.float32),   # (36,)
            torch.tensor(audio_local,  dtype=torch.float32),   # (8,)
            torch.tensor([stars_val],  dtype=torch.float32),   # (1,)
            torch.tensor(t_hand_ok,    dtype=torch.long),      # escalar
            torch.tensor(t_cut_ok,     dtype=torch.long),
            torch.tensor(t_col_ok,     dtype=torch.long),
            torch.tensor(orig_hand,    dtype=torch.long),      # target de correção
            torch.tensor(orig_cut,     dtype=torch.long),
            torch.tensor(orig_col,     dtype=torch.long),
        )


# ─────────────────────────────────────────────────────────────────
# Scheduler com Warmup  (inalterado)
# ─────────────────────────────────────────────────────────────────

class WarmupScheduler:
    def __init__(self, optimizer, warmup_epochs, base_lr, plateau_scheduler):
        self.optimizer         = optimizer
        self.warmup_epochs     = warmup_epochs
        self.base_lr           = base_lr
        self.plateau_scheduler = plateau_scheduler
        self.current_epoch     = 0

    def step(self, val_loss=None):
        self.current_epoch += 1
        if self.current_epoch <= self.warmup_epochs:
            lr = self.base_lr * (self.current_epoch / self.warmup_epochs)
            for pg in self.optimizer.param_groups:
                pg['lr'] = lr
        elif val_loss is not None:
            self.plateau_scheduler.step(val_loss)

    def get_lr(self):
        return self.optimizer.param_groups[0]['lr']


def _make_loader(dataset):
    weights = dataset.get_sample_weights()
    all_w   = np.resize(np.tile(weights, len(dataset) // len(dataset.files)), len(dataset))
    sampler = WeightedRandomSampler(
        torch.tensor(all_w, dtype=torch.float32),
        num_samples=len(dataset),
        replacement=True,
    )
    return DataLoader(
        dataset, batch_size=BATCH_SIZE, sampler=sampler,
        num_workers=NUM_WORKERS, persistent_workers=True, pin_memory=True,
    )


# ─────────────────────────────────────────────────────────────────
# Treino TimingNet  (inalterado)
# ─────────────────────────────────────────────────────────────────

def train_timing(device):
    print("\n" + "="*60)
    print("  FASE 1 — Treinando TimingNet")
    print("="*60)
    dataset    = TimingDataset(PROCESSED_DIR, SEQ_LEN)
    dataloader = _make_loader(dataset)
    model = get_timing_model().to(device)
    os.makedirs(MODELS_DIR, exist_ok=True)
    ckpt = os.path.join(MODELS_DIR, "timing_net_best.pth")
    if os.path.exists(ckpt):
        model.load_state_dict(torch.load(ckpt, map_location=device))
        print(f"  ✅ Retomando de {ckpt}\n")
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE, weight_decay=1e-5)
    plateau   = optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=3, factor=0.5)
    scheduler = WarmupScheduler(optimizer, warmup_epochs=3, base_lr=LEARNING_RATE,
                                plateau_scheduler=plateau)
    criterion = nn.BCEWithLogitsLoss(pos_weight=torch.tensor([10.0]).to(device))
    print(f"{'Epoch':>6} | {'Loss':>8} | {'LR':>10}")
    print("-" * 35)
    best_loss = float('inf')
    for epoch in range(EPOCHS_TIMING):
        model.train()
        acc_loss = 0.0
        for feats, t_timing, t_stars in dataloader:
            feats, t_timing, t_stars = feats.to(device), t_timing.to(device), t_stars.to(device)
            optimizer.zero_grad()
            loss = criterion(model(feats, t_stars), t_timing)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            acc_loss += loss.item()
        avg = acc_loss / len(dataloader)
        scheduler.step(avg)
        print(f"{epoch+1:>6} | {avg:>8.4f} | {scheduler.get_lr():>10.6f}")
        if avg < best_loss:
            best_loss = avg
            torch.save(model.state_dict(), ckpt)
    torch.save(model.state_dict(), os.path.join(MODELS_DIR, "timing_net_final.pth"))
    print(f"\n  TimingNet concluído. Melhor loss: {best_loss:.4f}")


# ─────────────────────────────────────────────────────────────────
# Treino NoteNet  (inalterado)
# ─────────────────────────────────────────────────────────────────

def train_notes(device):
    print("\n" + "="*60)
    print("  FASE 2 — Treinando NoteNet")
    print("="*60)
    dataset    = NoteDataset(PROCESSED_DIR, SEQ_LEN)
    dataloader = _make_loader(dataset)
    model = get_note_model().to(device)
    os.makedirs(MODELS_DIR, exist_ok=True)
    ckpt = os.path.join(MODELS_DIR, "note_net_best.pth")
    if os.path.exists(ckpt):
        model.load_state_dict(torch.load(ckpt, map_location=device))
        print(f"  ✅ Retomando de {ckpt}\n")
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE, weight_decay=1e-5)
    plateau   = optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=4, factor=0.5)
    scheduler = WarmupScheduler(optimizer, warmup_epochs=3, base_lr=LEARNING_RATE,
                                plateau_scheduler=plateau)
    criterion = nn.CrossEntropyLoss(ignore_index=-100)
    WEIGHTS   = {'hand': 1.0, 'col': 1.2, 'layer': 1.0, 'cut': 1.5, 'double': 0.8}
    print(f"{'Epoch':>6} | {'Loss':>8} | {'Hand':>7} | {'Col':>7} | "
          f"{'Layer':>7} | {'Cut':>7} | {'Dbl':>7} | {'LR':>10}")
    print("-" * 80)
    best_loss = float('inf')
    for epoch in range(EPOCHS_NOTE):
        model.train()
        acc = defaultdict(float)
        for audio, history, stars, t_hand, t_col, t_layer, t_cut, t_double in dataloader:
            audio, history, stars = audio.to(device), history.to(device), stars.to(device)
            t_hand, t_col, t_layer, t_cut, t_double = (
                t_hand.to(device), t_col.to(device), t_layer.to(device),
                t_cut.to(device),  t_double.to(device)
            )
            optimizer.zero_grad()
            out = model(audio, history, stars)
            def ce(logits, target):
                return criterion(logits.permute(0, 2, 1), target)
            l_hand   = ce(out['hand'],   t_hand)
            l_col    = ce(out['col'],    t_col)
            l_layer  = ce(out['layer'],  t_layer)
            l_cut    = ce(out['cut'],    t_cut)
            l_double = ce(out['double'], t_double)
            loss = (l_hand * WEIGHTS['hand'] + l_col * WEIGHTS['col']
                  + l_layer * WEIGHTS['layer'] + l_cut * WEIGHTS['cut']
                  + l_double * WEIGHTS['double'])
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            acc['total']  += loss.item()
            acc['hand']   += l_hand.item()
            acc['col']    += l_col.item()
            acc['layer']  += l_layer.item()
            acc['cut']    += l_cut.item()
            acc['double'] += l_double.item()
        n   = len(dataloader)
        avg = {k: v / n for k, v in acc.items()}
        scheduler.step(avg['total'])
        print(f"{epoch+1:>6} | {avg['total']:>8.4f} | {avg['hand']:>7.4f} | "
              f"{avg['col']:>7.4f} | {avg['layer']:>7.4f} | {avg['cut']:>7.4f} | "
              f"{avg['double']:>7.4f} | {scheduler.get_lr():>10.6f}")
        if avg['total'] < best_loss:
            best_loss = avg['total']
            torch.save(model.state_dict(), ckpt)
    torch.save(model.state_dict(), os.path.join(MODELS_DIR, "note_net_final.pth"))
    print(f"\n  NoteNet concluído. Melhor loss: {best_loss:.4f}")


# ─────────────────────────────────────────────────────────────────
# Treino FlowNet  (novo)
# ─────────────────────────────────────────────────────────────────

def train_flow(device):
    print("\n" + "="*60)
    print("  FASE 3 — Treinando FlowNet")
    print("="*60)

    dataset    = FlowDataset(PROCESSED_DIR)
    dataloader = _make_loader(dataset)

    model = get_flow_model().to(device)
    os.makedirs(MODELS_DIR, exist_ok=True)

    ckpt = os.path.join(MODELS_DIR, "flow_net_best.pth")
    if os.path.exists(ckpt):
        model.load_state_dict(torch.load(ckpt, map_location=device))
        print(f"  ✅ Retomando de {ckpt}\n")

    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE, weight_decay=1e-5)
    plateau   = optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=4, factor=0.5)
    scheduler = WarmupScheduler(optimizer, warmup_epochs=3, base_lr=LEARNING_RATE,
                                plateau_scheduler=plateau)

    criterion = nn.CrossEntropyLoss()

    # Pesos das losses:
    # hand_ok/cut_ok/col_ok são tarefas de detecção — peso maior para não ignorar
    # new_* são tarefas de correção — peso menor, só importam quando _ok=1
    W_OK  = 1.5   # peso dos heads de detecção
    W_NEW = 0.8   # peso dos heads de correção

    print(f"{'Epoch':>6} | {'Loss':>8} | {'HandOk':>7} | {'CutOk':>7} | "
          f"{'ColOk':>7} | {'NewH':>6} | {'NewCut':>7} | {'NewCol':>7} | {'LR':>10}")
    print("-" * 95)

    best_loss = float('inf')

    for epoch in range(EPOCHS_FLOW):
        model.train()
        acc = defaultdict(float)

        for ctx, audio, stars, t_hand_ok, t_cut_ok, t_col_ok, t_new_hand, t_new_cut, t_new_col \
                in dataloader:

            ctx, audio, stars = ctx.to(device), audio.to(device), stars.to(device)
            t_hand_ok, t_cut_ok, t_col_ok = (
                t_hand_ok.to(device), t_cut_ok.to(device), t_col_ok.to(device)
            )
            t_new_hand, t_new_cut, t_new_col = (
                t_new_hand.to(device), t_new_cut.to(device), t_new_col.to(device)
            )

            optimizer.zero_grad()
            out = model(ctx, audio, stars)

            # Losses de detecção (precisa corrigir ou não?)
            l_hand_ok = criterion(out['hand_ok'], t_hand_ok)
            l_cut_ok  = criterion(out['cut_ok'],  t_cut_ok)
            l_col_ok  = criterion(out['col_ok'],  t_col_ok)

            # Losses de correção (qual o valor correto?)
            l_new_hand = criterion(out['new_hand'], t_new_hand)
            l_new_cut  = criterion(out['new_cut'],  t_new_cut)
            l_new_col  = criterion(out['new_col'],  t_new_col)

            loss = (
                (l_hand_ok + l_cut_ok + l_col_ok) * W_OK
              + (l_new_hand + l_new_cut + l_new_col) * W_NEW
            )

            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

            acc['total']    += loss.item()
            acc['hand_ok']  += l_hand_ok.item()
            acc['cut_ok']   += l_cut_ok.item()
            acc['col_ok']   += l_col_ok.item()
            acc['new_hand'] += l_new_hand.item()
            acc['new_cut']  += l_new_cut.item()
            acc['new_col']  += l_new_col.item()

        n   = len(dataloader)
        avg = {k: v / n for k, v in acc.items()}
        scheduler.step(avg['total'])

        print(f"{epoch+1:>6} | {avg['total']:>8.4f} | {avg['hand_ok']:>7.4f} | "
              f"{avg['cut_ok']:>7.4f} | {avg['col_ok']:>7.4f} | {avg['new_hand']:>6.4f} | "
              f"{avg['new_cut']:>7.4f} | {avg['new_col']:>7.4f} | {scheduler.get_lr():>10.6f}")

        if avg['total'] < best_loss:
            best_loss = avg['total']
            torch.save(model.state_dict(), ckpt)

    torch.save(model.state_dict(), os.path.join(MODELS_DIR, "flow_net_final.pth"))
    print(f"\n  FlowNet concluído. Melhor loss: {best_loss:.4f}")


# ─────────────────────────────────────────────────────────────────
# Entry point
# ─────────────────────────────────────────────────────────────────

def train():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--phase",
                        choices=["timing", "notes", "flow", "all"],
                        default="all",
                        help="Fase a treinar. 'all' treina as três em sequência.")
    args   = parser.parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Dispositivo: {device}\n")

    if args.phase in ("timing", "all"):
        train_timing(device)
    if args.phase in ("notes", "all"):
        train_notes(device)
    if args.phase in ("flow", "all"):
        train_flow(device)

    print("\n✅ Treino completo.")
    print("   models/timing_net_best.pth  — TimingNet")
    print("   models/note_net_best.pth    — NoteNet")
    print("   models/flow_net_best.pth    — FlowNet")


if __name__ == "__main__":
    train()