"""
train.py — Pipeline de Treino V6

Quatro fases independentes:

  Fase 1 — TimingNet  : aprende QUANDO (binário por frame)
  Fase 2 — PlaceNet   : aprende ONDE (col/layer/mão) — só vê frames com nota
  Fase 3 — AngleNet   : aprende ÂNGULO (por mão, com histórico de paridade)
  Fase 4 — ViewNet    : aprende a avaliar jogabilidade (opcional)

Uso:
  python train.py                   # treina tudo em sequência
  python train.py --phase timing
  python train.py --phase place
  python train.py --phase angle
  python train.py --phase view
"""

import os
import argparse
import random
from collections import defaultdict

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler

from models import (
    get_timing_model, get_place_model, get_angle_model, get_view_model,
    CTX_FEATS, MEL_BINS, PLACE_WIN,
    ANGLE_HIST, NUM_CUTS, NUM_COLS, NUM_LAYERS,
    VIEW_WIN, VIEW_FEATS,
)

# ─────────────────────────────────────────────────────────────────
# Configuração
# ─────────────────────────────────────────────────────────────────

PROCESSED_DIR  = "data/processed"
MODELS_DIR     = "models"

BATCH_TIMING   = 128
BATCH_PLACE    = 512   # amostras pequenas (por nota), batch maior
BATCH_ANGLE    = 512
BATCH_VIEW     = 256

SEQ_LEN        = 512   # frames por sample do TimingNet
LR             = 7e-4
WEIGHT_DECAY   = 1e-5
NUM_WORKERS    = 6

EPOCHS_TIMING  = 40
EPOCHS_PLACE   = 50
EPOCHS_ANGLE   = 30
EPOCHS_VIEW    = 25

STAR_BINS = [
    (0.0,  4.0,  1.0),
    (4.0,  5.5,  1.2),
    (5.5,  7.0,  2.0),
    (7.0,  9.0,  3.0),
    (9.0, 99.0,  4.0),
]

PLACE_PAD = PLACE_WIN // 2   # = 3


# ─────────────────────────────────────────────────────────────────
# Utilitários compartilhados
# ─────────────────────────────────────────────────────────────────

def _list_bases(processed_dir):
    """Lista todos os arquivos base que têm mel+ctx+timing+notes+stars."""
    candidates = [
        f.replace('_mel.npy', '')
        for f in os.listdir(processed_dir)
        if f.endswith('_mel.npy')
    ]
    valid, stars = [], []
    for b in set(candidates):
        paths = [
            os.path.join(processed_dir, f"{b}_{s}.npy")
            for s in ['mel', 'ctx', 'timing', 'notes', 'stars']
        ]
        if all(os.path.exists(p) for p in paths):
            try:
                s = float(np.load(paths[4]).item())
                valid.append(b)
                stars.append(s)
            except Exception:
                pass
    return valid, np.array(stars, dtype=np.float32)


def _star_weights(stars):
    w = np.ones(len(stars), dtype=np.float32)
    for lo, hi, wt in STAR_BINS:
        w[(stars >= lo) & (stars < hi)] = wt
    return w


def _make_loader(dataset, batch_size):
    w = _star_weights(dataset.stars)
    all_w = np.resize(
        np.tile(w, (len(dataset) // max(len(dataset.files), 1) + 1)),
        len(dataset),
    )
    sampler = WeightedRandomSampler(
        torch.tensor(all_w, dtype=torch.float32),
        num_samples=len(dataset), replacement=True,
    )
    return DataLoader(
        dataset, batch_size=batch_size, sampler=sampler,
        num_workers=NUM_WORKERS, persistent_workers=True, pin_memory=True,
    )


def _ckpt(name):
    return os.path.join(MODELS_DIR, f"{name}_best.pth")


def _resume(model, name, device):
    path = _ckpt(name)
    if os.path.exists(path):
        model.load_state_dict(torch.load(path, map_location=device, weights_only=True))
        print(f"    ✅ Retomando de {path}")


def _save(model, name, suffix="best"):
    os.makedirs(MODELS_DIR, exist_ok=True)
    torch.save(model.state_dict(), os.path.join(MODELS_DIR, f"{name}_{suffix}.pth"))


class WarmupCosine:
    def __init__(self, opt, warmup, total, lr):
        self.opt = opt; self.warmup = warmup
        self.total = total; self.base = lr; self.epoch = 0

    def step(self):
        self.epoch += 1
        if self.epoch <= self.warmup:
            lr = self.base * self.epoch / self.warmup
        else:
            p = (self.epoch - self.warmup) / max(1, self.total - self.warmup)
            lr = self.base * 0.5 * (1 + np.cos(np.pi * p))
        for pg in self.opt.param_groups:
            pg['lr'] = lr

    def get_lr(self):
        return self.opt.param_groups[0]['lr']


# ─────────────────────────────────────────────────────────────────
# Dataset — TimingNet
# ─────────────────────────────────────────────────────────────────

class TimingDataset(Dataset):
    """
    Amostras de SEQ_LEN frames contíguos.
    Target: timing_targets (0 / 0.3 / 1.0) por frame.
    Cada arquivo é amostrado 10× por época para cobrir a música inteira.
    """

    def __init__(self, processed_dir, seq_len=SEQ_LEN):
        self.dir = processed_dir
        self.seq = seq_len
        self.files, self.stars = _list_bases(processed_dir)
        print(f"    TimingDataset : {len(self.files)} dificuldades")

    def __len__(self):
        return len(self.files) * 10

    def __getitem__(self, idx):
        b = self.files[idx % len(self.files)]
        ctx  = np.load(f"{self.dir}/{b}_ctx.npy",    mmap_mode='r')
        tim  = np.load(f"{self.dir}/{b}_timing.npy", mmap_mode='r')
        star = float(np.load(f"{self.dir}/{b}_stars.npy").item())

        T = ctx.shape[0]
        s = np.random.randint(0, max(1, T - self.seq))
        e = s + self.seq

        ctx_s = np.array(ctx[s:e], dtype=np.float32)
        tim_s = np.array(tim[s:e], dtype=np.float32).reshape(-1, 1)

        pad = self.seq - ctx_s.shape[0]
        if pad > 0:
            ctx_s = np.pad(ctx_s, ((0, pad), (0, 0)))
            tim_s = np.pad(tim_s, ((0, pad), (0, 0)))

        return (
            torch.tensor(ctx_s, dtype=torch.float32),
            torch.tensor(tim_s, dtype=torch.float32),
            torch.tensor([star], dtype=torch.float32),
        )


# ─────────────────────────────────────────────────────────────────
# Dataset — PlaceNet
# ─────────────────────────────────────────────────────────────────

class PlaceDataset(Dataset):
    """
    Uma amostra = uma nota real do dataset.

    Para cada nota selecionada aleatoriamente:
      - Monta janela de PLACE_WIN frames de mel+ctx ao redor do frame da nota
      - Targets: hand, col, layer, is_double

    Por que isso não colapsa:
      Cada exemplo É uma nota. O modelo nunca vê "nenhuma nota aqui".
      O gradiente é sempre sobre uma decisão de posicionamento real.
    """

    def __init__(self, processed_dir):
        self.dir = processed_dir
        self.files, self.stars = _list_bases(processed_dir)
        # Filtra arquivos com notas suficientes
        valid, vstars = [], []
        for b, s in zip(self.files, self.stars):
            n = np.load(f"{self.dir}/{b}_notes.npy", mmap_mode='r')
            if len(n) >= 4:
                valid.append(b); vstars.append(s)
        self.files = valid
        self.stars = np.array(vstars, dtype=np.float32)
        print(f"    PlaceDataset  : {len(self.files)} dificuldades")

    def __len__(self):
        return len(self.files) * 20

    def __getitem__(self, idx):
        b    = self.files[idx % len(self.files)]
        mel  = np.load(f"{self.dir}/{b}_mel.npy",   mmap_mode='r')
        ctx  = np.load(f"{self.dir}/{b}_ctx.npy",   mmap_mode='r')
        notes= np.load(f"{self.dir}/{b}_notes.npy", mmap_mode='r')
        star = float(np.load(f"{self.dir}/{b}_stars.npy").item())

        T = mel.shape[0]
        N = len(notes)
        ni = np.random.randint(0, N)
        note = notes[ni]

        # Frame desta nota
        fidx = int(float(note[4]) * T)  # beat_norm × num_frames
        fidx = max(PLACE_PAD, min(fidx, T - PLACE_PAD - 1))

        # Janela de ±PLACE_PAD frames
        mel_w = np.array(mel[fidx - PLACE_PAD: fidx + PLACE_PAD + 1], dtype=np.float32)
        ctx_w = np.array(ctx[fidx - PLACE_PAD: fidx + PLACE_PAD + 1], dtype=np.float32)

        # Pad se necessário (bordas)
        if mel_w.shape[0] < PLACE_WIN:
            pad = PLACE_WIN - mel_w.shape[0]
            mel_w = np.pad(mel_w, ((0, pad), (0, 0)))
            ctx_w = np.pad(ctx_w, ((0, pad), (0, 0)))

        # is_double: 1 se a próxima nota está muito próxima (~1/16 beat)
        is_double = 0
        if ni + 1 < N:
            gap = abs(float(notes[ni+1][4]) - float(note[4]))
            if gap < 0.02:
                is_double = 1

        return (
            torch.tensor(mel_w,        dtype=torch.float32),   # (PLACE_WIN, 64)
            torch.tensor(ctx_w,        dtype=torch.float32),   # (PLACE_WIN, 8)
            torch.tensor([star],       dtype=torch.float32),
            torch.tensor(int(note[0]), dtype=torch.long),      # hand
            torch.tensor(int(note[1]), dtype=torch.long),      # col
            torch.tensor(int(note[2]), dtype=torch.long),      # layer
            torch.tensor(is_double,    dtype=torch.long),
        )


# ─────────────────────────────────────────────────────────────────
# Dataset — AngleNet
# ─────────────────────────────────────────────────────────────────

class AngleDataset(Dataset):
    """
    Uma amostra = uma nota real, com histórico das últimas ANGLE_HIST notas
    da MESMA MÃO.

    Exclui DOTs (cut=8) como targets — AngleNet só aprende direcionais.
    """

    def __init__(self, processed_dir):
        self.dir = processed_dir
        self.files, self.stars = _list_bases(processed_dir)
        valid, vstars = [], []
        for b, s in zip(self.files, self.stars):
            n = np.load(f"{self.dir}/{b}_notes.npy", mmap_mode='r')
            if len(n) >= ANGLE_HIST + 1:
                valid.append(b); vstars.append(s)
        self.files = valid
        self.stars = np.array(vstars, dtype=np.float32)
        print(f"    AngleDataset  : {len(self.files)} dificuldades")

    def __len__(self):
        return len(self.files) * 20

    def __getitem__(self, idx):
        b     = self.files[idx % len(self.files)]
        notes = np.load(f"{self.dir}/{b}_notes.npy", mmap_mode='r')
        star  = float(np.load(f"{self.dir}/{b}_stars.npy").item())
        N = len(notes)

        # Separa por mão
        L = [i for i in range(N) if int(notes[i, 0]) == 0]
        R = [i for i in range(N) if int(notes[i, 0]) == 1]

        # Escolhe mão com histórico suficiente
        hand = random.randint(0, 1)
        idxs = L if hand == 0 else R
        if len(idxs) < ANGLE_HIST + 1:
            idxs = R if hand == 0 else L
        if len(idxs) < ANGLE_HIST + 1:
            return self._pad(star)

        # Nota alvo (com histórico disponível)
        for _ in range(10):
            tp = random.randint(ANGLE_HIST, len(idxs) - 1)
            cn = notes[idxs[tp]]
            if int(cn[3]) != 8:   # ignora DOTs como target
                break
        else:
            return self._pad(star)

        # Histórico
        ch = np.full(ANGLE_HIST, NUM_CUTS,   dtype=np.int64)
        oh = np.full(ANGLE_HIST, NUM_COLS,   dtype=np.int64)
        ah = np.full(ANGLE_HIST, NUM_LAYERS, dtype=np.int64)
        for h in range(ANGLE_HIST):
            hp = tp - ANGLE_HIST + h
            if hp >= 0:
                n = notes[idxs[hp]]
                ch[h] = int(n[3])
                oh[h] = int(n[1])
                ah[h] = int(n[2])

        pos  = np.array([cn[6], cn[7]], dtype=np.float32)
        gap  = np.array([float(cn[5])], dtype=np.float32)

        return (
            torch.tensor(ch,        dtype=torch.long),
            torch.tensor(oh,        dtype=torch.long),
            torch.tensor(ah,        dtype=torch.long),
            torch.tensor(pos,       dtype=torch.float32),
            torch.tensor(gap,       dtype=torch.float32),
            torch.tensor([star],    dtype=torch.float32),
            torch.tensor(int(cn[3]),dtype=torch.long),
        )

    def _pad(self, star):
        ch = torch.full((ANGLE_HIST,), NUM_CUTS,   dtype=torch.long)
        oh = torch.full((ANGLE_HIST,), NUM_COLS,   dtype=torch.long)
        ah = torch.full((ANGLE_HIST,), NUM_LAYERS, dtype=torch.long)
        return (
            ch, oh, ah,
            torch.tensor([0.33, 0.5], dtype=torch.float32),
            torch.tensor([0.5],       dtype=torch.float32),
            torch.tensor([star],      dtype=torch.float32),
            torch.tensor(0,           dtype=torch.long),
        )


# ─────────────────────────────────────────────────────────────────
# Dataset — ViewNet
# ─────────────────────────────────────────────────────────────────

class ViewDataset(Dataset):
    """
    Uma amostra = janela de VIEW_WIN notas consecutivas.
    40% das janelas são perturbadas artificialmente (qualidade=0).
    """

    PERTURB = 0.4

    def __init__(self, processed_dir):
        self.dir = processed_dir
        self.files, self.stars = _list_bases(processed_dir)
        valid, vstars = [], []
        for b, s in zip(self.files, self.stars):
            n = np.load(f"{self.dir}/{b}_notes.npy", mmap_mode='r')
            if len(n) >= VIEW_WIN:
                valid.append(b); vstars.append(s)
        self.files = valid
        self.stars = np.array(vstars, dtype=np.float32)
        print(f"    ViewDataset   : {len(self.files)} dificuldades")

    def __len__(self):
        return len(self.files) * 15

    def __getitem__(self, idx):
        b     = self.files[idx % len(self.files)]
        notes = np.load(f"{self.dir}/{b}_notes.npy", mmap_mode='r')
        star  = float(np.load(f"{self.dir}/{b}_stars.npy").item())
        N = len(notes)
        s = np.random.randint(0, max(1, N - VIEW_WIN))
        w = np.array(notes[s:s + VIEW_WIN], dtype=np.float32)
        if len(w) < VIEW_WIN:
            w = np.pad(w, ((0, VIEW_WIN - len(w)), (0, 0)))

        feat = np.stack([
            w[:, 0],       # hand
            w[:, 6],       # col_norm
            w[:, 7],       # layer_norm
            w[:, 3] / 8.0, # cut norm
            w[:, 5],       # beat_gap
        ], axis=1).astype(np.float32)

        # SPS aproximado
        total_beats = max(float(w[:, 5].sum()) * 8.0, 0.1)
        sps = min(VIEW_WIN / (total_beats * (60.0 / 120.0)) / 2, 10.0) / 10.0

        mask = np.zeros(VIEW_WIN, dtype=np.float32)
        quality = 1.0

        if random.random() < self.PERTURB:
            quality = 0.0
            n_perturb = random.randint(1, max(1, VIEW_WIN // 8))
            for _ in range(n_perturb):
                p = random.randint(0, VIEW_WIN - 1)
                feat[p, 0] = 1.0 - feat[p, 0]
                feat[p, 3] = random.randint(0, 7) / 8.0
                mask[p] = 1.0

        return (
            torch.tensor(feat,    dtype=torch.float32),
            torch.tensor([star],  dtype=torch.float32),
            torch.tensor([quality], dtype=torch.float32),
            torch.tensor([sps],   dtype=torch.float32),
            torch.tensor(mask,    dtype=torch.float32),
        )


# ─────────────────────────────────────────────────────────────────
# Fase 1 — TimingNet
# ─────────────────────────────────────────────────────────────────

def train_timing(device):
    print("\n" + "="*60)
    print("  FASE 1 — TimingNet (quando colocar notas)")
    print("="*60)
    ds  = TimingDataset(PROCESSED_DIR)
    dl  = _make_loader(ds, BATCH_TIMING)
    m   = get_timing_model().to(device)
    _resume(m, "timing_net", device)
    opt = optim.AdamW(m.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
    sch = WarmupCosine(opt, 3, EPOCHS_TIMING, LR)
    # pos_weight alto: notas são raras (~2% dos frames)
    crit = nn.BCEWithLogitsLoss(pos_weight=torch.tensor([12.0]).to(device))

    print(f"\n{'Ep':>5} | {'Loss':>8} | {'LR':>10}")
    print("-" * 30)
    best = float('inf')
    for ep in range(EPOCHS_TIMING):
        m.train()
        total = 0.0
        for ctx, tim, stars in dl:
            ctx, tim, stars = ctx.to(device), tim.to(device), stars.to(device)
            opt.zero_grad()
            loss = crit(m(ctx, stars), tim)
            loss.backward()
            nn.utils.clip_grad_norm_(m.parameters(), 1.0)
            opt.step()
            total += loss.item()
        avg = total / len(dl)
        sch.step()
        print(f"{ep+1:>5} | {avg:>8.4f} | {sch.get_lr():>10.6f}")
        if avg < best:
            best = avg; _save(m, "timing_net")
    _save(m, "timing_net", "final")
    print(f"\n  ✅ TimingNet concluído. Melhor loss: {best:.4f}")


# ─────────────────────────────────────────────────────────────────
# Fase 2 — PlaceNet
# ─────────────────────────────────────────────────────────────────

def train_place(device):
    print("\n" + "="*60)
    print("  FASE 2 — PlaceNet (onde colocar — col/layer/mão)")
    print("="*60)
    ds  = PlaceDataset(PROCESSED_DIR)
    dl  = _make_loader(ds, BATCH_PLACE)
    m   = get_place_model().to(device)
    _resume(m, "place_net", device)
    opt = optim.AdamW(m.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
    sch = WarmupCosine(opt, 3, EPOCHS_PLACE, LR)
    ce  = nn.CrossEntropyLoss()

    W = {'hand': 1.0, 'col': 1.5, 'layer': 1.0, 'double': 0.5}

    print(f"\n{'Ep':>5} | {'Total':>8} | {'Hand':>7} | {'Col':>7} | "
          f"{'Layer':>7} | {'ColAcc':>7} | {'LR':>10}")
    print("-" * 68)
    best = float('inf')
    for ep in range(EPOCHS_PLACE):
        m.train()
        acc = defaultdict(float)
        correct_col = total_col = 0
        for mw, cw, stars, t_hand, t_col, t_layer, t_dbl in dl:
            mw, cw, stars = mw.to(device), cw.to(device), stars.to(device)
            t_hand, t_col = t_hand.to(device), t_col.to(device)
            t_layer, t_dbl = t_layer.to(device), t_dbl.to(device)
            opt.zero_grad()
            out = m(mw, cw, stars)
            l_h = ce(out['hand'],      t_hand)
            l_c = ce(out['col'],       t_col)
            l_l = ce(out['layer'],     t_layer)
            l_d = ce(out['is_double'], t_dbl)
            loss = l_h*W['hand'] + l_c*W['col'] + l_l*W['layer'] + l_d*W['double']
            loss.backward()
            nn.utils.clip_grad_norm_(m.parameters(), 1.0)
            opt.step()
            acc['total'] += loss.item()
            acc['hand']  += l_h.item()
            acc['col']   += l_c.item()
            acc['layer'] += l_l.item()
            correct_col  += (out['col'].argmax(1) == t_col).sum().item()
            total_col    += t_col.size(0)
        n   = len(dl)
        avg = {k: v / n for k, v in acc.items()}
        col_acc = correct_col / max(total_col, 1)
        sch.step()
        print(f"{ep+1:>5} | {avg['total']:>8.4f} | {avg['hand']:>7.4f} | "
              f"{avg['col']:>7.4f} | {avg['layer']:>7.4f} | {col_acc:>7.3%} | "
              f"{sch.get_lr():>10.6f}")
        if avg['total'] < best:
            best = avg['total']; _save(m, "place_net")
    _save(m, "place_net", "final")
    print(f"\n  ✅ PlaceNet concluído. Melhor loss: {best:.4f}")


# ─────────────────────────────────────────────────────────────────
# Fase 3 — AngleNet
# ─────────────────────────────────────────────────────────────────

def train_angle(device):
    print("\n" + "="*60)
    print("  FASE 3 — AngleNet (ângulo de corte por mão)")
    print("="*60)
    ds  = AngleDataset(PROCESSED_DIR)
    dl  = _make_loader(ds, BATCH_ANGLE)
    m   = get_angle_model().to(device)
    _resume(m, "angle_net", device)
    opt  = optim.AdamW(m.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
    sch  = WarmupCosine(opt, 3, EPOCHS_ANGLE, LR)
    crit = nn.CrossEntropyLoss(label_smoothing=0.1)

    print(f"\n{'Ep':>5} | {'Loss':>8} | {'Acc':>8} | {'LR':>10}")
    print("-" * 38)
    best = float('inf')
    for ep in range(EPOCHS_ANGLE):
        m.train()
        total_loss = correct = samples = 0
        for ch, oh, ah, pos, gap, stars, target in dl:
            ch, oh, ah = ch.to(device), oh.to(device), ah.to(device)
            pos, gap, stars = pos.to(device), gap.to(device), stars.to(device)
            target = target.to(device)
            opt.zero_grad()
            logits = m(ch, oh, ah, pos, gap, stars)
            loss   = crit(logits, target)
            loss.backward()
            nn.utils.clip_grad_norm_(m.parameters(), 1.0)
            opt.step()
            total_loss += loss.item() * target.size(0)
            correct    += (logits.argmax(1) == target).sum().item()
            samples    += target.size(0)
        avg = total_loss / max(samples, 1)
        acc = correct / max(samples, 1)
        sch.step()
        print(f"{ep+1:>5} | {avg:>8.4f} | {acc:>8.3%} | {sch.get_lr():>10.6f}")
        if avg < best:
            best = avg; _save(m, "angle_net")
    _save(m, "angle_net", "final")
    print(f"\n  ✅ AngleNet concluído. Melhor loss: {best:.4f}")


# ─────────────────────────────────────────────────────────────────
# Fase 4 — ViewNet
# ─────────────────────────────────────────────────────────────────

def train_view(device):
    print("\n" + "="*60)
    print("  FASE 4 — ViewNet (avaliador de qualidade)")
    print("="*60)
    ds  = ViewDataset(PROCESSED_DIR)
    dl  = _make_loader(ds, BATCH_VIEW)
    m   = get_view_model().to(device)
    _resume(m, "view_net", device)
    opt = optim.AdamW(m.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
    sch = WarmupCosine(opt, 3, EPOCHS_VIEW, LR)
    bce = nn.BCEWithLogitsLoss()
    mse = nn.MSELoss()

    print(f"\n{'Ep':>5} | {'Total':>8} | {'Qual':>7} | {'SPS':>7} | {'Mask':>7} | {'LR':>10}")
    print("-" * 58)
    best = float('inf')
    for ep in range(EPOCHS_VIEW):
        m.train()
        acc = defaultdict(float)
        for feat, stars, qual, sps, mask in dl:
            feat, stars = feat.to(device), stars.to(device)
            qual, sps, mask = qual.to(device), sps.to(device), mask.to(device)
            opt.zero_grad()
            out = m(feat, stars)
            l_q = bce(out['quality'],      qual)
            l_s = mse(torch.sigmoid(out['sps_pred']), sps)
            l_m = bce(out['problem_mask'], mask)
            loss = l_q*2.0 + l_s*0.5 + l_m*1.0
            loss.backward()
            nn.utils.clip_grad_norm_(m.parameters(), 1.0)
            opt.step()
            acc['total'] += loss.item()
            acc['qual']  += l_q.item()
            acc['sps']   += l_s.item()
            acc['mask']  += l_m.item()
        n   = len(dl)
        avg = {k: v/n for k, v in acc.items()}
        sch.step()
        print(f"{ep+1:>5} | {avg['total']:>8.4f} | {avg['qual']:>7.4f} | "
              f"{avg['sps']:>7.4f} | {avg['mask']:>7.4f} | {sch.get_lr():>10.6f}")
        if avg['total'] < best:
            best = avg['total']; _save(m, "view_net")
    _save(m, "view_net", "final")
    print(f"\n  ✅ ViewNet concluído. Melhor loss: {best:.4f}")


# ─────────────────────────────────────────────────────────────────
# Entry point
# ─────────────────────────────────────────────────────────────────

def main():
    p = argparse.ArgumentParser(
        description="Treina os modelos do BS AI Mapper V6.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Fases:
  all     — Treina tudo em sequência (timing → place → angle → view)
  timing  — Só TimingNet   (quando colocar notas)
  place   — Só PlaceNet    (col/layer/mão por nota)
  angle   — Só AngleNet    (ângulo de corte por mão)
  view    — Só ViewNet     (avaliador de qualidade — opcional)

Exemplos:
  python train.py                    # treina tudo
  python train.py --phase angle      # retoma só AngleNet
        """,
    )
    p.add_argument("--phase",
                   choices=["all", "timing", "place", "angle", "view"],
                   default="all")
    args   = p.parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print("=" * 60)
    print("  BS AI MAPPER V6 — Pipeline de Treino")
    print("=" * 60)
    print(f"  Dispositivo : {device}")
    if device.type == "cuda":
        print(f"  GPU         : {torch.cuda.get_device_name(0)}")
        print(f"  VRAM        : {torch.cuda.get_device_properties(0).total_memory/1e9:.1f} GB")
    print(f"  Fase        : {args.phase}")
    print()

    os.makedirs(MODELS_DIR, exist_ok=True)

    if args.phase in ("all", "timing"): train_timing(device)
    if args.phase in ("all", "place"):  train_place(device)
    if args.phase in ("all", "angle"):  train_angle(device)
    if args.phase in ("all", "view"):   train_view(device)

    print("\n" + "="*60)
    print("  ✅ Treino completo.")
    print("="*60)
    for name in ["timing_net_best.pth", "place_net_best.pth",
                 "angle_net_best.pth", "view_net_best.pth"]:
        path = os.path.join(MODELS_DIR, name)
        status = "✅" if os.path.exists(path) else "❌"
        print(f"  {status}  {name}")


if __name__ == "__main__":
    main()
