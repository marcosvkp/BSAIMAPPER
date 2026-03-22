# BS AI Mapper V5 — MEMORY.md

> Referência rápida do sistema. Leia antes de alterar qualquer arquivo.

---

## Visão Geral

Sistema de geração de mapas Beat Saber com três modelos de IA que trabalham em sequência:

```
PatternModel → AngleNet → ViewNet → FlowFixer
    QUANDO         ÂNGULO     VALIDA     GARANTE
    ONDE?          por mão?   setor?     físico
    QUEM?
```

---

## Arquivos do Projeto

```
src/
  models.py              ← PatternModel + AngleNet + ViewNet (arquitetura)
  train.py               ← Pipeline de treino (fase 1/2/3)
  generate.py            ← Pipeline de geração (uso final)
  audio_processor.py     ← Extrai 72 features de áudio (mel+ctx)
  data_loader.py         ← Carrega mapas e gera arrays NumPy
  preprocess_data.py     ← Pré-processa todos os mapas para .npy
  flow_fixer.py          ← Validador físico V14 (determinístico, 100%)
  youtube_downloader.py  ← Download YouTube → MP3 + OGG/egg
  downloader.py          ← Download de mapas ranqueados (BeatSaver API)
  fetch_scoresaber_stars.py ← Busca estrelas na API do ScoreSaber
  scoresaber_api.py      ← Wrapper da API ScoreSaber
  parser/
    loader.py            ← Carrega Info.dat + dificuldade (.dat)
    beatmap.py           ← Parseamento V2 e V3
    objects.py           ← Dataclasses: Note, Obstacle, Event
    enums.py             ← NoteColor, NoteCutDirection, etc.
    base.py              ← BeatmapObject (base)

data/
  raw_maps/              ← Mapas baixados (1 pasta por mapa)
  processed/             ← Arrays .npy pré-processados
  map_hashes.json        ← Hashes de versão dos mapas baixados

models/
  pattern_model_best.pth ← PatternModel treinado
  angle_net_best.pth     ← AngleNet treinado
  view_net_best.pth      ← ViewNet treinado (opcional)

output/
  <nome_musica>/         ← Pasta com arquivos do mapa
  <nome_musica>.zip      ← Arquivo pronto para Beat Saber
```

---

## Features de Áudio

### mel_spec (64 valores por frame)
Espectrograma Mel em escala log, normalizado por bin.
- Captura timbre, instrumentação e textura harmônica
- Usado pelo PatternModel para posicionamento de notas
- 64 bins (não 128) = metade do custo computacional, ~90% da informação relevante

### ctx_feats (8 valores por frame)
| # | Nome            | Descrição                                      |
|---|-----------------|------------------------------------------------|
| 1 | onset_strength  | Força dos ataques — principal sinal de "tem nota" |
| 2 | onset_peaks     | Picos binários suavizados (~50ms)              |
| 3 | rms             | Energia local (volume)                         |
| 4 | beat_phase      | Fase cíclica no beat (0→1)                     |
| 5 | halfbeat_phase  | Fase no meio-beat                              |
| 6 | song_position   | Posição global na música (0→1)                 |
| 7 | is_drop         | Seção de alta energia (binário suave)          |
| 8 | is_breakdown    | Seção calma (binário suave)                    |

**Total: 72 features por frame** (43 fps a SR=22050, HOP=512)

---

## Modelos

### PatternModel (~2.1M params)
**Tarefa:** Decide QUANDO e ONDE colocar notas.

**Input:**
- `mel_spec`  : (B, T, 64) — espectrograma mel
- `ctx_feats` : (B, T, 8)  — features de contexto
- `stars`     : (B, 1)     — dificuldade alvo

**Output (por frame):**
- `has_note`  : (B, T, 1)  — há nota neste frame?
- `hand`      : (B, T, 2)  — mão esquerda(0) ou direita(1)?
- `col`       : (B, T, 4)  — coluna (0-3)
- `layer`     : (B, T, 3)  — camada (0-2)
- `is_double` : (B, T, 2)  — nota dupla?

**Arquitetura:** Conv1d × 3 → GRU bidirecional (hidden=256, 2 layers) → 5 heads

**Pesos:** `models/pattern_model_best.pth`
**Treinamento:** ~3-5 min/época, 40 épocas

---

### AngleNet (~800k params)
**Tarefa:** Decide o ângulo de corte de cada nota, operando POR MÃO.

**Por que por mão?**
O fluxo de swing (UP→DOWN→UP→...) é independente entre mãos. Misturar notas
das duas mãos no histórico polui completamente o sinal de paridade.

**Input (por nota):**
- `cut_history`   : (B, 12) long — últimos 12 cuts da mesma mão
- `col_history`   : (B, 12) long — últimas 12 colunas da mesma mão
- `layer_history` : (B, 12) long — últimas 12 camadas da mesma mão
- `pos_now`       : (B, 2)  float — [col/3, layer/2] normalizados
- `beat_gap`      : (B, 1)  float — beats desde nota anterior desta mão
- `stars`         : (B, 1)  float

**Output:** `cut_logits` : (B, 9) — logits para 9 direções de corte

**Arquitetura:** Embeddings separados (cut/col/layer) → GRU bidirecional (hidden=128)
→ fusão com pos_now + beat_gap → LayerNorm + GELU head

**Pesos:** `models/angle_net_best.pth`
**Treinamento:** ~1-2 min/época, 30 épocas

---

### ViewNet (~600k params)
**Tarefa:** Avalia a jogabilidade de janelas de 32 notas.

**Input (por janela):**
- `notes_window` : (B, 32, 5) — [hand, col/3, layer/2, cut/8, beat_gap]
- `stars`        : (B, 1)

**Output:**
- `quality`      : (B, 1)  — setor é jogável? (logit → sigmoid)
- `sps_pred`     : (B, 1)  — SPS previsto (logit → sigmoid × 10)
- `problem_mask` : (B, 32) — qual nota específica é problemática?

**Arquitetura:** Transformer Encoder (2 layers, 4 heads, d_model=64) +
encoding posicional aprendível → pooling → heads

**Pesos:** `models/view_net_best.pth`
**Treinamento:** ~1 min/época, 25 épocas

---

## Pipeline de Geração

```
URL YouTube
  ↓ youtube_downloader.py
MP3 + OGG + cover.png
  ↓ audio_processor.py
mel_spec (T,64) + ctx_feats (T,8) + energy (T,)
  ↓
┌─────────────────────────────────────────────────────┐
│ FASE 1: PatternModel                                 │
│  → notas com cut=8 (DOT), mão, col, layer definidos  │
└─────────────────────────────────────────────────────┘
  ↓
┌─────────────────────────────────────────────────────┐
│ FASE 2: AngleNet                                     │
│  → percorre notas em ordem cronológica               │
│  → estado separado por mão (HandState)               │
│  → substitui DOT por ângulo direcional               │
└─────────────────────────────────────────────────────┘
  ↓
┌─────────────────────────────────────────────────────┐
│ FASE 3: ViewNet (até 3 passes)                       │
│  → janela deslizante de 32 notas (passo=16)          │
│  → qualidade < 0.5 → identifica notas problemáticas  │
│  → re-aplica AngleNet com temperatura menor          │
└─────────────────────────────────────────────────────┘
  ↓
┌─────────────────────────────────────────────────────┐
│ FASE 4: FlowFixer V14                                │
│  → validação física determinística                   │
│  → garante: sem UP>UP, sem RIGHT>RIGHT, sem crossed  │
│  → adiciona bombas se necessário                     │
└─────────────────────────────────────────────────────┘
  ↓
output/<nome>.zip
```

---

## FlowFixer V14 — Regras Físicas Garantidas

Ordem de operações (CRÍTICA — não alterar):
1. `_remove_simultaneous` — remove tower stacks (mesma mão, mesmo timestamp)
2. `_remove_unreachable`  — remove notas inatingíveis (distância × tempo)
3. `_fix_edge`            — bordas da grade (**ANTES do swing**)
4. `_fix_swing`           — alternância + consistência de layer + horizontais
5. `_fix_crossed_arms`    — braços cruzados em simultâneas
6. `_fix_hitbox`          — doubles sobrepostos

**DOT é TRANSPARENTE:** não altera o estado de swing.
`UP > DOT > UP` é inválido e será corrigido.
`UP > DOT > DOWN` é válido e será mantido.

---

## Arquivos .npy Gerados pelo Preprocessamento

| Arquivo          | Shape            | Descrição                    |
|------------------|------------------|------------------------------|
| `{base}_mel.npy` | (T, 64)          | Espectrograma mel             |
| `{base}_ctx.npy` | (T, 8)           | Features de contexto         |
| `{base}_timing.npy` | (T,)          | Targets de timing (0/0.3/1.0)|
| `{base}_notes.npy`  | (N, 8)        | Sequência de notas            |
| `{base}_stars.npy`  | (1,)          | Estrelas ScoreSaber           |

### Colunas de `_notes.npy`
| Col | Nome       | Descrição                      |
|-----|------------|--------------------------------|
| 0   | hand       | 0=esquerda, 1=direita          |
| 1   | col        | Coluna (0-3)                   |
| 2   | layer      | Camada (0-2)                   |
| 3   | cut        | Direção de corte (0-8)         |
| 4   | beat_norm  | beat / total_beats (0→1)       |
| 5   | beat_gap   | (beats desde anterior)/8 (0→1)|
| 6   | col_norm   | col / 3.0                      |
| 7   | layer_norm | layer / 2.0                    |

---

## Comandos de Uso

### 1. Baixar mapas ranqueados
```bash
python downloader.py               # 1000 mapas (padrão)
```

### 2. Buscar estrelas do ScoreSaber
```bash
python fetch_scoresaber_stars.py
```

### 3. Pré-processar dados
```bash
python preprocess_data.py                 # processa novos mapas
python preprocess_data.py --force         # re-processa tudo
python preprocess_data.py --workers 12    # mais threads
```

### 4. Treinar modelos
```bash
python train.py                    # treina tudo (pattern → angle → view)
python train.py --phase pattern    # só PatternModel
python train.py --phase angle      # só AngleNet
python train.py --phase view       # só ViewNet
```

### 5. Gerar mapa
```bash
python generate.py                          # modo interativo
python generate.py --url URL --diffs 3,4,5 --stars 4.5,6.5,8.5
```

---

## Hardware e Tempos Estimados

| Etapa             | RTX 4070 + Ryzen 7 5700X | 1000 mapas |
|-------------------|--------------------------|------------|
| Preprocessamento  | CPU bound                | ~15-30 min |
| PatternModel      | ~3-5 min/época × 40      | ~2-3h total|
| AngleNet          | ~1-2 min/época × 30      | ~45-60 min |
| ViewNet           | ~1 min/época × 25        | ~25 min    |
| Geração (1 mapa)  | GPU                      | ~10-30s    |

---

## Decisões de Design e Justificativas

### Por que mel_spec separado de ctx_feats?
- `preprocess_data.py` salva `_mel.npy` e `_ctx.npy` separados
- mmap é mais eficiente quando arquivos menores
- Permite treinar só com ctx se mel não for necessário (e.g. estudos de ablação)

### Por que AngleNet opera por mão?
- Fluxo de paridade (UP→DOWN→UP) é independente entre mãos
- Misturar histórico das duas mãos = o modelo vê: UP(L)→DOWN(R)→UP(L) e aprende
  que "depois de DOWN sempre vem UP" — o que é errado pois o DOWN era da outra mão
- Solução: HandState separado por mão, histórico só da mesma mão

### Por que ViewNet usa Transformer e não GRU?
- Notas problemáticas podem ser identificadas por ATENÇÃO com notas distantes
  (ex: nota 5 e nota 28 criam um crossover — GRU perderia a nota 5 no contexto)
- Transformer com janela de 32 notas é eficiente e suficiente

### Por que FlowFixer mesmo com AngleNet?
- AngleNet é probabilístico — pode cometer erros
- FlowFixer é determinístico e garantido — nenhuma sequência inválida passa
- Os dois se complementam: AngleNet gera fluxo natural, FlowFixer garante físico

### Por que edge ANTES de swing no FlowFixer?
Se swing rodar primeiro e depois edge rodar:
- Swing corrige nota em layer=0 para DOWN_RIGHT
- Edge vê DOWN_RIGHT em layer=0 e converte para UP_RIGHT
- Swing foi desfeito — BUG
Com edge primeiro: nota chega com posição pré-validada, swing escolhe cut correto.

---

## Direções de Corte (Cut Direction)

| ID | Nome       | Símbolo |
|----|------------|---------|
| 0  | UP         | ↑       |
| 1  | DOWN       | ↓       |
| 2  | LEFT       | ←       |
| 3  | RIGHT      | →       |
| 4  | UP_LEFT    | ↖       |
| 5  | UP_RIGHT   | ↗       |
| 6  | DOWN_LEFT  | ↙       |
| 7  | DOWN_RIGHT | ↘       |
| 8  | DOT (ANY)  | •       |

---

## Histórico de Versões

| Versão | Principais Mudanças |
|--------|---------------------|
| V1-V2  | Modelo único (DirectorNet), PatternManager manual |
| V3     | TimingNet + NoteNet separados |
| V4     | FlowNet autoregressivo, FlowFixer V14 |
| **V5** | **PatternModel (mel+ctx), AngleNet por mão, ViewNet** |

### Mudanças V4 → V5
- **Removidos:** `models_optimized.py`, `generate_from_url.py`, `generate_optimized.py`, `pattern_manager.py`, `train_optimized.py`
- **Adicionados:** `models.py`, `generate.py`, `train.py`
- **Modificados:** `audio_processor.py` (adicionou mel_spec 64 bins), `data_loader.py` (novo schema 8 colunas), `preprocess_data.py` (salva _mel e _ctx separados), `youtube_downloader.py` (removeu path Windows hardcoded)
- **Mantidos inalterados:** `flow_fixer.py`, `parser/`, `downloader.py`, `fetch_scoresaber_stars.py`, `scoresaber_api.py`
