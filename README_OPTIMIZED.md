# 🚀 BSIAMapper V2 - Director AI (RTX 4070 Edition)

Esta versão evoluiu de um simples detector de beats para um **"Diretor de IA"** completo. Agora, a IA não apenas diz "quando" bater, mas também "como" (complexidade, verticalidade, ângulo), enquanto um sistema de regras robusto garante que o mapa seja jogável e divertido.

## 🌟 Novidades da V2

| Feature | V1 (Otimizado) | V2 (Director AI) |
|---------|----------------|------------------|
| **Inteligência** | Detectava apenas beats (On/Off). | Entende **Contexto, Grid, Memória e Ângulos**. |
| **Inputs** | Áudio apenas. | Áudio + **Grid Embedding + Histórico de Notas**. |
| **Controle** | Aleatório baseada em intensidade. | **Multiplicador de Dificuldade** e Thresholds Dinâmicos. |
| **Flow** | Regras básicas. | **FlowFixer V9** (Streams seguros, resets inteligentes). |
| **Autoavaliação** | Nenhuma. | **CriticNet** (Opcional) avalia a jogabilidade. |

---

## 🛠️ Como Usar

### 1. Pré-processamento
Gera features de áudio e metadados avançados (complexidade, verticalidade).
```bash
python src/preprocess_data.py
```

### 2. Treinamento (DirectorNet)
Treina o modelo principal com Mixed Precision e Batch Size grande (256+).
```bash
python src/train_optimized.py
```
*Cria `models/director_net_best.pth`.*

### 3. Geração de Mapas
Gera o mapa com controle total de dificuldade.
```bash
python src/generate_optimized.py
```
*Edite o arquivo para mudar o `difficulty_multiplier` (ex: 1.5 para Expert++).*

---

## 🧠 Arquitetura V2

### 1. DirectorNet (`models_optimized.py`)
- **Backbone**: CNN 1D + GRU Bidirecional + Self-Attention.
- **Inputs**: Espectrograma, Grid Embedding (onde está o foco?), Memória de Notas (o que veio antes?).
- **Outputs**: 
  - `Beat`: Probabilidade de nota.
  - `Complexity`: Chill, Dance ou Tech/Stream.
  - `Vertical`: Foco em baixo, meio ou cima.
  - `Angle`: Direção sugerida do corte (0-8).

### 2. PatternManager (`pattern_manager.py`)
- Recebe as "ordens" do Diretor (ex: "Faça um stream complexo na camada de cima").
- Escolhe padrões de um banco expandido (Stacks, Bursts, Sliders, Diagonais).
- Aplica regras de segurança (Vision Block, Paridade).

### 3. FlowFixer (`flow_fixer.py`)
- Pós-processamento que simula as mãos do jogador.
- Garante que não haja resets ruins em streams rápidos.
- Insere bombas táticas para forçar resets quando necessário.

---

## 💡 Dicas para sua RTX 4070

- **Batch Size**: O script está configurado para 256. Se sobrar VRAM, tente 512 em `train_optimized.py`.
- **Workers**: Use `num_workers=8` ou mais para alimentar a GPU rápido.
- **Mixed Precision**: Já ativado (`scaler`) para dobrar a velocidade de treino.

## 🔧 Customização Rápida

Quer mapas mais difíceis sem retreinar?
1. Abra `src/generate_optimized.py`.
2. Na chamada `generate_map_optimized`, mude `difficulty_multiplier` para `1.5` ou `2.0`.
3. Isso ajusta automaticamente a densidade, cooldowns e complexidade dos padrões.
