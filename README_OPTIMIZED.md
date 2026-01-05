# 🚀 BSIAMapper Otimizado (Single GPU Edition)

Esta versão foi reescrita para rodar eficientemente na sua **RTX 4070**, reduzindo o tempo de treino de "noites inteiras" para **minutos ou poucas horas**, mantendo a qualidade através de um sistema híbrido (IA + Regras).

## 📋 O que mudou?

| Feature | Antes (V4) | Agora (Otimizado) |
|---------|------------|-------------------|
| **Foco da IA** | Tentar adivinhar posição, cor e direção de cada nota. | Apenas detectar **QUANDO** ocorre um beat (Onset). |
| **Arquitetura** | CNN + LSTM Profunda (Pesada). | **CRNN Leve** (Conv1D + GRU). |
| **Padrões** | A IA tentava "inventar" padrões e errava muito. | **PatternManager** aplica padrões profissionais (Streams, Stacks) deterministicamente. |
| **Treino** | Sequências de 1000 frames, 100 epochs. | Sequências de 200 frames, 20 epochs. |
| **Tempo** | Horas/Dias. | **~15-30 Minutos**. |

---

## 🛠️ Como Usar

### 1. Pré-processamento (Se ainda não fez)
Se você já rodou isso antes, **não precisa rodar de novo**. O novo sistema lê os mesmos dados.
```bash
python src/preprocess_data.py
```

### 2. Treinamento Otimizado
Treina o novo modelo leve (`BeatNet`).
```bash
python src/train_optimized.py
```
*Isso vai criar o arquivo `models/beat_net_optimized.pth`.*

### 3. Geração de Mapas
Gera o mapa usando a IA para o ritmo e o `PatternManager` para o flow.
```bash
python src/generate_optimized.py
```
*O mapa sairá na pasta `output/OptimizedMap`.*

---

## 🧠 Estrutura dos Arquivos Novos

- **`src/models_optimized.py`**: Contém a `BeatNet`, uma rede neural enxuta focada apenas em achar o ritmo.
- **`src/pattern_manager.py`**: O "cérebro" determinístico. Contém regras de Beat Saber (alternância de mãos, flow, resets) e templates de padrões (streams, jumps). **Edite aqui se quiser mudar o estilo dos mapas.**
- **`src/train_optimized.py`**: Script de treino ultra-rápido. Usa "Lazy Loading" para não estourar a RAM e foca em janelas curtas onde a ação acontece.
- **`src/generate_optimized.py`**: Junta tudo. Pega o áudio -> IA acha os beats -> Pattern Manager coloca as notas -> Salva o arquivo.

## 💡 Dicas de Customização

Para mudar o estilo do mapa (ex: mais Tech ou mais Dance), você não precisa mais retreinar a IA! Apenas edite o `src/pattern_manager.py`:

1. Abra `src/pattern_manager.py`.
2. No método `get_pattern_for_intensity`, mude os pesos ou os padrões escolhidos.
3. Rode `generate_optimized.py` novamente.
