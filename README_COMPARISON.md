# Análise Comparativa Profunda: BSIAMapper (Seu Projeto) vs. InfernoSaber

Este documento detalha as diferenças arquiteturais, filosóficas e técnicas entre o seu projeto atual (`/src`) e o projeto de referência `infernosaber`. Esta análise foi atualizada para refletir o uso do seu modelo `DirectorNet`, o pipeline de geração avançado e as estratégias de treinamento.

## 1. Filosofia de Design

| Característica | BSIAMapper (Seu Projeto) | InfernoSaber |
| :--- | :--- | :--- |
| **Abordagem Principal** | **Híbrida (IA + Algoritmo):** A IA atua como um "Diretor" que define a intenção (intensidade, complexidade), enquanto algoritmos determinísticos constroem os padrões de notas. | **Pure Deep Learning (End-to-End Modular):** Tenta aprender tudo via redes neurais, separando o problema em módulos de predição de tempo e classificação de notas. |
| **Controle de Qualidade** | **Explícito:** Regras rígidas de "Flow" e "Paridade" garantem que o mapa seja jogável. | **Implícito/Estatístico:** Confia que o modelo aprendeu o que é jogável, com sanitização posterior. |
| **Flexibilidade** | **Alta:** Fácil adicionar novos padrões (ex: "Tech") apenas codificando-os no `PatternManager`, sem re-treinar a IA. | **Baixa:** Novos padrões exigem re-treinar o modelo com dados que contenham esses padrões. |

## 2. Arquitetura de IA (O Cérebro)

### BSIAMapper: `DirectorNet` (PyTorch)
*   **Conceito:** O modelo não coloca as notas. Ele "sente" a música e diz ao gerador o que fazer.
*   **Entrada:** Espectrograma Mel (84 features) + Features de Áudio Brutas.
*   **Estrutura:**
    *   **Backbone:** CNN 1D (extração de features) + GRU Bidirecional (contexto temporal). Mais leve e rápido que LSTMs profundas.
    *   **Multi-Head Output (3 Cabeças):**
        1.  **Beat Head:** Probabilidade de haver uma nota (0-1).
        2.  **Complexity Head:** Classifica a seção (Chill, Dance, Tech/Stream).
        3.  **Vertical Head:** Onde focar as notas (Baixo, Meio, Cima).
*   **Vantagem:** O modelo aprende conceitos musicais abstratos (ex: "aqui a música pede um stream complexo") em vez de tentar decorar posições de notas.

### InfernoSaber: Pipeline Modular (TensorFlow/Keras)
*   **Conceito:** Pipeline sequencial de modelos especializados.
*   **Processamento de Dados:** Trata o áudio como **Imagens**. Gera espectrogramas, redimensiona-os como imagens e usa CNNs 2D.
*   **Módulos:**
    1.  **Beat Predictor (TCN/LSTM):** Olha janelas de "imagens" de áudio e prevê *onde* ocorrem as batidas.
    2.  **Mapper (`lstm_half`):** Recebe o áudio codificado (via Autoencoder/CNN) + contexto temporal e decide *qual* nota colocar nos tempos previstos.
*   **Desvantagem:** Muito pesado. O tratamento de áudio como imagem de baixa resolução pode perder detalhes finos de timing que a CNN 1D do seu projeto captura melhor.

## 3. Estratégia de Treinamento

### BSIAMapper (`train_optimized.py`)
*   **Tipo:** Multi-Task Learning (Aprendizado Multi-Tarefa).
*   **Processo:**
    *   Um único script treina o `DirectorNet` para realizar 3 tarefas ao mesmo tempo.
    *   **Targets Dinâmicos:** O `DirectorDataset` calcula os targets (complexidade, verticalidade) *on-the-fly* a partir dos mapas originais, permitindo ajustes na lógica de treinamento sem reprocessar todos os dados.
    *   **Loss Function:** Soma ponderada de `BCEWithLogitsLoss` (Beat) e `CrossEntropyLoss` (Complexidade e Verticalidade).
*   **Eficiência:** Treina rápido e o modelo aprende correlações úteis (ex: seções complexas geralmente têm mais notas).

### InfernoSaber (`train_autoenc_music.py` + `train_bs_automapper.py`)
*   **Tipo:** Multi-Stage Learning (Aprendizado em Estágios).
*   **Processo:**
    1.  **Estágio 1 (Autoencoder):** Treina uma rede neural apenas para comprimir e descomprimir músicas (sem olhar para os mapas). O objetivo é ensinar a IA a "ouvir" e representar o áudio eficientemente.
    2.  **Estágio 2 (Mapper):** Usa a parte "Encoder" do modelo anterior para extrair features e treina uma LSTM para prever as notas.
*   **Complexidade:** Requer múltiplos scripts e gerenciamento de modelos salvos. Se o Autoencoder for ruim, o Mapper será ruim.

## 4. Geração de Mapas (O Corpo)

### BSIAMapper: Pipeline Construtivo (`generate_from_url.py`)
1.  **Análise:** O `DirectorNet` varre a música e gera curvas de probabilidade e classificação.
2.  **Pattern Manager (`pattern_manager.py`):**
    *   Recebe as instruções do Diretor (ex: "Alta intensidade, Complexidade Tech").
    *   Escolhe um padrão de uma biblioteca pré-codificada (`stream_burst`, `window_wipe`, `tech_angle`).
    *   Isso garante que os padrões sejam sempre estruturalmente perfeitos, pois são gerados por código, não "alucinados" pela IA.
3.  **Flow Fixer (`flow_fixer.py`):**
    *   Simula as mãos do jogador.
    *   Garante paridade (alternância de mãos).
    *   Corrige ângulos ruins e insere "Bombas de Reset" se o fluxo quebrar.
    *   Impede "Vision Blocks" e movimentos fisicamente impossíveis.

### InfernoSaber: Pipeline Preditivo (`gen_beats.py`)
1.  **Predição:** Gera uma lista de timestamps.
2.  **Classificação:** Para cada timestamp, o modelo cospe uma classe (tipo de nota, linha, camada).
3.  **Sanitização:** Remove notas sobrepostas e tenta corrigir erros óbvios, mas não tem a "consciência" de padrões complexos que o seu `PatternManager` tem.

## 5. Resumo Técnico para Prompts

*   **Seu Projeto (`src`):**
    *   **Linguagem:** Python 3.x, PyTorch.
    *   **Pontos Fortes:** Geração estruturada, padrões limpos, leve, fácil de ajustar dificuldade (basta mudar parâmetros no `PatternManager`).
    *   **Onde mexer:**
        *   Para novos estilos de mapa: Edite `PatternManager`.
        *   Para melhor detecção de ritmo: Treine o `DirectorNet` (Head de Beat).
        *   Para melhor jogabilidade: Ajuste o `FlowFixer`.

*   **InfernoSaber:**
    *   **Linguagem:** Python, TensorFlow/Keras.
    *   **Pontos Fortes:** Tenta capturar o estilo "humano" de forma mais orgânica (com erros e tudo), sistema completo com UI (legado).
    *   **Pontos Fracos:** Pesado, difícil de manter, arquitetura de áudio-como-imagem é datada.

**Conclusão:** O seu projeto evoluiu para uma arquitetura "Neuro-Simbólica" (Rede Neural para percepção + Lógica Simbólica para geração), o que é geralmente superior para tarefas que exigem regras rígidas como Beat Saber, comparado à abordagem "Pure Deep Learning" do InfernoSaber.
