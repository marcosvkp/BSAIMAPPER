# Beat Saber AI Mapper (BSIAMapper)

Este projeto é uma ferramenta avançada de Inteligência Artificial projetada para gerar automaticamente mapas (níveis) para o jogo **Beat Saber**.

## 🎯 Intuito do Projeto

O objetivo principal é criar mapas divertidos e jogáveis a partir de qualquer arquivo de música, utilizando Deep Learning para imitar os padrões de design encontrados em mapas rankeados pela comunidade. A IA analisa o áudio para entender o ritmo, intensidade e estrutura da música, traduzindo isso em padrões de blocos, direção de cortes e fluxo.

O sistema não apenas coloca notas aleatórias nas batidas; ele considera:
*   **Fluxo e Paridade**: Alternância correta entre as mãos e direção dos cortes.
*   **Estilos de Jogo**: Capacidade de gerar mapas com características diferentes (Speed, Tech, Accuracy, Standard).
*   **Prevenção de Erros**: Lógica para evitar "Vision Blocks" e padrões fisicamente impossíveis.

## 📂 Estrutura e Funcionalidades

Abaixo está a descrição do que cada módulo na pasta `src/` faz:

### Core (IA e Processamento)
*   **`src/models_optimized.py`**: Define a arquitetura da Rede Neural "DirectorNet". É um modelo multi-head (CNN + GRU) que prevê simultaneamente:
    *   Probabilidade de batida (Beat Detection).
    *   Complexidade do padrão (Chill, Dance, Tech).
    *   Viés vertical (se as notas devem ir para cima, meio ou baixo).
*   **`src/audio_processor.py`**: O "ouvido" da IA. Usa `librosa` e `ffmpeg` para:
    *   Detectar o BPM da música.
    *   Gerar Mel Spectrograms (representação visual do som).
    *   Criar grids rítmicos.
    *   Normalizar e adicionar silêncio (intro/outro) ao áudio.
*   **`src/generate_optimized.py`**: O script principal de geração.
    *   Carrega o modelo `DirectorNet`.
    *   Processa a música e gera as notas brutas.
    *   Invoca o `PatternManager` para traduzir as previsões da IA em padrões de notas.
    *   Invoca o `FlowFixer` para corrigir erros de paridade e adicionar bombas em resets.
    *   Empacota tudo em um arquivo ZIP pronto para o jogo.
*   **`src/pattern_manager.py`**: Gerencia a criação de padrões específicos (streams, jumps, sliders) com base na intensidade e complexidade ditadas pela IA. Mantém o estado básico de fluxo (onde estão as mãos).
*   **`src/flow_fixer.py`**: Um simulador de física e paridade pós-processamento.
    *   Analisa o mapa gerado nota por nota.
    *   Detecta quebras de fluxo (resets).
    *   Insere bombas apenas em pausas longas (> 3s) para forçar resets seguros.
    *   Corrige direções de corte impossíveis (ex: corte pra cima quando a mão já está em cima).

### Treinamento e Dados
*   **`src/downloader.py`**: Ferramenta para baixar mapas rankeados do BeatSaver para criar o dataset.
*   **`src/data_loader.py`**: Lê os arquivos dos mapas (`.dat`, `.json`) e converte em tensores para treinamento.
*   **`src/preprocess_data.py`**: Otimiza o dataset, salvando os mapas processados em arquivos `.npy` para carregamento rápido.
*   **`src/train_optimized.py`**: Script de treinamento da `DirectorNet`. Usa uma loss function ponderada para equilibrar a precisão do ritmo com a classificação de estilo e verticalidade.

### Utilitários
*   **`src/youtube_downloader.py`**: Permite baixar músicas do YouTube e converter automaticamente para `.mp3` e `.egg` (OGG), facilitando a criação de mapas para músicas novas.

## 🚀 Como Usar

1.  **Instalação**: Instale as dependências com `pip install -r requirements.txt`.
    *   Certifique-se de ter o `ffmpeg` instalado no sistema ou acessível pelo script.
2.  **Obter Música**:
    *   Coloque um arquivo `musica.mp3` na raiz do projeto.
    *   OU use `python src/youtube_downloader.py` para baixar direto do YouTube.
3.  **Treinar (Opcional)**:
    *   Se não tiver o modelo `models/director_net.pth`, execute `python src/train_optimized.py` (requer dataset processado na pasta `data/processed`).
4.  **Gerar Mapa**:
    *   Execute `python src/generate_optimized.py`.
    *   O script irá analisar a música, gerar o mapa, corrigir o fluxo e salvar o resultado em `output/DirectorMap.zip`.
5.  **Jogar**:
    *   Extraia ou copie o ZIP gerado para a pasta `Beat Saber_Data/CustomLevels` do seu jogo.

---
*Projeto desenvolvido para fins educacionais e de pesquisa em geração procedural de conteúdo com Deep Learning.*
