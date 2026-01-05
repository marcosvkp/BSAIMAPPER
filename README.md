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
*   **`src/model.py`**: Define a arquitetura da Rede Neural (CNN + LSTM). A CNN processa as características visuais do espectrograma do áudio, enquanto a LSTM entende a sequência temporal e o contexto musical.
*   **`src/audio_processor.py`**: O "ouvido" da IA. Usa `librosa` e `ffmpeg` para:
    *   Detectar o BPM da música.
    *   Gerar Mel Spectrograms (representação visual do som).
    *   Criar grids rítmicos.
    *   Normalizar e adicionar silêncio (intro/outro) ao áudio.
*   **`src/generate.py`**: O script final de uso.
    *   Carrega o modelo treinado.
    *   Processa uma música nova.
    *   Gera as notas e aplica algoritmos complexos de pós-processamento (regras de fluxo, cooldowns, estilos).
    *   Empacota tudo em um arquivo ZIP pronto para a pasta `CustomLevels` do Beat Saber.

### Treinamento e Dados
*   **`src/downloader.py`**: Ferramenta para baixar milhares de mapas rankeados do BeatSaver, criando a base de conhecimento da IA.
*   **`src/data_loader.py`**: Lê os arquivos complexos dos mapas (`.dat`, `.json`) e os converte em matrizes matemáticas que a IA consegue entender (Features de Áudio vs. Posição das Notas).
*   **`src/preprocess_data.py`**: Otimiza o treinamento. Processa todos os mapas baixados de uma vez, salvando-os em arquivos binários `.npy` para que o treinamento seja rápido e eficiente.
*   **`src/train.py`**: O "professor". Gerencia o ciclo de aprendizado da IA, ajustando os pesos da rede neural para minimizar erros e maximizar a diversidade e precisão dos mapas gerados.

### Utilitários
*   **`src/youtube_downloader.py`**: Facilita a vida do usuário, permitindo baixar músicas diretamente do YouTube e convertê-las automaticamente para os formatos necessários para gerar um mapa.

## 🚀 Como Usar (Básico)

1.  **Instalação**: Instale as dependências com `pip install -r requirements.txt`.
2.  **Obter Música**: Coloque um arquivo `musica.mp3` na raiz ou use o `src/youtube_downloader.py`.
3.  **Gerar Mapa**: Execute `src/generate.py`.
    *   Você pode configurar o estilo (Standard, Speed, Tech) editando a variável `TARGET_STYLE` no final do arquivo.
4.  **Jogar**: Pegue o arquivo ZIP gerado na pasta `output/` e coloque na pasta de mapas do seu Beat Saber.

---
*Projeto desenvolvido para fins educacionais e de pesquisa em geração procedural de conteúdo com Deep Learning.*
