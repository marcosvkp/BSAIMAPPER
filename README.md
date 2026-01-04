# Beat Saber AI Mapper

Este projeto visa criar uma IA capaz de gerar mapas de Beat Saber baseados em música.

## Objetivos

1.  **Treinamento com Mapas Rankeados**: Utilizar mapas da comunidade (BeatSaver) para aprender padrões e fluxo.
2.  **Processamento de Áudio**:
    *   Detectar BPM.
    *   Adicionar 3 segundos de silêncio no início e no fim da música.
3.  **Geração de Mapas**: Criar arquivos de nível compatíveis com Beat Saber.

## Estrutura do Projeto

*   `data/`: Armazenamento de mapas baixados e processados.
*   `src/`: Código fonte.
    *   `audio_processor.py`: Manipulação de áudio (BPM, silêncio).
    *   `data_loader.py`: Download e parsing de mapas.
    *   `model.py`: Definição do modelo de IA.
    *   `train.py`: Script de treinamento.
    *   `generate.py`: Script de geração.

## Como usar

(Instruções futuras)
