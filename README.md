
# ConfigHagging 🖐️🤖

Este projeto é um espaço de trabalho dedicado a experimentos de **Visão Computacional**, com foco principal no **Reconhecimento de Gestos** em tempo real usando MediaPipe e Machine Learning.

## 🚀 O que este projeto faz?

O objetivo principal é capturar marcos (*landmarks*) das mãos através de uma webcam, processar esses dados e treinar um modelo capaz de classificar gestos personalizados. Além do reconhecimento de gestos, o repositório contém notebooks para detecção de objetos, segmentação de imagens e classificação geral.

### Principais etapas:
1.  **Coleta de Dados:** Gravação das coordenadas (x, y, z) de 21 pontos da mão (`coletar_dados_maos.ipynb`).
2.  **Treinamento:** Utilização de um classificador *Random Forest* para aprender os padrões dos gestos colecionados (`treinar_modelo_gestos.py`).
3.  **Classificação em Tempo Real:** Integração do modelo treinado com a captura da webcam para identificar gestos ao vivo (`classificar_gestos_webcam.py`).

## 🛠️ Ferramentas Utilizadas

O projeto utiliza tecnologias modernas de IA e processamento de imagem:

-   **[MediaPipe](https://mediapipe.dev/):** Para rastreamento de mãos (*Hand Tracking*) e detecção de objetos.
-   **[Scikit-Learn](https://scikit-learn.org/):** Para criação e treinamento do modelo de Machine Learning (*Random Forest*).
-   **[OpenCV](https://opencv.org/):** Manipulação de vídeo e exibição das janelas de captura.
-   **[PyTorch](https://pytorch.org/) & [Transformers](https://huggingface.co/docs/transformers/index):** Para experimentos avançados de segmentação e visão computacional (CLIPSeg, Timm).
-   **[Pandas](https://pandas.pydata.org/) / [Numpy](https://numpy.org/):** Manipulação e análise de dados.
-   **[uv](https://github.com/astral-sh/uv):** Gerenciamento rápido de pacotes e ambientes Python.

## 🧪 Execução

O projeto possui um ponto de entrada unificado através do arquivo `main.py`. Você pode executar as principais funcionalidades via terminal:

- **Treinar o modelo**: `python main.py train`
- **Classificar gestos (Webcam)**: `python main.py classify`
- **Rodar testes de ambiente**: `python main.py test`

> **Nota:** Para a coleta de novos gestos, utilize os notebooks na pasta `notebooks/`.

## 📂 Estrutura do Projeto

A estrutura foi organizada seguindo boas práticas de desenvolvimento para facilitar a manutenção e legibilidade:

```text
├── assets/          # Arquivos de suporte (labels, classes)
├── data/            # Datasets (CSV) e imagens coletadas
├── models/          # Modelos treinados (.pkl) e modelos MediaPipe (.task, .tflite)
├── notebooks/       # Jupyter Notebooks para experimentos e coleta de dados
├── src/             # Código-fonte principal (Scripts de treino e classificação)
├── tests/           # Scripts de validação e testes de ambiente
├── main.py          # Ponto de entrada (CLI)
└── pyproject.toml   # Gerenciamento de dependências (uv/pip)
```

