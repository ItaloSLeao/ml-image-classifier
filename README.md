# Classificador de alimentos de pratos brasileiros

![PyTorch](https://img.shields.io/badge/PyTorch-2.x-ee4c2c)
[![Modelo no Hugging Face](https://img.shields.io/badge/%F0%9F%A4%97%20modelo-aitalo%2Fconvnext--tiny--comida--brasileira-yellow)](https://huggingface.co/aitalo/convnext-tiny-comida-brasileira)
[![Abrir no Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/ItaloSLeao/br-image-classifier/blob/main/notebooks/demo_colab.ipynb)
![Licença do código](https://img.shields.io/badge/c%C3%B3digo-MIT-green)
![Licença dos dados](https://img.shields.io/badge/dados-CC%20BY%204.0-lightgrey)

Fine-tuning de um ConvNeXt-Tiny para reconhecer 20 alimentos comuns no prato brasileiro (arroz,
feijão, farofa, mandioca, bife à milanesa e outros) a partir de recortes de fotos de pratos.

O projeto cobre o ciclo inteiro: preparação dos dados com split sem vazamento, dois baselines, treino
em GPU gratuita do Kaggle, avaliação, rejeição por baixa confiança, publicação no Hugging Face Hub e
demo em notebook.

## Resultados

Conjunto de teste: 131 recortes de 72 fotos que não aparecem no treino nem na validação.

<!-- RESULTADOS:INICIO -->
| Modelo | Acurácia | F1 macro | Acurácia top-3 | Exemplos de teste |
|---|---|---|---|---|
| Classe majoritária | 10,7% | 1,0% | 28,2% | 131 |
| ConvNeXt-Tiny congelado + cabeça linear | pendente | pendente | pendente | pendente |
| ConvNeXt-Tiny com fine-tuning | pendente | pendente | pendente | pendente |

Detalhes do modelo com fine-tuning: pendente (aguardando a rodada de treino).
<!-- RESULTADOS:FIM -->

As tabelas acima são geradas por `python -m src.report` a partir dos arquivos em `results/`.
A matriz de confusão e os erros mais confiantes de cada rodada ficam em `results/<nome>_confusao.png`
e `results/<nome>_erros.png`.

## Pipeline

```
Mendeley Data ──> src/data.py ──> recortes por máscara + data/splits.json (por foto, seed 42)
                                        │
                  src/train.py ─────────┤  ConvNeXt-Tiny: fine-tuning ou só a cabeça linear
                                        │  fp16, aumento leve, parada antecipada
                  src/evaluate.py ──────┤  acurácia, F1 macro e por classe, top-3,
                                        │  matriz de confusão, erros, limiar de rejeição
                  src/report.py ────────┤  tabelas do README e do model card
                  src/push.py ──────────┘  Hugging Face Hub
                  src/predict.py           predict(imagem) -> classe, confiança, top-3, rejeitado
```

1. **Dados.** O dataset traz fotos de pratos e uma máscara por alimento. Cada par vira um recorte do
   alimento. Como a mesma foto aparece em várias classes, o split é agrupado por foto de origem.
2. **Baselines.** Classe majoritária, e o mesmo backbone congelado com uma cabeça linear.
3. **Modelo.** ConvNeXt-Tiny pré-treinado no ImageNet, com todos os pesos ajustados.
4. **Rejeição.** `predict` recusa responder quando a confiança fica abaixo de um limiar calibrado na validação.

As justificativas estão em [docs/decisoes.md](docs/decisoes.md).

## Como reproduzir

### Treino no Kaggle (GPU T4)

1. Crie um notebook no Kaggle e importe `notebooks/kaggle_train.ipynb`.
2. Em *Settings*, ligue **Accelerator: GPU T4** e **Internet**.
3. Para publicar o modelo, adicione o segredo `HF_TOKEN` em *Add-ons > Secrets* (token com permissão de escrita).
4. Rode as células em ordem. Elas clonam o repositório, instalam as dependências, preparam os dados,
   treinam os baselines e o modelo, avaliam e geram `results.zip`.
5. Baixe `results.zip`, extraia na raiz do repositório e rode `python -m src.report` para atualizar as tabelas.

### Na sua máquina (CPU, testes leves)

```bash
python -m venv .venv && source .venv/bin/activate
pip install torch==2.15.0 torchvision==0.29.2 --index-url https://download.pytorch.org/whl/cpu
pip install -r requirements.txt
pytest                              # testes rápidos, sem download
python -m src.data                  # baixa o dataset (cerca de 80 MB) e gera os recortes
python -m src.evaluate --majority   # baseline de classe majoritária
```

### Inferência

```python
from src.predict import predict

predict("recorte.jpg")
# {'classe': ..., 'confianca': ..., 'top3': [...], 'rejeitado': False}
```

O demo interativo está em `notebooks/demo_colab.ipynb` e carrega o modelo direto do Hub, sem token.

## Limitações

- **Fundo preto.** No dataset os pratos já vêm recortados sobre fundo preto. O modelo nunca viu mesa,
  toalha ou talheres, e o desempenho em fotos comuns de celular não foi medido.
- **Só recortes.** O modelo espera um alimento ocupando quase todo o quadro. Uma foto de prato inteiro
  está fora do que ele aprendeu.
- **Teste pequeno.** São 131 recortes, e cinco classes têm três exemplos de teste ou menos. Nessas
  classes um único erro muda o F1 em dezenas de pontos.
- **Uma única rodada de treino.** Uma seed, sem intervalo de confiança e sem busca de hiperparâmetros.
  Diferenças pequenas entre modelos não são conclusivas.
- **Classes confundidas.** As confusões medidas aparecem na tabela acima e na matriz de confusão.
- **Classes desbalanceadas.** De 12 a 69 recortes de treino por classe, sem reponderação.
- **Domínio das fotos.** Todas vêm de uma única fonte, com iluminação e estilo parecidos. A página do
  dataset não informa a origem das fotos.
- **A rejeição não detecta "não é comida".** O limiar mede baixa confiança e foi calibrado só com
  imagens de comida. Uma imagem de outro domínio pode passar.
- **Backbone não comparado.** A escolha do ConvNeXt-Tiny é argumentada, não medida contra alternativas.

## Dados e licenças

- **Código:** MIT ([LICENSE](LICENSE)).
- **Dados:** de Carvalho, Marcos Alberto (2023), "Brazilian food images", Mendeley Data, V2,
  doi: [10.17632/7n36jtcpv3.2](https://doi.org/10.17632/7n36jtcpv3.2), licença
  [CC BY 4.0](https://creativecommons.org/licenses/by/4.0/). As imagens não são versionadas neste
  repositório; o código baixa da fonte. Os gráficos em `results/` contêm recortes do dataset,
  redistribuídos sob a mesma licença.
- **Modelo base:** `facebook/convnext-tiny-224`, Apache-2.0.

Detalhes em [docs/datasets.md](docs/datasets.md).

## Estrutura

```
src/         data.py, train.py, evaluate.py, predict.py, report.py, push.py, modeling.py
notebooks/   kaggle_train.ipynb (treino), demo_colab.ipynb (demo)
tests/       testes rápidos de dados, métricas e inferência
data/        splits.json (as imagens são baixadas, não versionadas)
results/     métricas em JSON, matrizes de confusão e tabelas geradas
docs/        decisoes.md, datasets.md
model_card.md
```
