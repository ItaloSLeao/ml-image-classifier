# Classificador de alimentos brasileiros

Modelo para reconhecer 20 alimentos em recortes de fotos de pratos, como arroz, feijão, farofa e mandioca. Ajustei um ConvNeXt-Tiny pré-treinado no ImageNet e comparei o resultado com dois modelos de referência: classe majoritária e o mesmo ConvNeXt com apenas a camada de classificação treinada.

O [modelo publicado no Hugging Face](https://huggingface.co/aitalo/convnext-tiny-comida-brasileira) retorna a classe, a confiança e as três previsões mais prováveis. Quando a confiança fica abaixo do limiar definido na validação, a previsão é rejeitada. O projeto foi desenvolvido para estudo e portfólio.

## Como usar

Envie um recorte em que **um alimento ocupa quase todo o quadro**. O modelo atribui uma única classe por imagem; ele não separa os alimentos de um prato inteiro.

### Demonstração no Colab

Abra a [demonstração no Colab](https://colab.research.google.com/github/ItaloSLeao/ml-image-classifier/blob/main/notebooks/demo_colab.ipynb) e execute as células em ordem. O notebook baixa o modelo e abre uma interface com imagens de exemplo. Não é necessário token nem GPU.

### Na sua máquina

Execute na raiz do repositório. As dependências foram testadas localmente com Python 3.14 e CPU:

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install --index-url https://download.pytorch.org/whl/cpu torch==2.15.0 torchvision==0.29.2
pip install -r requirements.txt
python -m src.predict recorte.jpg
```

Substitua `recorte.jpg` pelo caminho da imagem. Na primeira chamada, o modelo e o limiar são baixados do Hugging Face. Para usar em Python:

```python
from src.predict import predict

resultado = predict("recorte.jpg")
print(resultado)
```

A resposta contém `classe`, `confianca`, `top3` e `rejeitado`. Se a previsão for rejeitada, `classe` será `None` no Python e `null` na saída JSON. O limiar mede baixa confiança e não garante a detecção de imagens fora das 20 classes.

## Como reproduzir

### Treino no Kaggle

1. Importe [notebooks/kaggle_train.ipynb](notebooks/kaggle_train.ipynb) em um notebook do Kaggle.
2. Habilite GPU T4 e internet nas configurações.
3. Execute as células em ordem. Elas clonam o repositório, preparam os dados, treinam e avaliam os modelos e geram `results.zip`.
4. Baixe `results.zip`, extraia na raiz do repositório local e execute `python -m src.report` para atualizar as tabelas.

O notebook mantém o PyTorch com CUDA da imagem do Kaggle. As versões usadas no treino ficam registradas em `results/treino_<nome>.json`.

A publicação no Hugging Face é opcional. Para executar essa etapa, adicione o segredo `HF_TOKEN` no Kaggle com um token de escrita.

### Dados e avaliação local

Depois de instalar as dependências, execute:

```bash
pytest                              # testes rápidos, sem baixar dados ou modelos
python -m src.data                  # baixa cerca de 80 MB e gera os recortes
python -m src.evaluate --majority   # avalia a classe majoritária
```

O dataset traz fotos de pratos com uma máscara para cada alimento. Cada par de foto e máscara gera um recorte retangular com 5% de folga, preservando o conteúdo dentro da caixa. Como a mesma foto pode gerar vários recortes, separei os conjuntos por foto de origem para impedir que partes do mesmo prato apareçam no treino e no teste.

A divisão está em [data/splits.json](data/splits.json), com seed 42: 653 recortes de treino, 131 de validação e 131 de teste. Os testes verificam a separação entre fotos e a presença das classes nos três conjuntos.

O treino usa aumento de dados e parada antecipada, escolhendo o checkpoint pelo F1 macro na validação. Depois, a avaliação calcula as métricas e define o limiar de rejeição a partir da confiança dos acertos de validação. O teste não participa da escolha desse limiar. Os detalhes estão em [docs/decisoes.md](docs/decisoes.md).

## Resultados

Os três modelos foram avaliados nos mesmos 131 recortes de teste, provenientes de 72 fotos ausentes do treino e da validação.

<!-- RESULTADOS:INICIO -->
| Modelo | Acurácia | F1 macro | Acurácia top-3 | Exemplos de teste |
|---|---|---|---|---|
| Classe majoritária | 10,7% | 1,0% | 28,2% | 131 |
| ConvNeXt-Tiny congelado + cabeça linear | 77,9% | 76,0% | 94,7% | 131 |
| ConvNeXt-Tiny com fine-tuning | 94,7% | 93,7% | 99,2% | 131 |

### F1 por classe (modelo com fine-tuning)

| Classe | Precisão | Revocação | F1 | Exemplos de teste |
|---|---|---|---|---|
| macarrão | 100,0% | 66,7% | 80,0% | 3 |
| repolho roxo | 66,7% | 100,0% | 80,0% | 2 |
| frango | 87,5% | 87,5% | 87,5% | 8 |
| beterraba | 100,0% | 80,0% | 88,9% | 5 |
| bife à milanesa | 100,0% | 80,0% | 88,9% | 5 |
| ovo | 80,0% | 100,0% | 88,9% | 4 |
| purê | 100,0% | 80,0% | 88,9% | 5 |
| batata frita | 88,9% | 100,0% | 94,1% | 8 |
| cenoura | 100,0% | 90,0% | 94,7% | 10 |
| feijão | 100,0% | 90,9% | 95,2% | 11 |
| tomate | 91,7% | 100,0% | 95,7% | 11 |
| arroz | 92,3% | 100,0% | 96,0% | 12 |
| carne bovina | 92,3% | 100,0% | 96,0% | 12 |
| abóbora | 100,0% | 100,0% | 100,0% | 5 |
| farofa | 100,0% | 100,0% | 100,0% | 5 |
| macarrão ao molho vermelho | 100,0% | 100,0% | 100,0% | 4 |
| maionese de batata | 100,0% | 100,0% | 100,0% | 2 |
| mandioca | 100,0% | 100,0% | 100,0% | 2 |
| milho | 100,0% | 100,0% | 100,0% | 3 |
| salada | 100,0% | 100,0% | 100,0% | 14 |

### Confusões mais frequentes no teste

| Classe real | Predita como | Vezes |
|---|---|---|
| purê | ovo | 1 |
| macarrão | batata frita | 1 |
| frango | arroz | 1 |
| feijão | carne bovina | 1 |
| cenoura | tomate | 1 |

### Rejeição por baixa confiança

Limiar de confiança: 0,265 (quantil 0.05 da confiança dos acertos de validação).

| Conjunto | Cobertura | Acurácia nos aceitos | Erros rejeitados | Acertos rejeitados |
|---|---|---|---|---|
| Validação (n=131) | 92,4% | 95,9% | 37,5% | 5,7% |
| Teste (n=131) | 97,7% | 95,3% | 14,3% | 1,6% |

### Treino

14 épocas rodadas (melhor: 8), 1,1 min em Tesla T4, fp16 ligado, seed 42, uma única rodada.
<!-- RESULTADOS:FIM -->

As tabelas são geradas por `python -m src.report` a partir dos arquivos em [results/](results/). O script também atualiza [results/comparacao.md](results/comparacao.md) e [model_card.md](model_card.md). As matrizes de confusão e as imagens de erros ficam em `results/<nome>_confusao.png` e `results/<nome>_erros.png`.

Cobertura é a fração de previsões aceitas pelo limiar. A acurácia nos aceitos considera apenas essa fração; erros e acertos rejeitados indicam quanto de cada grupo foi descartado. No teste, o limiar rejeitou 14,3% dos erros, então a maioria das previsões incorretas continuou sendo aceita.

A rodada de fine-tuning usou PyTorch 2.11.0+cu130, torchvision 0.26.0+cu130 e transformers 5.23.0. Esses dados e o histórico por época estão em [results/treino_convnext-tiny-finetune.json](results/treino_convnext-tiny-finetune.json).

## Limitações

As fotos do dataset têm os pratos recortados sobre fundo preto. O desempenho em fotos comuns de celular, com mesa, talheres e outros fundos, não foi medido. A entrada esperada é um recorte de um alimento; fotos de pratos inteiros estão fora do cenário de treino.

O teste tem apenas 131 recortes, e cinco classes têm três exemplos ou menos. Nessas classes, um único erro muda bastante a métrica. O treino também é desbalanceado, com 12 a 69 recortes por classe, sem reponderação ou reamostragem.

Foi feita uma única rodada, com seed 42, sem intervalo de confiança nem busca de hiperparâmetros. O ConvNeXt-Tiny não foi comparado com outros modelos de arquitetura diferente. Diferenças pequenas entre resultados precisam ser interpretadas nesse contexto.

Todas as fotos vêm de uma única fonte, com iluminação e estilo parecidos; a página do dataset não informa a origem das fotos. O limiar foi calibrado apenas com alimentos desse conjunto. Uma imagem de outro domínio pode receber confiança alta e ser aceita.

## Dados e licenças

O código está sob licença [MIT](LICENSE). O modelo base é o `facebook/convnext-tiny-224`, sob Apache-2.0.

Os dados vêm de *Brazilian food images*, de Carvalho, Marcos Alberto (2023), Mendeley Data, V2, DOI [10.17632/7n36jtcpv3.2](https://doi.org/10.17632/7n36jtcpv3.2), sob [CC BY 4.0](https://creativecommons.org/licenses/by/4.0/).

As imagens originais não são versionadas neste repositório; o código as baixa da fonte. Neste projeto, os pares de foto e máscara foram convertidos em recortes e os nomes das classes foram traduzidos para o português. Os gráficos em `results/` e as imagens de exemplo contêm recortes redistribuídos sob a mesma licença, com a atribuição acima.

A descrição do conjunto, as classes e a divisão dos dados estão em [docs/datasets.md](docs/datasets.md).
