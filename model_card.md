---
license: apache-2.0
language:
- pt
library_name: transformers
pipeline_tag: image-classification
base_model: facebook/convnext-tiny-224
datasets:
- brazilian-food-images
metrics:
- accuracy
- f1
tags:
- image-classification
- food
- brazilian-food
- convnext
- vision
---

# ConvNeXt-Tiny para alimentos de pratos brasileiros

Fine-tuning do [`facebook/convnext-tiny-224`](https://huggingface.co/facebook/convnext-tiny-224) para
classificar **recortes de alimentos** em 20 classes comuns no prato do dia a dia brasileiro: abóbora,
arroz, batata frita, beterraba, bife à milanesa, carne bovina, cenoura, farofa, feijão, frango,
macarrão, macarrão ao molho vermelho, maionese de batata, mandioca, milho, ovo, purê, repolho roxo,
salada e tomate.

É um projeto de portfólio, treinado com 653 imagens. Não é um modelo para uso em produção.

## Uso

```python
from transformers import pipeline

clf = pipeline("image-classification", model="aitalo/convnext-tiny-comida-brasileira")
clf("recorte_do_alimento.jpg", top_k=3)
```

Com rejeição por baixa confiança, usando o limiar publicado em `limiar.json`:

```python
import json
import torch
from huggingface_hub import hf_hub_download
from PIL import Image
from transformers import AutoImageProcessor, AutoModelForImageClassification

repo = "aitalo/convnext-tiny-comida-brasileira"
model = AutoModelForImageClassification.from_pretrained(repo).eval()
processor = AutoImageProcessor.from_pretrained(repo)
limiar = json.load(open(hf_hub_download(repo, "limiar.json")))["limiar"]

image = Image.open("recorte_do_alimento.jpg").convert("RGB")
with torch.no_grad():
    probs = model(**processor(image, return_tensors="pt")).logits[0].softmax(-1)
conf, idx = probs.max(0)
print("rejeitado" if conf < limiar else model.config.id2label[idx.item()], float(conf))
```

A entrada esperada é uma imagem em que um único alimento ocupa quase todo o quadro.

## Dados de treino

[Brazilian food images](https://data.mendeley.com/datasets/7n36jtcpv3/2) (Mendeley Data): fotos de
pratos com máscara de segmentação por alimento. Cada par (foto, máscara) foi transformado em um
recorte retangular do alimento, o que dá 915 recortes de 505 fotos. O split é estratificado por classe
e agrupado por foto de origem, para que um mesmo prato não apareça em dois conjuntos: 653 recortes de
treino, 131 de validação e 131 de teste.

## Treino

- Entrada 224x224, redimensionada sem corte central
- AdamW, taxa de aprendizado 1e-4 com aquecimento e decaimento em cosseno, weight decay 0,05, label smoothing 0,1
- Batch 32, até 30 épocas, parada antecipada pelo F1 macro de validação
- Precisão mista (fp16) em uma GPU T4
- Aumento de dados leve: recorte aleatório, espelhamento, pequena variação de brilho, contraste e saturação

## Resultados

Medidos no conjunto de teste, comparando com dois baselines avaliados pelo mesmo código.

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

## Limitações

- **Fundo preto.** No dataset os pratos já vêm recortados sobre fundo preto. O modelo nunca viu mesa,
  toalha ou talheres, e o desempenho em fotos comuns de celular não foi medido.
- **Só recortes.** Uma foto de prato inteiro, com vários alimentos, está fora do que o modelo aprendeu.
- **Teste pequeno.** São 131 recortes de 72 fotos, e cinco classes têm três exemplos de teste ou menos.
  As métricas por classe mudam muito com um único erro.
- **Uma única rodada.** Um treino, uma seed, sem intervalo de confiança e sem busca de hiperparâmetros.
- **Classes desbalanceadas.** De 12 a 69 recortes de treino por classe.
- **Domínio das fotos.** Todas vêm de uma única fonte, com estilo visual parecido, e a página do
  dataset não informa a origem das fotos.
- **A rejeição não detecta "não é comida".** O limiar foi calibrado só com imagens de comida do próprio
  dataset. Uma imagem de outro domínio pode receber confiança alta.
- **Só 20 classes.** Qualquer outro alimento será classificado como uma delas ou rejeitado.

## Licenças

- **Pesos:** Apache-2.0, a mesma do modelo base.
- **Dados de treino:** de Carvalho, Marcos Alberto (2023), "Brazilian food images", Mendeley Data, V2,
  doi: [10.17632/7n36jtcpv3.2](https://doi.org/10.17632/7n36jtcpv3.2), licença
  [CC BY 4.0](https://creativecommons.org/licenses/by/4.0/). As imagens foram recortadas pelas máscaras
  e os nomes das classes traduzidos. As imagens em `exemplos/` e nos gráficos de `results/` são
  recortes desse dataset, redistribuídos sob a mesma licença. A página do dataset não informa quem
  tirou as fotos.
