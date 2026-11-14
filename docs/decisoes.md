# Decisões de projeto

## Tarefa: classificar recortes, não fotos inteiras

O dataset traz fotos de pratos com vários alimentos e uma máscara por alimento. Uma foto inteira não
tem uma classe só, então cada par (foto, máscara) vira um recorte: a caixa que envolve a máscara, com
5% de folga. O recorte é retangular e mantém o que estiver dentro da caixa, inclusive pedaços de
alimentos vizinhos. Não apago o que está fora da máscara porque na inferência não existe máscara.

Consequência: o modelo espera uma imagem em que **um alimento ocupa quase todo o quadro**. Uma foto de
prato inteiro está fora do que ele aprendeu.

## Split agrupado por foto

A mesma foto aparece em várias classes. Um split aleatório por recorte colocaria pedaços do mesmo
prato no treino e no teste, com a mesma iluminação e a mesma louça, e inflaria a métrica. O split usa
`StratifiedGroupKFold` do scikit-learn com o hash da foto como grupo e seed 42: primeiro separa 1/7
para teste, depois 1/6 do restante para validação. O resultado fica em `data/splits.json`, versionado,
e `check_splits` falha se uma foto aparecer em dois conjuntos ou se uma classe faltar em algum deles.

## Backbone: ConvNeXt-Tiny

Escolhi `facebook/convnext-tiny-224` (pré-treinado no ImageNet-1k, licença Apache-2.0) em vez do
ViT-Base por três motivos:

- **Poucos dados.** São 653 recortes de treino. O ConvNeXt-Tiny tem cerca de 28 milhões de parâmetros
  contra cerca de 86 milhões do ViT-Base, e redes convolucionais costumam precisar de menos dados para
  ajustar bem, por já embutirem localidade e invariância a translação.
- **Custo.** É cerca de três vezes menor, treina com folga dentro de 30 minutos numa T4 e roda em CPU
  no demo.
- **O sinal é textura e cor.** Distinguir arroz, farofa e purê depende de padrões locais, que é o
  ponto forte de convoluções.

Não comparei os dois backbones experimentalmente; a escolha é por argumento, não por medição.

## Pré-processamento

A imagem é redimensionada direto para 224x224 (bicúbica), sem corte central, e normalizada com média e
desvio do ImageNet. O corte central padrão do ConvNeXt descartaria as bordas de recortes que já estão
justos no alimento. O redimensionamento direto distorce a proporção de recortes alongados. O mesmo
processador é salvo junto com o modelo e usado na validação, no teste e na inferência.

## Treino

| Item | Valor |
|---|---|
| Otimizador | AdamW, weight decay 0,05 |
| Taxa de aprendizado | 1e-4 no fine-tuning, 1e-3 na cabeça linear; uma época de aquecimento e depois cosseno |
| Perda | entropia cruzada com label smoothing 0,1 |
| Batch | 32 |
| Épocas | até 30, com parada antecipada após 6 épocas sem melhora |
| Critério do melhor checkpoint | F1 macro na validação |
| Precisão mista | fp16 com `torch.autocast` quando há GPU |
| Aumento de dados | recorte aleatório redimensionado (60% a 100% da área), espelhamento horizontal e vertical, variação leve de brilho, contraste e saturação |

- O critério é o F1 macro, e não a acurácia, porque as classes são desbalanceadas (de 12 a 69 recortes
  de treino) e a acurácia premiaria acertar só as classes grandes.
- O aumento de cor é leve de propósito: a cor separa classes como cenoura, beterraba e repolho roxo.
  Não há variação de matiz.
- Não uso pesos por classe nem reamostragem. É uma simplificação; o efeito do desbalanceamento aparece
  no F1 por classe.
- Os hiperparâmetros são valores usuais para fine-tuning de ConvNeXt e **não foram ajustados** por busca.

## Baselines

1. **Classe majoritária:** prevê sempre a classe mais frequente do treino. No top-3, as três mais frequentes.
2. **Backbone congelado + cabeça linear:** o mesmo ConvNeXt-Tiny com todos os pesos fixos, treinando só
   a camada de classificação (15.380 parâmetros). Mede quanto as features do ImageNet já resolvem.

Os três são avaliados no mesmo conjunto de teste pelo mesmo código (`src/evaluate.py`).

## Limiar de rejeição

`predict` devolve `rejeitado: true` e `classe: null` quando a maior probabilidade fica abaixo de um limiar.

**Como o limiar é escolhido:** `src/evaluate.py` calcula a confiança (maior probabilidade softmax) de
cada exemplo de validação e toma o quantil 5% da confiança **dos acertos**. Ou seja, o limiar é o
maior valor que ainda aceita cerca de 95% das predições corretas na validação. Ele é gravado em
`limiar.json`, junto do modelo, com a cobertura e a fração de erros rejeitados na validação. As mesmas
estatísticas são medidas depois no teste, que não participa da escolha.

**O que ele não é:** um detector de "não é comida". O limiar foi calibrado só com imagens de comida do
próprio dataset, porque não há um conjunto de imagens fora do domínio com licença verificada neste
projeto. Redes neurais podem dar confiança alta para imagens que nunca viram, então uma foto de carro
pode passar pelo limiar. Além disso, a validação tem só 131 exemplos, então o quantil é uma estimativa
ruidosa.

## Dependências

`requirements.txt` fixa as versões usadas nos testes locais. No Kaggle, o notebook instala só
`transformers` e `huggingface_hub` nessas versões e mantém o `torch` e o `torchvision` da imagem,
porque reinstalar o PyTorch demora e pode quebrar a compatibilidade com o CUDA da máquina. As versões
realmente usadas em cada treino ficam registradas em `results/treino_<nome>.json`.

## Resultados sempre vêm de arquivos

Nenhuma métrica é digitada à mão. `src/report.py` lê `results/*.json` e reescreve as tabelas do README
e do model card; rodada que ainda não existe aparece como "pendente".
