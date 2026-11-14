# Dados

## Fonte e licença

- **Dataset:** Brazilian food images, de Marcos Alberto de Carvalho (Universidade Federal de Itajubá)
- **Onde:** Mendeley Data, <https://data.mendeley.com/datasets/7n36jtcpv3/2>
- **DOI:** `10.17632/7n36jtcpv3.2` (versão 2, de 02/05/2023)
- **Licença:** [CC BY 4.0](https://creativecommons.org/licenses/by/4.0/)

A CC BY 4.0 permite usar, modificar e redistribuir, inclusive em trabalhos derivados, desde que haja
atribuição, link para a licença e indicação das mudanças. A página do dataset ressalva que conteúdo
identificado como de terceiros pode exigir permissão adicional; nenhum arquivo está identificado assim.
A página não informa quem tirou as fotos.

**Atribuição:** de Carvalho, Marcos Alberto (2023), "Brazilian food images", Mendeley Data, V2,
doi: 10.17632/7n36jtcpv3.2. Licença CC BY 4.0.

## O que este repositório faz com os dados

- As imagens **não são versionadas aqui**. `python -m src.data` baixa os arquivos pela API pública da
  Mendeley e confere o SHA-256 de cada um.
- **Mudanças em relação ao original:** cada par (foto, máscara) vira um recorte retangular do alimento,
  e os nomes das classes foram traduzidos para o português.
- `data/splits.json` guarda apenas caminhos relativos, rótulos e um identificador da foto de origem.
- Os gráficos em `results/` e os exemplos do demo contêm alguns recortes do conjunto de teste,
  redistribuídos sob a mesma CC BY 4.0 com a atribuição acima.

## Estrutura do original

Todas as imagens têm 512x512 pixels. Em `training_set/<classe>/`, `<classe>NNNN.jpg` é a foto de um
prato e `<classe>NNNN_.jpg` é a máscara binária de um alimento daquele prato. Os pratos já vêm
recortados sobre **fundo preto**.

A mesma foto aparece em várias pastas, uma vez para cada alimento anotado: são 915 pares, mas só 505
fotos distintas, e 212 delas estão em mais de uma classe. Por isso o split é feito por foto de origem.

O original também traz um `test_set/` com 160 fotos sem rótulo nem máscara, que não é usado aqui, e
pastas `.git` enviadas por engano, que o download ignora.

## Classes e split

Split estratificado por classe e agrupado por foto, seed 42: 653 recortes de treino (361 fotos),
131 de validação (72 fotos) e 131 de teste (72 fotos).

| Classe | Pasta original | Treino | Validação | Teste |
|---|---|---|---|---|
| abóbora | pumpkin | 25 | 6 | 5 |
| arroz | rice | 56 | 12 | 12 |
| batata frita | fries | 37 | 8 | 8 |
| beterraba | beet | 23 | 5 | 5 |
| bife à milanesa | milanesesteak | 25 | 5 | 5 |
| carne bovina | beef | 61 | 12 | 12 |
| cenoura | carrot | 50 | 10 | 10 |
| farofa | farofa | 24 | 5 | 5 |
| feijão | beans | 57 | 11 | 11 |
| frango | chicken | 42 | 8 | 8 |
| macarrão | noodles | 18 | 4 | 3 |
| macarrão ao molho vermelho | pastaredsauce | 21 | 5 | 4 |
| maionese de batata | mayo(potato salad) | 12 | 3 | 2 |
| mandioca | cassava | 12 | 2 | 2 |
| milho | corn | 13 | 2 | 3 |
| ovo | egg | 20 | 3 | 4 |
| purê | mashed | 22 | 4 | 5 |
| repolho roxo | cabbage | 14 | 2 | 2 |
| salada | salad | 69 | 14 | 14 |
| tomate | tomato | 52 | 10 | 11 |
