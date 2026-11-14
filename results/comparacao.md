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
