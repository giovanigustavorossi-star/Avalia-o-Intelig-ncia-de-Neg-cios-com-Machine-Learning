# Desafio Final - Machine Learning

Projeto final da pós-graduação em Machine Learning.

O cenário é o de uma instituição financeira que precisa de três sistemas analíticos para apoiar
campanhas de marketing, a partir da base `clientes.csv` (1000 clientes):

1. **Predição de renda** - estimar a renda mensal do cliente (regressão)
2. **Classificação da categoria** - prever o nível do cliente (classificação multi-classe)
3. **Segmentação** - agrupar clientes em perfis naturais, sem rótulo prévio (clusterização)

## Estrutura do repositório

```
pos-ml-desafio/
├── datasets/
│   └── clientes.csv
├── desafio_1_renda.ipynb
├── desafio_2_categoria.ipynb
├── desafio_3_segmentacao.ipynb
└── README.md
```

## Adaptações em relação ao enunciado

O `clientes.csv` disponibilizado tem 26 colunas com nomes diferentes dos citados na tabela do
enunciado. Fiz o de-para abaixo e mantive a lógica pedida em cada etapa. Cada notebook explica
a adaptação na própria introdução.

| Enunciado | Coluna usada |
|---|---|
| `Tempo_Emprego` | `Tempo na Empresa` |
| `Dependentes` | `Pessoas em Casa` |
| `Score_Credito` | `Score` |
| `Estado_Civil` | `Estado Civil` |
| `Escolaridade` | não existe (uso `Cidade`, `Moradia` e demais cadastrais) |
| `Gasto_Mensal_Cartao` | não existe (o arquivo traz notas de preferência de consumo de 1 a 5) |
| `Categoria_Cliente` | nível extraído de `Principal Cartão`: Basic / Select / Elite / Platinum |

Duas decisões que vale destacar:

- **Desafio 2:** o enunciado proíbe usar a `Renda` porque ela trivializaria o problema.
  Nesta base a variável com esse papel é o **`Score`** — as faixas de score são exatamente os
  cortes entre os níveis do cartão (Basic 0-100, Select 101-199, Elite 201-398, Platinum 404+).
  Removi o `Score` e mantive a `Renda`, seguindo o espírito da regra. A verificação está na
  seção 1.4 do notebook.
- **Desafio 3:** o silhouette aponta **k = 2**, fora da faixa de 3 a 6 sugerida como
  referência. Testei várias combinações de 5 features numéricas e o resultado se repetiu em
  todas — é característica dos dados, não escolha de modelagem. Como o critério pedido é
  maximizar o silhouette, segui o resultado do Optuna e incluí uma análise complementar com
  k = 5 na seção 4.3.

## Como executar

Rodei tudo no Python 3.12. Funciona tanto no Jupyter local quanto no Google Colab.

```bash
pip install scikit-learn optuna joblib jupyter pandas matplotlib seaborn
jupyter notebook
```

Abra cada notebook e execute as células **na ordem, de cima para baixo**. Os notebooks leem o
dataset do caminho relativo `datasets/clientes.csv`, então é importante abrir o Jupyter a
partir da raiz do projeto.

No Colab, faça o upload da pasta `datasets/` para o mesmo diretório do notebook antes de rodar.

## O que cada notebook faz e o que gera

### `desafio_1_renda.ipynb` - Predição de Renda

Análise exploratória (histogramas, matriz de correlação, ANOVA por variável categórica),
pré-processamento com One-Hot dentro de um Pipeline, regressão linear, avaliação com
RMSE / MAE / R², validação cruzada de 5 folds, análise de resíduos e tabela de coeficientes.

Com as variáveis cruas a regressão linear chega apenas a R² 0,47, porque só o `Score` tem
correlação relevante com a renda. Aplicando `PolynomialFeatures(degree=2)` nas 5 variáveis
financeiras, o R² sobe para 0,86 — o modelo continua sendo `LinearRegression`, o que muda é o
conjunto de features.

- Métricas obtidas: **R² 0,86** no teste, RMSE ≈ R$ 5.550, MAE ≈ R$ 4.250, R² médio em
  validação cruzada 0,83
- Saída: **`modelo_renda.pkl`**

### `desafio_2_categoria.ipynb` - Classificação da Categoria

Distribuição das classes, boxplots por categoria, ANOVA e qui-quadrado, verificação do
vazamento do `Score`, split estratificado, Random Forest baseline comparado com um modelo
tunado via GridSearchCV (`n_estimators`, `max_depth`, `min_samples_leaf` e `class_weight`),
validação cruzada estratificada com `f1_macro`, classification report, matriz de confusão e
feature importance.

As classes são muito desbalanceadas (Basic 660 clientes, Platinum 28), e o `class_weight`
escolhido pelo GridSearchCV é o que explica o ganho de cerca de 10 pontos sobre o baseline.

- Métricas obtidas: **f1_macro 0,82** no teste (baseline 0,73), acurácia 0,87
- Saída: **`modelo_categoria.pkl`**

### `desafio_3_segmentacao.ipynb` - Segmentação de Clientes

Clusterização com K-Means usando 5 features numéricas (`Renda`, `Investimentos`, `Score`,
`Probabilidade Inadimplencia` e `Dívidas`). O número de clusters é otimizado com
**Optuna + GridSampler** sobre `range(2, 11)`, maximizando o `silhouette_score`. Inclui perfil
de cada cluster (médias e tamanho), nomes de negócio e análise complementar com k = 5.

Extra implementado: projeção **PCA 2D** dos clusters (só para visualização, não entra no
pipeline salvo).

- Resultado obtido: **k = 2** com **silhouette 0,30**
- Segmentos encontrados: Bons Pagadores (30% da base) e Alto Risco (70%)
- Saída: **`modelo_segmentacao.pkl`** (Pipeline completo: ColumnTransformer + KMeans)

## Observações

- Os arquivos `.pkl` são gerados ao executar os notebooks, por isso não estão versionados aqui.
- Todos os modelos usam `random_state=42` para os resultados serem reproduzíveis.
- Encoders e scalers são ajustados dentro de Pipelines, sempre depois do split treino/teste,
  para evitar vazamento de dados.

## Bibliotecas usadas

pandas, numpy, matplotlib, seaborn, scikit-learn, scipy, optuna e joblib.
