Desafio Final - Machine Learning
Projeto final da pós-graduação em Machine Learning.
O cenário é o de uma instituição financeira que precisa de três sistemas analíticos para apoiar
campanhas de marketing, a partir da base `clientes.csv`. São três notebooks independentes:
predição de renda (regressão), classificação da categoria do cliente (classificação
multi-classe) e segmentação de clientes (clusterização).
> **Observação:** o `clientes.csv` disponibilizado tem colunas com nomes diferentes dos citados
> no enunciado. O de-para e as decisões que precisei tomar por causa disso estão explicados na
> introdução de cada notebook.
Como executar
```bash
pip install scikit-learn optuna joblib jupyter pandas matplotlib seaborn
jupyter notebook
```
Abra cada notebook no Jupyter ou no Google Colab e execute todas as células em ordem, de cima
para baixo. Os notebooks leem o dataset do caminho `datasets/clientes.csv`, então é preciso
abrir o Jupyter a partir da raiz do projeto. No Colab, faça o upload da pasta `datasets/` para
o mesmo diretório do notebook antes de rodar.
O que cada notebook gera
`desafio_1_renda.ipynb` - Predição de Renda
Regressão linear para estimar a renda mensal do cliente.
Métricas: R² 0,86 no teste, RMSE ≈ R$ 5.550, MAE ≈ R$ 4.250, R² médio de 0,83 na validação
cruzada de 5 folds
Saída: `modelo_renda.pkl`
`desafio_2_categoria.ipynb` - Classificação da Categoria
Random Forest para prever o nível do cliente (Basic / Select / Elite / Platinum), com baseline
e tuning via GridSearchCV.
Métricas: f1_macro 0,82 no teste (baseline 0,73), acurácia 0,87
Saída: `modelo_categoria.pkl`
`desafio_3_segmentacao.ipynb` - Segmentação de Clientes
K-Means com o número de clusters otimizado via Optuna, maximizando o silhouette score.
Inclui visualização dos clusters com PCA 2D.
Resultado: k = 2 com silhouette 0,30
Saída: `modelo_segmentacao.pkl` (Pipeline completo: ColumnTransformer + KMeans)
Os arquivos `.pkl` são gerados ao executar os notebooks e não estão versionados no repositório.
