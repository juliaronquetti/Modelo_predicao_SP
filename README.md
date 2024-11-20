# Modelo_predicao_SP

#### [Pré-processamento](https://colab.research.google.com/github/juliaronquetti/Modelo_predicao_SP/blob/main/1_imoveis_preprocess.ipynb)
- Anonimizar dados (extrair informações de URL)
- Renomear arquivos e nomes de campos

## Dados

O conjunto de dados fornece informações para a previsão de preços de imóveis. Ele contém dados de 23.433 imóveis à venda na cidade de São Paulo (Brasil), anunciados durante o mês de abril de 2024. Os dados incluem o preço, mais de 30 características dos imóveis (m², número de quartos, garagem, etc.) e imagens dos anúncios, totalizando 605K imagens.

<br>
<img src="https://github.com/juliaronquetti/Modelo_predicao_SP/blob/main/imagens/fotos_thumbs.png?raw=true" width="640">
<br>

## [Preparação dos dados](https://colab.research.google.com/github/juliaronquetti/Modelo_predicao_SP/blob/main/2_imoveis_data_preparation.ipynb)
- Exploração dos dados
- Limpeza de dados nulos e inconsistentes
- Limpeza de alguns outliers
- Conversões de dados
- Codificação de dados (exceto o texto 'título')

## [Regressão Linear](https://colab.research.google.com/github/juliaronquetti/Modelo_predicao_SP/blob/main/3_imoveis_linear_regression.ipynb)
- Aplicar diferentes modelos lineares, utilizando apenas campos numéricos para a estimativa de preço
  - Modelo linear simples
  - Logaritmo dos campos numéricos ('price', 'area_util')
  - Logaritmo dos campos numéricos + iterações
  - Todos os casos são filtrados com coeficientes de regressão significativos

<br>

- **Melhor resultado**

| Model                   | R2	 | RMSE	      | MAE	      | MedAE	    | MAPE |
|-------------------------|------|------------|-----------|-----------|------|
| Linear log, inter (all) |	0.63 | 402047.49  | 210913.13	| 127497.35 |**0.22**|

[resultados detalhados](https://github.com/juliaronquetti/Modelo_predicao_SP/blob/main/resultados_detalhados/imoveis_results_linear_regressions.csv)


<br>
<br>
<img src="https://github.com/juliaronquetti/Modelo_predicao_SP/blob/main/imagens/imoveis_results_linear_regressions.png?raw=true" width="640">

<br>

As seguintes interações foram selecionadas e adicionadas ao modelo de regressão com coeficientes significativos:
`area util` $\times$ `quartos`, `area util` $\times$ `quartos`, `area util` $\times$ `categoria casas`, `area util` $\times$ `price`

## [Seleção Modelo ML](https://colab.research.google.com/github/juliaronquetti/Modelo_predicao_SP/blob/main/4_imoveis_select_ML_models.ipynb)
- Aplicar e selecionar entre diferentes modelos lineares (apenas campos numéricos são utilizados para estimativa de preço)
  
<br>

- **Melhores resultados**

| Model                   | R2	 | RMSE	      | MAE	      | MedAE	    | MAPE |
|-------------------------|------|------------|-----------|-----------|------|
| Gradient Boosting (all)     |	0.99 | 71283.11	  | 50316.73	| 35095.12	| 0.07 |
| **Random Forest     (all)**	| 0.96 | 130296.62	| 88421.16	| 57367.88	|**0.10**|
| K-Nearest Neighbors (all)	| 0.67	| 375741.02	| 260570.99 | 	172230.00	| 0.31 |

[Resultados detalhados](https://github.com/juliaronquetti/Modelo_predicao_SP/blob/main/resultados_detalhados/imoveis_results_ML_selection.csv)

<br>


Foi utilizada uma validação cruzada de 5 vezes (5-fold). Os 3 melhores resultados foram exibidos. Árvores de decisão apresentaram MAPE = 0, mas com grande overfitting, portanto foram excluídas. Regressões lineares e suas variantes (Ridge Regression, HuberRegressor, etc.) apresentaram resultados melhores do que K-Nearest Neighbors, com MAPE em torno de 0,30, mas foram excluídas, pois um resultado melhor, em torno de 0,22, foi obtido anteriormente com regressão linear com transformação logarítmica e interações. Random Forest foi selecionado, apesar de um MAPE um pouco menor do que o Gradient Boosting, pois mostra melhores resultados (RMSE) nos dados de teste (aqui, os resultados são sobre todos os dados). Para todas as execuções a seguir, foram utilizados Random Forests com 1.000 árvores e sem poda.

## [RandomForest: atributos numéricos](https://colab.research.google.com/github/juliaronquetti/Modelo_predicao_SP/blob/main/5_imoveis_ML_best_model_numeric.ipynb)
- Apply RandomForestRegression, only numeric fields are used for price estimation

<br>

- **Melhores resultados**

| Model                   | R2	 | RMSE	      | MAE	      | MedAE	    | MAPE |
|-------------------------|------|------------|-----------|-----------|------|
| RF Numeric Fields (all) |	0.98 | 95300.42	| 57290.74 |	29912.10 |**0.07** |

[Resultados detalhados](https://github.com/juliaronquetti/Modelo_predicao_SP/blob/main/resultados_detalhados/imoveis_results_ML_numeric.csv)

<br>
<br>
<img src="https://github.com/juliaronquetti/Modelo_predicao_SP/blob/main/imagens/imoveis_results_ML_numeric.png?raw=true" width="640">

O modelo de aprendizado de máquina selecionado apresenta um ganho muito significativo, com um erro médio de 7% (MAPE) na estimativa de preços, em comparação com 22% para o melhor modelo linear. Esse erro é muito próximo do encontrado na literatura [1][2].

## [RandomForest: numéricos + Yolo](https://colab.research.google.com/github/juliaronquetti/Modelo_predicao_SP/blob/main/6_imoveis_ML_best_model_numeric_yolo.ipynb)
- Aplicar RandomForestRegression, utilizando campos numéricos e objetos detectados pelo Yolo para estimativa de preço

<br>

- **Melhor resultado**

| Model                   | R2	 | RMSE	      | MAE	      | MedAE	    | MAPE |
|-------------------------|------|------------|-----------|-----------|------|
| RF Num + Yolo (all)	| 0.98	| 95801.03	| 57979.46	| 30782.84	| **0.07** |

[Resultados detalhados](https://github.com/juliaronquetti/Modelo_predicao_SP/blob/main/resultados_detalhados/imoveis_results_ML_numeric_yolo.csv)

<br>
<br>
<img src="https://github.com/juliaronquetti/Modelo_predicao_SP/blob/main/imagens/imoveis_results_ML_numeric_yolo.png?raw=true" width="640">

<br>

| nr | Atributo |	Mutual Information |
|----|---------|---------------------|
| 1	 | condominio	| 1.082172 | 
| 2	 | area_util	| 1.080046 | 
| 3	 | tipo_Padrão| 	1.042260 | 
| 4	 | **sink**	| 0.983158 | 
| 5	 | banheiros	| 0.944558 | 
| 6	 | quartos	| 0.902648 | 
| 7	 | iptu	| 0.878083 | 
| 8	 | vagas_na_garagem	| 0.835178 | 
| 9  | location	| 0.815770 | 
| 10 | 	**chair**	| 0.791374 | 
| 11 |	**toilet** |	0.788386 |
| 12 |	**potted plant** |	0.599620 |
| 13 |	**bed**	| 0.576171 |
| 14 |	**tv**	| 0.571253 |
| 15 |	**couch**	| 0.565151 |

A adição de objetos detectados pelo Yolo nas imagens como preditores não parece trazer nenhum ganho significativo na estimativa de preço em comparação com o modelo que utiliza apenas preditores numéricos. Apesar disso, diversos objetos detectados mostram um ganho informativo relevante, e entre os 15 atributos com maior ganho, 7 são objetos detectados nas imagens. Isso sugere que essa abordagem pode ser potencialmente útil, embora pareça necessário detectar mais objetos relevantes ao cenário de anúncios imobiliários, além daqueles identificados como padrão pelo Yolo.

## [RandomForest: numérico + texto](https://colab.research.google.com/github/juliaronquetti/Modelo_predicao_SP/blob/main/7_imoveis_ML_best_model_numeric_text.ipynb)
- Aplicar RandomForestRegression, utilizando campos numéricos e texto para estimativa de preço

<br>

- **Melhor resultado**

| Model                   | R2	 | RMSE	      | MAE	      | MedAE	    | MAPE |
|-------------------------|------|------------|-----------|-----------|------|
| RF Num + text (all)	| 0.98 | 93045.62	| 55591.42 | 29443.23	| **0.06** |

[Resultados detalhados](https://github.com/juliaronquetti/Modelo_predicao_SP/blob/main/resultados_detalhados/imoveis_results_ML_numeric_text.csv)

<br>
<br>
<img src="https://github.com/juliaronquetti/Modelo_predicao_SP/blob/main/imagens/imoveis_results_ML_numeric_text.png?raw=true" width="640">

<br>
Para codificar o texto, foi utilizada uma codificação TF-IDF (TfidfVectorizer), considerando apenas palavras individuais (unigramas) e um máximo de 1000 atributos. Apesar do resultado ligeiramente melhor em relação ao uso apenas de atributos numéricos, a diferença é bastante pequena e não parece significativa o suficiente. No entanto, isso sugere que a codificação com um maior número de n-gramas ou outros métodos de codificação (talvez o uso de LLMs) pode levar a resultados ainda melhores.

## [RandomForest: numérico + texto + yolo](https://colab.research.google.com/github/juliaronquetti/Modelo_predicao_SP/blob/main/8_imoveis_ML_best_model_numeric_text_yolo.ipynb)
- Aplicar RandomForestRegression, utilizando campos numéricos, texto e objetos detectados pelo Yolo para estimativa de preço

<br>

- **Melhor resultado**

| Model                   | R2	 | RMSE	      | MAE	      | MedAE	    | MAPE |
|-------------------------|------|------------|-----------|-----------|------|
| RF Num+text+Yolo (all)	| 0.98 |	92842.89 |	55535.63 |	29494.47 | **0.06** |

[Resultados detalhados](https://github.com/juliaronquetti/Modelo_predicao_SP/blob/main/resultados_detalhados/imoveis_results_ML_numeric_text_yolo.csv)

<br>
<br>
<img src="https://github.com/juliaronquetti/Modelo_predicao_SP/blob/main/imagens/imoveis_results_ML_numeric_text_yolo.png?raw=true" width="640">

<br>
Aqui, todos os dados são utilizados: campos numéricos, texto e objetos detectados nas imagens pelo Yolo como preditores. Embora apresente resultados melhores do que o uso de apenas preditores numéricos, a diferença é pequena em relação aos modelos que adicionam apenas texto ou objetos detectados nas imagens.

# Conclusão
Como conclusão, este trabalho mostra o seguinte:
1. Modelos de aprendizado de máquina podem fornecer resultados muito melhores do que os melhores modelos lineares, mesmo com interações e modelos robustos.
2. Modelos de ensemble, como Random Forest e Gradient Boosting, apresentam os melhores resultados entre muitos outros modelos de aprendizado de máquina.
3. A adição de objetos detectados em imagens e texto não trouxe uma diferença significativa em relação ao uso apenas de preditores numéricos, mas a diferença entre os modelos é inferior a 2% em relação ao RMSE médio dos modelos (apenas ML), e o menor RMSE é obtido usando todas as características, o que mostra a viabilidade da abordagem de usar objetos detectados nas imagens e texto como preditores. Para melhores resultados, é necessário detectar mais objetos relevantes nas imagens e utilizar uma codificação mais avançada da característica de texto (como o uso de n-grams > 1).

| Model |	RMSE % |
|-------------------------|------|
|	RF Numeric Fields (all)	|1.12	|
|	RF Num + Yolo (all)	|1.65	|
|	RF Num + text (all)	|-1.28	|
|	**RF Num+text+Yolo (all)** | **-1.49** |

Trabalhos futuros podem envolver, além da detecção de outros objetos nas imagens, o uso de LLMs para processamento de texto (embedding, análise de sentimento), o uso direto de imagens em modelos de redes convolucionais pré-treinados (VGG16, ResNet, Inception) e a construção de modelos hierárquicos (criando diferentes modelos segmentando os dados, por exemplo, por zona, número de quartos).

# Referências

[1] Marzagão, T., Ferreira, R., & Sales, L. (2021). A note on real estate appraisal in Brazil. Revista Brasileira de Economia, 75(1), 29-36.

[2] Poursaeed, O., Matera, T., & Belongie, S. (2018). Vision-based real estate price estimation. Machine Vision and Applications, 29(4), 667-676.

