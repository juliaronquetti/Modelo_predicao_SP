# Modelo_predicao_SP

#### [Pré-processamento](https://colab.research.google.com/github/juliaronquetti/Modelo_predicao_SP/blob/main/1_imoveis_preprocess.ipynb)
- Anonimizar dados (extrair informações de URL)
- Renomear arquivos e nomes de campos

## Dados

O conjunto de dados fornece informações para a previsão de preços de imóveis. Ele contém dados de 23.433 imóveis à venda na cidade de São Paulo (Brasil), anunciados durante o mês de abril de 2024. Os dados incluem o preço, mais de 30 características dos imóveis (m², número de quartos, garagem, etc.) e imagens dos anúncios, totalizando 407.567 imagens.

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

[detailed results](https://github.com/juliaronquetti/Modelo_predicao_SP/blob/main/imagens/imoveis_results_linear_regressions.png)


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

[Resultados detalhados](https://github.com/Rogerio-mack/Property-Price-Prediction-Sao-Paulo/blob/main/detailed_result_tables/imoveis_results_ML_selection.csv)

<br>

A 5-fold cross-validation were used. 3 Best results are showed. Decision trees give MAPE = 0, but with large overffiting, thus was excluded here. Linear Regression regressions and their variants (Ridge Regression, HuberRegressor etc.) give better results than K-Nearest Neighbors, but ~0.30. They are also excluded here since a better result, ~0.22, was obtained berfore with linear regression with log transformation and iteractions. Random Forest was selected, despite a little small MAPE than Gradient Boosting because it shows better results (RMSE) in test data (here, results are over all data). For all the following executions, Random Forests with with 1,000 trees and no pruning were used.

## [RandomForest: numeric features](https://colab.research.google.com/github/juliaronquetti/Modelo_predicao_SP/blob/main/5_imoveis_ML_best_model_numeric.ipynb)
- Apply RandomForestRegression, only numeric fields are used for price estimation

<br>

- **Best result**

| Model                   | R2	 | RMSE	      | MAE	      | MedAE	    | MAPE |
|-------------------------|------|------------|-----------|-----------|------|
| RF Numeric Fields (all) |	0.98 | 95300.42	| 57290.74 |	29912.10 |**0.07** |

[detailed results](https://github.com/Rogerio-mack/Property-Price-Prediction-Sao-Paulo/blob/main/detailed_result_tables/imoveis_results_ML_numeric.csv)

<br>
<br>
<img src="https://github.com/Rogerio-mack/Property-Price-Prediction-Sao-Paulo/blob/main/figures/imoveis_results_ML_numeric.png?raw=true" width="640">

The selected machine learning model presents a very significant gain, presenting an average of 7% (MAPE) error in price estimation compared to 22% for the best linear model. This is an error very close to that found in the literature [1][2].

## [RandomForest: numeric + Yolo](https://colab.research.google.com/github/juliaronquetti/Modelo_predicao_SP/blob/main/6_imoveis_ML_best_model_numeric_yolo.ipynb)
- Apply RandomForestRegression, numeric fields and Yolo detected objects are used for price estimation

<br>

- **Best result**

| Model                   | R2	 | RMSE	      | MAE	      | MedAE	    | MAPE |
|-------------------------|------|------------|-----------|-----------|------|
| RF Num + Yolo (all)	| 0.98	| 95801.03	| 57979.46	| 30782.84	| **0.07** |

[detailed results](https://github.com/Rogerio-mack/Property-Price-Prediction-Sao-Paulo/blob/main/detailed_result_tables/imoveis_results_ML_numeric_yolo.csv)

<br>
<br>
<img src="https://github.com/Rogerio-mack/Property-Price-Prediction-Sao-Paulo/blob/main/figures/imoveis_results_ML_numeric_yolo.png?raw=true" width="640">

<br>

| nr | Feature |	Mutual Information |
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

The add of Yolo detected objects in the images as predictors does not appear to bring any significant gain to price estimation compared to the model that uses only numerical predictors. Despite this, several detected objects show a significant gain in information, and among the 15 attributes with the highest gain, 7 objects are objects detected in the images. This suggests that this approach may be potentially useful, although it seems necessary detect more relevant objects to the real estate advertising scenario, in addition to those identified as standard by Yolo.

## [RandomForest: numeric + text](https://colab.research.google.com/github/juliaronquetti/Modelo_predicao_SP/blob/main/7_imoveis_ML_best_model_numeric_text.ipynb)
- Apply RandomForestRegression, numeric fields and text used for price estimation

<br>

- **Best result**

| Model                   | R2	 | RMSE	      | MAE	      | MedAE	    | MAPE |
|-------------------------|------|------------|-----------|-----------|------|
| RF Num + text (all)	| 0.98 | 93045.62	| 55591.42 | 29443.23	| **0.06** |

[detailed results](https://github.com/Rogerio-mack/Property-Price-Prediction-Sao-Paulo/blob/main/detailed_result_tables/imoveis_results_ML_numeric_text.csv)

<br>
<br>
<img src="https://github.com/Rogerio-mack/Property-Price-Prediction-Sao-Paulo/blob/main/figures/imoveis_results_ML_numeric_text.png?raw=true" width="640">

<br>
To encode the text, a TF-IDF (TfidfVectorizer) encode was used, only individual words (unigrams) and a maximum of 1000 features. Despite the better result with respect to the use of only numerical features, the difference is quite small and does not seem significant enough, but it suggests that encoding with a greater number of n-grams or other encoding methods (perhaps the use of LLMs) can lead to even better results.

## [RandomForest: numeric + text + yolo](https://colab.research.google.com/github/juliaronquetti/Modelo_predicao_SP/blob/main/8_imoveis_ML_best_model_numeric_text_yolo.ipynb)
- Apply RandomForestRegression, numeric fields, text and Yolo detected objects used for price estimation

<br>

- **Best result**

| Model                   | R2	 | RMSE	      | MAE	      | MedAE	    | MAPE |
|-------------------------|------|------------|-----------|-----------|------|
| RF Num+text+Yolo (all)	| 0.98 |	92842.89 |	55535.63 |	29494.47 | **0.06** |

[detailed results](https://github.com/Rogerio-mack/Property-Price-Prediction-Sao-Paulo/blob/main/detailed_result_tables/imoveis_results_ML_numeric_text_yolo.csv)

<br>
<br>
<img src="https://github.com/Rogerio-mack/Property-Price-Prediction-Sao-Paulo/blob/main/figures/imoveis_results_ML_numeric_text_yolo.png?raw=true" width="640">

<br>
Here all data is used, numeric, text field and Yolo detected objects images as predictors.  Although it presents better results than using only numerical predictors, the result has little difference in relation to models that add text or objects detected in the images.

# Conclusion
As a conclusion this work shows the follow:
1. Machine Learning models can give much better results than the best linear models, even with interactions and robust models
2. Ensemble models, such as Random Forest and Gradient Boosting, give best results among many other Machine Learning Models
3. The addition of objects detected from images and text did not bring a significant difference in relation to the use of only numerical predictors, but the difference between the models is less than 2% in relation to the average RMSE of the models (only ML) and the lowest RMSE is obtained using all features, which shows the viability of the approach of objects detected from images and text as predictors, being necessary for better results the detection of more relevant objects in the images and a more improved encoding of the text feature (such as the use of n-grams > 1).

| Model |	RMSE % |
|-------------------------|------|
|	RF Numeric Fields (all)	|1.12	|
|	RF Num + Yolo (all)	|1.65	|
|	RF Num + text (all)	|-1.28	|
|	**RF Num+text+Yolo (all)** | **-1.49** |

Future work may involve, in addition to the detection of other objects in the images, the use of LLMs for text processing (embedding, sentiment analysis), the direct use of images in pre-trained convolutional network models (VGG16, ResNet, Inception), and the construction of hierarchical models (creating several different models segmenting the data, for example by zone, number of bedrooms).

# References

[1] Marzagão, T., Ferreira, R., & Sales, L. (2021). A note on real estate appraisal in Brazil. Revista Brasileira de Economia, 75(1), 29-36.

[2] Poursaeed, O., Matera, T., & Belongie, S. (2018). Vision-based real estate price estimation. Machine Vision and Applications, 29(4), 667-676.

