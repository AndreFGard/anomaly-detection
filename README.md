# Anomaly Detection in Industrial Robotic Arms
**Sumary**
A work that analyzed the Casper Industrial Robotic Arm dataset and used feature engineering to enable a comparison of Anomaly Detection performance of classical models and a Deep-Learning model.


## Index
- [Summary](#summary)
- [Engenharia de Features 🇧🇷](#engenharia-de-features)
- [Feature Engineering 🏴󠁧󠁢󠁥󠁮󠁧󠁿](#feature-engineering)

## Summary
The automatic anomaly detection in industrial robots is essential for operational security and equipment integrity. This work aproaches failure identification in a UR3e robotic arm, using an IMU dataset ([Casper 2](https://www.kaggle.com/datasets/hkayan/industrial-robotic-arm-imu-data-casper-1-and-2)), composed of more than 870 thousand samples collected in realistic scenarios created to emulate real life failures. Our work was divided in exploratory data analysis, pre-processing - with feature engineering and windowing -  and modelling, with hyperparam tuning. Among the evaluated model architectures are Gaussian Mixture, Isolation Forest and Convolutional Autoencoder.

> A detecção automática de anomalias em robôs industriais é essencial para a segurança operacional e integridade dos equipamentos e máquinas utilizados. Este trabalho aborda a identificação de falhas em um braço robótico UR3e, usando um dataset IMU, baseado em unidades de medição inercial), o ([Casper 2](https://www.kaggle.com/datasets/hkayan/industrial-robotic-arm-imu-data-casper-1-and-2)), que é composto por mais de 870 mil amostras coletadas em cenários realistas criados para emular falhas da vida real. Nosso trabalho foi dividido em análise exploratória de dados, pré-processamento - com engenharia de features e criação de janelas deslizantes - e modelagem, com tunagem de hiperparâmetros. Entre as arquiteturas de modelos avaliadas, estão a Mistura Gaussiana (Gaussian Mixture Model), Floresta de Isolamento e Autoencoder Convolucional.


## Engenharia de Features
Devido à natureza distinta dos modelos avaliados, nossa estratégia de engenharia de features precisou ser adaptada. Desta forma, a extração manual de features se restringiu aos modelos probabilísticos e baseados em densidade, enquanto os dados crus da série temporal foram utilizados, sendo sujeitados apenas a enjanelamento deslizante e a normalização

É importante destacar a alta dimensionalidade do dataset usado, em que cada segundo consistia de 90 dimensões - 9 amostras, 10Hz. Ademais, esse número se multiplicava pela quantidade de segundos em cada janela. Assim, como modelos clássicos frequentemente são vítimas da Maldição da Dimensionalidade e são comumente menos capazes de aprender relações complexas entre os dados, a engenharia e seleção de features foi indispensável.

Em essência, esse processo no presente trabalho consistiu na criação de janelas e extração de features relativas a cada uma destas, atributos os quais enumeram-se abaixo, seguidos de uma explicação simplística de como eles podem ser úteis.

1. **Média**: representa o valor médio do sinal ao longo da janela, estando associada ao nível basal ou à tendência local do movimento. Alterações na média podem indicar mudanças sistemáticas no comportamento do sistema.
2. **Desvio padrão**: quantifica a variabilidade do sinal dentro da janela, sendo associado à intensidade das vibrações.
3. **RMS (Root Mean Square)**: mede a energia do sinal, combinando informações de magnitude e variabilidade, sendo sensível tanto a oscilações quanto a impactos.
4. **Pico-a-pico (*peak-to-peak*)**: corresponde à diferença entre os valores máximo e mínimo do sinal na janela, capturando grandes variações de amplitude.
5. **Curtose**: caracteriza o grau de impulsividade do sinal, indicando a presença de picos abruptos e eventos raros de grande magnitude.
6. **Fator de crista (*crest factor*)**: definido como a razão entre o valor de pico e o RMS, fornece uma medida normalizada da severidade dos picos em relação à energia média do sinal.
7. **Frequência dominante**: extraída a partir da transformada de Fourier da janela, corresponde à frequência com maior energia espectral, permitindo capturar características dinâmicas do sistema, como ressonâncias ou mudanças de regime de operação.

Isso levou a uma redução de * n atributos x tamanho da janela* dimensões (eg. 9 x 40) para (n atributos x 8) (eg. 72), no nosso tamanhho inicial de 40 amostras por janela (4s). No entanto, essas features ainda são altamente correlacionadas e redundantes. Assim, após análise de correlação linear, que manteve apenas 58 atributos, foi usado um PCA (ajustado apenas no conjunto de treino, com normalização), preservando 95% da variância mas resultando em apenas 15 atributos.

Esse processo, em conjunto com outros passos do pré-processamento, foi estruturado em dois *pipelines*. O primeiro, para modelos clássicos, levou em conta a necessidade de ajustar o tamanho da janela como hiperparâmetro, o que requer recomeçar a engenharia e seleção de features. O segundo, para modelos de Deep-Learning, restringiu-se ao enjanelamento deslizante e normalização. Testes preliminares durante a elaboração do trabalho mostraram que esse passo foi essencial para o melhor funcionamento dos modelos classicos utilizados.

## Feature engineering
Due to the different nature of the evaluated models, our feature engineering strategy had to be adapted. Therefore, manual feature extraction was applied exclusively to probabilistic and density-based models, while for Deep-Learning based models, a Convolutional Autoencoder, in our case, raw time series data was used, only being subjected to windowing and normalization.

It's important to highlight the high dimensionality of the dataset used. Each second would ammount to 90 dimensions - 9 dimensions per sample, 10Hz. Besides that, each window consisted of more than a second. Since classical models often fall victim to the Curse of Dimensionality and are usually less able to understand deep correlations in data, feature engineering and selection was mandatory. In essence, our feature engineering consisted of the creation of sliding windows and later feature extraction and selection per window. For each sensor and each window, the following features were extracted, followed by a simplistic explanation of what they might portrait:

1. Mean - Average signal value
2. Standard Deviation - Signal variability
3. Root Mean Square - Total signal energy, sensitive to impacts
5. Peak to Peak - Captures large variations, corresponds to the difference between lowest and highest value
6. Kurtosis - Signal impulsiveness and peaks
7. Crest Factor - Ratio between peak value and RMS, normalizing the peak's severity
8. Dominant Frequency - Extracted from the Fourier Transform.

This lead to a reduction from *features x window size* dimensions (eg. 9 x 40) to *8 x features* (eg. 72), in our initial window size of 40 samples (4s). However, these engineered features are still highly redundant and correlated. After linear correlation analysis, which mantained only 58 features, a Principal Component Analysis was fitted - exclusively on the training set and after normalization - to preserve 95% of variance and thereby a reduction from 58 to 15 features was obtained.

This process, along with other pre-processing steps, was made into two pipelines. One was for classical models, since the hyperparam tuning step required window size tuning, which required also restarting the feature engineering and selection steps. The second was for Deep-Learning models, which only covered windowing and normalization. Preliminary testing during the work's elaboration showed this process was highly effective in increasing the baseline efficiency of our classical models.

### Extra - Visualization site
A simple website was elaborated to quickly compare visually the behavior of the robotic arm under normal *versus* abnormal conditions. In red is the "abnormal arm", in which constant oscilation-like movements can be seen. 
https://amcd.andrefgard.duckdns.org/3djs.html
