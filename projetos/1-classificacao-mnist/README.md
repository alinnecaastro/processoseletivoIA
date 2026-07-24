# Classificação de Dígitos Manuscritos (MNIST)

Projeto desenvolvido como parte do desafio técnico de Edge AI. O objetivo é treinar uma Rede Neural Convolucional (CNN) para reconhecer dígitos manuscritos do conjunto de dados MNIST e otimizar o modelo para execução em dispositivos com recursos limitados utilizando TensorFlow Lite.

1️⃣ Resumo da Arquitetura do Modelo

O modelo desenvolvido em train_model.py utiliza uma Rede Neural Convolucional (CNN) composta por três blocos convolucionais. Cada bloco possui uma camada Conv2D com filtros de tamanho 3x3, função de ativação ReLU e preenchimento same, seguida por uma camada de BatchNormalization e uma camada de MaxPooling2D com janela 2x2.

Os blocos convolucionais utilizam, respectivamente, 32, 64 e 128 filtros. O aumento gradual da quantidade de filtros permite que o modelo aprenda desde características mais simples, como bordas e traços, até padrões mais complexos presentes nos dígitos manuscritos.

Após os blocos convolucionais, os dados são transformados em um vetor por meio da camada Flatten. Em seguida, é utilizada uma camada totalmente conectada Dense com 128 neurônios e ativação ReLU.

Antes da camada de saída, foi aplicada uma camada de Dropout com taxa de 0,5, que desativa aleatoriamente 50% dos neurônios durante o treinamento. Essa técnica ajuda a reduzir o sobreajuste e melhora a capacidade de generalização do modelo.

A camada de saída possui 10 neurônios, correspondentes aos dígitos de 0 a 9, e utiliza a função de ativação softmax, responsável por gerar a probabilidade prevista para cada classe.

O modelo foi compilado utilizando o otimizador Adam, a função de perda sparse_categorical_crossentropy e a métrica de acurácia.

Para a validação, as 60.000 imagens originalmente destinadas ao treinamento no conjunto MNIST foram embaralhadas de forma reproduzível utilizando a semente 42. Em seguida, foram separadas em 54.000 imagens para treinamento e 6.000 imagens para validação. As 10.000 imagens oficiais de teste do MNIST foram mantidas separadas para a avaliação final.

O treinamento foi configurado para executar por até 15 épocas, utilizando lotes de 128 imagens. Também foi aplicada a estratégia de parada antecipada, por meio do callback EarlyStopping, monitorando a perda de validação (val_loss). O treinamento é interrompido quando não ocorre melhoria durante três épocas consecutivas. A opção restore_best_weights=True restaura automaticamente os pesos correspondentes à melhor perda de validação encontrada durante o treinamento.

## 2️⃣ Bibliotecas Utilizadas

O desenvolvimento do projeto foi realizado em Python utilizando bibliotecas voltadas para aprendizado de máquina, processamento de imagens e computação numérica. As principais bibliotecas utilizadas foram:

| Biblioteca   | Versão                      | Finalidade                                                                                                  |
| ------------ | --------------------------- | ----------------------------------------------------------------------------------------------------------- |
| TensorFlow   | **2.21.0**                  | Construção, treinamento, avaliação e conversão do modelo para TensorFlow Lite.                              |
| Keras        | **3.12.3**                  | Criação da arquitetura da Rede Neural Convolucional (CNN), definição das camadas e treinamento do modelo.   |
| NumPy        | **1.26.4**                  | Manipulação de arrays, normalização dos dados e divisão do conjunto de treinamento e validação.             |
| Pillow (PIL) | Conforme `requirements.txt` | Leitura e pré-processamento de imagens utilizadas na inferência.                                            |
| OpenCV       | **4.10.0.84**               | Processamento de imagens, incluindo operações de preparação das imagens antes da classificação pelo modelo. |

Além dessas bibliotecas, foram utilizados módulos nativos da linguagem Python, como `os`, para configuração do ambiente de execução e manipulação de arquivos.

3️⃣ Técnica de Otimização do Modelo

Após o treinamento da Rede Neural Convolucional (CNN), o modelo salvo no formato Keras (model.h5) foi convertido para o formato TensorFlow Lite (model.tflite) utilizando uma técnica de otimização conhecida como Quantização de Faixa Dinâmica (Dynamic Range Quantization).

A conversão foi realizada por meio da classe TFLiteConverter, utilizando a configuração:

conversor.optimizations = [tf.lite.Optimize.DEFAULT]

Essa técnica aplica uma otimização pós-treinamento (Post-Training Quantization), reduzindo a precisão dos pesos compatíveis do modelo de ponto flutuante para um formato mais compacto durante a inferência. Como não foi fornecido um representative_dataset, a conversão utiliza automaticamente a quantização de faixa dinâmica.

O principal objetivo dessa otimização é reduzir o tamanho do arquivo do modelo e melhorar sua eficiência em dispositivos com recursos limitados, como microcontroladores, smartphones e sistemas embarcados, mantendo uma boa precisão na classificação dos dígitos manuscritos.

Ao final da conversão, o programa também calcula o tamanho dos arquivos model.h5 e model.tflite, apresentando a porcentagem de redução obtida após a otimização, permitindo comparar diretamente o ganho de armazenamento proporcionado pela técnica utilizada.

4️⃣ Resultados Obtidos

Ao final do treinamento, o modelo apresentou uma acurácia de validação de 98,92% e uma acurácia de teste de 98,93%, demonstrando excelente desempenho na classificação dos dígitos manuscritos do conjunto de dados MNIST.

Após a conversão do modelo para o formato TensorFlow Lite utilizando quantização de faixa dinâmica, foi observada uma redução significativa no tamanho do arquivo, tornando o modelo mais adequado para aplicações de Edge AI.

Resultados
Métrica	Valor
Perda de validação	0,0429
Acurácia de validação	98,92%
Acurácia de teste	98,93%
Tamanho dos arquivos
Arquivo	Tamanho
model.h5	2,84 MB
model.tflite	0,24 MB

A conversão para TensorFlow Lite reduziu o tamanho do modelo em aproximadamente 91,39%, mantendo um elevado nível de precisão. Esse resultado demonstra que a técnica de otimização foi eficaz, tornando o modelo mais leve e apropriado para execução em dispositivos com recursos computacionais limitados, sem comprometer significativamente seu desempenho.

## 6️⃣ Exemplo de Inferência

### Saída do terminal

```text
============================================================
INFERÊNCIA COM O MODELO TENSORFLOW LITE
============================================================
Amostra 01 | Predito: 7 | Real: 7 | Confiança: 100.00% | ACERTOU | Tempo: 0.149 ms
Amostra 02 | Predito: 2 | Real: 2 | Confiança: 100.00% | ACERTOU | Tempo: 0.119 ms
Amostra 03 | Predito: 1 | Real: 1 | Confiança: 100.00% | ACERTOU | Tempo: 0.099 ms
Amostra 04 | Predito: 0 | Real: 0 | Confiança: 100.00% | ACERTOU | Tempo: 0.097 ms
Amostra 05 | Predito: 4 | Real: 4 | Confiança: 100.00% | ACERTOU | Tempo: 0.094 ms
Amostra 06 | Predito: 1 | Real: 1 | Confiança: 100.00% | ACERTOU | Tempo: 0.102 ms
Amostra 07 | Predito: 4 | Real: 4 | Confiança: 99.98% | ACERTOU | Tempo: 0.096 ms
Amostra 08 | Predito: 9 | Real: 9 | Confiança: 100.00% | ACERTOU | Tempo: 0.095 ms
Amostra 09 | Predito: 5 | Real: 5 | Confiança: 100.00% | ACERTOU | Tempo: 0.094 ms
Amostra 10 | Predito: 9 | Real: 9 | Confiança: 100.00% | ACERTOU | Tempo: 0.094 ms

------------------------------------------------------------
RESUMO
------------------------------------------------------------
Acertos: 10/10
Acurácia nas amostras: 100.00%
Tempo médio por inferência: 0.104 ms
```

### Comentário

Nas dez amostras testadas, o modelo classificou corretamente todos os dígitos, alcançando **100% de acerto** durante a inferência. Esse resultado está de acordo com a alta acurácia obtida nas etapas de validação e teste do modelo. Além da precisão, chamou minha atenção o tempo de execução, com uma média de apenas **0,104 ms por inferência**, mostrando que o modelo ficou bastante leve e rápido após a conversão para TensorFlow Lite. Embora esse conjunto de amostras seja pequeno e não represente todo o conjunto de dados, os resultados indicam que o modelo apresentou um desempenho consistente e adequado para aplicações de Edge AI.
