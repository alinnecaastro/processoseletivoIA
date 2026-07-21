import os

# Força a execução do TensorFlow usando somente CPU.
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

import numpy as np
import tensorflow as tf


# Configura sementes para que os resultados sejam mais reproduzíveis.
SEMENTE = 42
np.random.seed(SEMENTE)
tf.random.set_seed(SEMENTE)


def carregar_dados():
    """
    Carrega o MNIST, normaliza as imagens e cria explicitamente
    os conjuntos de treinamento, validação e teste.
    """

    print("Carregando o dataset MNIST...")

    (x_treino_completo, y_treino_completo), (x_teste, y_teste) = (
        tf.keras.datasets.mnist.load_data()
    )

    # Normaliza os pixels de 0-255 para 0-1.
    x_treino_completo = x_treino_completo.astype("float32") / 255.0
    x_teste = x_teste.astype("float32") / 255.0

    # Adiciona o canal da imagem.
    # Antes: (quantidade, 28, 28)
    # Depois: (quantidade, 28, 28, 1)
    x_treino_completo = np.expand_dims(
        x_treino_completo,
        axis=-1
    )

    x_teste = np.expand_dims(
        x_teste,
        axis=-1
    )

    # Split explícito:
    # 55.000 imagens para treinamento
    # 5.000 imagens para validação
    x_treino = x_treino_completo[:-5000]
    y_treino = y_treino_completo[:-5000]

    x_validacao = x_treino_completo[-5000:]
    y_validacao = y_treino_completo[-5000:]

    print(f"Treinamento: {x_treino.shape}")
    print(f"Validação:   {x_validacao.shape}")
    print(f"Teste:       {x_teste.shape}")

    return (
        x_treino,
        y_treino,
        x_validacao,
        y_validacao,
        x_teste,
        y_teste,
    )


def criar_modelo():
    """
    Cria uma CNN simples com três blocos convolucionais.
    Cada bloco possui Conv2D, BatchNormalization e MaxPooling2D.
    """

    modelo = tf.keras.Sequential(
        [
            tf.keras.layers.Input(shape=(28, 28, 1)),

            # Primeiro bloco convolucional
            tf.keras.layers.Conv2D(
                filters=16,
                kernel_size=(3, 3),
                padding="same",
                activation="relu",
            ),
            tf.keras.layers.BatchNormalization(),
            tf.keras.layers.MaxPooling2D(
                pool_size=(2, 2)
            ),

            # Segundo bloco convolucional
            tf.keras.layers.Conv2D(
                filters=32,
                kernel_size=(3, 3),
                padding="same",
                activation="relu",
            ),
            tf.keras.layers.BatchNormalization(),
            tf.keras.layers.MaxPooling2D(
                pool_size=(2, 2)
            ),

            # Terceiro bloco convolucional
            tf.keras.layers.Conv2D(
                filters=64,
                kernel_size=(3, 3),
                padding="same",
                activation="relu",
            ),
            tf.keras.layers.BatchNormalization(),
            tf.keras.layers.MaxPooling2D(
                pool_size=(2, 2)
            ),

            # Transforma os mapas de características em um vetor.
            tf.keras.layers.Flatten(),

            tf.keras.layers.Dense(
                units=64,
                activation="relu"
            ),

            # Regularização antes da saída.
            tf.keras.layers.Dropout(rate=0.30),

            # Dez neurônios, um para cada dígito de 0 a 9.
            tf.keras.layers.Dense(
                units=10,
                activation="softmax"
            ),
        ],
        name="cnn_mnist",
    )

    modelo.compile(
        optimizer="adam",
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"],
    )

    return modelo


def main():
    (
        x_treino,
        y_treino,
        x_validacao,
        y_validacao,
        x_teste,
        y_teste,
    ) = carregar_dados()

    modelo = criar_modelo()

    print("\nArquitetura do modelo:")
    modelo.summary()

    early_stopping = tf.keras.callbacks.EarlyStopping(
        monitor="val_loss",
        mode="min",
        patience=3,
        restore_best_weights=True,
        verbose=1,
    )

    print("\nIniciando o treinamento...")

    historico = modelo.fit(
        x_treino,
        y_treino,
        validation_data=(
            x_validacao,
            y_validacao
        ),
        epochs=15,
        batch_size=128,
        callbacks=[early_stopping],
        verbose=1,
    )

    # Avalia o modelo restaurado na validação.
    perda_validacao, acuracia_validacao = modelo.evaluate(
        x_validacao,
        y_validacao,
        verbose=0,
    )

    melhor_acuracia_historica = max(
        historico.history["val_accuracy"]
    )

    print(
        "\nMelhor acurácia registrada durante o treinamento: "
        f"{melhor_acuracia_historica:.4%}"
    )

    print(
        f"Perda final de validação: "
        f"{perda_validacao:.4f}"
    )

    print(
        "Acurácia final de validação do modelo restaurado: "
        f"{acuracia_validacao:.4%}"
    )

    # Avaliação no conjunto de teste.
    perda_teste, acuracia_teste = modelo.evaluate(
        x_teste,
        y_teste,
        verbose=0,
    )

    print(f"Perda no teste: {perda_teste:.4f}")
    print(f"Acurácia no teste: {acuracia_teste:.4%}")

    # Salva o modelo.
    modelo.save("model.h5")

    tamanho_mb = os.path.getsize("model.h5") / (1024 * 1024)

    print("\nModelo salvo com sucesso!")
    print("Arquivo: model.h5")
    print(f"Tamanho: {tamanho_mb:.2f} MB")
        
if __name__ == "__main__":
    main()