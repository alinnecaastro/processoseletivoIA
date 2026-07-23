import os

# Força o TensorFlow a utilizar somente a CPU.
# Esta linha precisa ficar antes da importação do TensorFlow.
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

import numpy as np
import tensorflow as tf


CAMINHO_MODELO = "model.h5"
QUANTIDADE_EPOCAS = 15
TAMANHO_LOTE = 128
SEMENTE = 42


def carregar_dados():
    """
    Carrega o MNIST, normaliza as imagens e realiza
    uma divisão explícita entre treino e validação.
    """

    (x_treino_completo, y_treino_completo), (x_teste, y_teste) = (
        tf.keras.datasets.mnist.load_data()
    )

    # Normaliza os pixels de 0-255 para 0-1.
    x_treino_completo = x_treino_completo.astype("float32") / 255.0
    x_teste = x_teste.astype("float32") / 255.0

    # Adiciona o canal da imagem:
    # (28, 28) -> (28, 28, 1)
    x_treino_completo = np.expand_dims(
        x_treino_completo,
        axis=-1
    )

    x_teste = np.expand_dims(
        x_teste,
        axis=-1
    )

    # Embaralhamento reproduzível antes do split.
    gerador = np.random.default_rng(SEMENTE)
    indices = gerador.permutation(len(x_treino_completo))

    x_treino_completo = x_treino_completo[indices]
    y_treino_completo = y_treino_completo[indices]

    # 54.000 imagens para treino e 6.000 para validação.
    quantidade_validacao = 6000

    x_validacao = x_treino_completo[:quantidade_validacao]
    y_validacao = y_treino_completo[:quantidade_validacao]

    x_treino = x_treino_completo[quantidade_validacao:]
    y_treino = y_treino_completo[quantidade_validacao:]

    print(f"Imagens de treino: {len(x_treino)}")
    print(f"Imagens de validação: {len(x_validacao)}")
    print(f"Imagens de teste: {len(x_teste)}")

    return (
        x_treino,
        y_treino,
        x_validacao,
        y_validacao,
        x_teste,
        y_teste
    )


def criar_modelo():
    """
    Cria uma CNN com três blocos convolucionais.
    """

    modelo = tf.keras.Sequential([
        tf.keras.layers.Input(shape=(28, 28, 1)),

        # Primeiro bloco convolucional
        tf.keras.layers.Conv2D(
            filters=32,
            kernel_size=(3, 3),
            padding="same",
            activation="relu"
        ),
        tf.keras.layers.BatchNormalization(),
        tf.keras.layers.MaxPooling2D(pool_size=(2, 2)),

        # Segundo bloco convolucional
        tf.keras.layers.Conv2D(
            filters=64,
            kernel_size=(3, 3),
            padding="same",
            activation="relu"
        ),
        tf.keras.layers.BatchNormalization(),
        tf.keras.layers.MaxPooling2D(pool_size=(2, 2)),

        # Terceiro bloco convolucional
        tf.keras.layers.Conv2D(
            filters=128,
            kernel_size=(3, 3),
            padding="same",
            activation="relu"
        ),
        tf.keras.layers.BatchNormalization(),
        tf.keras.layers.MaxPooling2D(pool_size=(2, 2)),

        tf.keras.layers.Flatten(),

        tf.keras.layers.Dense(
            units=128,
            activation="relu"
        ),

        # Regularização antes da camada de saída
        tf.keras.layers.Dropout(rate=0.5),

        # Saída para os dígitos de 0 a 9
        tf.keras.layers.Dense(
            units=10,
            activation="softmax"
        )
    ])

    modelo.compile(
        optimizer="adam",
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"]
    )

    return modelo


def main():
    tf.random.set_seed(SEMENTE)
    np.random.seed(SEMENTE)

    (
        x_treino,
        y_treino,
        x_validacao,
        y_validacao,
        x_teste,
        y_teste
    ) = carregar_dados()

    modelo = criar_modelo()

    modelo.summary()

    parada_antecipada = tf.keras.callbacks.EarlyStopping(
        monitor="val_loss",
        patience=3,
        restore_best_weights=True,
        verbose=1
    )

    modelo.fit(
        x_treino,
        y_treino,
        validation_data=(x_validacao, y_validacao),
        epochs=QUANTIDADE_EPOCAS,
        batch_size=TAMANHO_LOTE,
        callbacks=[parada_antecipada],
        verbose=1
    )

    perda_validacao, acuracia_validacao = modelo.evaluate(
        x_validacao,
        y_validacao,
        verbose=0
    )

    print("\nResultado final")
    print(f"Perda de validação: {perda_validacao:.4f}")
    print(
        f"Acurácia de validação final: "
        f"{acuracia_validacao * 100:.2f}%"
    )

    perda_teste, acuracia_teste = modelo.evaluate(
        x_teste,
        y_teste,
        verbose=0
    )

    print(f"Acurácia de teste: {acuracia_teste * 100:.2f}%")

    modelo.save(CAMINHO_MODELO)

    print(f"\nModelo salvo com sucesso em: {CAMINHO_MODELO}")


if __name__ == "__main__":
    main()