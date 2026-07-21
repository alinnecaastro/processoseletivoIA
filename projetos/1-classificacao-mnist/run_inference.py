import os
import time

# Força o TensorFlow a usar somente a CPU.
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

import numpy as np
import tensorflow as tf


CAMINHO_MODELO = "model.tflite"
QUANTIDADE_AMOSTRAS = 10


def preparar_imagem_mnist(imagem):
    """
    Prepara uma imagem do conjunto MNIST.

    Formato original:
    (28, 28), valores entre 0 e 255.

    Formato final:
    (1, 28, 28, 1), valores entre 0 e 1.
    """

    imagem = imagem.astype("float32") / 255.0

    # Adiciona o canal da imagem:
    # (28, 28) -> (28, 28, 1)
    imagem = np.expand_dims(
        imagem,
        axis=-1,
    )

    # Adiciona a dimensão do lote:
    # (28, 28, 1) -> (1, 28, 28, 1)
    imagem = np.expand_dims(
        imagem,
        axis=0,
    )

    return imagem


def executar_inferencia(
    interpretador,
    imagem,
    indice_entrada,
    indice_saida,
    tipo_entrada,
):
    """
    Executa uma inferência utilizando o modelo TensorFlow Lite.
    """

    imagem = imagem.astype(tipo_entrada)

    interpretador.set_tensor(
        indice_entrada,
        imagem,
    )

    inicio = time.perf_counter()

    interpretador.invoke()

    fim = time.perf_counter()

    resultado = interpretador.get_tensor(
        indice_saida
    )[0]

    classe_predita = int(
        np.argmax(resultado)
    )

    confianca = float(
        np.max(resultado)
    ) * 100

    tempo_ms = (
        fim - inicio
    ) * 1000

    return (
        classe_predita,
        confianca,
        tempo_ms,
    )


def testar_amostras_mnist(
    interpretador,
    indice_entrada,
    indice_saida,
    tipo_entrada,
):
    """
    Executa inferência em amostras do conjunto de teste MNIST.
    """

    print("\n" + "=" * 60)
    print("INFERÊNCIA COM O MODELO TENSORFLOW LITE")
    print("=" * 60)

    # Carrega somente o conjunto de teste do MNIST.
    (_, _), (x_teste, y_teste) = (
        tf.keras.datasets.mnist.load_data()
    )

    quantidade = min(
        QUANTIDADE_AMOSTRAS,
        len(x_teste),
    )

    acertos = 0
    tempos = []

    # Aquecimento do interpretador.
    imagem_aquecimento = preparar_imagem_mnist(
        x_teste[0]
    ).astype(tipo_entrada)

    interpretador.set_tensor(
        indice_entrada,
        imagem_aquecimento,
    )

    interpretador.invoke()

    for indice in range(quantidade):
        imagem = preparar_imagem_mnist(
            x_teste[indice]
        )

        (
            classe_predita,
            confianca,
            tempo_ms,
        ) = executar_inferencia(
            interpretador,
            imagem,
            indice_entrada,
            indice_saida,
            tipo_entrada,
        )

        classe_real = int(
            y_teste[indice]
        )

        correto = (
            classe_predita == classe_real
        )

        if correto:
            acertos += 1
            situacao = "ACERTOU"
        else:
            situacao = "ERROU"

        tempos.append(tempo_ms)

        print(
            f"Amostra {indice + 1:02d} | "
            f"Predito: {classe_predita} | "
            f"Real: {classe_real} | "
            f"Confiança: {confianca:.2f}% | "
            f"{situacao} | "
            f"Tempo: {tempo_ms:.3f} ms"
        )

    acuracia = (
        acertos / quantidade
    ) * 100

    tempo_medio = float(
        np.mean(tempos)
    )

    print("\n" + "-" * 60)
    print("RESUMO")
    print("-" * 60)

    print(
        f"Acertos: {acertos}/{quantidade}"
    )

    print(
        f"Acurácia nas amostras: "
        f"{acuracia:.2f}%"
    )

    print(
        "Tempo médio por inferência: "
        f"{tempo_medio:.3f} ms"
    )


def main():
    """
    Carrega o model.tflite e executa as inferências.
    """

    if not os.path.exists(CAMINHO_MODELO):
        raise FileNotFoundError(
            f"O arquivo {CAMINHO_MODELO} não foi encontrado. "
            "Execute primeiro: python optimize_model.py"
        )

    print(
        f"Carregando o modelo: {CAMINHO_MODELO}"
    )

    interpretador = tf.lite.Interpreter(
        model_path=CAMINHO_MODELO
    )

    interpretador.allocate_tensors()

    detalhes_entrada = (
        interpretador.get_input_details()
    )

    detalhes_saida = (
        interpretador.get_output_details()
    )

    indice_entrada = detalhes_entrada[0][
        "index"
    ]

    indice_saida = detalhes_saida[0][
        "index"
    ]

    tipo_entrada = detalhes_entrada[0][
        "dtype"
    ]

    formato_entrada = tuple(
        detalhes_entrada[0]["shape"]
    )

    print(
        "Formato esperado da entrada:",
        formato_entrada,
    )

    print(
        "Tipo esperado da entrada:",
        tipo_entrada,
    )

    if formato_entrada != (1, 28, 28, 1):
        raise ValueError(
            "O modelo possui um formato de entrada inesperado: "
            f"{formato_entrada}"
        )

    testar_amostras_mnist(
        interpretador,
        indice_entrada,
        indice_saida,
        tipo_entrada,
    )


if __name__ == "__main__":
    main()