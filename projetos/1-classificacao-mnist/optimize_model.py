import os

# Força o TensorFlow a utilizar somente a CPU.
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

import tensorflow as tf


CAMINHO_MODELO_KERAS = "model.h5"
CAMINHO_MODELO_TFLITE = "model.tflite"


def main():
    """
    Carrega o modelo Keras, aplica quantização de faixa dinâmica
    e converte o modelo para TensorFlow Lite.
    """

    if not os.path.exists(CAMINHO_MODELO_KERAS):
        raise FileNotFoundError(
            f"O arquivo {CAMINHO_MODELO_KERAS} não foi encontrado. "
            "Execute primeiro: python train_model.py"
        )

    print(f"Carregando {CAMINHO_MODELO_KERAS}...")

    modelo = tf.keras.models.load_model(
        CAMINHO_MODELO_KERAS,
        compile=False,
    )

    print(
        "Convertendo para TensorFlow Lite com "
        "quantização de faixa dinâmica..."
    )

    # Cria o conversor a partir do modelo Keras.
    conversor = tf.lite.TFLiteConverter.from_keras_model(
        modelo
    )

    # Ativa a otimização pós-treinamento.
    # Sem representative_dataset, aplica quantização
    # de faixa dinâmica nos pesos compatíveis.
    conversor.optimizations = [
        tf.lite.Optimize.DEFAULT
    ]

    # Realiza a conversão.
    modelo_tflite = conversor.convert()

    # Salva o modelo convertido.
    with open(CAMINHO_MODELO_TFLITE, "wb") as arquivo:
        arquivo.write(modelo_tflite)

    tamanho_h5 = os.path.getsize(
        CAMINHO_MODELO_KERAS
    ) / (1024 * 1024)

    tamanho_tflite = os.path.getsize(
        CAMINHO_MODELO_TFLITE
    ) / (1024 * 1024)

    reducao = (
        1 - tamanho_tflite / tamanho_h5
    ) * 100

    print("\nConversão concluída com sucesso!")
    print(f"{CAMINHO_MODELO_KERAS}:     {tamanho_h5:.2f} MB")
    print(f"{CAMINHO_MODELO_TFLITE}: {tamanho_tflite:.2f} MB")
    print(f"Redução aproximada: {reducao:.2f}%")


if __name__ == "__main__":
    main()