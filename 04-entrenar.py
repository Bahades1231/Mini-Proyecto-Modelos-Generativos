import csv
import time
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf

from cargar_datos import crear_dataset
from modelo import crear_generador, crear_discriminador

EPOCAS = 40
LAMBDA_L1 = 100
PASOS_PRUEBA = None
GUARDAR_CADA = 5

CARPETA_RESULTADOS = Path("resultados")
CARPETA_MUESTRAS = CARPETA_RESULTADOS / "muestras"
CARPETA_CHECKPOINTS = CARPETA_RESULTADOS / "checkpoints"

generador = crear_generador()
discriminador = crear_discriminador()

optimizador_g = tf.keras.optimizers.Adam(2e-4, beta_1=0.5)
optimizador_d = tf.keras.optimizers.Adam(2e-4, beta_1=0.5)

perdida_bce = tf.keras.losses.BinaryCrossentropy(from_logits=True)


def perdida_generador(salida_disc_generada, generada, objetivo):
    perdida_gan = perdida_bce(tf.ones_like(salida_disc_generada), salida_disc_generada)
    perdida_l1 = tf.reduce_mean(tf.abs(objetivo - generada))
    total = perdida_gan + LAMBDA_L1 * perdida_l1
    return total, perdida_gan, perdida_l1


def perdida_discriminador(salida_real, salida_generada):
    p_real = perdida_bce(tf.ones_like(salida_real), salida_real)
    p_gen = perdida_bce(tf.zeros_like(salida_generada), salida_generada)
    return p_real + p_gen


@tf.function
def paso_entrenamiento(boceto, foto):
    with tf.GradientTape() as cinta_g, tf.GradientTape() as cinta_d:
        generada = generador(boceto, training=True)

        salida_real = discriminador([boceto, foto], training=True)
        salida_gen = discriminador([boceto, generada], training=True)

        perd_g, perd_gan, perd_l1 = perdida_generador(salida_gen, generada, foto)
        perd_d = perdida_discriminador(salida_real, salida_gen)

    grad_g = cinta_g.gradient(perd_g, generador.trainable_variables)
    grad_d = cinta_d.gradient(perd_d, discriminador.trainable_variables)
    optimizador_g.apply_gradients(zip(grad_g, generador.trainable_variables))
    optimizador_d.apply_gradients(zip(grad_d, discriminador.trainable_variables))
    return perd_g, perd_gan, perd_l1, perd_d


def guardar_muestras(epoca, muestras):
    """Guarda una imagen con boceto | foto real | foto generada."""
    fig, ejes = plt.subplots(len(muestras), 3, figsize=(9, 3 * len(muestras)))
    titulos = ["Boceto", "Foto real", "Generada"]
    for fila, (boceto, foto) in enumerate(muestras):
        generada = generador(boceto, training=True)
        for col, img in enumerate([boceto, foto, generada]):
            ejes[fila, col].imshow((img[0] + 1) / 2)
            ejes[fila, col].axis("off")
            if fila == 0:
                ejes[fila, col].set_title(titulos[col])
    plt.tight_layout()
    plt.savefig(CARPETA_MUESTRAS / f"epoca_{epoca:03d}.png")
    plt.close(fig)


def guardar_progreso(epoca):
    generador.save_weights(str(CARPETA_CHECKPOINTS / "generador.weights.h5"))
    discriminador.save_weights(str(CARPETA_CHECKPOINTS / "discriminador.weights.h5"))
    (CARPETA_CHECKPOINTS / "epoca.txt").write_text(str(epoca))


def cargar_progreso():
    """Si hay un entrenamiento previo, retoma desde la última época guardada."""
    archivo = CARPETA_CHECKPOINTS / "epoca.txt"
    if archivo.exists():
        generador.load_weights(str(CARPETA_CHECKPOINTS / "generador.weights.h5"))
        discriminador.load_weights(str(CARPETA_CHECKPOINTS / "discriminador.weights.h5"))
        epoca = int(archivo.read_text())
        print(f"Retomando desde la época {epoca}")
        return epoca
    return 0


def main():
    print("GPU disponible:", tf.config.list_physical_devices("GPU"))
    CARPETA_MUESTRAS.mkdir(parents=True, exist_ok=True)
    CARPETA_CHECKPOINTS.mkdir(parents=True, exist_ok=True)

    ds_train = crear_dataset("train", entrenamiento=True)
    ds_test = crear_dataset("test", entrenamiento=False)
    muestras = list(ds_test.take(3))

    epoca_inicial = cargar_progreso()
    archivo_csv = CARPETA_RESULTADOS / "perdidas.csv"
    if epoca_inicial == 0:
        archivo_csv.write_text("epoca,perdida_g,perdida_gan,perdida_l1,perdida_d\n")

    for epoca in range(epoca_inicial + 1, EPOCAS + 1):
        inicio = time.time()
        ds = ds_train.take(PASOS_PRUEBA) if PASOS_PRUEBA else ds_train

        valores = []
        for boceto, foto in ds:
            valores.append([float(v) for v in paso_entrenamiento(boceto, foto)])
        prom = np.mean(valores, axis=0)

        with open(archivo_csv, "a", newline="") as f:
            csv.writer(f).writerow([epoca, *prom])

        print(f"Época {epoca}/{EPOCAS} | {time.time() - inicio:.0f}s | "
              f"gen={prom[0]:.3f} (gan={prom[1]:.3f}, l1={prom[2]:.3f}) | "
              f"disc={prom[3]:.3f}")

        if epoca == 1 or epoca % GUARDAR_CADA == 0 or epoca == EPOCAS:
            guardar_muestras(epoca, muestras)
            guardar_progreso(epoca)

    generador.save(CARPETA_RESULTADOS / "generador.keras")
    print("Entrenamiento terminado. Modelo guardado en resultados/generador.keras")


if __name__ == "__main__":
    main()