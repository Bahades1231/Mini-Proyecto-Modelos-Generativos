from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf

from cargar_datos import crear_dataset
from modelo import crear_generador

BASE_DIR = Path(__file__).resolve().parent
CARPETA_RESULTADOS = BASE_DIR / "bosquejo-a-imagen" / "resultados"
PESOS = CARPETA_RESULTADOS / "checkpoints" / "generador.weights.h5"
N_EJEMPLOS = 8

tf.random.set_seed(0)


def a_01(img):
    """Lleva una imagen de [-1, 1] a [0, 1]."""
    return tf.clip_by_value((img + 1) / 2, 0.0, 1.0)


def calcular_metricas(generador, ds_test):
    l1_m11, l1_01 = [], []
    psnr_tf, psnr_manual, ssim_gen = [], [], []
    psnr_base, ssim_base = [], []

    for boceto, foto in ds_test:
        gen = generador(boceto, training=True)
        
        l1_m11.append(float(tf.reduce_mean(tf.abs(gen - foto))))

        g = (gen + 1.0) / 2.0
        f = (foto + 1.0) / 2.0
        b = (boceto + 1.0) / 2.0
        l1_01.append(float(tf.reduce_mean(tf.abs(g - f))))

        mse = float(tf.reduce_mean(tf.square(g - f)))
        psnr_manual.append(10.0 * np.log10(1.0 / mse))
        psnr_tf.append(float(tf.image.psnr(g, f, max_val=1.0)[0]))
        ssim_gen.append(float(tf.image.ssim(g, f, max_val=1.0)[0]))

        psnr_base.append(float(tf.image.psnr(b, f, max_val=1.0)[0]))
        ssim_base.append(float(tf.image.ssim(b, f, max_val=1.0)[0]))

    print(f"\nMétricas sobre {len(l1_m11)} imágenes de test")
    print(f"L1 en [-1,1] (comparable con el entrenamiento): {np.mean(l1_m11):.3f}")
    print(f"L1 en [0,1]:                                    {np.mean(l1_01):.3f}")
    print(f"PSNR generador (tf):     {np.mean(psnr_tf):.2f} dB")
    print(f"PSNR generador (manual): {np.mean(psnr_manual):.2f} dB")
    print(f"SSIM generador:          {np.mean(ssim_gen):.3f}")
    print(f"Referencia (boceto vs foto): PSNR {np.mean(psnr_base):.2f} dB | SSIM {np.mean(ssim_base):.3f}")

def guardar_comparacion(generador, ds_test):
    muestras = list(ds_test.take(N_EJEMPLOS))
    fig, ejes = plt.subplots(N_EJEMPLOS, 3, figsize=(9, 3 * N_EJEMPLOS))
    titulos = ["Boceto (entrada)", "Foto real", "Generada por el modelo"]
    for fila, (boceto, foto) in enumerate(muestras):
        generada = generador(boceto, training=True)
        for col, img in enumerate([boceto, foto, generada]):
            ejes[fila, col].imshow(a_01(img[0]))
            ejes[fila, col].axis("off")
            if fila == 0:
                ejes[fila, col].set_title(titulos[col])
    plt.tight_layout()
    salida = CARPETA_RESULTADOS / "validacion_comparacion.png"
    salida.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(salida, dpi=120)
    plt.close(fig)
    print("Comparación guardada en", salida)


def guardar_curvas():
    datos = np.genfromtxt(CARPETA_RESULTADOS / "perdidas.csv", delimiter=",", names=True)
    fig, ejes = plt.subplots(1, 2, figsize=(11, 4))

    ejes[0].plot(datos["epoca"], datos["perdida_l1"])
    ejes[0].set_title("Pérdida L1 del generador")
    ejes[0].set_xlabel("Época")

    ejes[1].plot(datos["epoca"], datos["perdida_gan"], label="Generador (GAN)")
    ejes[1].plot(datos["epoca"], datos["perdida_d"], label="Discriminador")
    ejes[1].set_title("Pérdidas adversarias")
    ejes[1].set_xlabel("Época")
    ejes[1].legend()

    plt.tight_layout()
    salida = CARPETA_RESULTADOS / "curvas_perdida.png"
    salida.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(salida, dpi=120)
    plt.close(fig)
    print("Curvas guardadas en", salida)


def main():
    generador = crear_generador()
    generador.load_weights(str(PESOS))
    ds_test = crear_dataset("test", entrenamiento=False)

    calcular_metricas(generador, ds_test)
    guardar_comparacion(generador, ds_test)
    guardar_curvas()


if __name__ == "__main__":
    main()