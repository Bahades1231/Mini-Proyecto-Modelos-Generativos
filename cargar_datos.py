from pathlib import Path

import matplotlib.pyplot as plt
import tensorflow as tf

CARPETA_PARES = Path("data/pares")
TAMAÑO = 256
BATCH_SIZE = 1
BUFFER_SIZE = 1000


def cargar_imagen(ruta):
    """Lee un PNG del disco y lo devuelve como tensor float32 (alto, ancho, 3)."""
    contenido = tf.io.read_file(ruta)
    imagen = tf.io.decode_png(contenido, channels=3)
    return tf.cast(imagen, tf.float32)


def cargar_par(ruta_boceto, ruta_foto):
    return cargar_imagen(ruta_boceto), cargar_imagen(ruta_foto)


def normalizar(boceto, foto):
    """Lleva los píxeles de [0, 255] a [-1, 1] (el generador usa tanh)."""
    return (boceto / 127.5) - 1.0, (foto / 127.5) - 1.0


def jitter_aleatorio(boceto, foto):
    """Aumento de datos: agranda a 286x286, recorta a 256x256 al azar y
    voltea horizontalmente. Se aplica IGUAL al boceto y a la foto."""
    boceto = tf.image.resize(boceto, [286, 286],
                             method=tf.image.ResizeMethod.NEAREST_NEIGHBOR)
    foto = tf.image.resize(foto, [286, 286],
                           method=tf.image.ResizeMethod.NEAREST_NEIGHBOR)

    apilado = tf.stack([boceto, foto], axis=0)
    recorte = tf.image.random_crop(apilado, size=[2, TAMAÑO, TAMAÑO, 3])
    boceto, foto = recorte[0], recorte[1]

    if tf.random.uniform(()) > 0.5:
        boceto = tf.image.flip_left_right(boceto)
        foto = tf.image.flip_left_right(foto)
    return boceto, foto


def crear_dataset(subconjunto, entrenamiento):
    """subconjunto: 'train' o 'test'. Devuelve un tf.data.Dataset de pares."""
    carpeta_b = CARPETA_PARES / subconjunto / "boceto"
    carpeta_f = CARPETA_PARES / subconjunto / "foto"

    nombres = sorted(p.name for p in carpeta_b.glob("*.png"))
    rutas_b = [str(carpeta_b / n) for n in nombres]
    rutas_f = [str(carpeta_f / n) for n in nombres]

    ds = tf.data.Dataset.from_tensor_slices((rutas_b, rutas_f))
    ds = ds.map(cargar_par, num_parallel_calls=tf.data.AUTOTUNE)
    if entrenamiento:
        ds = ds.map(jitter_aleatorio, num_parallel_calls=tf.data.AUTOTUNE)
    ds = ds.map(normalizar, num_parallel_calls=tf.data.AUTOTUNE)
    if entrenamiento:
        ds = ds.shuffle(BUFFER_SIZE)
    return ds.batch(BATCH_SIZE)


if __name__ == "__main__":
    ds = crear_dataset("train", entrenamiento=True)
    fig, ejes = plt.subplots(3, 2, figsize=(6, 9))
    for fila, (boceto, foto) in enumerate(ds.take(3)):
        ejes[fila, 0].imshow((boceto[0] + 1) / 2)
        ejes[fila, 0].set_title("Boceto (entrada)")
        ejes[fila, 1].imshow((foto[0] + 1) / 2)
        ejes[fila, 1].set_title("Foto (objetivo)")
        ejes[fila, 0].axis("off")
        ejes[fila, 1].axis("off")
    plt.tight_layout()
    plt.show()