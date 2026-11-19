import tensorflow as tf
from tensorflow.keras import layers

TAMAÑO = 256
CANALES = 3


def bloque_bajada(filtros, TAMAÑO_kernel, usar_batchnorm=True):
    """Convolución que reduce la imagen a la mitad (alto y ancho)."""
    inicializador = tf.random_normal_initializer(0.0, 0.02)
    bloque = tf.keras.Sequential()
    bloque.add(layers.Conv2D(filtros, TAMAÑO_kernel, strides=2, padding="same",
                             kernel_initializer=inicializador, use_bias=False))
    if usar_batchnorm:
        bloque.add(layers.BatchNormalization())
    bloque.add(layers.LeakyReLU())
    return bloque


def bloque_subida(filtros, TAMAÑO_kernel, usar_dropout=False):
    """Convolución transpuesta que duplica el tamaño de la imagen."""
    inicializador = tf.random_normal_initializer(0.0, 0.02)
    bloque = tf.keras.Sequential()
    bloque.add(layers.Conv2DTranspose(filtros, TAMAÑO_kernel, strides=2,
                                      padding="same",
                                      kernel_initializer=inicializador,
                                      use_bias=False))
    bloque.add(layers.BatchNormalization())
    if usar_dropout:
        bloque.add(layers.Dropout(0.5))
    bloque.add(layers.ReLU())
    return bloque


def crear_generador():
    """Generador U-Net: recibe un boceto y devuelve una foto de bota."""
    entradas = layers.Input(shape=[TAMAÑO, TAMAÑO, CANALES])

    pila_bajada = [
        bloque_bajada(64, 4, usar_batchnorm=False),
        bloque_bajada(128, 4),
        bloque_bajada(256, 4),
        bloque_bajada(512, 4),
        bloque_bajada(512, 4),
        bloque_bajada(512, 4),
        bloque_bajada(512, 4),
        bloque_bajada(512, 4),
    ]

    pila_subida = [
        bloque_subida(512, 4, usar_dropout=True),
        bloque_subida(512, 4, usar_dropout=True),
        bloque_subida(512, 4, usar_dropout=True),
        bloque_subida(512, 4),
        bloque_subida(256, 4),
        bloque_subida(128, 4),
        bloque_subida(64, 4),
    ]

    inicializador = tf.random_normal_initializer(0.0, 0.02)
    ultima = layers.Conv2DTranspose(CANALES, 4, strides=2, padding="same",
                                    kernel_initializer=inicializador,
                                    activation="tanh")

    x = entradas
    saltos = []
    for bajada in pila_bajada:
        x = bajada(x)
        saltos.append(x)

    saltos = reversed(saltos[:-1])

    for subida, salto in zip(pila_subida, saltos):
        x = subida(x)
        x = layers.Concatenate()([x, salto])

    x = ultima(x)
    return tf.keras.Model(inputs=entradas, outputs=x, name="generador")


def crear_discriminador():
    """Discriminador PatchGAN: recibe (boceto, imagen) y juzga parches de la imagen."""
    inicializador = tf.random_normal_initializer(0.0, 0.02)

    boceto = layers.Input(shape=[TAMAÑO, TAMAÑO, CANALES], name="boceto")
    imagen = layers.Input(shape=[TAMAÑO, TAMAÑO, CANALES], name="imagen")

    x = layers.Concatenate()([boceto, imagen])

    x = bloque_bajada(64, 4, usar_batchnorm=False)(x)
    x = bloque_bajada(128, 4)(x)
    x = bloque_bajada(256, 4)(x)

    x = layers.ZeroPadding2D()(x)
    x = layers.Conv2D(512, 4, strides=1, kernel_initializer=inicializador,
                      use_bias=False)(x)
    x = layers.BatchNormalization()(x)
    x = layers.LeakyReLU()(x)

    x = layers.ZeroPadding2D()(x)
    salida = layers.Conv2D(1, 4, strides=1,
                           kernel_initializer=inicializador)(x)

    return tf.keras.Model(inputs=[boceto, imagen], outputs=salida,
                          name="discriminador")


if __name__ == "__main__":
    generador = crear_generador()
    discriminador = crear_discriminador()
    generador.summary()
    discriminador.summary()

    boceto_prueba = tf.random.normal([1, TAMAÑO, TAMAÑO, CANALES])
    generada = generador(boceto_prueba, training=False)
    veredicto = discriminador([boceto_prueba, generada], training=False)
    print("Salida del generador:", generada.shape)
    print("Salida del discriminador:", veredicto.shape)