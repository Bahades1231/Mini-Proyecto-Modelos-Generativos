from pathlib import Path

import cv2
import gradio as gr
import numpy as np
import tensorflow as tf

from modelo import crear_generador

TAMAÑO = 256
BASE_DIR = Path(__file__).resolve().parent

POSIBLES = [
    BASE_DIR / "resultados" / "checkpoints" / "generador.weights.h5",
    BASE_DIR / "bosquejo-a-imagen" / "resultados" / "checkpoints" / "generador.weights.h5",
]
PESOS = next((p for p in POSIBLES if p.exists()), None)
if PESOS is None:
    raise FileNotFoundError("No encontré generador.weights.h5")

generador = crear_generador()
generador.load_weights(str(PESOS))
print("Pesos cargados desde", PESOS)


def recorte_central(img):
    alto, ancho = img.shape[:2]
    lado = min(alto, ancho)
    y, x = (alto - lado) // 2, (ancho - lado) // 2
    return img[y:y + lado, x:x + lado]


def crear_boceto(foto_rgb):
    """Mismo proceso que usaste para crear el dataset (Canny)."""
    gris = cv2.cvtColor(foto_rgb, cv2.COLOR_RGB2GRAY)
    gris = cv2.GaussianBlur(gris, (5, 5), 0)
    bordes = cv2.Canny(gris, 50, 150)
    boceto = cv2.bitwise_not(bordes)
    return cv2.cvtColor(boceto, cv2.COLOR_GRAY2RGB)


def generar(imagen, tipo_entrada):
    if imagen is None:
        return None, None
    img = recorte_central(imagen[:, :, :3])
    img = cv2.resize(img, (TAMAÑO, TAMAÑO), interpolation=cv2.INTER_AREA)

    if tipo_entrada.startswith("Foto"):
        boceto = crear_boceto(img)
    else:
        boceto = img

    x = boceto.astype(np.float32) / 127.5 - 1.0
    x = x[np.newaxis, ...]
    y = generador(x, training=True)[0].numpy()
    generada = ((y + 1) / 2).clip(0, 1)
    return boceto, (generada * 255).astype(np.uint8)


demo = gr.Interface(
    fn=generar,
    inputs=[
        gr.Image(label="Sube una imagen"),
        gr.Radio(["Foto (se convierte a boceto)", "Boceto ya dibujado"],
                 value="Foto (se convierte a boceto)", label="Tipo de entrada"),
    ],
    outputs=[gr.Image(label="Boceto usado"), gr.Image(label="Generada por el modelo")],
    title="Boceto a imagen (pix2pix)",
)

if __name__ == "__main__":
    demo.launch(share=True)