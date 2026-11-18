import random
from pathlib import Path

import cv2

CARPETA_ORIGINALES = Path("data/originales/Boots")
CARPETA_SALIDA = Path("data/pares")
TAMAÑO = 256
PORC_TEST = 0.1
SEMILLA = 42
MAX_FOTOS = 1000


def recorte_central(img):
    """Recorta la imagen a un cuadrado desde el centro."""
    alto, ancho = img.shape[:2]
    lado = min(alto, ancho)
    y = (alto - lado) // 2
    x = (ancho - lado) // 2
    return img[y:y + lado, x:x + lado]


def crear_boceto(foto):
    """Convierte una foto en un bosquejo de líneas negras sobre fondo blanco."""
    gris = cv2.cvtColor(foto, cv2.COLOR_BGR2GRAY)
    gris = cv2.GaussianBlur(gris, (5, 5), 0)
    bordes = cv2.Canny(gris, 50, 150)
    boceto = cv2.bitwise_not(bordes)
    return cv2.cvtColor(boceto, cv2.COLOR_GRAY2BGR)


def main():
    random.seed(SEMILLA)

    extensiones = {".jpg", ".jpeg", ".png"}
    rutas = sorted(p for p in CARPETA_ORIGINALES.rglob("*")
                   if p.suffix.lower() in extensiones)
    if not rutas:
        print(f"No encontré fotos en {CARPETA_ORIGINALES}")
        return

    random.shuffle(rutas)
    rutas = rutas[:MAX_FOTOS]
    n_test = max(1, int(len(rutas) * PORC_TEST))

    for subconjunto in ("train", "test"):
        for tipo in ("foto", "boceto"):
            (CARPETA_SALIDA / subconjunto / tipo).mkdir(parents=True, exist_ok=True)

    guardadas = 0
    for i, ruta in enumerate(rutas):
        foto = cv2.imread(str(ruta))
        if foto is None:
            print(f"No pude leer {ruta.name}, la salto")
            continue

        foto = recorte_central(foto)
        foto = cv2.resize(foto, (TAMAÑO, TAMAÑO), interpolation=cv2.INTER_AREA)
        boceto = crear_boceto(foto)

        subconjunto = "test" if i < n_test else "train"
        nombre = f"{i:05d}.png"
        cv2.imwrite(str(CARPETA_SALIDA / subconjunto / "foto" / nombre), foto)
        cv2.imwrite(str(CARPETA_SALIDA / subconjunto / "boceto" / nombre), boceto)
        guardadas += 1

    print(f"Listo: {guardadas} pares creados ({n_test} en test, el resto en train)")


if __name__ == "__main__":
    main()