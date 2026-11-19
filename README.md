# Bosquejo a imagen (Pix2Pix) con botas

Miniproyecto 3 de Redes Generativas · Universidad Autónoma de Occidente

Modelo generativo que convierte un **bosquejo de líneas** en la **foto de una bota**. Usa Pix2Pix (GAN condicional) implementado en TensorFlow/Keras: un generador **U-Net** y un discriminador **PatchGAN**, entrenados desde cero con un dataset armado por el grupo.

---

## Archivos pesados (Google Drive)

GitHub no admite archivos de más de 100 MB, por eso estos archivos están en Drive:

- **Dataset de pares boceto/foto (`pares.zip`):** https://drive.google.com/drive/folders/1bw8UdkC-Fo-0d2DkEVfvrKDvFil_Jesh?usp=sharing
- **Carpeta `bosquejo-a-imagen` completa (checkpoints, muestras, `perdidas.csv`, resultados, curvas y validación):** https://drive.google.com/drive/folders/1nr2pkG2iAFvPv1a3cCx5ZnjzQ_4goVeT?usp=sharing
- **Cuaderno de Colab usado para entrenar:** https://colab.research.google.com/drive/1YAK9VNaVkata8bEQkIR-X1IgxiwaHIm5?usp=sharing (Importante cambiar las carpetas o dejarlas tal cual como están en cada celda de codigo)
** NOTA: Si se necesita solamente revisar el modelo, se debe clonar el repositorio y tener las carpetas de data y bosquejo-a-imagen (se descomprimen dentro del repositorio clonado localmente) **

---

## Contenido del repositorio

| Archivo | Para qué sirve |
|---|---|
| `01-crear-dataset.py` | Convierte 1000 fotos de botas en pares boceto/foto de 256×256 (bocetos con Canny). Separa 900 pares en `train` y 100 en `test`. |
| `cargar_datos.py` | Lee los pares del disco, aplica aumento de datos (jitter y volteo), normaliza a [-1, 1] y arma el `tf.data.Dataset`. Lo importan 04 y 05. |
| `modelo.py` | Define el generador (U-Net, 54.4 M de parámetros) y el discriminador (PatchGAN, 2.77 M). Lo importan 04, 05 y `demo.py`. |
| `04-entrenar.py` | Entrena 40 épocas, guarda muestras, pesos y `perdidas.csv`. Retoma desde el último checkpoint si se interrumpe. |
| `05-validar.py` | Calcula L1, PSNR y SSIM sobre los 100 pares de test y genera la cuadrícula de comparación y las curvas de pérdida. |
| `demo.py` | Demo en vivo con Gradio: se sube una imagen y el modelo genera la bota. |
| `requirements.txt` | Dependencias del proyecto. |

> `modelo.py` y `cargar_datos.py` no se ejecutan solos: los importan los demás scripts, así que deben estar en la misma carpeta.

---

## Dataset

- Fotos de la categoría **Boots** de [UT Zappos50K](https://vision.cs.utexas.edu/projects/finegrained/utzap50k/).
- Se eligieron **1000 fotos al azar** (semilla 42), se recortaron al centro y se redujeron a **256×256**.
- El bosquejo de cada foto se genera con **Canny** (desenfoque gaussiano 5×5, umbrales 50 y 150), con líneas negras sobre fondo blanco.
- División: **900 pares de entrenamiento** y **100 de prueba**.

---

## Cómo ejecutarlo

### 1. Instalar dependencias

```bash
python -m venv venv
venv\Scripts\activate        # Windows
# source venv/bin/activate   # Linux / Mac
pip install -r requirements.txt
```

`requirements.txt` debe incluir al menos: `tensorflow`, `numpy`, `opencv-python`, `matplotlib`, `gradio`.

### 2. Preparar los datos

**Opción A: usar el dataset ya creado.** Descarga `pares.zip` del enlace de Drive y descomprímelo y crea la carpeta "data" dentro de `data/` pega pares.zip descomprimida, de modo que quede `data/pares/train/` y `data/pares/test/` (cada una con `boceto/` y `foto/`).

**Opción B: crearlo desde cero.** Descarga UT Zappos50K, copia la carpeta `Boots` a `data/originales/Boots` y ejecuta:

```bash
python 01-crear-dataset.py
```

### 3. Entrenar

```bash
python 04-entrenar.py
```

Se recomienda hacerlo en **Google Colab con GPU (T4)**: en CPU cada época toma unos 13 a 14 minutos (≈ 9 horas en total), y en Colab unos 2 minutos (≈ 72 minutos en total).
En tu carpeta de drive donde harás el entrenamiento (si se hace en collab) deben ir los .py de 04-entrenar.py, cargar_datos.py, modelo.py

Los resultados quedan en `resultados/`. Para una prueba rápida, en `04-entrenar.py` usa `EPOCAS = 2` y `PASOS_PRUEBA = 20`.

#### Entrenar en Google Colab

```python
from google.colab import drive
drive.mount('/content/drive')
```
```python
!mkdir -p /content/proyecto/data
!cp /content/drive/MyDrive/bosquejo-a-imagen/*.py /content/proyecto/
!unzip -q /content/drive/MyDrive/bosquejo-a-imagen/pares.zip -d /content/proyecto/data
%cd /content/proyecto
!ls data/pares
```
```python
!mkdir -p /content/drive/MyDrive/bosquejo-a-imagen/resultados
!ln -s /content/drive/MyDrive/bosquejo-a-imagen/resultados /content/proyecto/resultados
```
```python
!python 04-entrenar.py
```

### 4. Validar

Coloca los pesos entrenados (`generador.weights.h5`) en la carpeta `checkpoints` de resultados (descárgalos del Drive si no entrenaste tú) y ejecuta:

```bash
python 05-validar.py
```

Genera `validacion_comparacion.png` y `curvas_perdida.png` y muestra las métricas en consola.

### 5. Demo en vivo

```bash
pip install gradio
python demo.py
```

Abre el enlace local que aparece (`http://127.0.0.1:7860`). Se puede subir:

- **Una foto de bota:** se convierte a boceto con Canny y el modelo genera la foto.
- **Un boceto ya dibujado:** líneas negras sobre fondo blanco.

`demo.py` busca los pesos en `resultados/checkpoints/generador.weights.h5` o en `bosquejo-a-imagen/resultados/checkpoints/generador.weights.h5`.

---

## Parámetros del experimento

| Parámetro | Valor |
|---|---|
| Fotos usadas | 1000 (900 train / 100 test) |
| Tamaño de imagen | 256×256 |
| Épocas | 40 |
| Batch size | 1 |
| `LAMBDA_L1` | 100 |
| Optimizador | Adam (lr = 2e-4, beta_1 = 0.5), uno por red |
| Pérdida GAN | Entropía cruzada binaria (`from_logits=True`) |
| Aumento de datos | Recorte aleatorio (286→256) y volteo horizontal |
| Generador | U-Net, 54,425,859 parámetros |
| Discriminador | PatchGAN 30×30, 2,770,433 parámetros |

---

## Resultados

Métricas sobre los **100 pares de test** (imágenes que el modelo nunca vio al entrenar):

| Medida | Generador | Referencia (boceto vs foto) |
|---|---|---|
| L1 en [-1, 1] | 0.117 | — |
| L1 en [0, 1] | 0.058 | — |
| PSNR | 19.73 dB | 7.73 dB |
| SSIM | 0.748 | 0.636 |

Diferencia entre train y test (L1, 50 pares cada uno): **0.107 vs 0.122**, sin señales fuertes de sobreajuste.

![Comparación boceto, foto real y generada](imagenes/validacion_comparacion.png)

![Curvas de pérdida](imagenes/curvas_perdida.png)

**Observaciones:**

- El modelo respeta bien la forma: siluetas, cordones, hebillas y patrones del boceto aparecen en la bota generada.
- Las texturas son plausibles (cuero, piel, tela).
- El **color es el punto débil**: un boceto de líneas no contiene información de color, así que el modelo tiende a generar marrones o tonos apagados.

---

## Limitaciones

- Un solo entrenamiento, con un solo juego de hiperparámetros; no se probaron variantes.
- Los bosquejos los genera Canny. No se probó con dibujos hechos a mano, así que el resultado con ellos puede ser peor.
- Validación limitada a 100 imágenes del mismo dataset; no se usó FID ni evaluación con personas.
- Las métricas y la demo usan `training=True` (dropout activo, como en el tutorial de Pix2Pix), por lo que los resultados varían un poco entre ejecuciones.
- El entrenamiento no fija semilla aleatoria, por lo que repetirlo no da números idénticos.
- No se revisó si hay fotos casi idénticas entre train y test.

---

## Referencias

- Isola et al., *Image-to-Image Translation with Conditional Adversarial Networks* (Pix2Pix), 2017.
- [Tutorial de Pix2Pix de TensorFlow](https://www.tensorflow.org/tutorials/generative/pix2pix), en el que se basa la arquitectura.
- [UT Zappos50K](https://vision.cs.utexas.edu/projects/finegrained/utzap50k/), dataset de fotos de zapatos.
