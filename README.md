# Bosquejo a Imagen

Archivo pares.zip: https://drive.google.com/drive/folders/1bw8UdkC-Fo-0d2DkEVfvrKDvFil_Jesh?usp=sharing
Link collab de entrenamieno y validación: https://colab.research.google.com/drive/14hIf0qaXkEfg3stnjZBDzUDijcWPpEXf?usp=sharing
Link Dataset: https://vision.cs.utexas.edu/projects/finegrained/utzap50k/

Proyecto de modelos generativos basado en **Pix2Pix**, cuyo objetivo es transformar bocetos en imágenes realistas mediante una arquitectura de redes neuronales compuesta por un generador U-Net y un discriminador PatchGAN.

## 1. Estructura del proyecto

Los archivos principales son:

| Archivo | Función |
|---|---|
| `cargar_datos.py` | Carga las imágenes, aplica aumentación de datos y prepara los conjuntos de entrenamiento y prueba. |
| `modelo.py` | Define el generador U-Net y el discriminador PatchGAN. |
| `04-entrenar.py` | Entrena el modelo y guarda sus pesos, muestras y pérdidas. |
| `05-validar.py` | Evalúa el modelo mediante métricas y genera comparaciones y gráficas. |
| `demo.py` | Permite utilizar el modelo mediante una interfaz de Gradio. |
| `pares.zip` | Contiene el conjunto de imágenes emparejadas para entrenar y probar el modelo. |

### Estructura esperada de los datos

El archivo `pares.zip` contiene directamente las carpetas `train` y `test`. Cada una contiene dos carpetas: `boceto` y `foto`.

La estructura que debe tener el proyecto después de organizar los datos es:

```text
bosquejo-a-imagen/
│
├── 04-entrenar.py
├── 05-validar.py
├── cargar_datos.py
├── modelo.py
├── demo.py
├── pares.zip
│
├── data/
│   └── pares/
│       ├── train/
│       │   ├── boceto/
│       │   └── foto/
│       │
│       └── test/
│           ├── boceto/
│           └── foto/
│
└── resultados/
    ├── checkpoints/
    │   └── generador.weights.h5
    ├── perdidas.csv
    ├── validacion_comparacion.png
    └── curvas_perdida.png
```

**Importante:** el ZIP no contiene una carpeta llamada `pares`. Por eso, después de extraerlo, es necesario crear `data/pares` y colocar dentro las carpetas `train` y `test`. Esta organización es necesaria porque `cargar_datos.py` busca los datos en esa ubicación.

## 2. Instalación de dependencias

Para ejecutar el proyecto localmente, se recomienda utilizar un entorno virtual de Python.

Instalar las dependencias:

```bash
pip install tensorflow numpy opencv-python matplotlib gradio
```

En Google Colab, TensorFlow suele estar disponible en el entorno. Gradio también debe estar instalado para ejecutar la demostración; si no está disponible, puede instalarse con:

```python
!pip install gradio
```

## 3. Preparación del conjunto de datos

Antes de entrenar, se debe extraer `pares.zip` y organizar las carpetas de la siguiente manera:

```text
data/
└── pares/
    ├── train/
    │   ├── boceto/
    │   └── foto/
    │
    └── test/
        ├── boceto/
        └── foto/
```

- `train`: contiene los pares de imágenes utilizados para entrenar el modelo.
- `test`: contiene los pares reservados para evaluar el rendimiento del modelo.

El conjunto de datos utilizado está compuesto por 1000 pares de imágenes: 900 para entrenamiento y 100 para prueba.

## 4. Entrenamiento del modelo

El entrenamiento se realiza mediante `04-entrenar.py`.

Este script utiliza los datos de entrenamiento para que el generador aprenda a transformar bocetos en imágenes. El discriminador participa en el proceso comparando las imágenes generadas con las imágenes reales.

El entrenamiento está configurado para 40 épocas.

Durante este proceso se guardan:
- Los pesos del generador y los checkpoints correspondientes.
- Las muestras de imágenes generadas durante el entrenamiento.
- El archivo `perdidas.csv`, que contiene información de las pérdidas del modelo.

El entrenamiento puede reanudarse desde el último checkpoint disponible cuando el código lo contempla.

Para iniciar el entrenamiento localmente, ejecutar desde la carpeta principal del proyecto:

```bash
python 04-entrenar.py
```

### Entrenamiento en Google Colab

Se recomienda utilizar Google Colab con una GPU disponible, preferiblemente una Tesla T4.

**Paso 1. Conectar Google Drive**

```python
from google.colab import drive
drive.mount('/content/drive')
```

**Paso 2. Preparar el proyecto y copiar los archivos**

Se crea la carpeta de trabajo en Colab, se copian los scripts desde Drive y se extrae el conjunto de datos.

```python
!mkdir -p /content/proyecto/data
!cp /content/drive/MyDrive/bosquejo-a-imagen/*.py /content/proyecto/
!unzip -q /content/drive/MyDrive/bosquejo-a-imagen/pares.zip -d /content/proyecto/data
%cd /content/proyecto
```

**Paso 3. Organizar las carpetas del conjunto de datos**

Como el ZIP contiene directamente `train` y `test`, se deben colocar dentro de `data/pares`.

```python
!mkdir -p /content/proyecto/data/pares
!mv /content/proyecto/data/train /content/proyecto/data/pares/
!mv /content/proyecto/data/test /content/proyecto/data/pares/
```

**Paso 4. Preparar la carpeta de resultados**

Se crea la carpeta de resultados en Google Drive y se conecta con la carpeta de trabajo de Colab. De esta manera, los resultados del entrenamiento quedan almacenados en Drive.

```python
!mkdir -p /content/drive/MyDrive/bosquejo-a-imagen/resultados
!ln -s /content/drive/MyDrive/bosquejo-a-imagen/resultados /content/proyecto/resultados
```

**Paso 5. Crear la ruta adicional utilizada por la validación**

El script `05-validar.py` busca los resultados dentro de una carpeta llamada `bosquejo-a-imagen`. Para evitar errores de ubicación, se crea esa carpeta y se conecta con la misma carpeta de resultados de Drive.

```python
!mkdir -p /content/proyecto/bosquejo-a-imagen
!ln -s /content/drive/MyDrive/bosquejo-a-imagen/resultados /content/proyecto/bosquejo-a-imagen/resultados
```

**Paso 6. Ejecutar el entrenamiento**

```python
!python 04-entrenar.py
```

Al finalizar, los pesos y los resultados del entrenamiento estarán almacenados en Google Drive.

## 5. Validación del modelo

Una vez terminado el entrenamiento, se ejecuta `05-validar.py`.

Este script carga los pesos del generador y utiliza las imágenes del conjunto de prueba para evaluar la calidad de las imágenes producidas.

Calcula las siguientes métricas:

- **L1:** mide la diferencia promedio entre los píxeles de la imagen generada y la imagen real. Un valor menor indica menor diferencia.
- **PSNR:** mide la relación entre la señal y el ruido de la imagen. Generalmente, un valor mayor indica una mejor reconstrucción.
- **SSIM:** compara la similitud estructural entre las imágenes. Sus valores se interpretan entre 0 y 1; cuanto más cercano a 1, mayor similitud.

También genera dos archivos:

| Archivo | Contenido |
|---|---|
| `validacion_comparacion.png` | Presenta el boceto de entrada, la fotografía real y la imagen generada por el modelo. |
| `curvas_perdida.png` | Muestra las curvas de pérdida del generador y del discriminador durante el entrenamiento. |

Para ejecutar la validación en Google Colab, primero se copia el script desde Drive:

```python
!cp /content/drive/MyDrive/bosquejo-a-imagen/05-validar.py /content/proyecto/
```

Después se ejecuta:

```python
!python 05-validar.py
```

Los resultados se guardan en la carpeta `resultados` de Google Drive.

## 6. Demostración del modelo

El archivo `demo.py` permite utilizar el modelo mediante una interfaz de Gradio.

Su objetivo es facilitar la interacción con el generador: se proporciona un boceto como entrada y el modelo produce una imagen a partir de él.

Para ejecutar la demostración localmente:

```bash
python demo.py
```

En Google Colab, copiar el archivo desde Drive:

```python
!cp /content/drive/MyDrive/bosquejo-a-imagen/demo.py /content/proyecto/
```

Ejecutar la demostración:

```python
!python demo.py
```

La interfaz de Gradio permitirá cargar un boceto y observar el resultado generado.

## 7. ¿Cómo utilizar un modelo ya entrenado sin volver a entrenarlo?

**No es necesario ejecutar el entrenamiento cada vez que se quiera utilizar el modelo.**

Una vez que el entrenamiento ha finalizado, se pueden reutilizar los pesos guardados para realizar demostraciones o generar nuevas imágenes.

Para ello, es necesario conservar en Google Drive los archivos del modelo entrenado, especialmente:

```text
resultados/
└── checkpoints/
    └── generador.weights.h5
```

También se deben conservar los scripts necesarios para cargar el generador, como `modelo.py` y `demo.py`.

### Utilizar el modelo previamente entrenado en Google Colab

**Paso 1. Conectar Google Drive**

```python
from google.colab import drive
drive.mount('/content/drive')
```

**Paso 2. Copiar los scripts necesarios**

```python
!mkdir -p /content/proyecto
!cp /content/drive/MyDrive/bosquejo-a-imagen/*.py /content/proyecto/
%cd /content/proyecto
```

**Paso 3. Conectar la carpeta de resultados**

```python
!ln -s /content/drive/MyDrive/bosquejo-a-imagen/resultados /content/proyecto/resultados
!mkdir -p /content/proyecto/bosquejo-a-imagen
!ln -s /content/drive/MyDrive/bosquejo-a-imagen/resultados /content/proyecto/bosquejo-a-imagen/resultados
```

**Paso 4. Ejecutar la demostración**

```python
!python demo.py
```

De esta manera, se utiliza el generador con los pesos que ya fueron entrenados, sin necesidad de ejecutar nuevamente `04-entrenar.py`.

**Importante:** para esta demostración no es necesario extraer `pares.zip` ni ejecutar la validación. Solo se necesitan los scripts correspondientes y los pesos del modelo previamente entrenado.

## 8. Resultados del proyecto

Los resultados se almacenan en la carpeta `resultados`, que debe conservarse para poder utilizar posteriormente el modelo y consultar su rendimiento.

Entre los archivos más importantes se encuentran:

- `generador.weights.h5`: pesos del generador entrenado.
- `perdidas.csv`: registro de las pérdidas durante el entrenamiento.
- `validacion_comparacion.png`: comparación entre bocetos, fotografías reales e imágenes generadas.
- `curvas_perdida.png`: gráficas de las pérdidas del modelo.

Estos archivos permiten consultar el entrenamiento, evaluar el rendimiento y reutilizar el modelo sin necesidad de entrenarlo nuevamente.

---

## 9. Consideraciones para Google Colab

- Los archivos almacenados directamente en `/content/proyecto` son temporales y pueden desaparecer cuando se reinicia o termina la sesión de Colab.
- Los resultados guardados en Google Drive permanecen disponibles.
- Si se inicia una nueva sesión de Colab, es necesario volver a conectar Drive, copiar los scripts y crear nuevamente los accesos directos a la carpeta de resultados.
- Si se desea entrenar desde cero en una sesión nueva, también se debe volver a extraer y organizar el conjunto de datos.
- Si solamente se desea utilizar un modelo previamente entrenado, no es necesario cargar el conjunto de datos ni ejecutar el entrenamiento.
- Se recomienda conservar los checkpoints y el archivo `perdidas.csv` para mantener el historial y poder continuar o analizar el entrenamiento.

## 10. Orden general de ejecución

| Etapa | Archivo o acción | Objetivo |
|---|---|---|
| Preparación | `cargar_datos.py` y `pares.zip` | Preparar los datos. |
| Entrenamiento | `04-entrenar.py` | Entrenar el modelo y guardar sus pesos. |
| Validación | `05-validar.py` | Evaluar el modelo y generar métricas y gráficas. |
| Demostración | `demo.py` | Probar el modelo con nuevos bocetos. |

**Flujo recomendado:** preparar los datos → entrenar → validar → utilizar la demostración.

Si el modelo ya está entrenado, se puede ir directamente a la demostración utilizando los pesos guardados.
