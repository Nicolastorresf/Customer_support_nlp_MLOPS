# Proyecto: Clasificación de Tweets de Atención al Cliente 

## Autor
* **Nombre:** Nicolas Torres
* **Fecha:** Mayo 2025

## Descripción del Proyecto

Este proyecto implementa un pipeline de Machine Learning de extremo a extremo para clasificar tweets de un canal de atención al cliente. El objetivo es tomar mensajes de texto de usuarios, procesarlos a través de una serie de etapas (ingesta, preparación, preprocesamiento NLP, ingeniería de características, y pseudo-etiquetado mediante clustering) y finalmente entrenar un modelo de clasificación supervisado para asignar estos mensajes a categorías relevantes.

La solución está diseñada como un sistema batch, con un fuerte énfasis en la modularidad, reproducibilidad, trazabilidad y buenas prácticas de ingeniería de Machine Learning, siguiendo los lineamientos de la prueba técnica para Ingeniero de Machine Learning de Nequi.

El pipeline ha sido desarrollado en Python y utiliza diversas librerías estándar de data science y NLP.

## Dataset Utilizado

* **Nombre del Dataset:** Customer Support on Twitter
* **Fuente:** Kaggle
* **URL/Slug en Kaggle:** `thoughtvector/customer-support-on-twitter`,'https://www.kaggle.com/datasets/thoughtvector/customer-support-on-twitter' 
* **Archivo Principal en el Dataset:** `twcs.csv`
* **Descripción Breve:** Este dataset público contiene más de 2 millones de tweets que representan interacciones entre clientes y múltiples compañías a través de Twitter.
* **Filtrado Aplicado:** Para este proyecto, el dataset se filtra inicialmente para incluir únicamente los tweets entrantes de clientes (`inbound=True`), resultando en un conjunto de trabajo de aproximadamente 1.6 millones de registros, los cuales se dividen luego en conjuntos de `discovery` (entrenamiento), `validation` y `evaluation`.

## Estructura de Carpetas del Proyecto

```text
customer_support_nlp_MLOPS/
├── data/                     # Datos (IGNORADOS POR GIT gracias a .gitignore)
│   ├── 00_raw/               # Dataset crudo original (ej. twcs.csv)
│   ├── 01_ingested_splits/   # Splits iniciales (discovery_ingested.csv, etc.)
│   ├── 02_prepared_data/     # Datos después de preparación básica
│   ├── 03_preprocessed_text/ # Textos después de limpieza y preprocesamiento NLP
│   ├── 04_features/          # Embeddings (completos y reducidos .npy) y sus IDs (.csv)
│   ├── 05_clustering_outputs/# Asignaciones de clusters (.csv) y reportes de análisis (.txt)
│   └── 06_pseudo_labelled_data/ # Datasets finales con pseudo-etiquetas para clasificación
├── docs/                     # (OPCIONAL) Documentos de propuestas de diseño (.md)
│   ├── DECISIONES_PIPELINE.md
│   ├── PROPUESTA_ARQUITECTURA.md
│   ├── PROPUESTA_CI_CD_CT.md
│   ├── PROPUESTA_CALIDAD_VERSIONADO.md
│   └── PROPUESTA_MONITOREO_SEGURIDAD.md
├── models/                   # Modelos serializados y reportes 
│   ├── classification/       # Modelos de clasificación, encoders, reportes, análisis de errores
│   ├── clustering/           # Modelos KMeans ajustados 
│   └── feature_reduction/    # Modelos PCA ajustados 
├── notebooks/                # Jupyter notebooks para experimentación y análisis exploratorio
│   └── Clustering_Experimentation.ipynb
├── src/                      # Código fuente del pipeline de ML
│   ├── __init__.py
│   ├── all_emoticons_expanded.py # Módulo de utilidad para emojis
│   ├── chat_words.py         # Módulo de utilidad para abreviaturas de chat
│   ├── clustering.py         # Lógica de clustering (KMeans) y pseudo-etiquetado
│   ├── config.py             # Archivo de configuración centralizado (rutas, parámetros)
│   ├── data_ingestion.py     # Script para descarga y división inicial de datos
│   ├── data_preparation.py   # Script para preparación básica de datos
│   ├── feature_engineering.py# Script para generación de embeddings y reducción de dimensionalidad
│   ├── model_training.py     # Script para entrenamiento y evaluación del modelo de clasificación
│   └── preprocessing.py      # Script para limpieza y preprocesamiento de texto NLP
├── .gitignore                # Especifica los archivos y directorios a ignorar por Git
├── README.md                 # Este archivo
└── requirements.txt          # Dependencias del proyecto Python
## Prerrequisitos

* Python 3.10 (o la versión que estés usando)
* Git
* (Recomendado) Un entorno virtual para gestionar las dependencias del proyecto:
    * Para crear un entorno (ej. llamado `nequi_env`):
      ```bash
      python -m venv nequi_env
      ```
    * Para activar el entorno:
      * En Windows (Git Bash o CMD/Powershell):
        ```bash
        nequi_env\Scripts\activate
        ```
      * En Linux/macOS:
        ```bash
        source nequi_env/bin/activate
        ```

## Instalación de Dependencias

Una vez clonado el repositorio y con el entorno virtual activado (si se usa), instala todas las dependencias necesarias ejecutando el siguiente comando en la raíz del proyecto:

```bash
pip install -r requirements.txt

## Configuración

El pipeline es altamente configurable a través del archivo `src/config.py`. Este archivo centraliza:

* Rutas a los datos de entrada y salida para cada etapa del pipeline.
* Nombres de columnas clave utilizados consistentemente a través de los scripts.
* Parámetros para la ingeniería de características, como el modelo de Sentence Transformers a utilizar (`SENTENCE_TRANSFORMER_MODEL`) y el número de componentes para PCA (`PCA_N_COMPONENTS`).
* Parámetros para el clustering, incluyendo el número de clústeres para KMeans (`KMEANS_N_CLUSTERS`) y sus métodos de inicialización.
* Hiperparámetros para el modelo de clasificación (Regresión Logística: `LOGREG_C`, `LOGREG_CLASS_WEIGHT`).
* Configuración global de logging para trazabilidad.

**Antes de la primera ejecución, considera lo siguiente:**

1.  **Dataset de Kaggle:**
    * El script `src/data_ingestion.py` está configurado para intentar descargar el dataset `thoughtvector/customer-support-on-twitter` desde Kaggle utilizando `kagglehub`. Esto requiere que tengas tus credenciales de API de Kaggle configuradas en tu sistema (generalmente un archivo `kaggle.json` en `~/.kaggle/` en Linux/macOS o `C:\Users\<Usuario>\.kaggle\` en Windows, o mediante variables de entorno `KAGGLE_USERNAME` y `KAGGLE_KEY`).
    * **Alternativa Manual:** Si prefieres no usar la descarga automática o tienes problemas con ella, puedes descargar manualmente el archivo `twcs.csv` del dataset desde la página de Kaggle y colocarlo directamente en la carpeta `data/00_raw/` de este proyecto antes de ejecutar el pipeline. El script de ingestión lo detectará y procederá con la división.

2.  **Revisar `src/config.py` (Valores Clave):**
    * **`KMEANS_N_CLUSTERS`**: Este valor (actualmente configurado a `8` por defecto en el script `config.py` que proporcionaste) es crucial para el proceso de pseudo-etiquetado y define el número de categorías temáticas que se intentarán descubrir. Se determinó mediante experimentación documentada en el notebook `notebooks/Clustering_Experimentation.ipynb` (utilizando métricas como el método del codo y el coeficiente de silueta). Si deseas cambiar el número de categorías finales, este es el parámetro principal a modificar, y se recomienda re-ejecutar la experimentación de clustering para validar el nuevo `k`.
    * **`LOGREG_C` y `LOGREG_CLASS_WEIGHT`**: Estos son los hiperparámetros para el modelo de Regresión Logística. Los valores actuales (`C=10.0`, `class_weight='balanced'`) fueron seleccionados tras iteraciones que demostraron un buen rendimiento y generalización, como se evidencia en los resultados del modelo.

## Cómo Ejecutar el Pipeline Completo

Los scripts del pipeline están diseñados para ser ejecutados en secuencia desde la raíz del proyecto. Cada script toma la salida del anterior, procesa los datos y guarda sus resultados para la siguiente etapa. La mayoría de los scripts que operan sobre los diferentes conjuntos de datos (`data_preparation.py`, `preprocessing.py`, `feature_engineering.py`, `clustering.py`) aceptan el argumento `--dataset_type` con las opciones `discovery`, `validation`, `evaluation`, o `all` (para procesar todos los conjuntos aplicables en el orden correcto).

**Orden de Ejecución Recomendado:**

Para asegurar la consistencia (especialmente que los modelos de PCA y KMeans se ajusten solo con datos de `discovery` y se apliquen a los demás), se recomienda el siguiente orden:

1.  **Paso 1: Ingestión de Datos**
    * **Acción:** Descarga (si es necesario) el dataset crudo desde Kaggle y lo divide en los conjuntos iniciales `discovery`, `validation`, y `evaluation`.
    * **Comando:**
        ```bash
        python src/data_ingestion.py
        ```
    * **Salida:** Archivos `.csv` (ej. `discovery_ingested.csv`) en la carpeta `data/01_ingested_splits/`.

2.  **Paso 2: Preparación de Datos**
    * **Acción:** Realiza limpieza básica de datos, conversión de tipos y creación de características iniciales (como `tweet_length`).
    * **Comando:**
        ```bash
        python src/data_preparation.py --dataset_type all
        ```
    * **Salida:** Archivos `.csv` (ej. `discovery_prepared.csv`) en la carpeta `data/02_prepared_data/`.

3.  **Paso 3: Preprocesamiento de Texto NLP**
    * **Acción:** Aplica un conjunto exhaustivo de técnicas de limpieza de texto y normalización lingüística, incluyendo manejo de múltiples idiomas.
    * **Comando:**
        ```bash
        python src/preprocessing.py --dataset_type all
        ```
    * **Salida:** Archivos `.csv` (ej. `discovery_preprocessed.csv`) con el texto limpio en `data/03_preprocessed_text/`.

4.  **Paso 4: Ingeniería de Características**
    * **Acción:** Genera embeddings de texto usando un modelo pre-entrenado de Sentence Transformers y luego aplica reducción de dimensionalidad mediante PCA.
    * **Comando:**
        ```bash
        python src/feature_engineering.py --dataset_type all
        ```
    * **Importante:** El modelo PCA se ajusta **únicamente** con los datos de `discovery` y se guarda. Este mismo modelo PCA ajustado se utiliza luego para transformar los datos de `discovery`, `validation`, y `evaluation`, asegurando consistencia.
    * **Salida:** Embeddings completos y reducidos (`.npy`) y sus IDs (`.csv`) en `data/04_features/`. El objeto del modelo PCA ajustado (`.joblib`) se guarda en `models/feature_reduction/`.

5.  **Paso 5: Clustering y Pseudo-Etiquetado**
    * **Acción:** Aplica el algoritmo KMeans a los embeddings reducidos para agrupar los tweets en clústeres temáticos. Luego, asigna etiquetas de categoría legibles a estos clústeres basándose en un mapeo manual definido (ver `define_category_map()` en `src/clustering.py`).
    * **Nota:** El número de clústeres (`k`) para KMeans se define en `src/config.py` (variable `KMEANS_N_CLUSTERS`), idealmente determinado a través del análisis en `notebooks/Clustering_Experimentation.ipynb`.
    * **Comando:**
        ```bash
        python src/clustering.py --dataset_type all
        ```
    * **Importante:** El modelo KMeans se ajusta **únicamente** con los datos (embeddings reducidos) de `discovery` y se guarda. Este mismo modelo KMeans ajustado se utiliza para predecir los clústeres para los conjuntos `discovery`, `validation`, y `evaluation`.
    * **Salida:** Archivos con las asignaciones de clúster (`.csv`) en `data/05_clustering_outputs/`. Los datasets finales pseudo-etiquetados (`.csv`) se guardan en `data/06_pseudo_labelled_data/`. El objeto del modelo KMeans ajustado (`.joblib`) se guarda en `models/clustering/`.

6.  **Paso 6: Entrenamiento y Evaluación del Modelo de Clasificación**
    * **Acción:** Entrena un modelo de Regresión Logística utilizando los datos pseudo-etiquetados del conjunto `discovery` y lo evalúa en los conjuntos de entrenamiento, validación y evaluación final.
    * **Comando:**
        ```bash
        python src/model_training.py
        ```
    * **Salida:** El modelo de clasificación entrenado y el `LabelEncoder` correspondiente (`.joblib`) se guardan en `models/classification/`. Los reportes de clasificación detallados (`.txt`) y los archivos CSV con el análisis de errores para cada conjunto de datos también se guardan en esta carpeta.

## Resultados del Modelo de Clasificación

Tras la ejecución completa del pipeline con las configuraciones actuales (Regresión Logística con `C=10.0` y `class_weight='balanced'` según `src/config.py`, sobre embeddings reducidos a 50 dimensiones por PCA), el modelo demostró un rendimiento robusto y consistente:

| Conjunto de Datos       | Accuracy | Macro Avg F1-score | Weighted Avg F1-score |
|-------------------------|----------|--------------------|-----------------------|
| Entrenamiento (`discovery`) | ~0.94    | ~0.94              | ~0.94                 |
| Validación (`validation`)  | ~0.94    | ~0.94              | ~0.94                 |
| Evaluación (`evaluation`)  | ~0.94    | ~0.94              | ~0.94                 |

El rendimiento es notablemente similar a través de todos los conjuntos, lo que indica una buena capacidad de generalización del modelo y la ausencia de un sobreajuste significativo a los datos de entrenamiento. Esta consistencia subraya la efectividad del pipeline de preprocesamiento, ingeniería de características y pseudo-etiquetado implementado.

Los reportes de clasificación detallados por clase, así como los archivos CSV que contienen un análisis de los errores de clasificación específicos para cada conjunto, se encuentran disponibles en la carpeta `models/classification/` para una inspección más profunda.