# Proyecto: Clasificación de Tweets de Atención al Cliente (MLOps Nequi)

## Autor

***Nombre:** Nicolas Torres
***Fecha:** Mayo 2025

## Descripción del Proyecto

Este proyecto implementa un pipeline de Machine Learning de extremo a extremo para clasificar tweets de un canal de atención al cliente. El objetivo es tomar mensajes de texto de usuarios, procesarlos a través de una serie de etapas (ingesta, preparación, preprocesamiento NLP, ingeniería de características, y pseudo-etiquetado mediante clustering) y finalmente entrenar un modelo de clasificación supervisado para asignar estos mensajes a categorías relevantes.

La solución está diseñada como un sistema batch, con un fuerte énfasis en la modularidad, reproducibilidad, trazabilidad y buenas prácticas de ingeniería de Machine Learning, siguiendo los lineamientos de la prueba técnica para Ingeniero de Machine Learning de Nequi]. El pipeline ha sido desarrollado en Python y utiliza diversas librerías estándar de data science y NLP. El código fuente se organiza en scripts modulares dentro de la carpeta `src/`, orquestados por `src/run_full_pipeline.py`. La solución también incluye contenerización con Docker y flujos de CI/CD/CT utilizando GitHub Actions.

## Dataset Utilizado

* **Nombre del Dataset:** Customer Support on Twitter
* **Fuente:** Kaggle
* **URL/Slug en Kaggle:** `thoughtvector/customer-support-on-twitter` (https://www.kaggle.com/datasets/thoughtvector/customer-support-on-twitter)
* **Archivo Principal en el Dataset:** `twcs.csv`
* **Descripción Breve:** Este dataset público contiene aproximadamente 3 millones de tweets que representan interacciones entre clientes y múltiples compañías a través de Twitter.
* **Filtrado Aplicado:** Para este proyecto, el dataset se filtra inicialmente para incluir únicamente los tweets entrantes de clientes (`inbound=True`). Este subconjunto se divide luego en conjuntos de `discovery` (para entrenamiento y ajuste de modelos no supervisados), `validation` y `evaluation`.

## Estructura de Carpetas del Proyecto

```text
customer_support_nlp_MLOPS/
├── .github/                        # Workflows de GitHub Actions
│   └── workflows/
│       ├── ci_validation.yml       # Workflow de Integración Continua (linting, tests básicos)
│       └── manual_retrain_pipeline.yml # Workflow de Entrenamiento Continuo (manual)
├── data/                           # Datos (IGNORADOS POR GIT gracias a .gitignore)
│   ├── 00_raw/                     # Dataset crudo original (ej. twcs.csv)
│   ├── 01_ingested_splits/         # Splits iniciales (discovery_split_raw.csv, etc.)
│   ├── 02_prepared_data/         # Datos después de preparación básica
│   ├── 03_preprocessed_text/     # Textos después de limpieza y preprocesamiento NLP
│   ├── 04_features/                # Embeddings (completos y reducidos .npy) y sus IDs (.csv)
│   ├── 05_clustering_outputs/    # Asignaciones de clusters (.csv) y reportes de análisis (.txt)
│   └── 06_pseudo_labelled_data/  # Datasets finales con pseudo-etiquetas para clasificación
├── docs/                           # Documentos de propuestas de diseño y decisiones
│   ├── ARQUITECTURA_NUBE.md        # Propuesta de Arquitectura en la Nube y Orquestación
│   ├── PIPELINE_DATOS_MODELO.md    # Decisiones sobre el pipeline de datos y modelo
│   ├── CI_CD_CT.md                 # Detalles del pipeline de CI/CD/CT
│   └── MONITOREO_SEGURIDAD.md      # Propuesta de Monitoreo y Seguridad
├── models/                         # Modelos serializados y reportes
│   ├── classification/             # Modelos de clasificación, encoders, reportes, análisis de errores
│   ├── clustering/                 # Modelos KMeans ajustados
│   └── feature_reduction/          # Modelos PCA ajustados
├── notebooks/                      # Jupyter notebooks para experimentación y análisis exploratorio
│   └── Clustering_Experimentation.ipynb # Notebook de ejemplo para k-means
├── src/                            # Código fuente del pipeline de ML
│   ├── __init__.py
│   ├── all_emoticons_expanded.py # Módulo de utilidad para emojis
│   ├── chat_words.py             # Módulo de utilidad para abreviaturas de chat
│   ├── clustering.py             # Lógica de clustering (KMeans) y pseudo-etiquetado
│   ├── config.py                 # Archivo de configuración centralizado
│   ├── data_ingestion.py         # Script para descarga y división inicial de datos
│   ├── data_preparation.py       # Script para preparación básica de datos
│   ├── feature_engineering.py    # Script para generación de embeddings y reducción de dimensionalidad
│   ├── model_training.py         # Script para entrenamiento y evaluación del modelo de clasificación
│   ├── preprocessing.py          # Script para limpieza y preprocesamiento de texto NLP
│   └── run_full_pipeline.py      # Script principal para orquestar todo el pipeline
├── .dockerignore                   # Especifica archivos a ignorar por Docker en el build
├── .gitignore                      # Especifica archivos y directorios a ignorar por Git
├── Dockerfile                      # Define la imagen Docker para el pipeline
├── README.md                       # Este archivo
└── requirements.txt                # Dependencias del proyecto Python
```

## Prerrequisitos

* Python 3.12 (la versión utilizada para desarrollo y pruebas)
* Git
* Docker (para construir y ejecutar la imagen del pipeline)
* (Recomendado) Un entorno virtual para gestionar las dependencias del proyecto:

* Para crear un entorno (ej. llamado `nequi_env`):

```bash
        python -m venv nequi_env
```

***Para activar el entorno:**

***En Windows (Git Bash o CMD/Powershell):**

```bash
            nequi_env\Scripts\activate
```

***En Linux/macOS:**

```bash
            source nequi_env/bin/activate
```

## Instalación de Dependencias

Una vez clonado el repositorio y con el entorno virtual activado (si se usa), instala todas las dependencias necesarias ejecutando el siguiente comando en la raíz del proyecto:

```bash
pip install -r requirements.txt
```

## Configuración

El pipeline es altamente configurable a través del archivo src/config.py. Este archivo centraliza:
     *Rutas a los datos de entrada y salida para cada etapa del pipeline (con soporte para ejecución local o S3 basado en variables de entorno).
     *Nombres de columnas clave.
     *Parámetros para la ingeniería de características (modelo Sentence Transformers, componentes PCA).
     *Parámetros para el clustering (número de clústeres KMeans, métodos de inicialización).
     *Hiperparámetros para el modelo de clasificación (Regresión Logística: LOGREG_C = 1, LOGREG_CLASS_WEIGHT = 'balanced').
     *Configuración de logging, preprocesamiento y división de datos.

## Antes de la primera ejeción, considera lo siguiente

 1.**Dataset de Kaggle:**

*El script src/data_ingestion.py intentará descargar el dataset thoughtvector/customer-support-on-twitter desde Kaggle usando kagglehub. Esto requiere credenciales de API de Kaggle configuradas (kaggle.json o variables de entorno KAGGLE_USERNAME/KAGGLE_KEY).

* **Alternativa Manual:** Puedes descargar twcs.csv manualmente desde Kaggle y colocarlo en data/00_raw/ antes de ejecutar el pipeline.

2.**Revisar `src/config.py` (Valores Clave):**

* **`KMEANS_N_CLUSTERS`**: Actualmente configurado a `8` por defecto. Este valor es crucial para el pseudo-etiquetado y define el número de categorías temáticas. Se recomienda determinarlo mediante experimentación (ej. `notebooks/Clustering_Experimentation.ipynb`).
* **Otras Configuraciones**: Revisa las variables en `src/config.py` para ajustar el comportamiento de los diferentes módulos del pipeline según sea necesario (ej., rutas, modelo de embedding, parámetros de NLTK, etc.).

## Ejecución del Pipeline

**Opción 1:** Ejecución Completa Orquestada
Se ha provisto un script principal src/run_full_pipeline.py que ejecuta todas las etapas del pipeline en la secuencia correcta. Este es el método recomendado para una ejecución completa.

```bash
     python src/run_full_pipeline.py
```

o como módulo:

```bash
    python -m src.run_full_pipeline
```

***Este script se encargará de:**

1.Ingestión de datos.
2.Preparación de datos para los conjuntos discovery, validation y evaluation.
3.Preprocesamiento de texto NLP para todos los conjuntos.
4.Ingeniería de características (ajustando PCA en discovery y aplicándolo a los demás).
5.Clustering y pseudo-etiquetado (ajustando KMeans en discovery y aplicándolo a los demás, generando pseudo-etiquetas).
7.Entrenamiento y evaluación del modelo de clasificación.

**Opción 1:** Ejecución Individual de Scripts (Para Debugging o Pasos Específicos)
Si necesitas ejecutar una etapa específica o depurar, puedes ejecutar los scripts individuales. La mayoría aceptan el argumento --dataset_type (discovery, validation, evaluation, o all). Es crucial ejecutar los pasos en orden y asegurar que discovery se procese primero para etapas que ajustan modelos (PCA, KMeans).

1.Ingestión de Datos:

```Bash
python src/data_ingestion.py
```

2.**Preparación de Datos:**

```Bash
python src/data_preparation.py --dataset_type all
```

3.**Preprocesamiento de Texto NLP:**

```Bash
python src/preprocessing.py --dataset_type all
```

4.**Ingeniería de Características:**

```Bash
python src/feature_engineering.py --dataset_type all
```

(PCA se ajusta en discovery y se aplica al resto)

5.**Clustering y Pseudo-Etiquetado:**

```Bash
python src/clustering.py --dataset_type all
```

(KMeans se ajusta en discovery y se aplica al resto)

6.**Entrenamiento y Evaluación del Modelo:**

```Bash
python src/model_training.py
```

## Contenerización con Docker

El proyecto incluye un Dockerfile para construir una imagen Docker que contiene el pipeline y sus dependencias.

**Puntos Clave del Dockerfile:**
* Utiliza una imagen base `python:3.12-slim`.
* Establece el directorio de trabajo en `/app`.
* **Copia `requirements.docker.txt` como `requirements.txt` dentro del contexto de la build e instala las dependencias desde este archivo. Esto asegura que se instale una versión de PyTorch compatible con CPU, haciendo la imagen más portable.**
* Pre-descarga los recursos NLTK necesarios.
* Copia el código del proyecto (la carpeta `src/` y otros archivos necesarios) a la imagen.
* Define `python src/run_full_pipeline.py` como el `ENTRYPOINT`, por lo que el pipeline completo se ejecuta cuando se inicia un contenedor.

1.**Construir la Imagen:**
Desde la raíz del proyecto, ejecuta:

```Bash

docker build -t nequi_mlops_pipeline .
```

(Puedes cambiar nequi_mlops_pipeline por el nombre y etiqueta que prefieras).

2.**Ejecutar el Pipeline dentro del Contenedor:**

Una vez construida la imagen, puedes ejecutar el pipeline completo dentro del contenedor:

```Bash
docker run --rm nequi_mlops_pipeline
```

El ENTRYPOINT del Dockerfile está configurado para ejecutar python src/run_full_pipeline.py.

* Si necesitas mapear volúmenes para persistir los datos o modelos generados fuera del contenedor, o pasar variables de entorno, puedes hacerlo con las opciones de docker run. Por ejemplo, para guardar la carpeta data y models en tu host:

```Bash

docker run --rm \
  -v $(pwd)/data:/app/data \
  -v $(pwd)/models:/app/models \
  nequi_mlops_pipeline
```

* Asegúrate de que las rutas en src/config.py sean relativas al WORKDIR (/app) o configurables mediante variables de entorno para que funcionen correctamente dentro del contenedor. El config.py actual ya soporta variables de entorno para rutas base.

## CI/CD/CT con GitHub Actions

El repositorio está configurado con los siguientes workflows de GitHub Actions (ver carpeta .github/workflows/):

1. ci_validation.yml (Integración Continua):

* Disparador: Se ejecuta en cada push o pull_request a la rama main (o tu rama principal).

* Acciones:
  * Realiza checkout del código.
  * Configura el entorno Python (versión 3.12).
  * Instala las dependencias desde requirements.txt.
  * Ejecuta un linter (Flake8) para verificar la calidad del código en src/.
  * (Conceptual) Podría extenderse para ejecutar pruebas unitarias/integración y construir    la imagen Docker para validación.

2. manual_retrain_pipeline.yml (Entrenamiento y "Despliegue" Continuo Manual):

* Disparador: Se ejecuta manualmente desde la pestaña "Actions" del repositorio en GitHub (workflow_dispatch). Permite ingresar el log_level como parámetro.

* Acciones:
* Realiza checkout del código.
* Configura el entorno Python e instala dependencias.
* (Opcional) Puede configurarse con secretos de Kaggle para la descarga automática de datos.
* Ejecuta la secuencia completa de scripts del pipeline (Ingestión, Preparación,  Preprocesamiento, Ing. Características, Clustering, Entrenamiento del Modelo) utilizando los scripts de src/ (similar a run_full_pipeline.py pero directamente en el workflow).
*Construye una nueva imagen Docker con el código y los modelos/datos actualizados, etiquetándola con el github.run_id.
* Sube los artefactos generados (modelos, datos pseudo-etiquetados, reportes) a GitHub Actions para su inspección o descarga.
* Propósito: Este workflow sirve como un mecanismo para reentrenar el modelo con la última versión del código (y potencialmente nuevos datos si data_ingestion.py se adapta para ello) y empaquetar la solución actualizada.

## Resultados del Modelo de Clasificación

Tras la ejecución completa del pipeline con las configuraciones actuales (Regresión Logística con `C=10.0` y `class_weight='balanced'` según `src/config.py`, sobre embeddings reducidos a 50 dimensiones por PCA), el modelo demostró un rendimiento robusto y consistente:

````bash
| Conjunto de Datos       | Accuracy | Macro Avg F1-score | Weighted Avg F1-score |
|-------------------------|----------|--------------------|-----------------------|
| Entrenamiento (`discovery`) | ~0.94    | ~0.94              | ~0.94                 |
| Validación (`validation`)  | ~0.94    | ~0.94              | ~0.94                 |
| Evaluación (`evaluation`)  | ~0.94    | ~0.94              | ~0.94                 |
````

El rendimiento es notablemente similar a través de todos los conjuntos, lo que indica una buena capacidad de generalización del modelo y la ausencia de un sobreajuste significativo a los datos de entrenamiento. Esta consistencia subraya la efectividad del pipeline de preprocesamiento, ingeniería de características y pseudo-etiquetado implementado.