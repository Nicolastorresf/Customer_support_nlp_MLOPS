# Documentación de Decisiones Técnicas del Pipeline

Este documento detalla las decisiones clave tomadas durante el diseño e implementación del pipeline de Machine Learning para la clasificación de tweets de atención al cliente.

## 1. Dataset Seleccionado

* **Dataset:** Customer Support on Twitter (Kaggle: `thoughtvector/customer-support-on-twitter`)
* **Justificación de la Elección:**
    * **Volumen de Datos:** Cumple con el requisito de más de 1 millón de registros, permitiendo trabajar con un volumen de datos realista.
    * **Relevancia para el Problema:** Contiene interacciones reales de atención al cliente en Twitter, lo cual es directamente aplicable al caso de uso de Nequi.
    * **Disponibilidad Pública:** Fácilmente accesible y reproducible.
    * **Riqueza de Información:** Incluye metadatos como `inbound` (para filtrar tweets de clientes), `author_id`, y el texto de los tweets, que son útiles para el análisis y preprocesamiento.
* **Filtrado Inicial:** Se decidió filtrar los datos para quedarse únicamente con los tweets entrantes (`inbound=True`), ya que el objetivo es clasificar los mensajes de los clientes. Esto resultó en aproximadamente 1.6 millones de tweets.

## 2. Preprocesamiento de Texto (`src/preprocessing.py`)

Se implementó un pipeline de preprocesamiento de texto robusto y multilingüe con los siguientes pasos y justificaciones:

* **Detección de Idioma:** Se utiliza `langdetect` para identificar el idioma principal de cada tweet. Esto permite aplicar tratamientos específicos (como stopwords y lematización) de forma más precisa. Se manejan casos donde la detección puede fallar o el texto es demasiado corto.
* **Limpieza Inicial:**
    * **Minúsculas:** Estandarización básica.
    * **Eliminación de HTML, URLs, Menciones de Twitter (`@usuario`), Hashtags (`#tema`):** Estos elementos suelen ser ruido para la clasificación semántica o pueden ser tratados de forma especial (ej. hashtags como características, aunque en este pipeline se optó por eliminarlos para simplificar el modelo base).
* **Manejo de Elementos Específicos del Lenguaje de Chat/Twitter:**
    * **Expansión de Emojis (`emot` y `all_emoticons_expanded.py`):** Los emojis se convierten a su representación textual (ej. ":)" -> "cara sonriente emoji") para capturar su significado semántico en lugar de tratarlos como caracteres especiales o eliminarlos. Se utiliza un diccionario personalizado para emoticonos textuales comunes.
    * **Expansión de Abreviaturas de Chat (`chat_words.py`):** Se expanden abreviaturas comunes (ej. "tqm" -> "te quiero mucho") para normalizar el texto.
* **Normalización Adicional:**
    * **Eliminación de Puntuación y Caracteres Especiales:** Se eliminan para reducir la dimensionalidad del vocabulario y enfocarse en las palabras.
    * **Eliminación de Números:** Se consideró que los números no eran cruciales para las categorías temáticas definidas.
    * **Eliminación de Espacios Extra:** Para limpiar el texto.
* **Corrección Ortográfica (`pyspellchecker`):** Se aplica una corrección ortográfica básica para el idioma detectado. Esto ayuda a agrupar palabras mal escritas con sus formas correctas. Se aplica con cautela para no introducir errores.
* **Tokenización (`nltk.word_tokenize`):** Divide el texto en palabras individuales (tokens).
* **Eliminación de Stopwords (`nltk.corpus.stopwords`):** Se eliminan palabras comunes (artículos, preposiciones) que no suelen aportar significado distintivo para la clasificación. Se hace de forma sensible al idioma. (Controlado por `config.PREPROCESSING_REMOVE_STOPWORDS`).
* **Lematización (`nltk.WordNetLemmatizer` y `pos_tag`):** Reduce las palabras a su forma base o lema (ej. "running" -> "run"). Se prefiere sobre el stemming porque el lema suele ser una palabra real, conservando mejor el significado. Se utiliza `pos_tag` para mejorar la precisión de la lematización.
* **Manejo de Textos Cortos/Vacíos:** Los textos que quedan vacíos o muy cortos después del preprocesamiento se marcan o manejan para evitar errores en etapas posteriores.

**Justificación General del Preprocesamiento:** El objetivo era normalizar y limpiar el texto lo máximo posible para reducir el ruido, estandarizar el vocabulario y facilitar que los modelos de embedding y clasificación capturen la semántica subyacente de manera más efectiva.

## 3. Ingeniería de Características (`src/feature_engineering.py`)

* **Embeddings de Sentencias (Sentence Transformers):**
    * **Modelo Seleccionado:** `config.SENTENCE_TRANSFORMER_MODEL` (actualmente `"paraphrase-multilingual-MiniLM-L12-v2"`).
    * **Justificación:** Se eligió un modelo de Sentence Transformers pre-entrenado y multilingüe porque:
        * Capturan el significado semántico a nivel de oración/documento, lo cual es más rico que TF-IDF o Word2Vec simple para este tipo de tarea.
        * El modelo `paraphrase-multilingual-MiniLM-L12-v2` es eficiente y ofrece un buen equilibrio entre rendimiento y tamaño, además de soportar múltiples idiomas presentes en el dataset.
    * Se generan embeddings para el texto preprocesado de cada tweet.
* **Reducción de Dimensionalidad (PCA):**
    * **Técnica:** Análisis de Componentes Principales (PCA) de `scikit-learn`.
    * **Número de Componentes (`config.PCA_N_COMPONENTS`):** Se configuró a `50`. Esta elección se basó en [Menciona aquí cómo llegaste a 50. ¿Fue experimentación, un valor común, o revisaste la varianza explicada? Si tienes la varianza explicada por 50 componentes, menciónala. Ej: "Se eligieron 50 componentes, que explicaban aproximadamente X% de la varianza, ofreciendo una reducción significativa de la dimensionalidad manteniendo información relevante."].
    * **Justificación:** Los embeddings de Sentence Transformers suelen tener una alta dimensionalidad (ej. 384 para MiniLM). PCA se utiliza para:
        * Reducir el costo computacional de las etapas posteriores (clustering, entrenamiento del clasificador).
        * Potencialmente eliminar ruido y redundancia en los embeddings.
        * Mitigar la "maldición de la dimensionalidad".
    * **Consistencia:** El modelo PCA se ajusta **únicamente** con los embeddings del conjunto `discovery` y luego se guarda. Este mismo modelo ajustado se utiliza para transformar los embeddings de los conjuntos `discovery`, `validation` y `evaluation`, asegurando que todos los datos se proyecten al mismo espacio de características de baja dimensión.
* **Guardado de Artefactos:**
    * Se guardan los embeddings completos, los reducidos por PCA y sus IDs correspondientes.
    * Se guarda el objeto del modelo PCA ajustado (`pca_model_fitted.joblib`) para su uso consistente.

## 4. Pseudo-Etiquetado mediante Clustering (`src/clustering.py` y `notebooks/Clustering_Experimentation.ipynb`)

Dado que el dataset original no venía con categorías de atención al cliente predefinidas, se optó por un enfoque de pseudo-etiquetado basado en clustering para descubrir estas categorías.

* **Algoritmo de Clustering:** K-Means (`sklearn.cluster.KMeans`).
    * **Justificación:** Es un algoritmo popular, eficiente para grandes datasets y relativamente fácil de interpretar.
* **Determinación del Número de Clústeres (`k`):**
    * Se realizó un análisis exploratorio en `notebooks/Clustering_Experimentation.ipynb` utilizando los embeddings reducidos del conjunto `discovery`.
    * Se evaluaron métricas como el **Método del Codo (WCSS)**, el **Coeficiente de Silueta**, el **Índice Calinski-Harabasz** y el **Índice Davies-Bouldin** para un rango de valores de `k` (de 2 a 15).
    * **Selección de `k=8` (`config.KMEANS_N_CLUSTERS`):** Este valor se eligió porque [Explica brevemente tu razonamiento basado en las métricas. Ej: "mostró un buen 'codo' en la gráfica WCSS, un pico en el Coeficiente de Silueta y una interpretación semántica coherente de los 8 clústeres resultantes."].
* **Análisis Cualitativo y Mapeo de Categorías:**
    * Para cada uno de los 8 clústeres obtenidos del conjunto `discovery`, se generaron palabras clave frecuentes (unigramas y bigramas) y se inspeccionaron tweets de muestra.
    * Basado en este análisis cualitativo, se asignó manualmente un nombre de categoría descriptivo a cada `cluster_id`. Este mapeo está codificado en la función `define_category_map()` dentro de `src/clustering.py`. Las categorías definidas son:
        * `0: "Consultas_Problemas_Productos_Pedidos"`
        * `1: "Agradecimiento_Cliente"`
        * `2: "Feedback_General_Expresivo_Emojis"`
        * `3: "Soporte_Tecnico_Fallas_SW_HW_Servicios"`
        * `4: "Contenido_Baja_Informacion_o_No_Procesable"`
        * `5: "Gestion_Cuentas_Pedidos_Atencion_Cliente"`
        * `6: "Problemas_Consultas_Servicios_Transporte"`
        * `7: "Dialogo_Interaccion_Soporte_General"`
* **Consistencia del Proceso:**
    * El modelo KMeans se ajusta **únicamente** con los embeddings reducidos del conjunto `discovery` utilizando `k=8` y se guarda.
    * Este mismo modelo KMeans ajustado se utiliza para `.predict()` las asignaciones de clúster para los conjuntos `discovery`, `validation` y `evaluation`.
    * El `define_category_map()` se aplica consistentemente a estas asignaciones para generar los archivos CSV pseudo-etiquetados.
* **Resultado:** Este proceso genera datasets (`_pseudo_labelled_... .csv`) para entrenamiento, validación y evaluación con una columna de categoría consistente.

## 5. Modelo de Clasificación (`src/model_training.py`)

* **Algoritmo Seleccionado:** Regresión Logística (`sklearn.linear_model.LogisticRegression`).
    * **Justificación:**
        * Cumple con el requisito de un "modelo básico" para la prueba.
        * Es eficiente para entrenar, interpretable y suele funcionar bien como línea base en problemas de clasificación de texto con buenas características.
        * Es robusto y fácil de desplegar.
* **Características de Entrada:** Embeddings reducidos a 50 dimensiones por PCA.
* **Hiperparámetros Clave:**
    * **`C=10.0` (`config.LOGREG_C`):** Este valor para el inverso de la fuerza de regularización fue seleccionado tras observar que ofrecía un excelente rendimiento sin sobreajuste significativo (métricas de entrenamiento, validación y evaluación muy similares y altas, ~0.94 F1-score). Un valor más alto indica menos regularización.
    * **`class_weight='balanced'` (`config.LOGREG_CLASS_WEIGHT`):** Se utiliza para ajustar los pesos de las clases de manera inversamente proporcional a sus frecuencias. Esto es crucial dado que las categorías generadas por el clustering pueden tener un número de muestras desbalanceado. Ayuda al modelo a prestar atención adecuada a las clases menos frecuentes.
    * **`solver='liblinear'`:** Adecuado para el tamaño del dataset y soporta regularización L1/L2 (aunque se usa L2 por defecto).
    * **`max_iter=1000`:** Para asegurar la convergencia.
* **Codificación de Etiquetas:** Se utiliza `sklearn.preprocessing.LabelEncoder`, ajustado únicamente con las etiquetas del conjunto de entrenamiento (`discovery`) y luego usado para transformar las etiquetas de los conjuntos de validación y evaluación.
* **Evaluación:**
    * Se generan reportes de clasificación completos (precisión, recall, F1-score por clase, promedios macro y ponderado, accuracy) para los conjuntos de entrenamiento, validación y evaluación.
    * Se implementó la generación de archivos CSV con los errores de clasificación para un análisis más detallado.
* **Resultados:** El modelo alcanzó un F1-score macro promedio de ~0.94 de forma consistente en los tres conjuntos, indicando una buena generalización y un pipeline de pseudo-etiquetado efectivo.

## 6. Proceso de Diagnóstico y Mejora (Iteraciones Clave)

Es importante destacar que el alto rendimiento actual del modelo es el resultado de un proceso iterativo de diagnóstico y mejora:

1.  **Resultados Iniciales Deficientes:** Las primeras ejecuciones del modelo de clasificación arrojaron F1-scores muy bajos (cercanos a 0.05) y una inconsistencia drástica entre el rendimiento en validación y evaluación para algunas clases.
2.  **Identificación de Causa Raíz:** El análisis reveló que la causa principal de estos problemas era la **falta de consistencia en la aplicación de los transformadores (PCA y KMeans) a través de los diferentes conjuntos de datos.** Específicamente, PCA y KMeans se estaban reajustando (re-entrenando) en cada conjunto (`discovery`, `validation`, `evaluation`) por separado.
    * Esto significaba que el espacio de características de PCA no era el mismo para entrenamiento y prueba.
    * Más críticamente, los `cluster_id` generados por KMeans no tenían el mismo significado semántico en los diferentes conjuntos, pero se les aplicaba el mismo mapeo manual a nombres de categoría, resultando en etiquetas incorrectas para validación y evaluación.
3.  **Solución Implementada:** Se refactorizaron los scripts `src/feature_engineering.py` y `src/clustering.py` para asegurar que:
    * El modelo PCA se ajusta **únicamente** en `discovery` y se guarda. Este modelo guardado se carga para transformar `validation` y `evaluation`.
    * El modelo KMeans se ajusta **únicamente** en `discovery` (con el `k` óptimo) y se guarda. Este modelo guardado se carga para predecir clústeres en `validation` y `evaluation`.
    * El `define_category_map` se aplica de forma consistente a estas predicciones de clúster.
4.  **Ajuste de Hiperparámetros del Clasificador:** Se experimentó con los hiperparámetros de `LogisticRegression`, especialmente `C` y `class_weight`, llegando a los valores actuales que ofrecen un buen rendimiento.
5.  **Resultado de la Mejora:** Tras estos cambios, el rendimiento del modelo mejoró drásticamente al ~0.94 F1-score en todos los conjuntos, demostrando la importancia de un pipeline de ML consistente y correctamente implementado.

Este proceso de depuración es una parte fundamental de la ingeniería de Machine Learning y demuestra la capacidad de identificar, diagnosticar y resolver problemas en el pipeline.