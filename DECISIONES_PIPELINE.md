# Documentación de Decisiones Técnicas del Pipeline

Este documento detalla las decisiones clave tomadas durante el diseño e implementación del pipeline de Machine Learning para la clasificación de tweets de atención al cliente.

## 1. Dataset Seleccionado

* **Dataset:** Customer Support on Twitter (Kaggle: `thoughtvector/customer-support-on-twitter`)
* **Justificación de la Elección:**
    * **Volumen de Datos:** Cumple con el requisito de más de 1 millón de registros, permitiendo trabajar con un volumen de datos realista.
    * **Relevancia para el Problema:** Contiene interacciones reales de atención al cliente en Twitter, lo cual es directamente aplicable al caso de uso de Nequi.
    * **Disponibilidad Pública:** Fácilmente accesible y reproducible.
    * **Riqueza de Información:** Incluye metadatos como `inbound` (para filtrar tweets de clientes), `author_id`, y el texto de los tweets, que son útiles para el análisis y preprocesamiento.
* **Filtrado Inicial:** Se decidió filtrar los datos para quedarse únicamente con los tweets entrantes (`inbound=True`), ya que el objetivo es clasificar los mensajes de los clientes. Esto resultó en aproximadamente 1.6 millones de tweets antes de la división en conjuntos.

## 2. Preprocesamiento de Texto (`src/preprocessing.py`)

Se implementó un pipeline de preprocesamiento de texto robusto y multilingüe con los siguientes pasos y justificaciones:

* **Detección de Idioma:** Se utiliza `langdetect` para identificar el idioma principal de cada tweet. Esto permite aplicar tratamientos específicos (como stopwords y lematización) de forma más precisa. Se manejan casos donde la detección puede fallar o el texto es demasiado corto.
* **Limpieza Inicial:**
    * **Minúsculas:** Estandarización básica.
    * **Eliminación de HTML, URLs, Menciones de Twitter (`@usuario`), Hashtags (`#tema`):** Estos elementos suelen ser ruido para la clasificación semántica.
* **Manejo de Elementos Específicos del Lenguaje de Chat/Twitter:**
    * **Expansión de Emojis (`emot` y `all_emoticons_expanded.py`):** Los emojis se convierten a su representación textual para capturar su significado semántico. Se utiliza un diccionario personalizado para emoticonos textuales.
    * **Expansión de Abreviaturas de Chat (`chat_words.py`):** Se expanden abreviaturas comunes para normalizar el texto.
* **Normalización Adicional:**
    * **Eliminación de Puntuación y Caracteres Especiales:** Para reducir la dimensionalidad del vocabulario.
    * **Eliminación de Números:** Se consideró que no eran cruciales.
    * **Eliminación de Espacios Extra:** Para limpiar el texto.
* **Corrección Ortográfica (`pyspellchecker`):** Aplicada con cautela para el idioma detectado.
* **Tokenización (`nltk.word_tokenize`):** Divide el texto en palabras.
* **Eliminación de Stopwords (`nltk.corpus.stopwords`):** Sensible al idioma, controlada por `config.PREPROCESSING_REMOVE_STOPWORDS`.
* **Lematización (`nltk.WordNetLemmatizer` y `pos_tag`):** Reduce palabras a su forma base, usando `pos_tag` para mejorar precisión.
* **Manejo de Textos Cortos/Vacíos:** Para evitar errores posteriores.

**Justificación General del Preprocesamiento:** Normalizar y limpiar el texto para reducir ruido, estandarizar vocabulario y facilitar la captura de semántica por los modelos.

## 3. Ingeniería de Características (`src/feature_engineering.py`)

* **Embeddings de Sentencias (Sentence Transformers):**
    * **Modelo Seleccionado:** `config.SENTENCE_TRANSFORMER_MODEL` (actualmente `"paraphrase-multilingual-MiniLM-L12-v2"`).
    * **Justificación:** Modelo pre-entrenado y multilingüe que captura significado semántico a nivel de oración, eficiente y con buen equilibrio rendimiento/tamaño.
* **Reducción de Dimensionalidad (PCA):**
    * **Técnica:** Análisis de Componentes Principales (PCA) de `scikit-learn`.
    * **Número de Componentes (`config.PCA_N_COMPONENTS`):** Configurado a `50`. Esta elección se basó en la necesidad de reducir significativamente la dimensionalidad de los embeddings (originalmente 384D) para mejorar la eficiencia computacional del clustering y el entrenamiento del clasificador, manteniendo al mismo tiempo suficiente información para la separación de clases temáticas. *Si tienes la cifra de varianza explicada del notebook o de la ejecución del pipeline, añádela aquí (ej: "logrando explicar aproximadamente X% de la varianza").*
    * **Justificación:** Reducir costo computacional, eliminar ruido/redundancia y mitigar la "maldición de la dimensionalidad".
    * **Consistencia:** El modelo PCA se ajusta **únicamente** con los embeddings del conjunto `discovery` y se guarda, aplicándose luego a todos los conjuntos (`discovery`, `validation`, `evaluation`).
* **Guardado de Artefactos:** Se guardan embeddings completos, reducidos, IDs y el modelo PCA.

## 4. Pseudo-Etiquetado mediante Clustering (`src/clustering.py` y `notebooks/Clustering_Experimentation.ipynb`)

Dado que el dataset original no venía con categorías de atención al cliente predefinidas, se optó por un enfoque de pseudo-etiquetado basado en clustering para descubrir estas categorías.

* **Algoritmo de Clustering:** K-Means (`sklearn.cluster.KMeans`).
    * **Justificación:** Popular, eficiente para grandes datasets y relativamente fácil de interpretar.
* **Determinación del Número de Clústeres (`k`):**
    * Se realizó un análisis exploratorio en `notebooks/Clustering_Experimentation.ipynb` utilizando los embeddings reducidos por PCA (50 dimensiones) del conjunto `discovery`.
    * Se evaluaron métricas para un rango de `k` (de 5 a 20 según la ejecución del notebook):
        * **Método del Codo (WCSS - Inertia):** Mostró una disminución progresiva sin un "codo" extremadamente marcado, pero con un cambio de pendiente ligeramente más notorio alrededor de k=8, sugiriendo una disminución en la ganancia marginal de reducción de WCSS a partir de este punto.
        * **Coeficiente de Silueta (usando distancia coseno, sobre una muestra de 50,000 puntos):** Alcanzó su valor más alto en k=5 (0.1249), pero los valores se mantuvieron razonablemente consistentes y aceptables en la región de k=8 (0.1151) a k=11 (0.1165), antes de mostrar más variabilidad.
        * **Índice Calinski-Harabasz (sobre una muestra):** El valor más alto se observó en k=5 (2593.22) y descendió consistentemente al aumentar `k`. k=8 (1984.36) representó un punto intermedio en este descenso.
        * **Índice Davies-Bouldin (menor es mejor, sobre una muestra):** Mostró una tendencia general a disminuir (mejorar) con `k` mayores, alcanzando un mínimo local en k=14 (2.8706). k=8 (3.1758) fue parte de una tendencia descendente respecto a k más bajos como k=5 (3.1343).
    * **Selección de `k=8` (`config.KMEANS_N_CLUSTERS`):** Este valor se eligió considerando un equilibrio entre las métricas y la necesidad de interpretabilidad semántica para las categorías de atención al cliente. k=8 ofrecía una segmentación razonable donde la WCSS comenzaba a estabilizarse, y las otras métricas, aunque no óptimas individualmente en k=8, eran aceptables en conjunto, evitando una sobresegmentación o una subsegmentación excesiva. La interpretabilidad y el tamaño de los clústeres resultantes también jugaron un papel.
* **Análisis Cualitativo y Mapeo de Categorías:**
    * Para cada uno de los 8 clústeres del conjunto `discovery`, se generaron palabras clave y se inspeccionaron tweets de muestra (como se evidencia en `Clustering_Experimentation.ipynb` y el reporte generado por `src/clustering.py`).
    * Basado en este análisis, se asignó manualmente un nombre de categoría descriptivo a cada `cluster_id` en la función `define_category_map()` en `src/clustering.py`. Las categorías definidas son:
        * `0: "Consultas_Problemas_Productos_Pedidos"`
        * `1: "Agradecimiento_Cliente"`
        * `2: "Feedback_General_Expresivo_Emojis"`
        * `3: "Soporte_Tecnico_Fallas_SW_HW_Servicios"`
        * `4: "Contenido_Baja_Informacion_o_No_Procesable"`
        * `5: "Gestion_Cuentas_Pedidos_Atencion_Cliente"`
        * `6: "Problemas_Consultas_Servicios_Transporte"`
        * `7: "Dialogo_Interaccion_Soporte_General"`
       
* **Consistencia del Proceso:** El modelo KMeans se ajusta únicamente en `discovery` y se aplica consistentemente a los demás conjuntos.
* **Resultado:** Generación de datasets pseudo-etiquetados para entrenamiento, validación y evaluación.

## 5. Modelo de Clasificación (`src/model_training.py`)

* **Algoritmo Seleccionado:** Regresión Logística (`sklearn.linear_model.LogisticRegression`).
    * **Justificación:** Cumple con el requisito de "modelo básico", eficiente, interpretable y buen baseline para clasificación de texto con buenas características.
* **Características de Entrada:** Embeddings reducidos a 50 dimensiones por PCA.
* **Hiperparámetros Clave (según `config.py`):**
    * **`LOGREG_C=1.0`:** El valor para el inverso de la fuerza de regularización (originalmente 1 en `config.py`). *Asegúrate de que este valor sea el que usaste para obtener los resultados que reportarás. Si experimentaste y 10.0 dio mejores resultados y actualizaste `config.py` a 10.0, entonces pon 10.0 aquí. Si `config.py` sigue en 1.0, usa 1.0.*
    * **`LOGREG_CLASS_WEIGHT='balanced'`:** Para ajustar pesos de clases desbalanceadas.
    * **`solver='liblinear'` y `max_iter=1000`:** Para asegurar la convergencia y compatibilidad.
* **Codificación de Etiquetas:** `sklearn.preprocessing.LabelEncoder`, ajustado en `discovery`.
* **Evaluación:** Reportes de clasificación completos y análisis de errores para todos los conjuntos.
* **Resultados:** *Actualiza esta sección con tus métricas finales después de ejecutar el pipeline con la configuración final (ej: "El modelo alcanzó un F1-score macro promedio de ~0.9X...")*.

## 6. Proceso de Diagnóstico y Mejora (Iteraciones Clave)

Es importante destacar que el rendimiento actual del modelo es el resultado de un proceso iterativo de diagnóstico y mejora:

1.  **Resultados Iniciales Deficientes:** Las primeras ejecuciones del modelo de clasificación arrojaron F1-scores muy bajos (cercanos a 0.05) y una inconsistencia drástica entre el rendimiento en validación y evaluación para algunas clases.
2.  **Identificación de Causa Raíz:** El análisis reveló que la causa principal de estos problemas era la **falta de consistencia en la aplicación de los transformadores (PCA y KMeans) a través de los diferentes conjuntos de datos.** Específicamente, PCA y KMeans se estaban reajustando (re-entrenando) en cada conjunto (`discovery`, `validation`, `evaluation`) por separado.
    * Esto significaba que el espacio de características de PCA no era el mismo para entrenamiento y prueba.
    * Más críticamente, los `cluster_id` generados por KMeans no tenían el mismo significado semántico en los diferentes conjuntos, pero se les aplicaba el mismo mapeo manual a nombres de categoría, resultando en etiquetas incorrectas para validación y evaluación.
3.  **Solución Implementada:** Se refactorizaron los scripts `src/feature_engineering.py` y `src/clustering.py` para asegurar que:
    * El modelo PCA se ajusta **únicamente** en `discovery` y se guarda. Este modelo guardado se carga para transformar `validation` y `evaluation`.
    * El modelo KMeans se ajusta **únicamente** en `discovery` (con el `k` óptimo) y se guarda. Este modelo guardado se carga para predecir clústeres en `validation` y `evaluation`.
    * El `define_category_map` se aplica de forma consistente a estas predicciones de clúster.
4.  **Ajuste de Hiperparámetros del Clasificador:** Se experimentó con los hiperparámetros de `LogisticRegression`, especialmente `C` y `class_weight`, llegando a los valores actuales que ofrecen un buen rendimiento.
5.  **Resultado de la Mejora:** Tras estos cambios, el rendimiento del modelo mejoró drásticamente, demostrando la importancia de un pipeline de ML consistente y correctamente implementado. *(Actualiza con tus resultados finales aquí también)*.

Este proceso de depuración es una parte fundamental de la ingeniería de Machine Learning y demuestra la capacidad de identificar, diagnosticar y resolver problemas en el pipeline.