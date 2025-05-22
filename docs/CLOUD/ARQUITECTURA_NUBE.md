# 🌩️ ARQUITECTURA EN LA NUBE - MLOps NLP EN AWS

## ✨ Introducción

Esta arquitectura representa una solución MLOps completa y escalable para el procesamiento de datos de texto (tweets) y la clasificación automática en categorías predefinidas mediante técnicas de NLP y Machine Learning. Está implementada sobre servicios administrados de AWS, permitiendo trazabilidad, automatización, seguridad y eficiencia operativa. La solución incluye flujos de entrenamiento, inferencia y evolución del modelo.

---

## 🗂️ Descripción General de Componentes

### 📥 1. Ingesta y Almacenamiento Inicial

| Servicio       | Función |
|----------------|--------|
| `S3 - Raw`     | Zona de aterrizaje para archivos JSON/CSV. Configurado con versionado y cifrado. |
| `AWS Lambda / AWS Glue` | Validación del esquema, metadatos y calidad de los datos entrantes. |
| `AWS Glue Data Catalog` | Catálogo de metadatos de los datasets. |
| `S3 - Ingested` | Almacén de datos ya validados y enriquecidos. |

---

### 🧠 2. Feature Engineering (NLP)

| Servicio       | Función |
|----------------|--------|
| `AWS Batch / Fargate` | Ejecuta contenedores Docker con scripts Python (`data_prep`, `preprocessing`, `feature_engineering`). |
| `S3 - Processed` | Contenedor de los datos procesados, listos para clustering o entrenamiento. |
| `Athena`       | Exploración interactiva y validación de calidad de datos. |

---

### 🧪 3. Pseudo-Etiquetado (Clustering)

| Servicio       | Función |
|----------------|--------|
| `AWS Batch`    | Ejecuta `clustering.py` (con KMeans, UMAP). |
| `S3 - Pseudo-Labeled` | Almacena los datos con pseudo-etiquetas generadas automáticamente. |

---

### 🎯 4. Entrenamiento del Clasificador

| Servicio       | Función |
|----------------|--------|
| `SageMaker Training Job` | Entrena un modelo de clasificación basado en los datos pseudo-etiquetados. |
| `S3 - Model Artifacts`   | Guarda los modelos (.joblib, métricas, etc.). |
| `SageMaker Model Registry` | Versiona, evalúa y aprueba modelos entrenados. |

---

### 🤖 5. Inferencia Batch

| Servicio       | Función |
|----------------|--------|
| `S3 - New Data` | Nuevos lotes de datos para inferencia. |
| `SageMaker Batch Transform` | Ejecuta inferencia batch usando el modelo aprobado. |
| `S3 - Predictions` | Almacena las predicciones generadas. |
| `RDS / DynamoDB` | Bases de datos opcionales para consulta o dashboards. |

---

### 🔁 6. Orquestación y Monitoreo

| Servicio       | Función |
|----------------|--------|
| `AWS Step Functions` | Orquesta cada etapa del pipeline (validación, FE, clustering, entrenamiento, inferencia). |
| `Amazon CloudWatch` | Centraliza logs, métricas y dashboards. |
| `EventBridge` | Dispara pipelines por cron o eventos (subida a S3, cambios en parámetros, etc.). |

---

### 🛠️ 7. CI/CD y Seguridad

| Servicio       | Función |
|----------------|--------|
| `CodeCommit / GitHub` | Repositorio de código fuente (scripts, Dockerfiles, IaC). |
| `CodePipeline / GitHub Actions` | Automatiza testing, construcción de imágenes y despliegue del pipeline. |
| `Amazon ECR`   | Registro de imágenes Docker usadas por Batch/Fargate. |
| `AWS KMS, GuardDuty, Tags` | Seguridad, control de costos y protección de artefactos y datos. |

---

### ♻️ 8. Evolución de Categorías y Reentrenamiento

| Componente | Función |
|------------|--------|
| `S3 - New Data for Clustering` | Acumulación periódica de datos recientes no etiquetados. |
| `Exploratory Clustering Job (Batch)` | Genera un nuevo resumen de clusters para detectar nuevas posibles categorías. |
| `Cluster Report + Summary (S3)` | Informe que debe ser validado por expertos. |
| `Validación Humana (Lambda)` | Paso semiautomatizado donde el analista decide cambios en categorías. |
| `Actualizar category_map()` | Cambia la lógica de mapeo de clusters a etiquetas. |
| `Step Function - Reentrenamiento` | Ejecuta nuevo entrenamiento con el esquema de categorías actualizado y despliega el modelo actualizado. |

---

## 📌 Consideraciones Técnicas

- Todos los scripts se ejecutan en contenedores Docker para garantizar reproducibilidad.
- Los datos están cifrados en reposo (SSE-KMS) y en tránsito (TLS).
- El pipeline es **modular**, **escalable** y **versionado**, cumpliendo los principios clave de **MLOps**.
- El ciclo de vida del modelo está controlado por una combinación de `SageMaker Model Registry`, `CodePipeline`, `Step Functions` y `CloudWatch`.

---

## 🚀 Conclusión

Esta arquitectura permite una gestión eficiente, escalable y segura del flujo de Machine Learning para procesamiento de lenguaje natural. Incorpora automatización, trazabilidad, versionamiento, monitoreo y adaptabilidad al cambio, lo que la hace ideal tanto para entornos de prueba técnica como para entornos de producción real.
