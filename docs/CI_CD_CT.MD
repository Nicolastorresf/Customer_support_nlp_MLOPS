# CI/CD/CT en el Proyecto de Clasificación de Tweets

## 📄 Introducción
El objetivo del pipeline CI/CD/CT implementado es garantizar la calidad del código, la automatización del reentrenamiento y empaquetado del modelo, y su preparación para despliegue. Esto se logra mediante la integración de GitHub Actions como sistema de orquestación, lo cual permite mantener buenas prácticas de ingeniería de software y Machine Learning dentro del flujo de desarrollo y operaciones.

---

## 📆 Workflow `ci_validation.yml` (Integración Continua)

### ✅ Propósito
Este workflow verifica automáticamente la calidad y consistencia del código en cada modificación relevante sobre la rama `main`.

### 🔎 Disparadores
- `push` a `main`
- `pull_request` hacia `main`

### ⚖️ Pasos Clave
1. **`checkout`**: Clona el repositorio.
2. **`setup-python`**: Configura Python 3.12.
3. **Instalación de Dependencias**: Usando `requirements.txt`.
4. **`flake8` (Linting)**: Revisa la calidad de código en la carpeta `src/`.

### ✨ Extensiones Propuestas
- Incorporar pruebas unitarias utilizando `pytest`.
- Validación de notebooks con `nbval` o `papermill`.
- Construcción de imagen Docker en entorno de prueba.
- Ejecución de tests básicos sobre la imagen construida.

---

## 🛠️ Workflow `manual_retrain_pipeline.yml` (Entrenamiento y "Despliegue" Continuo)

### ✅ Propósito
Automatizar el reentrenamiento del modelo desde cero, ejecutando el pipeline completo de procesamiento, generación de features, pseudo-etiquetado y clasificación, junto con la creación de una nueva imagen Docker conteniendo los artefactos generados.

### 🔎 Disparador
- Manual mediante `workflow_dispatch`, con un parámetro `log_level` como input.

### ⚖️ Pasos Clave
1. **`checkout` del repositorio**.
2. **Configuración del entorno**: Instalación de dependencias.
3. **Ejecución de scripts del pipeline**, en orden:
   - `data_ingestion.py`
   - `data_preparation.py`
   - `preprocessing.py`
   - `feature_engineering.py`
   - `clustering.py`
   - `model_training.py`
4. **Construcción de imagen Docker** con los artefactos generados (modelos, datos, reportes).
5. **Etiquetado de la imagen** con `github.run_id`.
6. **Subida de artefactos** al job de GitHub Actions (accesible desde la pestaña "Actions").

### 🌿 Estrategia de Reentrenamiento
Gracias a que `data_ingestion.py` puede re-descargar o manejar nuevos datos, y todo el pipeline se ejecuta desde cero, este workflow permite realizar un reentrenamiento completo con:
- Cambios en el código
- Nuevos datos
- Nuevas configuraciones de clustering o clasificación

Esto facilita la evolución del modelo sin necesidad de intervenciones manuales más allá de ejecutar el workflow.

---

## 🏠 Detalle del Despliegue (Integración con Arquitectura de Nube)

### 🌐 Rol de la Imagen Docker Construida
La imagen Docker generada por `manual_retrain_pipeline.yml` contiene:
- Código fuente actualizado
- Modelos entrenados
- Artefactos intermedios y finales del pipeline

Esta imagen se puede subir a un registro como **Amazon ECR**, y desde allí integrarse en la arquitectura propuesta en AWS (ver `ARQUITECTURA_NUBE.md` y `mlops_architecture.py`).

### ⚙️ Escenarios de Despliegue Posibles
- **AWS Batch / Fargate**: Ejecutar inferencia batch sobre nuevos datos usando esta imagen.
- **SageMaker Batch Transform**: Cargar la imagen como contenedor personalizado y ejecutar inferencias en lotes.
- **Actualización de modelo**: Si el modelo se registra en `SageMaker Model Registry`, puede versionarse y seleccionarse para su uso en etapas de inferencia o evaluación futuras.

Este enfoque asegura que la versión de modelo entrenada pueda desplegarse de forma coherente en diferentes entornos sin necesidad de reentrenamiento adicional.

---

## 🚀 Conclusión
El diseño del pipeline CI/CD/CT del proyecto permite mantener la calidad del código, facilitar el reentrenamiento bajo demanda y habilitar un despliegue controlado de versiones del modelo, maximizando la reproducibilidad y trazabilidad. La integración con servicios de nube como Amazon ECR, SageMaker y AWS Batch asegura su aplicabilidad en contextos reales de producción.
