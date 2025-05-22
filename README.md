# 🧠 Clasificación Automatizada de Mensajes de Atención al Cliente | MLOps NLP

## 📌 Resumen 

Este proyecto implementa un sistema completo de Machine Learning en batch para clasificar mensajes de texto enviados por usuarios a un canal de atención al cliente. A través de técnicas modernas de NLP y MLOps, se construyó un pipeline modular, escalable y reproducible que cubre todas las etapas del ciclo de vida de un modelo: desde la ingesta de datos hasta el despliegue en un entorno de producción basado en la nube.

> 🔍 Este repositorio fue desarrollado como una solución técnica integral para un caso de uso realista de sistemas NLP en producción, incorporando buenas prácticas de ingeniería y automatización en entornos empresariales.

---

## 🧱 Componentes Clave del Proyecto

| Componente | Descripción |
|-----------|-------------|
| 📥 Dataset | [Customer Support on Twitter (Kaggle)](https://www.kaggle.com/datasets/thoughtvector/customer-support-on-twitter), ~1.6M tweets filtrados de clientes reales |
| ⚙️ Pipeline NLP | Ingesta, preprocesamiento multilingüe, generación de embeddings, clustering (pseudo-etiquetado), y clasificación |
| 🤖 Modelo ML | Regresión Logística con pseudo-etiquetas semánticas derivadas por KMeans |
| 🐳 Contenerización | Dockerfile preparado para ejecutar el pipeline completo en entornos reproducibles |
| 🔁 CI/CD/CT | GitHub Actions con workflows para validación continua y reentrenamiento manual |
| ☁️ Arquitectura Cloud | Propuesta detallada en AWS (Batch, Fargate, SageMaker, S3, Step Functions, etc.) |
| 📊 Monitoreo & Seguridad | Diseño de métricas operativas, alertas, y controles IAM/KMS aplicables a producción |

---

## 🚀 ¿Qué Puedes Hacer con Este Repositorio?

- Reentrenar el modelo automáticamente con nuevos datos
- Validar la calidad del código antes de mergear cambios
- Ejecutar inferencias batch en la nube
- Explorar un pipeline de NLP modular y extensible
- Analizar una propuesta completa de arquitectura MLOps en AWS

---

## 📁 Estructura del Proyecto

```text
customer_support_nlp_MLOPS/
├── src/                  # Código fuente del pipeline (scripts modulares)
├── data/                 # Datasets en diferentes etapas (raw, procesado, features)
├── models/               # Modelos entrenados y reportes
├── notebooks/            # Exploración y clustering experimental
├── .github/workflows/    # Workflows de CI/CD/CT
├── docs/                 # Propuestas técnicas y decisiones arquitectónicas
└── Dockerfile            # Imagen Docker para ejecución reproducible
```

---

## 🧪 Cómo Ejecutar el Pipeline

1. Clona el repositorio y activa tu entorno virtual:
```bash
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

2. (Opcional) Coloca el dataset descargado en `data/00_raw/twcs.csv`.

3. Ejecuta el pipeline completo:
```bash
python src/run_full_pipeline.py
```

---

## 🐙 Automatización con GitHub Actions

Este repositorio incluye dos workflows:

- **`ci_validation.yml`**: Valida código automáticamente con Flake8 en cada push/pull_request a `main`.
- **`manual_retrain_pipeline.yml`**: Permite reentrenar todo el sistema y construir una imagen Docker lista para despliegue.

> Más detalles en [`docs/CI_CD_CT.md`](docs/CI_CD_CT.MD)

---

## ☁️ Propuesta de Arquitectura en la Nube

La solución está diseñada para ejecutarse en AWS, con componentes desacoplados y orquestados mediante Step Functions. Soporta inferencia batch, reentrenamiento, trazabilidad de versiones y seguridad avanzada.

> Ver: [`Descripción de la arquitectura`](docs/CLOUD/ARQUITECTURA_NUBE.md)

> Ver: [Diagrama de la arquitectura](docs/CLOUD/mlops_nlp_aws_architecture.png)
---

## 🧠 Decisiones Técnicas

Cada paso del pipeline fue cuidadosamente documentado, incluyendo:

- Justificación del preprocesamiento NLP multilingüe
- Selección del modelo y parámetros
- Diagnóstico y refactorización de errores en transformaciones

> Detalles completos en [`docs/DECISIONES_PIPELINE.md`](docs/DECISIONES_PIPELINE.md)

---

## 📊 Monitoreo y Seguridad

Se propone una estrategia basada en:

- CloudWatch, SNS y SageMaker Model Monitor
- IAM de mínimo privilegio, cifrado SSE-KMS
- Escaneo automático de imágenes Docker y revisión de acceso a servicios

> Ver [`docs/MONITOREO_SEGURIDAD.md`](docs/MONITOREO_SEGURIDAD.md)

---

## 📈 Resultados del Modelo

| Conjunto | Accuracy | F1 Score (Macro) |
|----------|----------|------------------|
| Entrenamiento | ~0.94 | ~0.94 |
| Validación    | ~0.94 | ~0.94 |
| Evaluación    | ~0.94 | ~0.94 |

> Estas métricas reflejan un pipeline sólido, balanceado y sin overfitting.

---

## 📝 Contribuciones & Licencia

Este repositorio fue desarrollado como una prueba técnica individual. Puedes usarlo como base para proyectos reales o educativos.

---

## ✅ Checklist de Requisitos Cumplidos

- [x] Dataset > 1M registros públicos
- [x] Pipeline batch completo (ingesta → inferencia)
- [x] Clasificación multilingüe con pseudo-etiquetado
- [x] Contenerización (Dockerfile)
- [x] Workflows CI/CD/CT funcionales
- [x] Arquitectura cloud documentada
- [x] Propuesta de monitoreo y seguridad
- [x] Documentación profesional
