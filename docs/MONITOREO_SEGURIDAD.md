# MONITOREO Y SEGURIDAD EN ARQUITECTURA MLOps - AWS

## ✨ Introducción

Este documento complementa la arquitectura propuesta para el sistema de clasificación de texto basado en NLP desplegado en AWS. Se enfoca en las estrategias de monitoreo operativo y medidas de seguridad adoptadas para garantizar la confiabilidad, trazabilidad y protección de los datos y modelos en producción.

---

## 🔍 Estrategia de Monitoreo Operativo

### ⚖️ Métricas Clave por Etapa

| Etapa                    | Métricas Relevantes                                                              |
| ------------------------ | -------------------------------------------------------------------------------- |
| Ingesta de Datos         | Archivos procesados, errores de validación, latencia promedio, volumen total     |
| Feature Engineering      | Tiempo de procesamiento, % de nulos, distribución de embeddings, fallos en batch |
| Clustering / Pseudo-etq. | Número de clusters generados, varianza intra-cluster, logs de resumen            |
| Entrenamiento            | Accuracy, F1-score, tiempo de entrenamiento, número de épocas, overfitting       |
| Inferencia Batch         | Tiempo por lote, confianza promedio, % de clasificación en "Otros"               |
| Costos                   | Costos por servicio (SageMaker, Fargate, S3), uso por job                        |

### ⚙️ Herramientas AWS Utilizadas

- **Amazon CloudWatch**:
  - **Logs**: Para Lambda, Batch, SageMaker.
  - **Metrics**: Personalizadas y de sistema.
  - **Alarms**: Basadas en umbrales críticos.
  - **Dashboards**: Visualización de KPIs por etapa.

- **SageMaker Model Monitor**:
  - Detección de *data drift*, *model drift*, *bias* y *feature skew* en datos de inferencia.

### ⚠️ Alertas

- **CloudWatch Alarms + SNS**:
  - Alarmas automáticas por:
    - Errores en jobs (estado fallido)
    - Confianza de predicción anormalmente baja
    - Costos diarios > presupuesto esperado
  - Enviadas a canales como email, Slack o SMS mediante Amazon SNS.

### ⟳ Acciones Correctivas Automatizables

- **Step Functions**:
  - Reintentos automáticos
  - Saltos condicionales por error
  - Fallback a jobs alternos

- **SageMaker Pipelines (opcional)**:
  - Ejecución de rollback a versión anterior del modelo en caso de degradación del rendimiento

- **Auto Scaling (Batch/Fargate)**:
  - Escalado basado en backlog de tareas o uso de CPU/Memory

---

## 🔒 Estrategia de Seguridad

### 🔐 IAM (Identity and Access Management)

- **Principio de mínimo privilegio**:
  - Cada servicio tiene roles dedicados con permisos estrictamente necesarios.

- **Políticas IAM por etapa**:
  - Lambda solo accede a S3 Raw e Ingested
  - SageMaker solo accede a Processed y Model Artifacts
  - Glue tiene acceso de lectura/escritura al Data Catalog

### 🔐 Seguridad de Datos

- **En reposo**:
  - S3: Cifrado SSE-S3 o SSE-KMS habilitado por defecto
  - EBS: Cifrado habilitado si se usan notebooks SageMaker

- **En tránsito**:
  - Todo el tráfico entre servicios y usuarios va sobre HTTPS (TLS 1.2 o superior)

### 🚧 Seguridad de Red

- **VPC y Subredes**:
  - Componentes críticos como SageMaker y Batch corren en subredes privadas

- **Security Groups y NACLs**:
  - Control de acceso a puertos y rangos IP

- **VPC Endpoints para S3 y DynamoDB**:
  - Evita exposición a Internet pública

### 🔑 Gestión de Secretos

- **AWS Secrets Manager o Parameter Store**:
  - Uso para tokens de APIs externas, claves de BD, variables sensibles de ejecución
  - Integración nativa con Lambda, SageMaker, Fargate mediante IAM

### 🛡️ Seguridad del Código y Artefactos

- **ECR Scan**:
  - Imágenes Docker son escaneadas automáticamente por vulnerabilidades

- **CI/CD Seguro**:
  - CodePipeline valida firma, permisos y rama antes de desplegar

### 🔍 Logging y Auditoría

- **CloudTrail**:
  - Registro de todas las llamadas a la API en la cuenta

- **CloudWatch Logs**:
  - Accesos a datos, ejecuciones de jobs y logs de aplicación centralizados

### 🧭 Auditoría Continua con CloudSploit

- **CloudSploit by Aqua Security**:
  - Se integra como herramienta externa de auditoría de seguridad (*CSPM - Cloud Security Posture Management*).
  - Analiza la configuración de múltiples servicios AWS (S3, IAM, RDS, Lambda, etc.) en busca de malas prácticas, exposiciones accidentales o configuraciones débiles.
  - Ideal para entornos en crecimiento, donde mantener la seguridad manualmente se vuelve inviable.

- **Ventajas**:
  - No requiere agentes ni instalación en los recursos.
  - Detecta automáticamente buckets públicos, claves IAM sin rotar, roles excesivos y configuraciones de red expuestas.
  - Entrega reportes periódicos y alertas sobre vulnerabilidades comunes.
  - Complementa servicios nativos como AWS Config y Security Hub.

- **Modo de integración**:
  - Se conecta a la cuenta AWS mediante un rol IAM de solo lectura (provisto por CloudSploit).
  - Se puede configurar para escanear diariamente o bajo demanda.
  - Resultados pueden enviarse a correo o integrarse a un SIEM.

---

## 🚀 Conclusión

Esta estrategia asegura un entorno de Machine Learning gobernado, confiable y seguro en AWS, con mecanismos automatizados para el monitoreo de calidad, respuesta a incidentes, y protección de los datos en cada fase del ciclo de vida del modelo.
