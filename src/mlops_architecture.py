import os # <--- 1. Importar os
from diagrams import Cluster, Diagram
from diagrams.aws.analytics import Athena, Glue
from diagrams.aws.compute import Batch, Lambda, Fargate, ECR
from diagrams.aws.ml import Sagemaker
from diagrams.aws.database import Dynamodb, RDS
from diagrams.aws.devtools import Codecommit, Codepipeline
from diagrams.aws.integration import StepFunctions, Eventbridge
from diagrams.aws.management import Cloudwatch
from diagrams.aws.security import KMS
from diagrams.aws.storage import S3


script_dir = os.path.dirname(__file__) 
project_root = os.path.dirname(script_dir) 
output_directory = os.path.join(project_root, "docs", "CLOUD")
output_filename = "mlops_nlp_aws_architecture" 

os.makedirs(output_directory, exist_ok=True)

with Diagram("Arquitectura MLOps NLP en AWS", 
             show=True, 
             filename=os.path.join(output_directory, output_filename), 
             outformat="png"):
    
    fuente_datos = S3("Usuarios JSON/CSV")

    with Cluster("Ingesta y Almacenamiento"):
        s3_raw = S3("S3 - Raw")
        val_lambda = Lambda("Validación\n(Lambda/Glue)")
        glue_catalog = Glue("Glue Catalog")
        s3_ingested = S3("S3 - Ingested")
        fuente_datos >> s3_raw >> val_lambda >> s3_ingested
        val_lambda >> glue_catalog

    with Cluster("Feature Engineering"):
        fe_batch = Batch("Feature Eng.\n(Batch/Fargate)")
        s3_processed = S3("S3 - Processed")
        athena = Athena("Athena")
        s3_ingested >> fe_batch >> s3_processed >> athena

    with Cluster("Pseudo-Etiquetado"):
        cluster_job = Batch("Clustering\n(KMeans/UMAP)")
        s3_pseudo = S3("S3 - Pseudo-Labeled")
        s3_processed >> cluster_job >> s3_pseudo

    with Cluster("Entrenamiento"):
        training = Sagemaker("SageMaker Training")
        model_artifacts = S3("Model Artifacts")
        model_registry = Sagemaker("Model Registry")
        [s3_pseudo, s3_processed] >> training
        training >> [model_artifacts, model_registry]

    with Cluster("Inferencia"):
        new_data = S3("New Data\n(Processed)")
        batch_infer = Sagemaker("Batch Transform")
        predictions = S3("Predictions")
        rds = RDS("RDS")
        dynamo = Dynamodb("DynamoDB")
        [new_data, model_artifacts, model_registry] >> batch_infer
        batch_infer >> predictions >> [rds, dynamo]

    with Cluster("Orquestación y Monitoreo"):
        step_fn = StepFunctions("Step Functions")
        cw = Cloudwatch("CloudWatch")
        eb = Eventbridge("EventBridge")
        [val_lambda, fe_batch, cluster_job, training, batch_infer] >> cw
        step_fn >> [val_lambda, fe_batch, cluster_job, training, batch_infer, cw]
        eb >> step_fn

    with Cluster("CI/CD + Seguridad"):
        repo = Codecommit("CodeCommit / GitHub")
        pipeline = Codepipeline("CodePipeline / GH Actions")
        docker_repo = ECR("ECR")
        kms = KMS("KMS / Tags / GuardDuty")
        repo >> pipeline >> [step_fn, docker_repo]
        [fe_batch, cluster_job, training, batch_infer] >> docker_repo
        [s3_raw, s3_ingested, s3_processed, s3_pseudo, model_artifacts, predictions] >> kms

print(f"Diagrama guardado en: {os.path.join(output_directory, output_filename)}.png")