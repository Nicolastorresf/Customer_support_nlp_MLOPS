# Diagrama de Arquitectura del Sistema ML

Aquí está el diagrama de la arquitectura:

```flowchart TD
    classDef s3Style fill:#D35400,stroke:#A04000,stroke-width:3px,color:white
    classDef lambdaStyle fill:#F39C12,stroke:#B9770E,stroke-width:2px,color:black
    linkStyle default stroke:#777,stroke-width:2px

    subgraph subGraph1["Ingesta y Almacenamiento Inicial"]
        direction LR // Opcional: cambia la dirección dentro del subgráfico
        B(["AWS S3 - Bucket Raw<br>(Zona de aterrizaje)"]):::s3Style
        C(["AWS Lambda o AWS Glue Job (Validación/Transformación Inicial)"]):::lambdaStyle
        D(["AWS S3 - Bucket Ingested<br>(Datos validados/limpios)"]):::s3Style
    end

    B --> C
    C --> D