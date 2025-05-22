# Usar una imagen base oficial de Python
FROM python:3.12-slim

# Establecer variables de entorno para Python
ENV PYTHONDONTWRITEBYTECODE 1
ENV PYTHONUNBUFFERED 1

# Establecer el directorio de trabajo dentro del contenedor
WORKDIR /app

# Copiar el archivo de requerimientos primero para aprovechar el cache de Docker
COPY requirements.txt .

# Instalar las dependencias de Python
RUN pip install --no-cache-dir -r requirements.txt

# (Opcional, pero recomendado si tu script de preprocesamiento descarga recursos NLTK)
# Pre-descargar recursos NLTK para que estén en la imagen y no se descarguen cada vez
RUN python -m nltk.downloader stopwords punkt wordnet omw-1.4 averaged_perceptron_tagger

# Copiar todo el contenido del proyecto al directorio de trabajo /app
# Asegúrate de tener un .dockerignore para excluir archivos/carpetas innecesarios (como .git, data/, models/)
COPY . .
# Si prefieres copiar selectivamente:
# COPY src/ ./src/
# COPY notebooks/ ./notebooks/ # Si algún script los usa, aunque generalmente no para pipelines
# COPY config.py ./config.py # Si config.py estuviera en la raíz y no en src/


 ENTRYPOINT ["python", "src/run_full_pipeline.py"] 
# O dejarlo para que se especifique en el comando `docker run` o en GitHub Actions.