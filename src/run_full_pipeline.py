# src/run_full_pipeline.py

import logging

# Importar el módulo de configuración y configurar el logging primero
try:
    from . import config
    config.setup_logging() # Configura el logging tan pronto como sea posible
    from . import data_ingestion
    from . import data_preparation
    from . import preprocessing # Contiene main_text_preprocessing_pipeline
    from . import feature_engineering
    from . import clustering
    from . import model_training
except ImportError as e:
    # Este bloque es para ayudar a la ejecución directa del script si las importaciones relativas fallan
    # y los módulos están en el mismo directorio (lo cual es el caso de src/)
    import config
    config.setup_logging()
    import data_ingestion
    import data_preparation
    import preprocessing
    import feature_engineering
    import clustering
    import model_training
    logging.warning(f"Se recurrió a importaciones directas debido a: {e}. Asegúrate de que esto sea intencional.")


def run_pipeline():
    """
    Ejecuta el pipeline completo de MLOps secuencialmente.
    """
    logging.info("############################################################")
    logging.info("### INICIANDO EJECUCIÓN COMPLETA DEL PIPELINE DE MLOPS ###")
    logging.info("############################################################")

    # --- PASO 1: INGESTIÓN DE DATOS ---
    logging.info("\n----- ETAPA 1: Ingestión de Datos -----")
    try:
        data_ingestion.main_data_pipeline() #
        logging.info("----- ETAPA 1: Ingestión de Datos COMPLETADA -----")
    except Exception as e:
        logging.error(f"Error en la ETAPA 1 (Ingestión de Datos): {e}", exc_info=True)
        return # Detener el pipeline si este paso crucial falla

    dataset_types_ordered = ["discovery", "validation", "evaluation"]

    # --- PASO 2: PREPARACIÓN DE DATOS ---
    logging.info("\n----- ETAPA 2: Preparación de Datos -----")
    try:
        for ds_type in dataset_types_ordered:
            logging.info(f"--- Preparando datos para: {ds_type} ---")
            data_preparation.main_data_preparation_pipeline(dataset_type_to_process=ds_type) #
        logging.info("----- ETAPA 2: Preparación de Datos COMPLETADA -----")
    except Exception as e:
        logging.error(f"Error en la ETAPA 2 (Preparación de Datos): {e}", exc_info=True)
        return

    # --- PASO 3: PREPROCESAMIENTO DE TEXTO ---
    logging.info("\n----- ETAPA 3: Preprocesamiento de Texto -----")
    try:
        for ds_type in dataset_types_ordered:
            logging.info(f"--- Preprocesando texto para: {ds_type} ---")
            preprocessing.main_text_preprocessing_pipeline(dataset_type_to_process=ds_type) #
        logging.info("----- ETAPA 3: Preprocesamiento de Texto COMPLETADO -----")
    except Exception as e:
        logging.error(f"Error en la ETAPA 3 (Preprocesamiento de Texto): {e}", exc_info=True)
        return

    # --- PASO 4: INGENIERÍA DE CARACTERÍSTICAS (Embeddings y PCA) ---
    logging.info("\n----- ETAPA 4: Ingeniería de Características -----")
    try:
        # Es crucial procesar 'discovery' primero para ajustar PCA
        logging.info("--- Generando características para: discovery (para ajustar PCA) ---")
        feature_engineering.main_feature_engineering_pipeline(dataset_type_to_process="discovery") #
        for ds_type in ["validation", "evaluation"]:
            logging.info(f"--- Generando características para: {ds_type} (aplicando PCA ajustado) ---")
            feature_engineering.main_feature_engineering_pipeline(dataset_type_to_process=ds_type) #
        logging.info("----- ETAPA 4: Ingeniería de Características COMPLETADA -----")
    except Exception as e:
        logging.error(f"Error en la ETAPA 4 (Ingeniería de Características): {e}", exc_info=True)
        return

    # --- PASO 5: CLUSTERING Y PSEUDO-ETIQUETADO ---
    logging.info("\n----- ETAPA 5: Clustering y Pseudo-Etiquetado -----")
    try:
        k_value_for_discovery = config.KMEANS_N_CLUSTERS #

        logging.info(f"--- Clustering y reporte para: discovery (k={k_value_for_discovery}, generando pseudo-etiquetas) ---")
        # Ejecutar para discovery, asegurando que se generen las pseudo-etiquetas
        clustering.main_clustering_pipeline( #
            dataset_type="discovery",
            run_analysis_report_only=False, # Queremos las pseudo-etiquetas
            k_for_discovery_training=k_value_for_discovery
        )

        # Para validation y evaluation, se aplica el modelo de discovery y el mapeo de categorías.
        # No se re-entrena k-means, k_for_discovery_training es None.
        for ds_type in ["validation", "evaluation"]:
            logging.info(f"--- Aplicando clustering y mapeo para: {ds_type} ---")
            clustering.main_clustering_pipeline( #
                dataset_type=ds_type,
                run_analysis_report_only=False, # Aplicar mapeo de categorías
                k_for_discovery_training=None # Usar el modelo de discovery ya entrenado
            )
        logging.info("----- ETAPA 5: Clustering y Pseudo-Etiquetado COMPLETADO -----")
    except Exception as e:
        logging.error(f"Error en la ETAPA 5 (Clustering y Pseudo-Etiquetado): {e}", exc_info=True)
        return

    # --- PASO 6: ENTRENAMIENTO Y EVALUACIÓN DEL MODELO DE CLASIFICACIÓN ---
    logging.info("\n----- ETAPA 6: Entrenamiento y Evaluación del Modelo -----")
    try:
        # La función main_model_training_pipeline internamente carga los datos
        # 'discovery', 'validation', y 'evaluation' según sus nombres por defecto.
        model_training.main_model_training_pipeline() #
        logging.info("----- ETAPA 6: Entrenamiento y Evaluación del Modelo COMPLETADO -----")
    except Exception as e:
        logging.error(f"Error en la ETAPA 6 (Entrenamiento y Evaluación del Modelo): {e}", exc_info=True)
        return

    logging.info("\n##############################################################")
    logging.info("### EJECUCIÓN COMPLETA DEL PIPELINE DE MLOPS FINALIZADA ###")
    logging.info("##############################################################")

if __name__ == "__main__":
    # Validaciones básicas de config (puedes expandir esto)
    essential_configs = [
        'KAGGLE_DATASET_SLUG', 'RAW_TWCS_CSV_PATH', #
        'PREPARATION_INPUT_PATHS', 'PREPARED_OUTPUT_PATHS', 'PREPARED_DATA_COLUMNS', #
        'PREPROCESSING_INPUT_PATHS', 'PREPROCESSED_OUTPUT_PATHS', #
        'FE_INPUT_PATHS', 'EMBEDDINGS_FULL_OUTPUT_PATHS', 'EMBEDDINGS_IDS_OUTPUT_PATHS', #
        'EMBEDDINGS_REDUCED_OUTPUT_PATHS', 'PCA_MODEL_OUTPUT_PATH', 'SENTENCE_TRANSFORMER_MODEL', #
        'CLUSTERING_INPUT_EMBEDDINGS_PATHS', 'CLUSTERING_INPUT_IDS_PATHS', #
        'CLUSTER_MODEL_OUTPUT_PATHS', 'PSEUDO_LABELLED_DISCOVERY_PATH', #
        'PSEUDO_LABELLED_VALIDATION_PATH', 'PSEUDO_LABELLED_EVALUATION_PATH', #
        'MODELS_DIR' #
    ]
    missing_configs = [attr for attr in essential_configs if not hasattr(config, attr) or getattr(config, attr) is None]

    if missing_configs:
        logging.critical(f"Faltan configuraciones esenciales en config.py: {', '.join(missing_configs)}. "
                         "El pipeline no puede continuar.")
    else:
        run_pipeline()