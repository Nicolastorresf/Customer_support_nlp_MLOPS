# src/feature_engineering.py

import pandas as pd
import numpy as np
import os
import logging
import torch # sentence-transformers usa torch por debajo
from sentence_transformers import SentenceTransformer # pip install sentence-transformers
from sklearn.decomposition import PCA
import joblib # Para guardar y cargar el modelo PCA
from typing import Tuple, List, Optional, Dict 
import argparse

# Importar configuraciones y constantes compartidas
try:
    from . import config 
except ImportError:
    import config 

# Configurar logging
config.setup_logging()

def generate_embeddings(
    df: pd.DataFrame,
    text_column: str,
    id_column: str,
    model_name: str,
    batch_size: int = 32, 
    device: Optional[str] = None
) -> Tuple[Optional[np.ndarray], Optional[pd.Series]]:
    """
    Genera embeddings para los textos en un DataFrame usando un modelo SentenceTransformer.
    """
    if text_column not in df.columns:
        logging.error(f"La columna de texto '{text_column}' no se encuentra en el DataFrame.")
        return None, None
    if id_column not in df.columns:
        logging.error(f"La columna de ID '{id_column}' no se encuentra en el DataFrame.")
        return None, None

    texts = df[text_column].fillna("").astype(str).tolist()
    ids = df[id_column]

    if not texts:
        logging.warning("La lista de textos para generar embeddings está vacía.")
        return np.array([]).reshape(0,0), pd.Series([], dtype=ids.dtype, name=ids.name)

    logging.info(f"Cargando el modelo SentenceTransformer: {model_name}")
    try:
        if device is None:
            device = 'cuda' if torch.cuda.is_available() else 'cpu'
        logging.info(f"Usando dispositivo para SentenceTransformer: {device}")
        model = SentenceTransformer(model_name, device=device)
    except Exception as e:
        logging.error(f"Error al cargar el modelo SentenceTransformer '{model_name}': {e}")
        return None, None

    logging.info(f"Generando embeddings para {len(texts)} textos (batch_size={batch_size}). Esto puede tardar...")
    try:
        embeddings = model.encode(texts, batch_size=batch_size, show_progress_bar=True)
        logging.info(f"Embeddings generados. Forma del array: {embeddings.shape}")
        return embeddings, ids
    except Exception as e:
        logging.error(f"Error durante la generación de embeddings: {e}")
        return None, None

def save_data(data_array: np.ndarray, output_path: str) -> bool:
    """Guarda un NumPy array en la ruta especificada."""
    try:
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        np.save(output_path, data_array)
        logging.info(f"Array guardado en: {output_path}")
        return True
    except Exception as e:
        logging.error(f"Error al guardar array en {output_path}: {e}")
        return False

def save_ids_csv(ids_series: pd.Series, output_path: str, id_column_name: str) -> bool:
    """Guarda una Serie de Pandas (IDs) como CSV."""
    try:
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        ids_series.to_frame(name=id_column_name).to_csv(output_path, index=False)
        logging.info(f"IDs correspondientes guardados en: {output_path}")
        return True
    except Exception as e:
        logging.error(f"Error al guardar IDs en {output_path}: {e}")
        return False

def apply_pca_dimensionality_reduction(
    embeddings: np.ndarray,
    dataset_type: str, 
    n_components: int = config.PCA_N_COMPONENTS,
    random_state: int = config.RANDOM_STATE
) -> Optional[np.ndarray]:
    """
    Aplica PCA. Ajusta en 'discovery' y guarda el modelo.
    Para otros conjuntos ('validation', 'evaluation'), carga el modelo PCA ajustado y transforma.
    """
    if embeddings is None or embeddings.shape[0] == 0:
        logging.warning(f"No hay embeddings para PCA en el conjunto {dataset_type}.")
        return None

    effective_n_components = min(n_components, embeddings.shape[0], embeddings.shape[1])
    if effective_n_components < 1: 
        logging.error(f"Número de componentes para PCA ({effective_n_components}) es inválido para la forma de los datos {embeddings.shape} en el conjunto {dataset_type}.")
        return None

    pca_model_path = config.PCA_MODEL_OUTPUT_PATH 
    os.makedirs(os.path.dirname(pca_model_path), exist_ok=True) 

    if dataset_type == "discovery":
        logging.info(f"Ajustando PCA con {effective_n_components} componentes en el conjunto 'discovery'...")
        try:
            reducer = PCA(n_components=effective_n_components, random_state=random_state) 
            reduced_embeddings = reducer.fit_transform(embeddings) 
            joblib.dump(reducer, pca_model_path) 
            logging.info(f"Modelo PCA ajustado y guardado en: {pca_model_path}")
            if hasattr(reducer, 'explained_variance_ratio_'): 
                logging.info(f"Varianza explicada por PCA (discovery): {reducer.explained_variance_ratio_.sum():.4f}")
        except Exception as e:
            logging.error(f"Error ajustando PCA en 'discovery': {e}")
            return None
    else: # Para 'validation' y 'evaluation'
        logging.info(f"Cargando modelo PCA ajustado desde {pca_model_path} para transformar el conjunto '{dataset_type}'...")
        if not os.path.exists(pca_model_path):
            logging.error(f"Modelo PCA no encontrado en {pca_model_path}. Ejecuta primero el pipeline para 'discovery'.")
            return None
        try:
            reducer = joblib.load(pca_model_path) 
            reduced_embeddings = reducer.transform(embeddings) 
            if hasattr(reducer, 'explained_variance_ratio_'):
                 logging.info(f"Varianza explicada por el modelo PCA cargado (ajustado en discovery): {reducer.explained_variance_ratio_.sum():.4f}")
        except Exception as e:
            logging.error(f"Error transformando con PCA en '{dataset_type}': {e}")
            return None
            
    logging.info(f"Reducción con PCA para '{dataset_type}' completada. Nueva forma: {reduced_embeddings.shape}")
    return reduced_embeddings


def main_feature_engineering_pipeline(dataset_type_to_process: str):
    """
    Pipeline principal para la ingeniería de características para un tipo de dataset.
    PCA se ajusta en 'discovery' y se aplica consistentemente a los demás.
    """
    logging.info(f"===== Iniciando Pipeline de Ingeniería de Características para el conjunto: '{dataset_type_to_process}' =====")

    input_path = config.FE_INPUT_PATHS.get(dataset_type_to_process) 
    embeddings_full_output_path = config.EMBEDDINGS_FULL_OUTPUT_PATHS.get(dataset_type_to_process) 
    embeddings_ids_output_path = config.EMBEDDINGS_IDS_OUTPUT_PATHS.get(dataset_type_to_process) 
    embeddings_reduced_output_path = config.EMBEDDINGS_REDUCED_OUTPUT_PATHS.get(dataset_type_to_process) 

    if not all([input_path, embeddings_full_output_path, embeddings_ids_output_path]): 
        logging.error(f"Una o más rutas de E/S principales no están definidas en config.py para FE del conjunto '{dataset_type_to_process}'.")
        return
    if config.FE_PERFORM_DIMENSIONALITY_REDUCTION and not embeddings_reduced_output_path: 
        logging.error(f"Ruta de salida para embeddings reducidos no definida para '{dataset_type_to_process}', pero reducción habilitada.")
        return

    os.makedirs(config.FEATURES_DIR, exist_ok=True) 
    logging.info(f"Directorio de salida para Ingeniería de Características (embeddings) asegurado: {config.FEATURES_DIR}")
    
    logging.info(f"Cargando datos preprocesados desde: {input_path}")
    try:
        df_preprocessed = pd.read_csv(input_path) 
        logging.info(f"Datos preprocesados para '{dataset_type_to_process}' cargados: {len(df_preprocessed)} filas.")
    except FileNotFoundError:
        logging.error(f"Archivo no encontrado: {input_path}. Ejecuta primero el script de preprocesamiento para este conjunto.")
        return
    except Exception as e:
        logging.error(f"Error cargando {input_path}: {e}")
        return

    if df_preprocessed.empty or config.COL_PREPROCESSED_TEXT not in df_preprocessed.columns: 
        logging.error(f"DataFrame para '{dataset_type_to_process}' vacío o sin la columna '{config.COL_PREPROCESSED_TEXT}'.")
        return
    if config.COL_TWEET_ID not in df_preprocessed.columns: 
        logging.error(f"La columna de ID '{config.COL_TWEET_ID}' no se encuentra en el DataFrame. No se pueden guardar los IDs de los embeddings.")
        return

    # 1. Generar Embeddings Completos (Alta Dimensionalidad)
    full_embeddings, tweet_ids = generate_embeddings( 
        df=df_preprocessed,
        text_column=config.COL_PREPROCESSED_TEXT, 
        id_column=config.COL_TWEET_ID, 
        model_name=config.SENTENCE_TRANSFORMER_MODEL 
    )

    if full_embeddings is None or tweet_ids is None or full_embeddings.size == 0: 
        logging.error(f"Fallo en la generación de embeddings para '{dataset_type_to_process}'. Terminando esta etapa.")
        return

    save_data(full_embeddings, embeddings_full_output_path) 
    save_ids_csv(tweet_ids, embeddings_ids_output_path, config.COL_TWEET_ID) 

    # 2. Reducción de Dimensionalidad (controlada por config.py)
    if config.FE_PERFORM_DIMENSIONALITY_REDUCTION: 
        if full_embeddings is not None and full_embeddings.size > 0: 
            logging.info(f"Técnica de reducción configurada en config.py: '{config.FE_DIM_REDUCTION_TECHNIQUE.upper()}'") 
            
            reduced_embeddings = None 
            if config.FE_DIM_REDUCTION_TECHNIQUE == 'pca': 
                reduced_embeddings = apply_pca_dimensionality_reduction(
                    embeddings=full_embeddings,
                    dataset_type=dataset_type_to_process 
                    # Los parámetros n_components y random_state se toman de config.py
                    # a través de los valores por defecto de la función.
                )
            elif config.FE_DIM_REDUCTION_TECHNIQUE == 'umap': 
                logging.warning("La técnica UMAP está configurada, pero este script está enfocado en PCA. "
                                "Si deseas usar UMAP, asegúrate de que la función para UMAP (similar a apply_pca_dimensionality_reduction) exista y sea llamada aquí.")
            else:
                logging.error(f"Técnica de reducción '{config.FE_DIM_REDUCTION_TECHNIQUE}' no soportada o no implementada en este script.")

            if reduced_embeddings is not None and reduced_embeddings.size > 0: 
                save_data(reduced_embeddings, embeddings_reduced_output_path) 
            else:
                logging.warning(f"La reducción de dimensionalidad ({config.FE_DIM_REDUCTION_TECHNIQUE}) no produjo resultados para '{dataset_type_to_process}'.")
        else:
            logging.warning("No hay embeddings completos para realizar la reducción de dimensionalidad.")
    else:
        logging.info("Reducción de dimensionalidad omitida según la configuración (FE_PERFORM_DIMENSIONALITY_REDUCTION = False).")

    logging.info(f"===== Pipeline de Ingeniería de Características para '{dataset_type_to_process}' Finalizado =====")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Ejecuta el pipeline de ingeniería de características para un tipo de conjunto específico o todos."
    )
    parser.add_argument(
        "--dataset_type",
        type=str,
        required=False,
        choices=['discovery', 'validation', 'evaluation', 'all'],
        default='all', 
        help="Tipo de dataset a procesar. Opciones: 'discovery', 'validation', 'evaluation', 'all' (para procesar todos los definidos en config)."
    )
    args = parser.parse_args()

    # Validaciones de configuración esenciales
    if not config.SENTENCE_TRANSFORMER_MODEL: 
        logging.critical("El modelo SENTENCE_TRANSFORMER_MODEL no está definido en config.py")
    elif not hasattr(config, 'FE_INPUT_PATHS') or \
         not hasattr(config, 'EMBEDDINGS_FULL_OUTPUT_PATHS') or \
         not hasattr(config, 'EMBEDDINGS_IDS_OUTPUT_PATHS') or \
         not hasattr(config, 'EMBEDDINGS_REDUCED_OUTPUT_PATHS') or \
         not hasattr(config, 'PCA_MODEL_OUTPUT_PATH'): 
        logging.critical("Una o más constantes de ruta para feature_engineering (incl. PCA_MODEL_OUTPUT_PATH) no fueron encontradas en config.py.")
    elif config.FE_PERFORM_DIMENSIONALITY_REDUCTION and config.FE_DIM_REDUCTION_TECHNIQUE not in ['pca', 'umap']: 
        logging.critical(f"Técnica de reducción de dimensionalidad '{config.FE_DIM_REDUCTION_TECHNIQUE}' no es válida. Elegir 'pca' o 'umap' en config.py.")
    else:
        dataset_types_to_process_in_order = []
        if args.dataset_type == 'all':
            # Asegurar que 'discovery' se procese primero si 'all' es seleccionado
            # Esto es crucial porque PCA y KMeans se ajustan en 'discovery'
            dataset_types_to_process_in_order = ['discovery']
            for dt_key in config.FE_INPUT_PATHS.keys():
                if dt_key != 'discovery':
                    dataset_types_to_process_in_order.append(dt_key)
            logging.info(f"Procesando todos los tipos de dataset para ingeniería de características en orden: {dataset_types_to_process_in_order}...")
        else:
            dataset_types_to_process_in_order = [args.dataset_type]
            logging.info(f"Procesando dataset '{args.dataset_type}' para ingeniería de características...")

        for set_type_key in dataset_types_to_process_in_order:
            if set_type_key in config.FE_INPUT_PATHS:
                 main_feature_engineering_pipeline(dataset_type_to_process=set_type_key)
            else:
                logging.warning(f"El tipo de dataset '{set_type_key}' no tiene una ruta de entrada definida en FE_INPUT_PATHS. Omitiendo.")