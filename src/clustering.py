# src/clustering.py

import pandas as pd
import numpy as np
import os
import logging
from sklearn.cluster import KMeans
from sklearn.feature_extraction.text import CountVectorizer 
import joblib 
from typing import Optional, Dict, List, Any, Tuple 
import argparse
from tqdm.auto import tqdm

# Importar configuraciones y constantes compartidas
try:
    from . import config
except ImportError:
    import config

# Configurar logging
config.setup_logging()

def load_embeddings_and_ids(
    embeddings_path: str,
    ids_path: str,
    id_column_name: str = config.COL_TWEET_ID
) -> Optional[Tuple[np.ndarray, pd.DataFrame]]:
    """Carga los embeddings y los IDs correspondientes."""
    logging.info(f"Cargando embeddings desde: {embeddings_path}")
    try:
        embeddings = np.load(embeddings_path)
        if embeddings.size == 0:
            logging.error(f"El archivo de embeddings {embeddings_path} está vacío.")
            return None, None
        logging.info(f"Embeddings cargados. Forma: {embeddings.shape}")
    except FileNotFoundError:
        logging.error(f"Archivo de embeddings no encontrado: {embeddings_path}")
        return None, None
    except Exception as e:
        logging.error(f"Error cargando embeddings desde {embeddings_path}: {e}")
        return None, None

    logging.info(f"Cargando IDs desde: {ids_path}")
    try:
        df_ids = pd.read_csv(ids_path) 
        if id_column_name not in df_ids.columns:
            logging.error(f"La columna de ID '{id_column_name}' no se encuentra en {ids_path}")
            return None, None
        if df_ids.empty:
            logging.error(f"El archivo de IDs {ids_path} está vacío.")
            return None, None
        logging.info(f"IDs cargados. Número de IDs: {len(df_ids)}")
    except FileNotFoundError:
        logging.error(f"Archivo de IDs no encontrado: {ids_path}")
        return None, None
    except Exception as e:
        logging.error(f"Error cargando IDs desde {ids_path}: {e}")
        return None, None

    if embeddings.shape[0] != len(df_ids):
        logging.error(
            f"El número de embeddings ({embeddings.shape[0]}) no coincide con el número de IDs ({len(df_ids)})."
            "Asegúrate de que provengan de la misma ejecución de feature_engineering.py."
        )
        return None, None
    return embeddings, df_ids


def apply_kmeans_clustering(
    embeddings: np.ndarray,
    dataset_type: str, 
    n_clusters_for_training: int, # k a usar si dataset_type es 'discovery'
    discovery_kmeans_model_path_template: str, # Ej: "models/discovery_clustering_model_kmeans_k{k}.joblib"
    random_state: int = config.RANDOM_STATE,
    init_method: str = config.KMEANS_INIT_METHOD,
    n_init_val: Any = config.KMEANS_N_INIT,
    max_iter_val: int = config.KMEANS_MAX_ITER
) -> Optional[Tuple[np.ndarray, Optional[KMeans]]]:
    """
    Aplica KMeans. Ajusta en 'discovery' con n_clusters_for_training y guarda el modelo.
    Para otros conjuntos, carga el modelo de 'discovery' (ajustado con config.KMEANS_N_CLUSTERS) y predice.
    """
    if embeddings is None:
        logging.error(f"Embeddings son None para el conjunto {dataset_type}.")
        return None, None
    
    # Ruta del modelo KMEANS de discovery (el que se usará para aplicar a validation/evaluation)
    # Se asume que KMEANS_N_CLUSTERS en config.py es el k óptimo de discovery.
    discovery_model_path_to_load = discovery_kmeans_model_path_template.format(k=config.KMEANS_N_CLUSTERS)

    if dataset_type == "discovery":
        if embeddings.shape[0] < n_clusters_for_training:
            logging.error(f"No hay suficientes muestras ({embeddings.shape[0]}) para formar {n_clusters_for_training} clusters con K-Means en 'discovery'.")
            return None, None
        
        logging.info(f"Ajustando K-Means con k={n_clusters_for_training} en el conjunto 'discovery'...")
        logging.info(f"Parámetros K-Means: init='{init_method}', n_init='{n_init_val}', max_iter={max_iter_val}")
        try:
            # Ruta para guardar el modelo de discovery específico para ESTE k de entrenamiento
            current_discovery_model_save_path = discovery_kmeans_model_path_template.format(k=n_clusters_for_training)
            os.makedirs(os.path.dirname(current_discovery_model_save_path), exist_ok=True)
            
            kmeans = KMeans(
                n_clusters=n_clusters_for_training,
                init=init_method,
                n_init=n_init_val,
                max_iter=max_iter_val,
                random_state=random_state,
                verbose=0
            )
            cluster_labels = kmeans.fit_predict(embeddings)
            joblib.dump(kmeans, current_discovery_model_save_path)
            logging.info(f"Modelo K-Means (discovery k={n_clusters_for_training}) ajustado y guardado en: {current_discovery_model_save_path}")
            return cluster_labels, kmeans
        except Exception as e:
            logging.error(f"Error ajustando K-Means en 'discovery': {e}")
            return None, None
    else: # Para 'validation' y 'evaluation'
        if embeddings.shape[0] == 0:
            logging.warning(f"Embeddings vacíos para el conjunto {dataset_type}. No se pueden predecir clusters.")
            return np.array([]), None

        logging.info(f"Cargando modelo K-Means (ajustado en discovery con k={config.KMEANS_N_CLUSTERS}) desde {discovery_model_path_to_load} para predecir en '{dataset_type}'...")
        if not os.path.exists(discovery_model_path_to_load):
            logging.error(f"Modelo K-Means de discovery no encontrado en {discovery_model_path_to_load}. Ejecuta primero el pipeline para 'discovery' con k={config.KMEANS_N_CLUSTERS}.")
            return None, None
        try:
            kmeans_model = joblib.load(discovery_model_path_to_load)
            cluster_labels = kmeans_model.predict(embeddings)
            return cluster_labels, kmeans_model
        except Exception as e:
            logging.error(f"Error prediciendo con K-Means en '{dataset_type}': {e}")
            return None, None

def get_top_frequent_keywords_per_cluster(
    df_with_text_and_cluster: pd.DataFrame, 
    cluster_id_col: str, 
    text_col_for_keywords: str, 
    target_cluster_id: int, 
    n_top_words: int = 10
) -> List[str]:
    cluster_texts = df_with_text_and_cluster[df_with_text_and_cluster[cluster_id_col] == target_cluster_id][text_col_for_keywords].dropna().astype(str).tolist()
    if not cluster_texts or len(cluster_texts) < 1:
        logging.debug(f"No hay textos en el cluster {target_cluster_id} para calcular frecuencia de palabras (textos: {len(cluster_texts)}).")
        return ["(No hay textos suficientes para keywords)"]
    try:
        stopwords = 'english' if not config.PREPROCESSING_REMOVE_STOPWORDS else None
        vectorizer = CountVectorizer(
            max_features=2000, 
            ngram_range=(1, 2), 
            stop_words=stopwords
        )
        word_count_matrix = vectorizer.fit_transform(cluster_texts)
        sum_words = word_count_matrix.sum(axis=0) 
        words_freq = [(word, sum_words[0, idx]) for word, idx in vectorizer.vocabulary_.items()]
        words_freq = sorted(words_freq, key = lambda x: x[1], reverse=True)
        return [word for word, freq in words_freq[:n_top_words]]
    except ValueError as ve:
         logging.warning(f"Error de CountVectorizer (prob. vocabulario vacío) para cluster {target_cluster_id}: {ve}")
         return ["(Vocabulario vacío o error en CountVectorizer)"]
    except Exception as e:
        logging.error(f"Error calculando frecuencia de palabras para cluster {target_cluster_id}: {e}")
        return ["(Error al calcular keywords)"]

def generate_cluster_analysis_report(
    df_with_clusters: pd.DataFrame, 
    output_report_path: str,
    n_samples_per_cluster: int = 50 
):
    """Genera un archivo de texto con muestras y keywords por cluster para análisis manual."""
    logging.info(f"Generando reporte de análisis de clusters en: {output_report_path}")
    all_clusters_info_string = ""
    
    text_cols_to_show = [col for col in [config.COL_TEXT, config.COL_PREPROCESSED_TEXT] if col in df_with_clusters.columns]
    unique_cluster_ids = sorted(df_with_clusters['cluster_id'].unique())

    for cid in tqdm(unique_cluster_ids, desc="Generando Reporte de Clusters"):
        cluster_df = df_with_clusters[df_with_clusters['cluster_id'] == cid]
        cluster_size = len(cluster_df)
        
        header = f"\n\n================ CLUSTER ID: {cid} (Total Tweets: {cluster_size}) ================\n"
        all_clusters_info_string += header
        
        samples_to_take = min(n_samples_per_cluster, cluster_size)
        if samples_to_take > 0:
            sample_tweets = cluster_df.sample(samples_to_take, random_state=config.RANDOM_STATE)
            all_clusters_info_string += f"--- {samples_to_take} Muestras de Tweets ---\n"
            for idx_sample, row_sample in sample_tweets.iterrows():
                all_clusters_info_string += f"Tweet ID: {row_sample.get(config.COL_TWEET_ID, 'N/A')}\n"
                for tc in text_cols_to_show:
                    text_content = row_sample.get(tc, "")
                    display_text = "[TEXTO ES NaN]" if pd.isna(text_content) else \
                                   "[TEXTO VACÍO O SOLO ESPACIOS]" if not str(text_content).strip() else \
                                   str(text_content)
                    all_clusters_info_string += f"  {tc}: {display_text}\n" 
                all_clusters_info_string += "------------------------------------\n"
        
        keywords = get_top_frequent_keywords_per_cluster(
            df_with_clusters, 'cluster_id', config.COL_PREPROCESSED_TEXT, cid, n_top_words=15
        )
        all_clusters_info_string += f"  Palabras/Frases Clave Más Frecuentes: {keywords}\n"
        all_clusters_info_string += f"  >> Nombre de Categoría para Cluster {cid}: [TU_NOMBRE_AQUÍ]\n"

    try:
        os.makedirs(os.path.dirname(output_report_path), exist_ok=True)
        with open(output_report_path, "w", encoding="utf-8") as f:
            f.write(all_clusters_info_string)
        logging.info(f"Reporte de análisis de clusters guardado en: {output_report_path}")
    except Exception as e:
        logging.error(f"No se pudo guardar el reporte de análisis de clusters: {e}")

def define_category_map() -> Dict[int, str]:
    """
    Define el mapeo de cluster_id a nombre de categoría.
    !!! ESTA FUNCIÓN ES DONDE TÚ CODIFICAS TUS DECISIONES DESPUÉS DEL ANÁLISIS DEL REPORTE !!!
    """
    category_map = {
        0: "Consultas_Problemas_Productos_Pedidos",
        1: "Agradecimiento_Cliente",
        2: "Feedback_General_Expresivo_Emojis",
        3: "Soporte_Tecnico_Fallas_SW_HW_Servicios",
        4: "Contenido_Baja_Informacion_o_No_Procesable",
        5: "Gestion_Cuentas_Pedidos_Atencion_Cliente",
        6: "Problemas_Consultas_Servicios_Transporte",
        7: "Dialogo_Interaccion_Soporte_General"
    }
    logging.info(f"Usando el siguiente mapeo de categorías definido manualmente: {category_map}")
    return category_map

def main_clustering_pipeline(dataset_type: str, run_analysis_report_only: bool = False, k_for_discovery_training: Optional[int] = None):
    logging.info(f"===== Iniciando Pipeline de Clustering para el conjunto: '{dataset_type}' =====")
    
    embeddings_input_path = config.CLUSTERING_INPUT_EMBEDDINGS_PATHS.get(dataset_type)
    ids_input_path = config.CLUSTERING_INPUT_IDS_PATHS.get(dataset_type)
    
    # k a usar para entrenar KMEANS si es 'discovery'. Si no se pasa, usa config.KMEANS_N_CLUSTERS.
    k_train_val = k_for_discovery_training if k_for_discovery_training is not None else config.KMEANS_N_CLUSTERS
    
    # Ruta base para el modelo KMEANS de discovery (para guardar si es discovery, para cargar si es val/eval)
    # El nombre siempre refleja el k con el que se *debería* haber entrenado en discovery (config.KMEANS_N_CLUSTERS)
    # o el k experimental si se está generando solo un reporte para discovery.
    k_for_discovery_model_file = k_train_val if dataset_type == "discovery" and run_analysis_report_only and k_for_discovery_training is not None else config.KMEANS_N_CLUSTERS
    discovery_kmeans_model_path = config.CLUSTER_MODEL_OUTPUT_PATHS.get("discovery", "").replace(
        f"kmeans_k{config.KMEANS_N_CLUSTERS}", f"kmeans_k{k_for_discovery_model_file}"
    )
    if not discovery_kmeans_model_path: # Fallback si la ruta base de discovery no está en config
        discovery_kmeans_model_path = os.path.join(config.MODELS_DIR, f"discovery_clustering_model_kmeans_k{k_for_discovery_model_file}.joblib")

    # Ruta de salida de las asignaciones de cluster para el dataset_type actual
    # El sufijo del archivo de asignaciones siempre usa KMEANS_N_CLUSTERS de config
    # porque se asume que ese es el modelo 'oficial' de discovery que se aplica.
    assignments_suffix = f"kmeans_k{config.KMEANS_N_CLUSTERS}"
    assignments_output_path = os.path.join(config.CLUSTERING_OUTPUTS_DIR, f"{dataset_type}_cluster_assignments_{assignments_suffix}.csv")

    # Ruta de salida del dataset pseudo-etiquetado
    pseudo_labelled_output_path = None
    if dataset_type == "discovery":
        pseudo_labelled_output_path = config.PSEUDO_LABELLED_DISCOVERY_PATH
    elif dataset_type == "validation":
        pseudo_labelled_output_path = config.PSEUDO_LABELLED_VALIDATION_PATH
    elif dataset_type == "evaluation":
        pseudo_labelled_output_path = config.PSEUDO_LABELLED_EVALUATION_PATH

    if not all([embeddings_input_path, ids_input_path, assignments_output_path]):
        logging.error(f"Una o más rutas de E/S para clustering del conjunto '{dataset_type}' no están definidas.")
        return

    embeddings, df_ids = load_embeddings_and_ids(embeddings_input_path, ids_input_path)
    if embeddings is None or df_ids is None: return

    logging.info(f"Procesando clustering para '{dataset_type}'. k para entrenamiento (si discovery): {k_train_val}. Modelo base (para carga val/eval): k={config.KMEANS_N_CLUSTERS}.")
    
    cluster_labels, _ = apply_kmeans_clustering(
        embeddings=embeddings, 
        dataset_type=dataset_type,
        n_clusters_for_training=k_train_val, # Usado solo si dataset_type es 'discovery'
        discovery_kmeans_model_path_template=config.CLUSTER_MODEL_OUTPUT_PATHS.get("discovery_template", os.path.join(config.MODELS_DIR, "discovery_clustering_model_kmeans_k{k}.joblib"))
       
    )

    if cluster_labels is None:
        logging.error(f"No se generaron etiquetas de cluster para '{dataset_type}'.")
        return

    df_results = df_ids.copy()
    df_results['cluster_id'] = cluster_labels
    
    try:
        os.makedirs(os.path.dirname(assignments_output_path), exist_ok=True)
        df_results.to_csv(assignments_output_path, index=False, encoding='utf-8')
        logging.info(f"Asignaciones de cluster para '{dataset_type}' guardadas en: {assignments_output_path}")
    except Exception as e:
        logging.error(f"Error guardando asignaciones de cluster para '{dataset_type}': {e}")
        return

    preprocessed_text_path = config.PREPROCESSED_OUTPUT_PATHS.get(dataset_type)
    if not preprocessed_text_path or not os.path.exists(preprocessed_text_path):
        logging.error(f"No se encontró el archivo de texto preprocesado: {preprocessed_text_path} para '{dataset_type}'.")
        return
    
    df_texts_data = pd.read_csv(preprocessed_text_path)
    df_full_analysis = pd.merge(df_results, df_texts_data, on=config.COL_TWEET_ID, how='left')

    if dataset_type == "discovery" and run_analysis_report_only:
        report_k_suffix = f"kmeans_k{k_train_val}" # El reporte usa el k con el que se entrenó esta vez
        report_path = os.path.join(config.CLUSTERING_OUTPUTS_DIR, f"discovery_{report_k_suffix}_analysis_report.txt")
        generate_cluster_analysis_report(df_full_analysis, report_path, n_samples_per_cluster=50)
        logging.info(f"Reporte de análisis para 'discovery' (k={k_train_val}) generado. Revisa '{report_path}' y define 'define_category_map()'.")
        return 

    category_map = define_category_map()
    df_full_analysis[config.COL_CATEGORY] = df_full_analysis['cluster_id'].map(category_map)

    unmapped_count = df_full_analysis[config.COL_CATEGORY].isnull().sum()
    if unmapped_count > 0:
        logging.warning(f"{unmapped_count} tweets no pudieron ser mapeados a una categoría. IDs de cluster no mapeados: "
                        f"{df_full_analysis[df_full_analysis[config.COL_CATEGORY].isnull()]['cluster_id'].unique().tolist()}")

    logging.info(f"Distribución de las categorías finales para '{dataset_type}':\n{df_full_analysis[config.COL_CATEGORY].value_counts(dropna=False)}")

    if pseudo_labelled_output_path:
        cols_to_save = [
            config.COL_TWEET_ID, config.COL_TEXT, config.COL_PREPROCESSED_TEXT, 
            'cluster_id', config.COL_CATEGORY
        ]
        for col in config.INITIAL_COLS_FROM_RAW:
            if col not in cols_to_save and col in df_full_analysis.columns:
                cols_to_save.append(col)
        
        cols_to_save = [col for col in cols_to_save if col in df_full_analysis.columns]
        df_pseudo_labelled = df_full_analysis[cols_to_save]
        
        try:
            os.makedirs(os.path.dirname(pseudo_labelled_output_path), exist_ok=True)
            df_pseudo_labelled.to_csv(pseudo_labelled_output_path, index=False, encoding='utf-8')
            logging.info(f"Dataset pseudo-etiquetado para '{dataset_type}' guardado en: {pseudo_labelled_output_path}")
        except Exception as e:
            logging.error(f"Error guardando dataset pseudo-etiquetado para '{dataset_type}': {e}")
    else:
        logging.warning(f"Ruta de salida para dataset pseudo-etiquetado de '{dataset_type}' no definida.")

    logging.info(f"===== Pipeline de Clustering y Etiquetado para '{dataset_type}' Finalizado =====")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Ejecuta el pipeline de clustering K-Means y etiquetado. "
                    "Genera un reporte de análisis o un dataset pseudo-etiquetado."
    )
    parser.add_argument(
        "--dataset_type",
        type=str,
        default="all",
        choices=['discovery', 'validation', 'evaluation', 'all'],
        help="Dataset a procesar. 'all' procesará discovery, luego aplicará a validation y evaluation."
    )
    parser.add_argument(
        "--k", 
        type=int,
        default=None, 
        help=(
            f"Número específico de clusters (k) para entrenar K-Means en 'discovery'. "
            f"Si no se provee, usará config.KMEANS_N_CLUSTERS (actualmente: {config.KMEANS_N_CLUSTERS})."
        )
    )
    parser.add_argument(
        "--generate_report_only",
        action='store_true',
        help=(
            "Si se establece (y --dataset_type es 'discovery'), solo genera el reporte de análisis de clusters "
            "y el modelo de clustering, pero no el dataset pseudo-etiquetado final. "
            "Ignorado para dataset_type 'validation' o 'evaluation'."
        )
    )
    args = parser.parse_args()

    config_ok = True
    if not all([
        config.CLUSTERING_INPUT_EMBEDDINGS_PATHS.get("discovery"),
        config.CLUSTER_MODEL_OUTPUT_PATHS.get("discovery_template", os.path.join(config.MODELS_DIR, "discovery_clustering_model_kmeans_k{k}.joblib")), # Asegura que la plantilla exista o haya un fallback
        config.PSEUDO_LABELLED_DISCOVERY_PATH,
        config.PSEUDO_LABELLED_VALIDATION_PATH,
        config.PSEUDO_LABELLED_EVALUATION_PATH
    ]):
        logging.critical("Faltan una o más rutas de configuración esenciales para clustering en config.py.")
        config_ok = False
        
    if not hasattr(config, 'KMEANS_N_CLUSTERS'):
        logging.critical("KMEANS_N_CLUSTERS no definido en config.py.")
        config_ok = False
        
    if not config_ok:
        logging.error("Terminando debido a configuraciones faltantes. Revisa config.py.")
    else:
        k_for_discovery_train_run = args.k if args.k is not None else config.KMEANS_N_CLUSTERS
        
        if args.k is not None and args.k != config.KMEANS_N_CLUSTERS:
            logging.info(
                f"Se usará k={args.k} desde la línea de comandos para 'discovery', "
                f"sobrescribiendo config.KMEANS_N_CLUSTERS ({config.KMEANS_N_CLUSTERS}) para el entrenamiento de discovery en esta ejecución."
            )
        elif args.k is None:
             logging.info(f"Usando k={config.KMEANS_N_CLUSTERS} desde config.py para K-Means en 'discovery'.")


        if args.dataset_type == 'all':
            logging.info(f"Procesando 'discovery' con K-Means (k para entrenamiento: {k_for_discovery_train_run})...")
            main_clustering_pipeline(
                dataset_type="discovery", 
                run_analysis_report_only=args.generate_report_only,
                k_for_discovery_training=k_for_discovery_train_run 
            )
            
            if not args.generate_report_only:
                # Validation y Evaluation siempre usan el modelo de discovery entrenado con config.KMEANS_N_CLUSTERS
                logging.info(f"Aplicando modelo K-Means de discovery (k={config.KMEANS_N_CLUSTERS}) y mapeo a 'validation'...")
                main_clustering_pipeline(dataset_type="validation", run_analysis_report_only=False, k_for_discovery_training=None)
                
                logging.info(f"Aplicando modelo K-Means de discovery (k={config.KMEANS_N_CLUSTERS}) y mapeo a 'evaluation'...")
                main_clustering_pipeline(dataset_type="evaluation", run_analysis_report_only=False, k_for_discovery_training=None)
        else: 
            k_param_specific = k_for_discovery_train_run if args.dataset_type == "discovery" else None
            report_only_specific = args.generate_report_only if args.dataset_type == "discovery" else False
            
            main_clustering_pipeline(
                dataset_type=args.dataset_type,
                run_analysis_report_only=report_only_specific,
                k_for_discovery_training=k_param_specific
            )