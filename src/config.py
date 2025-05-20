# src/config.py

import os
import logging
import string
import pandas as pd # Para la lógica de KMEANS_N_INIT




# --- Configuración Sensible al Entorno (para AWS/Docker) ---
S3_BUCKET_NAME = os.getenv("S3_BUCKET")
S3_DATA_PREFIX = os.getenv("S3_DATA_PREFIX", "data") # Default 'data'

# --- Definiciones de Rutas Base ---
CURRENT_FILE_DIR = os.path.dirname(os.path.abspath(__file__)) # Directorio 'src'
PROJECT_ROOT_LOCAL = os.path.dirname(CURRENT_FILE_DIR) # Raíz del proyecto local

if S3_BUCKET_NAME:
    _s3_prefix_cleaned = S3_DATA_PREFIX.strip('/')
    BASE_DATA_PATH = f"s3://{S3_BUCKET_NAME}"
    if _s3_prefix_cleaned:
        BASE_DATA_PATH = f"{BASE_DATA_PATH}/{_s3_prefix_cleaned}"
    # logging.info(f"Usando rutas base S3: {BASE_DATA_PATH}") # Se logueará cuando se llame a setup_logging
else:
    BASE_DATA_PATH = os.path.join(PROJECT_ROOT_LOCAL, "data")
    # logging.info(f"Usando rutas base locales: {BASE_DATA_PATH}")

# Directorios principales de datos con prefijos numéricos
RAW_DATA_DIR = os.getenv("APP_RAW_DATA_DIR", os.path.join(BASE_DATA_PATH, "00_raw"))
INGESTED_SPLITS_DIR = os.getenv("APP_INGESTED_SPLITS_DIR", os.path.join(BASE_DATA_PATH, "01_ingested_splits"))
PREPARED_DATA_DIR = os.getenv("APP_PREPARED_DATA_DIR", os.path.join(BASE_DATA_PATH, "02_prepared_data"))
PREPROCESSED_TEXT_DIR = os.getenv("APP_PREPROCESSED_TEXT_DIR", os.path.join(BASE_DATA_PATH, "03_preprocessed_text"))
CLUSTERING_OUTPUTS_DIR = os.getenv("APP_CLUSTERING_OUTPUTS_DIR", os.path.join(BASE_DATA_PATH, "05_clustering_outputs"))
PSEUDO_LABELLED_DATA_DIR = os.getenv("APP_PSEUDO_LABELLED_DATA_DIR", os.path.join(BASE_DATA_PATH, "06_pseudo_labelled_data"))
RESOURCES_DIR = CURRENT_FILE_DIR # 'src' para chat_words.py, all_emoticons_expanded.py

MODELS_DIR = os.getenv("APP_MODELS_DIR", os.path.join(PROJECT_ROOT_LOCAL, "models"))
FEATURES_DIR = os.getenv("APP_FEATURES_DIR", os.path.join(BASE_DATA_PATH, "04_features"))
PCA_MODEL_OUTPUT_PATH = os.path.join(MODELS_DIR, "feature_reduction", "pca_model_fitted.joblib")

# --- Nombres de Columnas Clave ---
COL_TWEET_ID = 'tweet_id'
COL_AUTHOR_ID = 'author_id'
COL_INBOUND = 'inbound'
COL_TEXT = 'text'
COL_CREATED_AT = 'created_at'
COL_IN_REPLY_TO_TWEET_ID = 'in_response_to_tweet_id'
COL_RESPONSE_TWEET_ID = 'response_tweet_id'
COL_TWEET_LENGTH = 'tweet_length'
COL_PREPROCESSED_TEXT = 'text_preprocessed'
COL_DETECTED_LANGUAGE = 'detected_language'
COL_CATEGORY = 'category'

# --- Parámetros de Ingestión y División (data_ingestion.py) ---
KAGGLE_DATASET_SLUG = os.getenv("APP_KAGGLE_DATASET_SLUG", "thoughtvector/customer-support-on-twitter")
EXPECTED_CSV_FILENAME_IN_KAGGLE_ARCHIVE = "twcs.csv"
KAGGLE_SUBFOLDER_IN_ARCHIVE = "twcs"

INITIAL_COLS_FROM_RAW = [
    COL_TWEET_ID, COL_AUTHOR_ID, COL_INBOUND, COL_TEXT,
    COL_CREATED_AT, COL_IN_REPLY_TO_TWEET_ID, COL_RESPONSE_TWEET_ID
]

def get_bool_env(var_name: str, default_value: bool) -> bool:
    val = os.getenv(var_name)
    if val is None:
        return default_value
    return val.lower() in ('true', '1', 't', 'yes', 'y')

DISCOVERY_TRAIN_SIZE: float = float(os.getenv("APP_DISCOVERY_TRAIN_SIZE", "0.70"))
_default_val_size_rel = (0.20 / (1.0 - DISCOVERY_TRAIN_SIZE)) if (1.0 - DISCOVERY_TRAIN_SIZE) > 0 else 0
VALIDATION_SIZE_REL_TO_REMAINING: float = float(os.getenv("APP_VALIDATION_SIZE_REL", str(_default_val_size_rel)))
RANDOM_STATE: int = int(os.getenv("APP_RANDOM_STATE", "42"))

RAW_TWCS_CSV_PATH = os.path.join(RAW_DATA_DIR, EXPECTED_CSV_FILENAME_IN_KAGGLE_ARCHIVE)

INGESTED_DISCOVERY_SET_PATH = os.path.join(INGESTED_SPLITS_DIR, "discovery_split_raw.csv")
INGESTED_VALIDATION_SET_PATH = os.path.join(INGESTED_SPLITS_DIR, "validation_split_raw.csv")
INGESTED_EVALUATION_SET_PATH = os.path.join(INGESTED_SPLITS_DIR, "evaluation_split_raw.csv")

# --- Parámetros de Preparación de Datos (data_preparation.py) ---
PREPARED_DATA_COLUMNS = INITIAL_COLS_FROM_RAW + [COL_TWEET_LENGTH]
PREPARATION_INPUT_PATHS = {
    "discovery": INGESTED_DISCOVERY_SET_PATH,
    "validation": INGESTED_VALIDATION_SET_PATH,
    "evaluation": INGESTED_EVALUATION_SET_PATH,
}
PREPARED_OUTPUT_PATHS = {
    "discovery": os.path.join(PREPARED_DATA_DIR, "discovery_prepared.csv"),
    "validation": os.path.join(PREPARED_DATA_DIR, "validation_prepared.csv"),
    "evaluation": os.path.join(PREPARED_DATA_DIR, "evaluation_prepared.csv"),
}

# --- Parámetros de Preprocesamiento (preprocessing.py) ---
PREPROCESSING_INPUT_PATHS = {
    "discovery": PREPARED_OUTPUT_PATHS["discovery"],
    "validation": PREPARED_OUTPUT_PATHS["validation"],
    "evaluation": PREPARED_OUTPUT_PATHS["evaluation"],
}
PREPROCESSED_OUTPUT_PATHS = {
    "discovery": os.path.join(PREPROCESSED_TEXT_DIR, "discovery_preprocessed.csv"),
    "validation": os.path.join(PREPROCESSED_TEXT_DIR, "validation_preprocessed.csv"),
    "evaluation": os.path.join(PREPROCESSED_TEXT_DIR, "evaluation_preprocessed.csv"),
}

CHAT_WORDS_FILE = os.path.join(RESOURCES_DIR, "chat_words.py")
EMOTICONS_FILE = os.path.join(RESOURCES_DIR, "all_emoticons_expanded.py")
DEFAULT_FALLBACK_LANG = os.getenv("APP_DEFAULT_FALLBACK_LANG", "en")
PUNCT_TO_REMOVE = string.punctuation.replace("_", "")

PREPROCESSING_DO_LOWERCASE = get_bool_env("APP_PREPROCESSING_DO_LOWERCASE", True)
PREPROCESSING_REMOVE_HTML = get_bool_env("APP_PREPROCESSING_REMOVE_HTML", True)
PREPROCESSING_REMOVE_URLS = get_bool_env("APP_PREPROCESSING_REMOVE_URLS", True)
PREPROCESSING_REMOVE_MENTIONS_HASHTAGS = get_bool_env("APP_PREPROCESSING_REMOVE_MENTIONS_HASHTAGS", True)
PREPROCESSING_EXPAND_CHAT_WORDS = get_bool_env("APP_PREPROCESSING_EXPAND_CHAT_WORDS", True)
PREPROCESSING_CONVERT_EMOJIS_EMOTICONS = get_bool_env("APP_PREPROCESSING_CONVERT_EMOJIS_EMOTICONS", True)
PREPROCESSING_REMOVE_PUNCTUATION = get_bool_env("APP_PREPROCESSING_REMOVE_PUNCTUATION", True)
PREPROCESSING_REMOVE_STOPWORDS = get_bool_env("APP_PREPROCESSING_REMOVE_STOPWORDS", False)
PREPROCESSING_ENABLE_SPELL_CORRECTION = get_bool_env("APP_PREPROCESSING_ENABLE_SPELL_CORRECTION", False)
PREPROCESSING_ENABLE_LEMMATIZATION = get_bool_env("APP_PREPROCESSING_ENABLE_LEMMATIZATION", True)

LANG_MAP = {
    'en': {'name': 'english', 'omw': 'eng', 'spellchecker': 'en', 'nltk_stopwords': 'english'},
    'es': {'name': 'spanish', 'omw': 'spa', 'spellchecker': 'es', 'nltk_stopwords': 'spanish'},
    'fr': {'name': 'french', 'omw': 'fra', 'spellchecker': 'fr', 'nltk_stopwords': 'french'},
    'de': {'name': 'german', 'omw': 'deu', 'spellchecker': 'de', 'nltk_stopwords': 'german'},
    'pt': {'name': 'portuguese', 'omw': 'por', 'spellchecker': 'pt', 'nltk_stopwords': 'portuguese'},
    'it': {'name': 'italian', 'omw': 'ita', 'spellchecker': 'it', 'nltk_stopwords': 'italian'},
    'nl': {'name': 'dutch', 'omw': 'nld', 'spellchecker': 'nl', 'nltk_stopwords': 'dutch'},
    'ru': {'name': 'russian', 'omw': 'rus', 'spellchecker': 'ru', 'nltk_stopwords': 'russian'},
}
SUPPORTED_LANGS_FOR_SPECIFIC_PROCESSING = list(LANG_MAP.keys())
NLTK_RESOURCES_NEEDED = {
    'corpora/stopwords': 'stopwords', 'tokenizers/punkt': 'punkt',
    'taggers/averaged_perceptron_tagger': 'averaged_perceptron_tagger',
    'corpora/wordnet': 'wordnet', 'corpora/omw-1.4': 'omw-1.4'
}

# --- Parámetros de Ingeniería de Características (feature_engineering.py) ---
FE_INPUT_PATHS = PREPROCESSED_OUTPUT_PATHS # Usa las salidas preprocesadas
SENTENCE_TRANSFORMER_MODEL = os.getenv("APP_SENTENCE_TRANSFORMER_MODEL", 'paraphrase-multilingual-MiniLM-L12-v2')

EMBEDDINGS_FULL_OUTPUT_PATHS = {
    "discovery": os.path.join(FEATURES_DIR, "discovery_embeddings_full.npy"),
    "validation": os.path.join(FEATURES_DIR, "validation_embeddings_full.npy"),
    "evaluation": os.path.join(FEATURES_DIR, "evaluation_embeddings_full.npy"),
}
EMBEDDINGS_IDS_OUTPUT_PATHS = {
    "discovery": os.path.join(FEATURES_DIR, "discovery_embeddings_ids.csv"),
    "validation": os.path.join(FEATURES_DIR, "validation_embeddings_ids.csv"),
    "evaluation": os.path.join(FEATURES_DIR, "evaluation_embeddings_ids.csv"),
}

FE_PERFORM_DIMENSIONALITY_REDUCTION = get_bool_env("APP_FE_PERFORM_DIM_REDUCTION", True)
FE_DIM_REDUCTION_TECHNIQUE = os.getenv("APP_DIM_REDUCTION_TECHNIQUE", "pca").lower()
PCA_N_COMPONENTS = int(os.getenv("APP_PCA_N_COMPONENTS", "50"))


UMAP_N_COMPONENTS = int(os.getenv("APP_UMAP_N_COMPONENTS", str(PCA_N_COMPONENTS)))
UMAP_N_NEIGHBORS = int(os.getenv("APP_UMAP_N_NEIGHBORS", "15"))
UMAP_MIN_DIST = float(os.getenv("APP_UMAP_MIN_DIST", "0.1"))
UMAP_METRIC = os.getenv("APP_UMAP_METRIC", "cosine")

if FE_DIM_REDUCTION_TECHNIQUE == 'pca':
    _n_components_for_suffix = PCA_N_COMPONENTS
elif FE_DIM_REDUCTION_TECHNIQUE == 'umap':
    _n_components_for_suffix = UMAP_N_COMPONENTS
else:
    _n_components_for_suffix = "unknown_tech"
    logging.warning(f"Técnica de reducción desconocida '{FE_DIM_REDUCTION_TECHNIQUE}', usando sufijo 'unknown_tech'.")
DIM_REDUCTION_SUFFIX = f"{FE_DIM_REDUCTION_TECHNIQUE}_{_n_components_for_suffix}d"

EMBEDDINGS_REDUCED_OUTPUT_PATHS = {
    "discovery": os.path.join(FEATURES_DIR, f"discovery_embeddings_reduced_{DIM_REDUCTION_SUFFIX}.npy"),
    "validation": os.path.join(FEATURES_DIR, f"validation_embeddings_reduced_{DIM_REDUCTION_SUFFIX}.npy"),
    "evaluation": os.path.join(FEATURES_DIR, f"evaluation_embeddings_reduced_{DIM_REDUCTION_SUFFIX}.npy"),
}

# --- Parámetros de Clustering (clustering.py) ---
CLUSTERING_INPUT_EMBEDDINGS_PATHS = EMBEDDINGS_REDUCED_OUTPUT_PATHS
CLUSTERING_INPUT_IDS_PATHS = EMBEDDINGS_IDS_OUTPUT_PATHS
CLUSTERING_ALGORITHM = os.getenv("APP_CLUSTERING_ALGORITHM", "kmeans").lower()

KMEANS_N_CLUSTERS = int(os.getenv("APP_KMEANS_N_CLUSTERS", "8"))
KMEANS_INIT_METHOD = os.getenv("APP_KMEANS_INIT_METHOD", 'k-means++')
try:
    from sklearn import __version__ as sklearn_version
    from packaging.version import parse as parse_version
    if parse_version(sklearn_version) >= parse_version("1.4"): _n_init_default = 'auto'
    else: _n_init_default = 10
except ImportError: _n_init_default = 10
KMEANS_N_INIT_STR_OR_INT = os.getenv("APP_KMEANS_N_INIT", str(_n_init_default))
KMEANS_N_INIT = int(KMEANS_N_INIT_STR_OR_INT) if KMEANS_N_INIT_STR_OR_INT != 'auto' else 'auto'
KMEANS_MAX_ITER = int(os.getenv("APP_KMEANS_MAX_ITER", "300"))

_clustering_method_suffix = f"kmeans_k{KMEANS_N_CLUSTERS}" if CLUSTERING_ALGORITHM == 'kmeans' else CLUSTERING_ALGORITHM

CLUSTER_ASSIGNMENTS_OUTPUT_PATHS = {
    "discovery": os.path.join(CLUSTERING_OUTPUTS_DIR, f"discovery_cluster_assignments_{_clustering_method_suffix}.csv"),
}
CLUSTER_MODEL_OUTPUT_PATHS = {
    "discovery": os.path.join(MODELS_DIR, f"discovery_clustering_model_{_clustering_method_suffix}.joblib"),
}

# --- Pseudo-Etiquetado ---
PSEUDO_LABELLED_DISCOVERY_PATH = os.path.join(PSEUDO_LABELLED_DATA_DIR, f"discovery_pseudo_labelled_{_clustering_method_suffix}.csv")
PSEUDO_LABELLED_VALIDATION_PATH = os.path.join(PSEUDO_LABELLED_DATA_DIR, f"validation_pseudo_labelled_{_clustering_method_suffix}.csv")
PSEUDO_LABELLED_EVALUATION_PATH = os.path.join(PSEUDO_LABELLED_DATA_DIR, f"evaluation_pseudo_labelled_{_clustering_method_suffix}.csv")

# Hiperparámetros del Modelo de Clasificación
LOGREG_C = 1 
LOGREG_CLASS_WEIGHT = 'balanced'

# --- Configuración de Logging Global ---
LOGGING_LEVEL = logging.INFO 
LOGGING_FORMAT = '%(asctime)s - %(levelname)s - %(name)s - %(module)s.%(funcName)s:%(lineno)d - %(message)s'
LOGGING_DATE_FORMAT = '%Y-%m-%d %H:%M:%S'

def setup_logging():
    root_logger = logging.getLogger()
    if not root_logger.hasHandlers():
        log_level_str = os.getenv("APP_LOGGING_LEVEL", "INFO").upper()
        log_level_int = getattr(logging, log_level_str, logging.INFO)
        logging.basicConfig(level=log_level_int, format=LOGGING_FORMAT, datefmt=LOGGING_DATE_FORMAT)
    # Loguear rutas base DESPUÉS de que el logging esté configurado
    if not getattr(setup_logging, "has_logged_paths", False): # Para loguear solo una vez
        if S3_BUCKET_NAME: 
            logging.info(f"Configuración de S3: Bucket='{S3_BUCKET_NAME}', Prefijo de Datos='{S3_DATA_PREFIX}'")
        logging.info(f"Ruta base para datos (local o S3): {BASE_DATA_PATH}")
        logging.info(f"Directorio de Modelos: {MODELS_DIR}")
        setattr(setup_logging, "has_logged_paths", True)
