# src/data_ingestion.py

import pandas as pd
from sklearn.model_selection import train_test_split
import os
import logging
from typing import List, Optional
import kagglehub
import shutil # Necesario para copiar el archivo descargado

# Importar configuraciones y constantes compartidas
try:
    # Si ejecutas como parte de un paquete (ej. python -m src.data_ingestion)
    from . import config
except ImportError:
    # Si ejecutas directamente el script (python src/data_ingestion.py)
    # y config.py está en el mismo directorio 'src' o en PYTHONPATH
    import config

# Configurar logging (una sola vez a través de la función en config)
config.setup_logging()


def ensure_raw_dataset_exists(
    dataset_slug: str,
    expected_filename_in_archive: str,
    kaggle_subfolder_in_archive: Optional[str],
    target_raw_path: str
) -> Optional[str]:
    """
    Asegura que el dataset crudo exista en target_raw_path.
    Si no existe, lo descarga de KaggleHub, lo busca (considerando subcarpetas)
    y lo copia a target_raw_path.

    Args:
        dataset_slug: Slug del dataset en Kaggle.
        expected_filename_in_archive: Nombre del archivo CSV esperado DENTRO del archivo/directorio de Kaggle.
        kaggle_subfolder_in_archive: Subcarpeta opcional dentro de la descarga de Kaggle donde podría estar el CSV.
        target_raw_path: Ruta final deseada para el archivo CSV crudo (ej. en data/raw/).

    Returns:
        Ruta al archivo crudo si tiene éxito, sino None.
    """
    # Crear el directorio data/raw si no existe
    os.makedirs(os.path.dirname(target_raw_path), exist_ok=True)

    if os.path.exists(target_raw_path):
        logging.info(f"Dataset crudo ya existe en: {target_raw_path}")
        return target_raw_path
    
    logging.info(f"Dataset crudo no encontrado en {target_raw_path}. Descargando de Kaggle: {dataset_slug}")
    try:
        # kagglehub.dataset_download() devuelve la ruta al directorio de la versión descargada
        download_version_dir = kagglehub.dataset_download(dataset_slug)
        logging.info(f"Dataset descargado por KaggleHub en (caché): {download_version_dir}")

        # Construir la ruta al archivo CSV dentro de la estructura descargada por KaggleHub
        source_csv_path_in_cache = download_version_dir
        if kaggle_subfolder_in_archive: # Si se espera que el CSV esté en una subcarpeta
            source_csv_path_in_cache = os.path.join(source_csv_path_in_cache, kaggle_subfolder_in_archive)
        source_csv_path_in_cache = os.path.join(source_csv_path_in_cache, expected_filename_in_archive)
        
        if os.path.exists(source_csv_path_in_cache):
            logging.info(f"Archivo CSV de Kaggle encontrado en la caché: {source_csv_path_in_cache}")
            # Copiar el archivo a nuestra carpeta data/raw/
            shutil.copy(source_csv_path_in_cache, target_raw_path)
            logging.info(f"Dataset copiado a la ubicación cruda local: {target_raw_path}")
            return target_raw_path
        else:
            logging.error(f"Archivo CSV esperado '{expected_filename_in_archive}' no encontrado en la ruta construida de la caché de Kaggle: {source_csv_path_in_cache}")
            # Listar contenido para ayudar a depurar
            try:
                logging.info(f"Contenido del directorio base de descarga de Kaggle '{download_version_dir}': {os.listdir(download_version_dir)}")
                if kaggle_subfolder_in_archive:
                    subfolder_path = os.path.join(download_version_dir, kaggle_subfolder_in_archive)
                    if os.path.isdir(subfolder_path):
                        logging.info(f"Contenido de la subcarpeta '{subfolder_path}': {os.listdir(subfolder_path)}")
            except Exception as list_e:
                logging.error(f"No se pudo listar el contenido para depuración: {list_e}")
            return None
            
    except Exception as e:
        logging.error(f"Error durante la descarga o copia del dataset '{dataset_slug}': {e}")
        logging.error("Asegúrate de tener 'kaggle.json' configurado o las variables de entorno KAGGLE_USERNAME/KAGGLE_KEY.")
        return None

def filter_and_split_dataset(
    raw_path: str, # Ruta al CSV en data/raw/
    discovery_path: str, # Ruta de salida para el split de descubrimiento en data/dividido/
    validation_path: str, # Ruta de salida para el split de validación en data/dividido/
    evaluation_path: str, # Ruta de salida para el split de evaluación en data/dividido/
    cols_to_load: Optional[List[str]] = None,
    col_inbound_name: str = config.COL_INBOUND,
    col_text_name: str = config.COL_TEXT
) -> bool:
    """
    Carga el dataset crudo, filtra tweets de clientes, realiza limpieza estructural básica del texto,
    divide el dataset y guarda los splits.
    """
    logging.info(f"Iniciando filtrado y división del dataset desde: {raw_path}")
    
    try:
        df_full: pd.DataFrame = pd.read_csv(raw_path, usecols=cols_to_load)
        logging.info(f"Dataset crudo cargado con {len(df_full)} filas y {len(df_full.columns)} columnas.")
        if cols_to_load:
            logging.info(f"Columnas cargadas: {df_full.columns.tolist()}")
    except FileNotFoundError:
        logging.error(f"El archivo CSV crudo {raw_path} no fue encontrado.")
        return False
    except ValueError as ve: 
        logging.error(f"Error al leer o parsear columnas específicas desde {raw_path}: {ve}")
        return False
    except Exception as e:
        logging.error(f"Ocurrió un error inesperado al cargar {raw_path}: {e}")
        return False

    if col_inbound_name not in df_full.columns:
        logging.error(f"La columna requerida para filtrar '{col_inbound_name}' no se encuentra en el dataset.")
        return False
        
    customer_tweets_df = df_full[df_full[col_inbound_name] == True].copy()
    # Opcional: Liberar memoria si df_full es muy grande y ya no se necesita.
    # del df_full
    # import gc; gc.collect()
    
    if customer_tweets_df.empty:
        logging.warning(f"No se encontraron tweets de clientes (columna '{col_inbound_name}' == True). No se generarán archivos de salida divididos.")
        return True # Se considera exitoso porque no hubo error, aunque no se generen datos.

    logging.info(f"Se encontraron {len(customer_tweets_df)} tweets de clientes.")

    if col_text_name not in customer_tweets_df.columns:
        logging.error(f"La columna de texto '{col_text_name}' no se encuentra en el dataframe de tweets de clientes.")
        return False

    # Limpieza estructural básica de la columna de texto
    customer_tweets_df[col_text_name] = customer_tweets_df[col_text_name].fillna('')
    customer_tweets_df[col_text_name] = customer_tweets_df[col_text_name].astype(str)
    customer_tweets_df = customer_tweets_df[customer_tweets_df[col_text_name].str.strip() != '']
    
    logging.info(f"Tweets de clientes después de limpiar NaNs/vacíos en '{col_text_name}': {len(customer_tweets_df)}.")

    if len(customer_tweets_df) < 3: # Necesario para al menos 1 muestra en train, val, test (o train, temp -> val, test)
        logging.warning("No hay suficientes tweets de clientes después de la limpieza (<3) para dividir en 3 conjuntos. No se generarán archivos de salida divididos.")
        return True

    logging.info(f"Dividiendo {len(customer_tweets_df)} tweets de clientes en conjuntos de Descubrimiento, Validación y Evaluación.")
    
    discovery_df, temp_df = train_test_split(
        customer_tweets_df,
        train_size=config.DISCOVERY_TRAIN_SIZE,
        random_state=config.RANDOM_STATE,
    )

    if len(temp_df) < 2: # Necesitamos al menos una muestra para validación y otra para evaluación desde temp_df
        logging.warning(f"No hay suficientes datos en temp_df ({len(temp_df)}) para dividir en Validación y Evaluación.")
        logging.warning(f"Se asignará todo temp_df ({len(temp_df)} filas) a Validación. El conjunto de Evaluación estará vacío.")
        validation_df = temp_df.copy() if not temp_df.empty else pd.DataFrame(columns=customer_tweets_df.columns)
        evaluation_df = pd.DataFrame(columns=customer_tweets_df.columns) 
    else:
        validation_df, evaluation_df = train_test_split(
            temp_df,
            train_size=config.VALIDATION_SIZE_REL_TO_REMAINING,
            random_state=config.RANDOM_STATE 
        )

    logging.info(f"Tamaño del conjunto de Descubrimiento (ingestado): {len(discovery_df)} filas ({len(discovery_df)/len(customer_tweets_df)*100 if len(customer_tweets_df)>0 else 0:.1f}%).")
    logging.info(f"Tamaño del conjunto de Validación (ingestado): {len(validation_df)} filas ({len(validation_df)/len(customer_tweets_df)*100 if len(customer_tweets_df)>0 else 0:.1f}%).")
    logging.info(f"Tamaño del conjunto de Evaluación (ingestado): {len(evaluation_df)} filas ({len(evaluation_df)/len(customer_tweets_df)*100 if len(customer_tweets_df)>0 else 0:.1f}%).")

    try:
        # Crear el directorio de salida para los splits (ej. data/dividido/)
        os.makedirs(os.path.dirname(discovery_path), exist_ok=True) # Asegura que el directorio exista

        discovery_df.to_csv(discovery_path, index=False, encoding='utf-8')
        logging.info(f"Conjunto de Descubrimiento (ingestado) guardado en: {discovery_path}")

        if not validation_df.empty:
            validation_df.to_csv(validation_path, index=False, encoding='utf-8')
            logging.info(f"Conjunto de Validación (ingestado) guardado en: {validation_path}")
        else:
            logging.info(f"Conjunto de Validación (ingestado) está vacío. No se guardará el archivo: {validation_path}")

        if not evaluation_df.empty:
            evaluation_df.to_csv(evaluation_path, index=False, encoding='utf-8')
            logging.info(f"Conjunto de Evaluación (ingestado) guardado en: {evaluation_path}")
        else:
            logging.info(f"Conjunto de Evaluación (ingestado) está vacío. No se guardará el archivo: {evaluation_path}")
            
        return True
    except IOError as ioe:
        logging.error(f"Error de E/S al guardar los archivos CSV divididos: {ioe}")
        return False
    except Exception as e:
        logging.error(f"Ocurrió un error inesperado al guardar los archivos CSV divididos: {e}")
        return False

def main_data_pipeline():
    """
    Orquesta la descarga del dataset, su copia a data/raw y su posterior filtrado y división.
    """
    logging.info("===== Iniciando Pipeline de Ingestión de Datos (Flujo con data/raw y data/dividido) =====")
    
    # Paso 1: Asegurar que el dataset crudo (twcs.csv) esté en config.RAW_TWCS_CSV_PATH (data/raw/twcs.csv)
    raw_dataset_path = ensure_raw_dataset_exists(
        config.KAGGLE_DATASET_SLUG,
        config.EXPECTED_CSV_FILENAME_IN_KAGGLE_ARCHIVE, # ej. "twcs.csv"
        config.KAGGLE_SUBFOLDER_IN_ARCHIVE, # ej. "twcs" o None
        config.RAW_TWCS_CSV_PATH # ej. "data/raw/twcs.csv"
    )

    if raw_dataset_path:
        # Paso 2: Filtrar el dataset crudo de data/raw/ y dividirlo en data/dividido/
        logging.info(f"Dataset crudo disponible en: {raw_dataset_path}. Procediendo a filtrar y dividir.")
        success = filter_and_split_dataset(
            raw_dataset_path, # Lee desde data/raw/twcs.csv
            config.INGESTED_DISCOVERY_SET_PATH, # Guarda en data/dividido/discovery_split_raw.csv
            config.INGESTED_VALIDATION_SET_PATH, # Guarda en data/dividido/validation_split_raw.csv
            config.INGESTED_EVALUATION_SET_PATH, # Guarda en data/dividido/evaluation_split_raw.csv
            cols_to_load=config.INITIAL_COLS_FROM_RAW,
            col_inbound_name=config.COL_INBOUND,
            col_text_name=config.COL_TEXT
        )
        if success:
            logging.info("Filtrado y división de datos completado con éxito (o no se generaron datos por falta de tweets de clientes).")
        else:
            logging.error("Pipeline de ingestión de datos encontró errores durante el filtrado/división.")
    else:
        logging.error("No se pudo asegurar la existencia del dataset crudo. El pipeline de ingestión no puede continuar.")

    logging.info("===== Pipeline de Ingestión de Datos Finalizado =====")

if __name__ == "__main__":
    # Validación simple de configuración antes de ejecutar el pipeline
    if not config.KAGGLE_DATASET_SLUG or not config.EXPECTED_CSV_FILENAME_IN_KAGGLE_ARCHIVE:
        logging.error("KAGGLE_DATASET_SLUG o EXPECTED_CSV_FILENAME_IN_KAGGLE_ARCHIVE no están definidos en config.py.")
    elif config.COL_INBOUND not in config.INITIAL_COLS_FROM_RAW or \
         config.COL_TEXT not in config.INITIAL_COLS_FROM_RAW:
        logging.error(
            f"Las columnas clave '{config.COL_INBOUND}' y/o '{config.COL_TEXT}' no están en la lista "
            f"INITIAL_COLS_FROM_RAW ({config.INITIAL_COLS_FROM_RAW}) en config.py. "
            "Asegúrate de que INITIAL_COLS_FROM_RAW contenga estas columnas y que los nombres de las constantes sean correctos."
        )
    else:
        main_data_pipeline()