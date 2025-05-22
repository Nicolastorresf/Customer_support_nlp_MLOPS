# src/data_preparation.py

import pandas as pd
import os
import logging
from typing import List, Optional
import argparse # Para argumentos de línea de comandos

# Importar configuraciones y constantes compartidas
try:
    # Si ejecutas como parte de un paquete (ej. python -m src.data_preparation)
    from . import config
except ImportError:
    # Si ejecutas directamente el script (python src/data_preparation.py)
    # y config.py está en el mismo directorio 'src' o en PYTHONPATH
    import config

# Configurar logging (una sola vez a través de la función en config)
config.setup_logging()


def prepare_dataset(
    input_csv_path: str,
    output_csv_path: str,
    columns_to_keep_after_preparation: List[str]
) -> bool:
    """
    Carga un conjunto de datos, realiza preparaciones adicionales definidas y lo guarda.
    Preparaciones:
    1. Convierte la columna COL_CREATED_AT a formato datetime.
    2. Elimina duplicados basados en COL_TWEET_ID.
    3. Crea la columna COL_TWEET_LENGTH a partir de COL_TEXT.
    4. Selecciona y reordena las columnas finales según columns_to_keep_after_preparation.

    Args:
        input_csv_path: Ruta al archivo CSV de entrada (ej. un archivo _ingested.csv).
        output_csv_path: Ruta para guardar el archivo CSV preparado.
        columns_to_keep_after_preparation: Lista de nombres de columnas que deben
                                           estar en el archivo de salida final.

    Returns:
        True si la preparación fue exitosa y se guardó el archivo, False en caso contrario.
    """
    logging.info(f"Iniciando la preparación del dataset desde: {input_csv_path}")

    try:
        df = pd.read_csv(input_csv_path)
        logging.info(f"Dataset '{os.path.basename(input_csv_path)}' cargado con {len(df)} filas y {len(df.columns)} columnas.")
    except FileNotFoundError:
        logging.error(f"Archivo de entrada no encontrado: {input_csv_path}")
        return False
    except Exception as e:
        logging.error(f"Error cargando el dataset desde {input_csv_path}: {e}")
        return False

    if df.empty:
        logging.warning(f"El dataset de entrada {input_csv_path} está vacío. No se realizará la preparación.")
        return True

    # --- INICIO DE TRANSFORMACIONES DE PREPARACIÓN ---

    # 1. Convertir COL_CREATED_AT a datetime
    if config.COL_CREATED_AT in df.columns:
        logging.info(f"Paso 1: Convirtiendo la columna '{config.COL_CREATED_AT}' a datetime.")
        df[config.COL_CREATED_AT] = pd.to_datetime(df[config.COL_CREATED_AT], errors='coerce')
        logging.info(f"Columna '{config.COL_CREATED_AT}' convertida a datetime.")
    elif config.COL_CREATED_AT in columns_to_keep_after_preparation:
        logging.warning(f"La columna '{config.COL_CREATED_AT}' se esperaba pero no se encontró en el dataset de entrada '{input_csv_path}'.")

    # 2. Eliminar duplicados basados en COL_TWEET_ID
    if config.COL_TWEET_ID in df.columns:
        initial_rows = len(df)
        logging.info(f"Paso 2: Eliminando duplicados basados en '{config.COL_TWEET_ID}'. Filas iniciales: {initial_rows}.")
        df = df.drop_duplicates(subset=[config.COL_TWEET_ID], keep='first')
        rows_dropped = initial_rows - len(df)
        logging.info(f"Duplicados por '{config.COL_TWEET_ID}' eliminados. Filas restantes: {len(df)} (se eliminaron {rows_dropped} filas).")
    elif config.COL_TWEET_ID in columns_to_keep_after_preparation:
        logging.warning(f"La columna '{config.COL_TWEET_ID}' para eliminación de duplicados no se encontró en '{input_csv_path}'.")

    # 3. Crear la columna COL_TWEET_LENGTH
    if config.COL_TEXT in df.columns:
        logging.info(f"Paso 3: Creando la columna '{config.COL_TWEET_LENGTH}'.")
        df[config.COL_TEXT] = df[config.COL_TEXT].astype(str)
        df[config.COL_TWEET_LENGTH] = df[config.COL_TEXT].str.len()
        logging.info(f"Columna '{config.COL_TWEET_LENGTH}' creada.")
    elif config.COL_TWEET_LENGTH in columns_to_keep_after_preparation:
        logging.warning(f"La columna '{config.COL_TEXT}' no se encontró. No se pudo crear '{config.COL_TWEET_LENGTH}'.")

    # --- FIN DE TRANSFORMACIONES DE PREPARACIÓN ---

    # 4. Seleccionar y reordenar las columnas finales
    logging.info(f"Paso 4: Seleccionando y reordenando columnas finales según la definición en 'config.PREPARED_DATA_COLUMNS'.")
    
    final_columns_present_in_df = [col for col in columns_to_keep_after_preparation if col in df.columns]
    
    missing_cols_for_output = [col for col in columns_to_keep_after_preparation if col not in df.columns]
    if missing_cols_for_output:
        logging.warning(f"Las siguientes columnas definidas en 'config.PREPARED_DATA_COLUMNS' no se encontraron "
                        f"o no se pudieron crear en el DataFrame actual y serán omitidas de la salida: {missing_cols_for_output}")

    if not final_columns_present_in_df:
        logging.error("Ninguna de las columnas especificadas en 'columns_to_keep_after_preparation' existe en el DataFrame. No se puede guardar un archivo sin esquema.")
        return False
        
    df_prepared = df[final_columns_present_in_df].copy()
    logging.info(f"Dataset preparado con {len(df_prepared)} filas y columnas: {df_prepared.columns.tolist()}.")

    try:
        os.makedirs(os.path.dirname(output_csv_path), exist_ok=True)
        df_prepared.to_csv(output_csv_path, index=False, encoding='utf-8')
        logging.info(f"Dataset preparado guardado exitosamente en: {output_csv_path}")
        return True
    except IOError as ioe:
        logging.error(f"Error de E/S al guardar el archivo CSV preparado en {output_csv_path}: {ioe}")
        return False
    except Exception as e:
        logging.error(f"Ocurrió un error inesperado al guardar el CSV preparado en {output_csv_path}: {e}")
        return False

def main_data_preparation_pipeline(dataset_type_to_process: str):
    """
    Orquesta la preparación del conjunto de datos especificado.
    """
    logging.info(f"===== Iniciando Pipeline de Preparación de Datos para el conjunto: '{dataset_type_to_process}' =====")
    
    input_path = config.PREPARATION_INPUT_PATHS.get(dataset_type_to_process)
    output_path = config.PREPARED_OUTPUT_PATHS.get(dataset_type_to_process)
    columns_to_keep = config.PREPARED_DATA_COLUMNS # Usar la definición de config.py

    if not input_path or not output_path:
        logging.error(f"Rutas de entrada/salida para el conjunto '{dataset_type_to_process}' no definidas correctamente en config.py. Omitiendo este conjunto.")
        return # O podrías lanzar una excepción si es un error crítico

    logging.info(f"--- Preparando conjunto de '{dataset_type_to_process}' ---")
    if os.path.exists(input_path):
        success = prepare_dataset(
            input_path,
            output_path,
            columns_to_keep # Pasa la lista de columnas definidas en config
        )
        if not success:
            logging.error(f"Falló la preparación para el conjunto de '{dataset_type_to_process}' (entrada: {input_path}).")
        else:
            logging.info(f"Preparación del conjunto de '{dataset_type_to_process}' completada y guardada en {output_path}.")
    else:
        logging.warning(f"Archivo de entrada para el conjunto de '{dataset_type_to_process}' no encontrado: {input_path}. Omitiendo preparación.")
            
    logging.info(f"===== Pipeline de Preparación de Datos para '{dataset_type_to_process}' Finalizado =====")

if __name__ == "__main__":
    # Configurar el parser de argumentos
    parser = argparse.ArgumentParser(
        description="Ejecuta el pipeline de preparación de datos para un tipo de conjunto específico (discovery, validation, evaluation) o todos."
    )
    parser.add_argument(
        "--dataset_type",
        type=str,
        required=False, # Lo hacemos opcional, si no se da, procesa todos
        choices=['discovery', 'validation', 'evaluation', 'all'],
        default='all', # Valor por defecto si no se especifica
        help="Tipo de dataset a procesar. Opciones: 'discovery', 'validation', 'evaluation', 'all' (para procesar todos los definidos en config)."
    )
    
    args = parser.parse_args()
    
    # Validaciones de configuración esenciales
    if not hasattr(config, 'PREPARATION_INPUT_PATHS') or \
       not hasattr(config, 'PREPARED_OUTPUT_PATHS') or \
       not hasattr(config, 'PREPARED_DATA_COLUMNS'):
        logging.critical("Constantes de ruta o columnas para data_preparation no encontradas en config.py. Revisa tu archivo de configuración.")
    else:
        if args.dataset_type == 'all':
            logging.info("Procesando todos los tipos de dataset definidos en config.PREPARATION_INPUT_PATHS...")
            for set_type_key in config.PREPARATION_INPUT_PATHS.keys():
                main_data_preparation_pipeline(dataset_type_to_process=set_type_key)
        else:
            main_data_preparation_pipeline(dataset_type_to_process=args.dataset_type)