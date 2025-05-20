# src/model_training.py

import pandas as pd
import numpy as np
import os
import logging
import joblib
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import classification_report, accuracy_score 
import argparse
from typing import Dict, Any, Tuple, Optional

try:
    from . import config
except ImportError:
    import config

config.setup_logging()

def load_final_data_for_model(
    dataset_type: str 
) -> Optional[Tuple[np.ndarray, pd.Series, pd.DataFrame]]:
    """
    Carga los embeddings reducidos y el archivo pseudo-etiquetado correspondiente
    para obtener las características (X), las etiquetas (y_text) y el DataFrame completo con contexto.
    """
    logging.info(f"Cargando datos finales para el conjunto: '{dataset_type}'")

    embeddings_path = config.EMBEDDINGS_REDUCED_OUTPUT_PATHS.get(dataset_type)
    ids_embeddings_path = config.EMBEDDINGS_IDS_OUTPUT_PATHS.get(dataset_type)
    
    labelled_data_path = None
    if dataset_type == "discovery":
        labelled_data_path = config.PSEUDO_LABELLED_DISCOVERY_PATH
    elif dataset_type == "validation":
        labelled_data_path = config.PSEUDO_LABELLED_VALIDATION_PATH
    elif dataset_type == "evaluation":
        labelled_data_path = config.PSEUDO_LABELLED_EVALUATION_PATH
    else:
        logging.error(f"Tipo de dataset '{dataset_type}' no reconocido para cargar datos de modelo.")
        return None, None, None

    if not all([embeddings_path, ids_embeddings_path, labelled_data_path]):
        logging.error(f"Una o más rutas de entrada para '{dataset_type}' no están definidas en config.py.")
        return None, None, None
    
    missing_files = []
    if not os.path.exists(embeddings_path): missing_files.append(f"Embeddings: '{embeddings_path}'")
    if not os.path.exists(ids_embeddings_path): missing_files.append(f"IDs Embeddings: '{ids_embeddings_path}'")
    if not os.path.exists(labelled_data_path): missing_files.append(f"Etiquetas: '{labelled_data_path}'")

    if missing_files:
        logging.error(f"Faltan archivos de entrada para el conjunto '{dataset_type}': {', '.join(missing_files)}. Asegúrate de haber ejecutado todos los scripts anteriores del pipeline.")
        return None, None, None

    try:
        embeddings = np.load(embeddings_path)
        df_ids_embeddings = pd.read_csv(ids_embeddings_path) 
        df_labelled_input = pd.read_csv(labelled_data_path) # Este es el archivo con las pseudo-etiquetas y, idealmente, contexto.    

        logging.info(f"Embeddings para '{dataset_type}' cargados. Forma: {embeddings.shape}")
        logging.info(f"IDs para embeddings de '{dataset_type}' cargados: {len(df_ids_embeddings)} IDs.")
        logging.info(f"Datos etiquetados para '{dataset_type}' cargados: {len(df_labelled_input)} filas.")

        if len(df_ids_embeddings) != embeddings.shape[0]:
            logging.error(f"Inconsistencia crítica: Longitud de IDs ({len(df_ids_embeddings)}) y embeddings ({embeddings.shape[0]}) no coincide para '{dataset_type}'.")
            return None, None, None
        
        # Crear un DataFrame temporal de embeddings con el índice original (COL_TWEET_ID)
        # Esto nos permite luego unir por COL_TWEET_ID y asegurar que los embeddings están alineados
        # con las filas de df_labelled_input.
        df_embeddings_with_id = pd.DataFrame(embeddings)
        df_embeddings_with_id[config.COL_TWEET_ID] = df_ids_embeddings[config.COL_TWEET_ID]
        df_embeddings_with_id = df_embeddings_with_id.set_index(config.COL_TWEET_ID)


        # Unir los datos etiquetados (que tienen COL_TWEET_ID y COL_CATEGORY, y otros como COL_TEXT)
        # con los IDs de los embeddings para filtrar y asegurar el orden.
        # 'inner' join para quedarnos solo con los IDs que tienen tanto embeddings como etiquetas/contexto.
        df_merged_final = pd.merge(
            df_ids_embeddings[[config.COL_TWEET_ID]], # Tomar solo los IDs de embeddings para el orden
            df_labelled_input, # Este DataFrame debe contener COL_TWEET_ID, COL_CATEGORY, y contexto como COL_TEXT
            on=config.COL_TWEET_ID,
            how='inner'
        )

        # Ahora, usar los IDs de df_merged_final para seleccionar los embeddings correctos y en el orden correcto
        final_ids_for_model = df_merged_final[config.COL_TWEET_ID]
        
        # Verificar si todos los final_ids_for_model están en el índice de df_embeddings_with_id
        missing_embedding_ids = final_ids_for_model[~final_ids_for_model.isin(df_embeddings_with_id.index)]
        if not missing_embedding_ids.empty:
            logging.error(f"Algunos IDs en datos etiquetados finales no tienen embeddings correspondientes para '{dataset_type}'. "
                          f"IDs faltantes (primeros 5): {missing_embedding_ids.tolist()[:5]}. "
                          "Esto no debería suceder si los archivos de IDs de embeddings y etiquetas son consistentes.")
            # Forzar consistencia eliminando los que no tienen embeddings
            df_merged_final = df_merged_final[df_merged_final[config.COL_TWEET_ID].isin(df_embeddings_with_id.index)]
            final_ids_for_model = df_merged_final[config.COL_TWEET_ID]

        embeddings_aligned = df_embeddings_with_id.loc[final_ids_for_model].values
        # df_merged_final ya está alineado con final_ids_for_model, y por ende con embeddings_aligned
        df_full_data_aligned = df_merged_final 
        labels_aligned_text = df_full_data_aligned[config.COL_CATEGORY]


        if embeddings_aligned.shape[0] != len(labels_aligned_text):
             logging.warning(f"Discrepancia de tamaño después del alineamiento final para '{dataset_type}'. "
                             f"Embeddings alineados: {embeddings_aligned.shape[0]}, "
                             f"Etiquetas alineadas: {len(labels_aligned_text)}. "
                             "Revisa la lógica de merge y filtrado.")
        
        if labels_aligned_text.isnull().any():
            nan_labels_count = labels_aligned_text.isnull().sum()
            logging.warning(f"Se encontraron {nan_labels_count} NaNs en las etiquetas para '{dataset_type}' después del merge. "
                            f"IDs con etiqueta NaN (primeros 5): {df_full_data_aligned[labels_aligned_text.isnull()][config.COL_TWEET_ID].tolist()[:5]}")
            not_nan_mask = labels_aligned_text.notna()
            embeddings_aligned = embeddings_aligned[not_nan_mask]
            labels_aligned_text = labels_aligned_text[not_nan_mask]
            df_full_data_aligned = df_full_data_aligned[not_nan_mask].reset_index(drop=True) # Resetear índice después de filtrar
            labels_aligned_text = labels_aligned_text.reset_index(drop=True) # Asegurar que y_data_text también tenga índice reseteado
            logging.info(f"Filas con etiquetas NaN eliminadas. Muestras restantes para '{dataset_type}': {len(labels_aligned_text)}")

        if len(labels_aligned_text) == 0:
            logging.error(f"No quedaron datos después de procesar etiquetas para '{dataset_type}'.")
            return None, None, None

        # Devolver y_data (etiquetas) como Serie de texto, y el DataFrame completo para contexto
        return embeddings_aligned, labels_aligned_text, df_full_data_aligned

    except Exception as e:
        logging.error(f"Error cargando features, etiquetas y contexto para '{dataset_type}': {e}", exc_info=True)
        return None, None, None


def train_classification_model(
    X_train: np.ndarray,
    y_train_text: pd.Series, 
    model_output_basepath: str
) -> Optional[Tuple[Any, LabelEncoder]]:
    """
    Entrena un modelo de Regresión Logística y guarda el modelo y el LabelEncoder.
    """
    logging.info(f"Iniciando entrenamiento del modelo de clasificación con {X_train.shape[0]} muestras de entrenamiento.")
    
    label_encoder = LabelEncoder()
    y_train_encoded = label_encoder.fit_transform(y_train_text) 
    num_classes = len(label_encoder.classes_)
    logging.info(f"Etiquetas codificadas. {num_classes} clases encontradas: {label_encoder.classes_.tolist()}")

    encoder_path = f"{model_output_basepath}_label_encoder.joblib"
    try:
        os.makedirs(os.path.dirname(encoder_path), exist_ok=True)
        joblib.dump(label_encoder, encoder_path)
        logging.info(f"LabelEncoder guardado en: {encoder_path}")
    except Exception as e:
        logging.error(f"Error guardando LabelEncoder: {e}")
        return None, None 

    model = LogisticRegression(
        C=config.LOGREG_C,
        random_state=config.RANDOM_STATE,
        solver='liblinear', 
        max_iter=1000,   
        class_weight=config.LOGREG_CLASS_WEIGHT 
    )
    
    try:
        logging.info("Ajustando el modelo de Regresión Logística...")
        model.fit(X_train, y_train_encoded)
        logging.info("Modelo de clasificación entrenado exitosamente.")
        
        model_path = f"{model_output_basepath}_model.joblib"
        joblib.dump(model, model_path)
        logging.info(f"Modelo de clasificación guardado en: {model_path}")
        return model, label_encoder
    except Exception as e:
        logging.error(f"Error durante el entrenamiento del modelo de clasificación: {e}", exc_info=True)
        return None, None


def evaluate_classification_model(
    model: Any,
    label_encoder: LabelEncoder,
    X_data: np.ndarray,
    y_data_text: pd.Series, 
    df_full_context_data: pd.DataFrame, 
    dataset_name: str
):
    """Evalúa el modelo en un conjunto de datos y muestra/guarda el reporte y análisis de errores."""
    if model is None or label_encoder is None or X_data is None or X_data.size == 0 or \
       y_data_text is None or y_data_text.empty or \
       df_full_context_data is None or df_full_context_data.empty:
        logging.warning(f"Datos insuficientes o vacíos para evaluar el modelo en el conjunto de {dataset_name}.")
        return

    logging.info(f"--- Evaluando Modelo en el Conjunto de {dataset_name} ---")
    
    # Re-verificar alineación y consistencia de los datos de entrada a esta función
    if len(X_data) != len(y_data_text) or len(X_data) != len(df_full_context_data):
        logging.error(f"Discrepancia crítica en la longitud de los datos para evaluación en '{dataset_name}'. "
                      f"X_data: {len(X_data)}, y_data_text: {len(y_data_text)}, df_full_context_data: {len(df_full_context_data)}. "
                      "Omitiendo evaluación para este conjunto. Revisa la lógica de carga y alineación de datos.")
        return

    try:
        y_data_encoded = label_encoder.transform(y_data_text) 
    except ValueError as e:
        logging.error(f"Error transformando etiquetas para el conjunto {dataset_name}: {e}")
        unique_labels = y_data_text.unique()
        logging.error(f"Clases en LabelEncoder: {label_encoder.classes_.tolist()}")
        logging.error(f"Etiquetas únicas en y_data_text ({len(unique_labels)}): {unique_labels.tolist()[:20]} (primeras 20 únicas)")
        
        # Encontrar etiquetas en y_data_text que no están en el LabelEncoder
        unknown_labels = set(unique_labels) - set(label_encoder.classes_)
        if unknown_labels:
            logging.error(f"Etiquetas desconocidas encontradas en '{dataset_name}' que no estaban en entrenamiento (primeras 5): {list(unknown_labels)[:5]}")
        
        logging.error("Asegúrate de que todas las etiquetas en este conjunto de datos ya existían en el conjunto de entrenamiento y no hay NaNs.")
        return

    try:
        predictions_encoded = model.predict(X_data)
        accuracy = accuracy_score(y_data_encoded, predictions_encoded)
        logging.info(f"Accuracy en {dataset_name}: {accuracy:.4f}")
        
        report_text = classification_report(
            y_data_encoded, 
            predictions_encoded, 
            target_names=label_encoder.classes_, 
            zero_division=0,
            output_dict=False 
        )
        logging.info(f"Reporte de Clasificación en {dataset_name}:\n{report_text}")
        
        report_filename_safe = dataset_name.lower().replace(' ', '_').replace('(', '').replace(')', '')
        report_output_path = os.path.join(config.MODELS_DIR, "classification", f"{report_filename_safe}_classification_report.txt") 
        os.makedirs(os.path.dirname(report_output_path), exist_ok=True)
        with open(report_output_path, "w", encoding="utf-8") as f:
            f.write(f"Reporte de Clasificación para: {dataset_name}\n")
            f.write(f"Accuracy: {accuracy:.4f}\n\n")
            f.write(report_text)
        logging.info(f"Reporte de clasificación para '{dataset_name}' guardado en: {report_output_path}")

        # --- Análisis de Errores ---
        # Asegurar que los índices de y_data_text y df_full_context_data sean consistentes para iloc
        # Si fueron reseteados en load_final_data_for_model, esto debería estar bien.
        y_data_text_for_errors = y_data_text.reset_index(drop=True)
        df_context_for_errors = df_full_context_data.reset_index(drop=True)

        misclassified_indices = np.where(y_data_encoded != predictions_encoded)[0]
        
        if len(misclassified_indices) > 0:
            logging.info(f"Recopilando {len(misclassified_indices)} errores de clasificación para el conjunto {dataset_name}...")
            predictions_text = label_encoder.inverse_transform(predictions_encoded)

            error_data = []
            for i in misclassified_indices:
                # Verificar que el índice i sea válido para todos los DataFrames/Series
                if i < len(y_data_text_for_errors) and i < len(predictions_text) and i < len(df_context_for_errors):
                    true_label_text = y_data_text_for_errors.iloc[i] 
                    predicted_label_text = predictions_text[i]
                    original_record = df_context_for_errors.iloc[i]

                    error_data.append({
                        'tweet_id': original_record.get(config.COL_TWEET_ID, 'N/A'),
                        'original_text': original_record.get(config.COL_TEXT, ''), 
                        'preprocessed_text': original_record.get(config.COL_PREPROCESSED_TEXT, ''), 
                        'true_category': true_label_text,
                        'predicted_category': predicted_label_text
                    })
                else:
                    logging.warning(f"Índice {i} fuera de rango durante la recopilación de errores. Omitiendo este error.")

            if error_data: # Solo guardar si se recopilaron datos de error
                df_errors = pd.DataFrame(error_data)
                errors_filename = f"{report_filename_safe}_classification_errors.csv"
                errors_output_path = os.path.join(config.MODELS_DIR, "classification", errors_filename) 
                df_errors.to_csv(errors_output_path, index=False, encoding='utf-8')
                logging.info(f"Análisis de errores para '{dataset_name}' guardado en: {errors_output_path}")
        else:
            logging.info(f"No se encontraron errores de clasificación en el conjunto {dataset_name}.")

    except Exception as e:
        logging.error(f"Error durante la evaluación del modelo en {dataset_name}: {e}", exc_info=True)


def main_model_training_pipeline(
    train_set_type: str = "discovery",
    val_set_type: Optional[str] = "validation",
    eval_set_type: Optional[str] = "evaluation"
):
    logging.info("===== Iniciando Pipeline de Entrenamiento de Modelo de Clasificación =====")

    # --- 1. Cargar Datos de Entrenamiento ---
    X_train, y_train_text, df_train_full_data = load_final_data_for_model(
        dataset_type=train_set_type
    )
    if X_train is None or y_train_text is None or df_train_full_data is None:
        logging.critical(f"No se pudieron cargar los datos de entrenamiento para '{train_set_type}'. Terminando.")
        return

    # --- 2. Entrenar Modelo ---
    # Usar _clustering_method_suffix de config si existe, sino un default
    clustering_suffix = getattr(config, '_clustering_method_suffix', 'kmeans_kUnknown')
    model_base_name = f"{train_set_type}_text_classifier_logreg_{clustering_suffix}"
    model_output_basepath = os.path.join(config.MODELS_DIR, "classification", model_base_name)
    
    trained_model, fitted_label_encoder = train_classification_model(X_train, y_train_text, model_output_basepath)
    if trained_model is None or fitted_label_encoder is None:
        logging.critical("Fallo en el entrenamiento del modelo. Terminando.")
        return

    # --- 3. Evaluar en Conjunto de Entrenamiento (Discovery) ---
    logging.info(f"--- Evaluando Modelo en el Conjunto de Entrenamiento ({train_set_type}) ---")
    evaluate_classification_model(
        model=trained_model, 
        label_encoder=fitted_label_encoder, 
        X_data=X_train, 
        y_data_text=y_train_text, 
        df_full_context_data=df_train_full_data,
        dataset_name=f"Entrenamiento ({train_set_type})"
    )

    # --- 4. Evaluar en Conjunto de Validación (si se proporciona) ---
    if val_set_type:
        X_val, y_val_text, df_val_full_data = load_final_data_for_model(dataset_type=val_set_type)
        if X_val is not None and y_val_text is not None and df_val_full_data is not None:
            evaluate_classification_model(
                model=trained_model, 
                label_encoder=fitted_label_encoder, 
                X_data=X_val, 
                y_data_text=y_val_text, 
                df_full_context_data=df_val_full_data,
                dataset_name=f"Validacion ({val_set_type})"
            )
        else:
            logging.warning(f"No se cargaron datos de validación para '{val_set_type}' o están vacíos. Evaluación de validación omitida.")
            
    # --- 5. Evaluar en Conjunto de Evaluación Final (si se proporciona) ---
    if eval_set_type:
        X_eval, y_eval_text, df_eval_full_data = load_final_data_for_model(dataset_type=eval_set_type)
        if X_eval is not None and y_eval_text is not None and df_eval_full_data is not None:
            evaluate_classification_model(
                model=trained_model, 
                label_encoder=fitted_label_encoder, 
                X_data=X_eval, 
                y_data_text=y_eval_text, 
                df_full_context_data=df_eval_full_data,
                dataset_name=f"Evaluacion Final ({eval_set_type})"
            )
        else:
            logging.warning(f"No se cargaron datos de evaluación para '{eval_set_type}' o están vacíos. Evaluación final omitida.")

    logging.info("===== Pipeline de Entrenamiento de Modelo de Clasificación Finalizado =====")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Entrena y evalúa un modelo de clasificación de texto.")
  
    args = parser.parse_args()

    logging.info(f"Usando Logistic Regression con C={config.LOGREG_C}, class_weight='{config.LOGREG_CLASS_WEIGHT}'")

    # Validaciones básicas de config
    required_paths = ['MODELS_DIR', 'PSEUDO_LABELLED_DISCOVERY_PATH', 'PSEUDO_LABELLED_VALIDATION_PATH', 'PSEUDO_LABELLED_EVALUATION_PATH']
    missing_configs = [path for path in required_paths if not hasattr(config, path) or getattr(config, path) is None]
    
    if missing_configs:
        logging.critical(f"Faltan las siguientes configuraciones de ruta en config.py: {', '.join(missing_configs)}.")
    else:
        main_model_training_pipeline()
        