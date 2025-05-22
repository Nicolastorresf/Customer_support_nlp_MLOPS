# src/preprocessing.py

import pandas as pd
import re
import string
from bs4 import BeautifulSoup
import nltk
from nltk.corpus import stopwords as nltk_stopwords
from nltk.corpus import wordnet
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
from nltk import pos_tag
from spellchecker import SpellChecker
from langdetect import detect, LangDetectException, DetectorFactory, detect_langs
import os
import emot
import logging
import argparse
from typing import List, Optional, Dict, Any

# Importar configuraciones y constantes compartidas
try:
    from . import config # Si ejecutas como parte de un paquete
except ImportError:
    import config # Si ejecutas directamente el script

# Configurar logging (una sola vez a través de la función en config)
config.setup_logging()

# --- Función para asegurar recursos NLTK ---
def ensure_nltk_resources():
    """
    Verifica y descarga los recursos de NLTK necesarios definidos en config.py.
    Debe ser llamada una vez al inicio del script.
    """
    logging.info("Verificando y descargando recursos NLTK necesarios...")
    if hasattr(config, 'NLTK_RESOURCES_NEEDED') and isinstance(config.NLTK_RESOURCES_NEEDED, dict):
        for resource_path_nltk, resource_name_nltk in config.NLTK_RESOURCES_NEEDED.items():
            try:
                nltk.data.find(resource_path_nltk)
                logging.info(f"Recurso NLTK '{resource_name_nltk}' ya se encuentra.")
            except LookupError:
                logging.info(f"Recurso NLTK '{resource_name_nltk}' no encontrado. Intentando descarga...")
                try:
                    nltk.download(resource_name_nltk, quiet=False)
                    logging.info(f"Recurso NLTK '{resource_name_nltk}' descargado exitosamente.")
                except Exception as e_download:
                    logging.error(f"Fallo al descargar el recurso NLTK '{resource_name_nltk}': {e_download}")
            except Exception as e_find:
                logging.error(f"Error inesperado verificando el recurso NLTK '{resource_name_nltk}': {e_find}")
    else:
        logging.warning("config.NLTK_RESOURCES_NEEDED no está definido en config.py o no es un diccionario. "
                        "No se descargarán recursos NLTK automáticamente. Esto podría causar errores más adelante.")
    logging.info("Verificación de recursos NLTK completada.")

# --- Llamar a la función de descarga de NLTK inmediatamente ---
ensure_nltk_resources()

# --- Configuración Global y Semillas ---
DetectorFactory.seed = 0
SCRIPT_START_TIME = pd.Timestamp.now(tz='America/Bogota')
logging.info(f"Script de Preprocesamiento Multilingüe Iniciado a las {SCRIPT_START_TIME} (Control por config.py)")

# --- Carga de Recursos Lingüísticos Externos ---
try:
    from chat_words import ALL_CHAT_WORDS_MAP
    logging.info(f"Mapas de chat words cargados para {len(ALL_CHAT_WORDS_MAP)} idiomas desde '{config.CHAT_WORDS_FILE}'.")
except ImportError as e:
    logging.warning(f"ADVERTENCIA: No se pudo cargar ALL_CHAT_WORDS_MAP desde '{config.CHAT_WORDS_FILE}': {e}. Usando valores por defecto.")
    ALL_CHAT_WORDS_MAP = {'en': {"lol": "laughing out loud"}, 'es': {"pq": "porque"}}

for lang_code_default in config.LANG_MAP.keys():
    if lang_code_default not in ALL_CHAT_WORDS_MAP:
        ALL_CHAT_WORDS_MAP[lang_code_default] = {}

try:
    from all_emoticons_expanded import ALL_EMOTICONS_EXPANDED
    logging.info(f"Mapas de emoticonos cargados para {len(ALL_EMOTICONS_EXPANDED)} idiomas desde '{config.EMOTICONS_FILE}'.")
except ImportError as e:
    logging.warning(f"ADVERTENCIA: No se pudo cargar ALL_EMOTICONS_EXPANDED desde '{config.EMOTICONS_FILE}': {e}. Usando valores por defecto.")
    ALL_EMOTICONS_EXPANDED = {'en': {':)': 'happy face emoji'}, 'es': {':)': 'cara feliz emoji'}}

for lang_code_default in config.LANG_MAP.keys():
    if lang_code_default not in ALL_EMOTICONS_EXPANDED:
        ALL_EMOTICONS_EXPANDED[lang_code_default] = {}

# --- Inicialización de Herramientas y Recursos Pre-cargados ---
emot_analyzer = emot.emot()
lemmatizer = WordNetLemmatizer()
SPELL_CHECKERS_CACHE: Dict[str, Optional[SpellChecker]] = {}

ALL_STOPWORDS_SETS: Dict[str, set] = {}
for lang_code, details in config.LANG_MAP.items():
    nltk_stop_name = details.get('nltk_stopwords')
    if nltk_stop_name:
        try:
            ALL_STOPWORDS_SETS[lang_code] = set(nltk_stopwords.words(nltk_stop_name))
            logging.info(f"Stopwords de NLTK para '{lang_code}' ('{nltk_stop_name}') cargadas exitosamente.")
        except LookupError:
            logging.error(f"Stopwords de NLTK para '{lang_code}' ('{nltk_stop_name}') NO SE ENCONTRARON dentro del corpus 'stopwords' de NLTK. "
                          f"Verifica que '{nltk_stop_name}' sea un idioma con stopwords disponibles en NLTK. Se usará un conjunto vacío.")
            ALL_STOPWORDS_SETS[lang_code] = set()
        except Exception as e_load:
            logging.error(f"Error inesperado al cargar stopwords para '{lang_code}' ('{nltk_stop_name}'): {e_load}. Se usará un conjunto vacío.")
            ALL_STOPWORDS_SETS[lang_code] = set()
    else:
        ALL_STOPWORDS_SETS[lang_code] = set()
logging.info(f"Stopwords (o intento) cargadas para {len(ALL_STOPWORDS_SETS)} idiomas.")

# --- Definiciones de Funciones de Limpieza ---
def remove_html_content(text_with_html: str) -> str:
    if isinstance(text_with_html, str) and '<' in text_with_html and '>' in text_with_html:
        try:
            return BeautifulSoup(text_with_html, "lxml").text
        except Exception as e:
            logging.warning(f"BeautifulSoup falló al procesar un texto que parecía HTML: {e}. Texto: '{text_with_html[:100]}...'")
            return text_with_html
    return text_with_html

def remove_urls_from_text(text_with_urls: str) -> str:
    return re.sub(r'https?://\S+|www\.\S+', ' ', text_with_urls)

def very_basic_cleaning(text_to_clean: str) -> str:
    if not isinstance(text_to_clean, str):
        logging.warning(f"very_basic_cleaning recibió un tipo no string: {type(text_to_clean)}. Devolviendo string vacío.")
        return ""
    
    processed_text = text_to_clean

    if config.PREPROCESSING_DO_LOWERCASE:
        processed_text = processed_text.lower()
    if config.PREPROCESSING_REMOVE_HTML:
        processed_text = remove_html_content(processed_text)
    if config.PREPROCESSING_REMOVE_URLS:
        processed_text = remove_urls_from_text(processed_text)
    if config.PREPROCESSING_REMOVE_MENTIONS_HASHTAGS:
        processed_text = re.sub(r'@\w+', ' ', processed_text) 
        processed_text = re.sub(r'#\w+', ' ', processed_text) 
    
    processed_text = re.sub(r'\s+', ' ', processed_text).strip()
    return processed_text

def detect_language_robust(text_cleaned: str) -> str:
    if not text_cleaned or text_cleaned.isspace(): return "unknown"
    try:
        detections = detect_langs(text_cleaned)
        if detections: return detections[0].lang
    except LangDetectException:
        return "unknown"
    except Exception as e:
        logging.warning(f"Error inesperado en detect_language_robust: {e}. Texto: '{text_cleaned[:50]}...'. Devolviendo 'unknown'.")
        return "unknown"
    return "unknown"

def format_neutral_emoticon_tag(name_from_emot: str, prefix: str) -> str:
    cleaned_name = re.sub(r'[^a-zA-Z0-9_]', '', name_from_emot.strip(":").replace(" ", "_").replace("-", "_").lower())
    return f"{prefix}_{cleaned_name}"

def convert_emojis_emoticons_adaptive(text_to_convert: str, lang_code_to_use: str, emot_analyzer_instance) -> str:
    processed_text = str(text_to_convert)
    lang_emoticon_map = ALL_EMOTICONS_EXPANDED.get(lang_code_to_use, ALL_EMOTICONS_EXPANDED.get(config.DEFAULT_FALLBACK_LANG, {}))
    try: 
        em_info = emot_analyzer_instance.emoji(processed_text)
        if em_info.get('flag'):
            for i in range(len(em_info['value']) - 1, -1, -1):
                emoji_val, emoji_mean_en = em_info['value'][i], em_info['mean'][i]
                start, end = em_info['location'][i]
                replacement = lang_emoticon_map.get(emoji_val) or \
                              lang_emoticon_map.get(emoji_mean_en) or \
                              format_neutral_emoticon_tag(emoji_mean_en, "emoji")
                processed_text = processed_text[:start] + f" {replacement} " + processed_text[end:]
    except Exception: pass 
    try: 
        emc_info = emot_analyzer_instance.emoticons(processed_text)
        if emc_info.get('flag'):
            for i in range(len(emc_info['value']) - 1, -1, -1):
                emoticon_val, emoticon_mean_en = emc_info['value'][i], emc_info['mean'][i]
                start, end = emc_info['location'][i]
                replacement = lang_emoticon_map.get(emoticon_val) or \
                              lang_emoticon_map.get(emoticon_mean_en) or \
                              format_neutral_emoticon_tag(emoticon_mean_en, "emoticon")
                processed_text = processed_text[:start] + f" {replacement} " + processed_text[end:]
    except Exception: pass
    return processed_text.strip()

def chat_words_conversion_adaptive(text_to_convert: str, lang_code_to_use: str) -> str:
    chat_map_to_use = ALL_CHAT_WORDS_MAP.get(lang_code_to_use, ALL_CHAT_WORDS_MAP.get(config.DEFAULT_FALLBACK_LANG, {}))
    if not chat_map_to_use: return text_to_convert
    words = text_to_convert.split()
    new_words = [chat_map_to_use.get(word.lower(), word) for word in words]
    return " ".join(new_words)

def remove_punctuation_custom(text_to_clean: str) -> str:
    return text_to_clean.translate(str.maketrans('', '', config.PUNCT_TO_REMOVE))

def remove_stopwords_adaptive(text_to_process: str, lang_code_to_use: str) -> str:
    stop_words = ALL_STOPWORDS_SETS.get(lang_code_to_use)
    if not stop_words: return text_to_process
    try:
        words = word_tokenize(text_to_process)
        return " ".join([w for w in words if w.lower() not in stop_words])
    except LookupError:
        logging.warning("NLTK: El tokenizador 'punkt' podría faltar. Devolviendo texto original para remove_stopwords.")
        return text_to_process
    except Exception as e:
        logging.error(f"Error en remove_stopwords_adaptive: {e}. Texto: '{text_to_process[:50]}...'. Devolviendo texto original.")
        return text_to_process

def get_spell_checker_instance(lang_code: str, spell_checkers_cache_dict: Dict[str, Optional[SpellChecker]]) -> Optional[SpellChecker]:
    if lang_code not in spell_checkers_cache_dict:
        lang_details = config.LANG_MAP.get(lang_code)
        sc_lang_name = lang_details.get('spellchecker') if lang_details else None
        if sc_lang_name:
            try:
                spell_checkers_cache_dict[lang_code] = SpellChecker(language=sc_lang_name)
            except Exception as e:
                logging.warning(f"No se pudo inicializar SpellChecker para '{lang_code}' (mapa: '{sc_lang_name}'): {e}.")
                spell_checkers_cache_dict[lang_code] = None
        else:
            spell_checkers_cache_dict[lang_code] = None
    return spell_checkers_cache_dict.get(lang_code)

def correct_spellings_adaptive(text_to_correct: str, lang_code_to_use: str, spell_checkers_cache_dict: Dict[str, Optional[SpellChecker]]) -> str:
    spell_checker = get_spell_checker_instance(lang_code_to_use, spell_checkers_cache_dict)
    if not spell_checker: return text_to_correct
    try:
        words = word_tokenize(text_to_correct)
        corrected_words = []
        for w in words:
            if len(w) > 2 and w.isalpha() and not w.isupper() and not any(c.isdigit() for c in w):
                original_case_word = w
                word_lower = w.lower()
                if word_lower not in spell_checker:
                    correction = spell_checker.correction(word_lower)
                    if correction and correction != word_lower:
                        if original_case_word[0].isupper() and len(original_case_word) > 1:
                            corrected_words.append(correction[0].upper() + correction[1:])
                        elif original_case_word[0].isupper():
                            corrected_words.append(correction.upper())
                        else:
                            corrected_words.append(correction)
                    else:
                        corrected_words.append(original_case_word)
                else:
                    corrected_words.append(original_case_word)
            else:
                corrected_words.append(w)
        return " ".join(corrected_words)
    except LookupError:
        logging.warning("NLTK: El tokenizador 'punkt' podría faltar. Devolviendo texto original para correct_spellings.")
        return text_to_correct
    except Exception as e:
        logging.error(f"Error en correct_spellings_adaptive: {e}. Texto: '{text_to_correct[:50]}...'. Devolviendo texto original.")
        return text_to_correct

def get_wordnet_pos_adaptive(word: str, lang_code_to_use: str) -> str:
    if lang_code_to_use == 'en': # WordNet POS tagging es más robusto para inglés en NLTK
        try:
            tag = pos_tag([word])[0][1][0].upper()
            tag_dict = {"J": wordnet.ADJ, "N": wordnet.NOUN, "V": wordnet.VERB, "R": wordnet.ADV}
            return tag_dict.get(tag, wordnet.NOUN) # Default a NOUN si no se reconoce
        except LookupError: # En caso de que falte 'averaged_perceptron_tagger'
             return wordnet.NOUN
    return wordnet.NOUN # Para otros idiomas, o si falla el tagger, default a NOUN

def lemmatize_text_adaptive(text_to_lemmatize: str, lang_code_to_use: str, lemmatizer_instance) -> str:
    lang_details = config.LANG_MAP.get(lang_code_to_use)
    omw_lang = lang_details.get('omw') if lang_details else None # 'omw' es el código de idioma para Open Multilingual Wordnet
    
    if not omw_lang: # Si el idioma no tiene un mapeo para OMW, no se puede lematizar con WordNetLemmatizer
        return text_to_lemmatize
    
    try:
        words = word_tokenize(text_to_lemmatize)
        lemmatized_words = []
        for w in words:
            pos = get_wordnet_pos_adaptive(w, lang_code_to_use)
            try:
                # Para idiomas diferentes al inglés, se debe especificar el 'lang' si el lemmatizer lo soporta y está configurado con OMW
                if lang_code_to_use != 'en':
                    lemma = lemmatizer_instance.lemmatize(w, pos=pos, lang=omw_lang)
                else:
                    lemma = lemmatizer_instance.lemmatize(w, pos=pos) # Para inglés, lang no es necesario explícitamente
                lemmatized_words.append(lemma)
            except Exception: # Captura errores específicos de lematización por palabra
                lemmatized_words.append(w) # Si falla, mantener la palabra original
        return " ".join(lemmatized_words)
    except LookupError: # Si falta 'punkt' para word_tokenize o recursos de WordNet/OMW
        logging.warning("NLTK: Recursos para lematización (punkt, wordnet, omw-1.4) podrían faltar. Devolviendo texto original.")
        return text_to_lemmatize
    except Exception as e:
        logging.error(f"Error en lemmatize_text_adaptive: {e}. Texto: '{text_to_lemmatize[:50]}...'. Devolviendo texto original.")
        return text_to_lemmatize

def remove_extra_spaces(text_to_clean: str) -> str:
    return re.sub(r'\s+', ' ', text_to_clean).strip()

# --- Función Auxiliar para Procesar Fila (usada con df.apply) ---
def _process_text_row(row: pd.Series,
                      emot_analyzer_instance, 
                      lemmatizer_instance, 
                      spell_checkers_cache_dict: Dict[str, Optional[SpellChecker]],
                      enable_spell_correction_flag: bool, 
                      enable_lemmatization_flag: bool) -> str:
    
    current_text_for_nlp = row['text_cleaned_initial'] 
    detected_lang_code = row['detected_language_initial']

    lang_to_use_for_processing = config.DEFAULT_FALLBACK_LANG
    if detected_lang_code in config.SUPPORTED_LANGS_FOR_SPECIFIC_PROCESSING:
        lang_to_use_for_processing = detected_lang_code
    
    if config.PREPROCESSING_CONVERT_EMOJIS_EMOTICONS:
        current_text_for_nlp = convert_emojis_emoticons_adaptive(current_text_for_nlp, lang_to_use_for_processing, emot_analyzer_instance)
    
    if config.PREPROCESSING_EXPAND_CHAT_WORDS:
        current_text_for_nlp = chat_words_conversion_adaptive(current_text_for_nlp, lang_to_use_for_processing)
    
    if config.PREPROCESSING_REMOVE_PUNCTUATION:
        current_text_for_nlp = remove_punctuation_custom(current_text_for_nlp)
    
    # Pasos específicos del idioma solo si el idioma es soportado y detectado
    if lang_to_use_for_processing in config.SUPPORTED_LANGS_FOR_SPECIFIC_PROCESSING:
        if config.PREPROCESSING_REMOVE_STOPWORDS:
            current_text_for_nlp = remove_stopwords_adaptive(current_text_for_nlp, lang_to_use_for_processing)
        
        if enable_spell_correction_flag:
            current_text_for_nlp = correct_spellings_adaptive(current_text_for_nlp, lang_to_use_for_processing, spell_checkers_cache_dict)
        
        if enable_lemmatization_flag and config.LANG_MAP.get(lang_to_use_for_processing, {}).get('omw'):
            current_text_for_nlp = lemmatize_text_adaptive(current_text_for_nlp, lang_to_use_for_processing, lemmatizer_instance)
            
    current_text_for_nlp = remove_extra_spaces(current_text_for_nlp)
    return current_text_for_nlp

# --- Pipeline Principal de Preprocesamiento ---
def preprocess_pipeline(df_input: pd.DataFrame, 
                        text_column_name: str = config.COL_TEXT,
                        enable_spell_correction_param: bool = False, 
                        enable_lemmatization_param: bool = True) -> pd.DataFrame:
    
    if not isinstance(df_input, pd.DataFrame):
        logging.error("La entrada para preprocess_pipeline debe ser un DataFrame de Pandas.")
        raise ValueError("La entrada debe ser un DataFrame de Pandas.")
    if text_column_name not in df_input.columns:
        logging.error(f"La columna '{text_column_name}' no se encuentra en el DataFrame de entrada.")
        raise ValueError(f"La columna '{text_column_name}' no se encuentra en el DataFrame.")

    logging.info(f"Pipeline de preprocesamiento iniciado para {len(df_input)} textos...")
    df = df_input.copy()

    logging.info("Realizando limpieza básica inicial (controlada por config) y detección de idioma...")
    df['text_cleaned_initial'] = df[text_column_name].astype(str).apply(very_basic_cleaning)
    df['detected_language_initial'] = df['text_cleaned_initial'].apply(detect_language_robust)
    
    current_spell_checkers_cache: Dict[str, Optional[SpellChecker]] = {} 

    logging.info("Aplicando procesamiento adaptativo avanzado (controlado por config)...")
    
    try:
        from tqdm.auto import tqdm 
        tqdm.pandas(desc="Preprocesando textos") 
        apply_method = df.progress_apply 
    except ImportError:
        logging.info("tqdm no instalado, no se mostrarán barras de progreso para df.apply.")
        apply_method = df.apply

    df[config.COL_PREPROCESSED_TEXT] = apply_method(
        _process_text_row,
        axis=1,
        emot_analyzer_instance=emot_analyzer,
        lemmatizer_instance=lemmatizer,
        spell_checkers_cache_dict=current_spell_checkers_cache,
        enable_spell_correction_flag=enable_spell_correction_param,
        enable_lemmatization_flag=enable_lemmatization_param
    )
    
    df[config.COL_DETECTED_LANGUAGE] = df['detected_language_initial']
    logging.info("Pipeline de preprocesamiento completado.")
    return df[[config.COL_PREPROCESSED_TEXT, config.COL_DETECTED_LANGUAGE]]


def main_text_preprocessing_pipeline(dataset_type_to_process: str):
    """
    Orquesta el preprocesamiento de texto para el conjunto de datos especificado.
    """
    logging.info(f"===== Iniciando Pipeline de Preprocesamiento de Texto para el conjunto: '{dataset_type_to_process}' =====")

    # La descarga de NLTK ya se hizo al inicio con ensure_nltk_resources()

    input_path = config.PREPROCESSING_INPUT_PATHS.get(dataset_type_to_process)
    output_path = config.PREPROCESSED_OUTPUT_PATHS.get(dataset_type_to_process)

    if not input_path or not output_path:
        logging.critical(f"Rutas de entrada/salida para preprocesamiento del conjunto '{dataset_type_to_process}' no definidas en config.py. Saliendo.")
        return

    os.makedirs(config.PREPROCESSED_TEXT_DIR, exist_ok=True)

    try:
        df_main = pd.read_csv(input_path)
        logging.info(f"Dataset '{dataset_type_to_process}' cargado para preprocesamiento: {input_path}, Filas: {len(df_main)}")
    except FileNotFoundError:
        logging.error(f"Archivo de entrada no encontrado: {input_path}. Ejecuta los scripts anteriores (data_ingestion, data_preparation).")
        logging.info("Creando DataFrame de ejemplo para demostración del preprocesamiento.")
        data_ejemplo = {
            'text_id': [f"id_{i+1}" for i in range(2)],
            config.COL_TEXT: [
                "This is a #GREAT example with a URL http://example.com and @mention, LOL! :)",
                "Este es otro ejemplo en ESPAÑOL, ¿por qué? xq sí... TQM ;-P <3"
            ]
        }
        df_main = pd.DataFrame(data_ejemplo)
        if config.COL_TWEET_ID not in df_main.columns and 'text_id' in df_main.columns:
             df_main.rename(columns={'text_id': config.COL_TWEET_ID}, inplace=True)

    except Exception as e:
        logging.critical(f"Error crítico cargando CSV para preprocesamiento ({input_path}): {e}. Saliendo.")
        exit(1)

    if config.COL_TEXT not in df_main.columns:
        logging.critical(f"Columna '{config.COL_TEXT}' no encontrada en el DataFrame de entrada '{input_path}'. Saliendo.")
        exit(1)
    
    if df_main.empty:
        logging.warning(f"DataFrame para el conjunto '{dataset_type_to_process}' está vacío. No se realizará preprocesamiento.")
        empty_df_cols = list(df_main.columns) + [config.COL_PREPROCESSED_TEXT, config.COL_DETECTED_LANGUAGE]
        pd.DataFrame(columns=empty_df_cols).to_csv(output_path, index=False, encoding='utf-8')
        logging.info(f"Archivo preprocesado vacío guardado para '{dataset_type_to_process}' en: {output_path}")
    else:
        df_processed_output_cols = preprocess_pipeline(
            df_main,
            text_column_name=config.COL_TEXT,
            enable_spell_correction_param=config.PREPROCESSING_ENABLE_SPELL_CORRECTION,
            enable_lemmatization_param=config.PREPROCESSING_ENABLE_LEMMATIZATION
        )

        df_main[config.COL_PREPROCESSED_TEXT] = df_processed_output_cols[config.COL_PREPROCESSED_TEXT]
        df_main[config.COL_DETECTED_LANGUAGE] = df_processed_output_cols[config.COL_DETECTED_LANGUAGE]
        
        logging.info(f"\n--- Ejemplo de Resultados del Preprocesamiento para '{dataset_type_to_process}' (primeras filas) ---")
        num_examples_to_show = min(len(df_main), 5)
        for i in range(num_examples_to_show):
            id_col_name = config.COL_TWEET_ID if config.COL_TWEET_ID in df_main.columns else 'text_id'
            text_id_val = df_main.get(id_col_name, pd.Series(df_main.index, name='text_id_fallback'))[i]
            original_text_val = df_main[config.COL_TEXT].iloc[i]
            detected_lang_val = df_main[config.COL_DETECTED_LANGUAGE].iloc[i]
            preprocessed_text_val = df_main[config.COL_PREPROCESSED_TEXT].iloc[i]

            logging.info(f"\n--- Fila {df_main.index[i]} (ID: {text_id_val}) ---")
            logging.info(f"Original:     '{str(original_text_val)[:100].replace(chr(10), ' ').strip()}'")
            logging.info(f"Detectado:    {detected_lang_val}")
            logging.info(f"Preprocesado: '{str(preprocessed_text_val)[:100].replace(chr(10), ' ').strip()}'")

        try:
            df_main.to_csv(output_path, index=False, encoding='utf-8')
            logging.info(f"\nDataset '{dataset_type_to_process}' preprocesado guardado en: {output_path}")
        except Exception as e:
            logging.error(f"Error guardando CSV preprocesado para '{dataset_type_to_process}' en {output_path}: {e}")

    logging.info(f"===== Pipeline de Preprocesamiento de Texto para '{dataset_type_to_process}' Finalizado =====")


# --- Bloque Principal de Ejecución ---
if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Ejecuta el pipeline de preprocesamiento de texto NLP para un tipo de conjunto específico o todos."
    )
    parser.add_argument(
        "--dataset_type",
        type=str,
        required=False,
        choices=['discovery', 'validation', 'evaluation', 'all'],
        default='all', 
        help="Tipo de dataset a preprocesar. Opciones: 'discovery', 'validation', 'evaluation', 'all' (para procesar todos los definidos en config)."
    )
    args = parser.parse_args()

    if not hasattr(config, 'PREPROCESSING_INPUT_PATHS') or \
       not hasattr(config, 'PREPROCESSED_OUTPUT_PATHS'):
        logging.critical("Constantes de ruta PREPROCESSING_INPUT_PATHS o PREPROCESSED_OUTPUT_PATHS no encontradas en config.py.")
    else:
        if args.dataset_type == 'all':
            logging.info("Procesando todos los tipos de dataset definidos en config.PREPROCESSING_INPUT_PATHS...")
            for set_type_key in config.PREPROCESSING_INPUT_PATHS.keys():
                main_text_preprocessing_pipeline(dataset_type_to_process=set_type_key)
        else:
            main_text_preprocessing_pipeline(dataset_type_to_process=args.dataset_type)

    script_end_time = pd.Timestamp.now(tz='America/Bogota')
    logging.info(f"Ejecución completa de src/preprocessing.py Finalizada a las {script_end_time}")
    logging.info(f"Tiempo total de ejecución del script: {script_end_time - SCRIPT_START_TIME}")