import streamlit as st
import pandas as pd
import numpy as np
import json
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, Normalizer, label_binarize, OneHotEncoder, LabelEncoder
from sklearn.metrics import roc_auc_score, roc_curve, confusion_matrix, classification_report, accuracy_score, precision_score, recall_score, f1_score
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, AdaBoostClassifier
from xgboost import XGBClassifier
from sklearn.impute import SimpleImputer
import lightgbm as lgb
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
import os
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline

# Función para identificar tipos de columnas
def identify_column_types(dataframe):
    """
    Identifica automáticamente qué columnas son numéricas y cuáles son categóricas.
    Devuelve un diccionario con las columnas clasificadas por tipo.
    """
    column_types = {
        'numeric': [],
        'categorical': [],
        'binary': [],
        'date': [],
        'text': []
    }
    
    # Lista de posibles nombres de columnas categóricas binarias comunes
    possible_binary_columns = ['sex', 'gender', 'is_', 'has_', 'smoking', 'deceased', 'dead', 'alive']
    
    for column in dataframe.columns:
        # Contar valores nulos
        null_count = dataframe[column].isnull().sum()
        # Obtener valores no nulos para análisis
        non_null_values = dataframe[column].dropna()
        
        if len(non_null_values) == 0:
            # Si la columna está completamente vacía, la marcamos como categórica por defecto
            column_types['categorical'].append(column)
            continue
        
        # Detectar fechas
        if pd.api.types.is_datetime64_any_dtype(dataframe[column]):
            column_types['date'].append(column)
            continue
        
        # Para columnas con tipo object (texto)
        if dataframe[column].dtype == object:
            # Detectar columnas de sexo o binarias comunes ('F'/'M', 'YES'/'NO', etc.)
            col_lower = column.lower()
            
            # Verificar si el nombre de la columna sugiere una variable binaria
            is_likely_binary = any(binary_name in col_lower for binary_name in possible_binary_columns)
            
            # Verificar si los valores son típicos de variables binarias (F/M, Y/N, YES/NO, etc.)
            unique_values = dataframe[column].unique()
            unique_values_lower = [str(x).lower() if x is not None else '' for x in unique_values]
            
            binary_value_pairs = [
                {'f', 'm'}, {'female', 'male'}, 
                {'y', 'n'}, {'yes', 'no'}, {'true', 'false'},
                {'t', 'f'}, {'0', '1'}, {'positive', 'negative'}
            ]
            
            is_binary_values = any(
                set(unique_values_lower).issubset(pair) for pair in binary_value_pairs
            )
            
            # Si parece binaria por nombre o valores
            if (is_likely_binary or is_binary_values) and dataframe[column].nunique() <= 2:
                column_types['binary'].append(column)
            # Si tiene pocos valores únicos, probablemente sea categórica
            elif dataframe[column].nunique() < 20:
                column_types['categorical'].append(column)
            else:
                # Si tiene muchos valores únicos, podría ser texto
                column_types['text'].append(column)
            
            continue
            
        # Verificar si es numérica
        if pd.api.types.is_numeric_dtype(dataframe[column]):
            # Verificar si es binaria (solo tiene 2 valores únicos)
            if dataframe[column].nunique() <= 2:
                column_types['binary'].append(column)
            else:
                column_types['numeric'].append(column)
        else:
            # Si no encaja en las categorías anteriores, la marcamos como categórica
            column_types['categorical'].append(column)
                
    return column_types

# Preprocesador que maneja columnas categóricas y numéricas automáticamente
def create_preprocessor(X, categorical_cols, numeric_cols, binary_cols=None):
    """
    Crea un preprocesador que maneja automáticamente columnas categóricas y numéricas.
    
    Args:
        X: DataFrame con las características
        categorical_cols: Lista de columnas categóricas
        numeric_cols: Lista de columnas numéricas
        binary_cols: Lista de columnas binarias (opcional)
    """
    # Verificar que haya columnas para procesar
    if not categorical_cols and not numeric_cols and not binary_cols:
        return None
    
    transformers = []
    
    # Procesar columnas categóricas
    if categorical_cols:
        categorical_transformer = Pipeline(steps=[
            ('onehot', OneHotEncoder(handle_unknown='ignore', sparse_output=False))
        ])
        transformers.append(('cat', categorical_transformer, categorical_cols))
    
    # Procesar columnas binarias (usar LabelEncoder para binarias)
    if binary_cols:
        binary_transformer = Pipeline(steps=[
            ('label', LabelEncoder())
        ])
        # Nota: Como ColumnTransformer no permite LabelEncoder directamente,
        # tendremos que procesar estas columnas por separado
    
    # Procesar columnas numéricas
    if numeric_cols:
        numeric_transformer = Pipeline(steps=[
            ('scaler', StandardScaler())
        ])
        transformers.append(('num', numeric_transformer, numeric_cols))
    
    # Crear el preprocesador
    preprocessor = ColumnTransformer(
        transformers=transformers,
        remainder='passthrough'  # Pasar otras columnas sin cambios
    )
    
    return preprocessor

# Configuración global
PRODUCTION_MODE = False  # Cambiar a False para modo desarrollo con warnings

# Suprimir todos los warnings de forma completa desde el inicio
os.environ['PYTHONWARNINGS'] = 'ignore'
warnings.filterwarnings('ignore')

# Establecer la configuración de la página antes de cualquier otro código de Streamlit
st.set_page_config(
    page_title="OMINIS ROC Lab", 
    layout="wide", 
    initial_sidebar_state="expanded",
    page_icon="https://ominis.org/favicon.ico" 
)

# Personalizar tema para modo producción (antes de cualquier renderizado)
primary_color = "#4682B4"  # Azul OMINIS
background_color = "#F0F2F6"
secondary_background_color = "#FFFFFF"
text_color = "#262730"
font = "sans-serif"

# Aplicar modo de producción: ocultar elementos y personalizar CSS
if PRODUCTION_MODE:
    # CSS para ocultar elementos y personalizar la apariencia
    st.markdown("""
    <style>
    /* Ocultar elementos de depuración y desarrollo */
    #MainMenu {visibility: hidden !important;}
    footer {visibility: hidden !important;}
    .stDeployButton {display: none !important;}
    .viewerBadge_container__1QSob {display: none !important;}
    div[data-testid="stToolbar"] {visibility: hidden !important;}
    div[data-testid="stDecoration"] {visibility: hidden !important;}
    div[data-testid="stStatusWidget"] {visibility: hidden !important;}
    
    /* Warning messages */
    div.stWarning {display: none !important;}
    
    /* Ocultar específicamente el mensaje sobre language_selector */
    div[data-testid="stAppMessageContainer"] {display: none !important;}
    .element-container div[data-testid="stNotification"] {display: none !important;}
    div.stException {display: none !important;}
    
    /* Ocultar mensaje de 'Calling st.rerun() within a callback' */
    .stAlert {display: none !important;}
    div[data-baseweb="notification"] {display: none !important;}
    div[data-testid="callbackError"] {display: none !important;}
    div.stWarningMsg {display: none !important;}
    
    /* Mejorar los estilos de los widgets */
    div.stButton > button {
        background-color: #4682B4;
        color: white;
        border-radius: 4px;
        border: none;
        padding: 8px 16px;
        font-weight: bold;
    }
    div.stButton > button:hover {
        background-color: #3A6EA5;
    }
    
    /* Mejorar apariencia de los selectores */
    div.stSelectbox > div > div {
        background-color: white;
        border-radius: 4px;
    }
    
    /* Mejorar apariencia de los sliders */
    div.stSlider > div {
        padding-top: 1rem;
        padding-bottom: 1rem;
    }
    
    /* Mejorar aspecto general */
    .reportview-container .main .block-container {
        padding-top: 2rem;
        padding-bottom: 2rem;
    }
    </style>
    """, unsafe_allow_html=True)

def load_data(file):
    """
    Carga un archivo de datos en formato CSV o Excel.
    Detecta automáticamente los tipos de columnas y muestra información relevante.
    """
    # Detectar el tipo de archivo basado en la extensión
    file_extension = file.name.split('.')[-1].lower()
    
    if file_extension in ['csv', 'txt']:
        # Intentar diferentes separadores y encodings para CSV
        try:
            # Primero intentar con coma
            df = pd.read_csv(file, sep=',')
        except:
            try:
                # Luego intentar con punto y coma (común en países con coma decimal)
                df = pd.read_csv(file, sep=';')
            except:
                try:
                    # Intentar con tabulador
                    df = pd.read_csv(file, sep='\t')
                except:
                    # Último intento con auto-detección
                    df = pd.read_csv(file, sep=None, engine='python')
    elif file_extension in ['xlsx', 'xls']:
        # Cargar archivo Excel
        df = pd.read_excel(file)
    else:
        raise ValueError(f"Formato de archivo no soportado: {file_extension}. Use CSV o Excel.")
    
    return df

# Load language files based on user selection
def load_lang(lang_code):
    with open(f'lang_{lang_code}.json', 'r') as lang_file:
        return json.load(lang_file)

# Obtener parámetros de URL
# Uso de la API actualizada de query_params

# Establecer el código de idioma desde parámetros de URL o utilizar el valor predeterminado
try:
    # Obtener el parámetro de idioma de la URL
    if 'lang' in st.query_params and st.query_params['lang'] in ['es', 'en']:
        lang_code = st.query_params['lang']
    else:
        lang_code = 'es'  # Valor predeterminado
except Exception as e:
    # En caso de error con query_params, usar el valor predeterminado
    if not PRODUCTION_MODE:
        st.warning(f"Error al obtener parámetros de URL: {e}")
    lang_code = 'es'

# Función para cambiar el idioma - evitamos usar st.rerun() directamente
def change_language():
    try:
        selected_language = st.session_state.language_selector
        language_options = {"Español": "es", "English": "en"}
        new_lang = language_options[selected_language]
        # Solo actualizamos el parámetro de URL sin llamar a rerun()
        st.query_params['lang'] = new_lang
    except Exception as e:
        if not PRODUCTION_MODE:
            st.warning(f"Error al cambiar el idioma: {e}")

# Guardar el código de idioma en session_state para que el selector muestre el valor correcto
language_options_reverse = {"es": "Español", "en": "English"}
if 'language_selector' not in st.session_state:
    st.session_state.language_selector = language_options_reverse[lang_code]

# Cargar el archivo de idioma
lang = load_lang(lang_code)

# Define available models
models = {
    'Logistic Regression': LogisticRegression,
    'KNN': KNeighborsClassifier,
    'Naive Bayes': GaussianNB,
    'SVM': SVC,
    'Decision Tree': DecisionTreeClassifier,
    'Random Forest': RandomForestClassifier,
    'XGBoost': XGBClassifier,
    'Gradient Boosting': GradientBoostingClassifier,
    'AdaBoost': AdaBoostClassifier,
    'LightGBM': lgb.LGBMClassifier
}

# Load data
@st.cache_data
def load_data(file):
    """
    Carga un archivo de datos en formato CSV o Excel.
    Detecta automáticamente los tipos de columnas y muestra información relevante.
    """
    # Detectar el tipo de archivo basado en la extensión
    file_extension = file.name.split('.')[-1].lower()
    
    if file_extension in ['csv', 'txt']:
        # Intentar diferentes separadores y encodings para CSV
        try:
            # Primero intentar con coma
            df = pd.read_csv(file, sep=',')
        except:
            try:
                # Luego intentar con punto y coma (común en países con coma decimal)
                df = pd.read_csv(file, sep=';')
            except:
                try:
                    # Intentar con tabulador
                    df = pd.read_csv(file, sep='\t')
                except:
                    # Último intento con auto-detección
                    df = pd.read_csv(file, sep=None, engine='python')
    elif file_extension in ['xlsx', 'xls']:
        # Cargar archivo Excel
        df = pd.read_excel(file)
    else:
        raise ValueError(f"Formato de archivo no soportado: {file_extension}. Use CSV o Excel.")
    
    return df


header1, header2, header3 = st.columns([2, 3, 2])
with header1:
    st.image('https://funsalud.org.mx/wp-content/uploads/2022/09/logotipo.png', width=200)
    st.markdown("<div style='text-align: left;'><a href='https://www.ominis.org' target='_blank' style='color: #4682B4; text-decoration: none; margin-left: 30px;'>www.ominis.org</a></div>", unsafe_allow_html=True)
with header2:
    st.title(lang['title'])
    st.write(lang['subtitle'])
with header3:
    # Selector de idioma con enfoque sin usar callbacks que llamen a rerun()
    language_options = {"Español": "es", "English": "en"}
    current_language = language_options_reverse.get(lang_code, "Español")
    
    # Si el idioma cambia, actualizamos la URL directamente con un enlace
    col1, col2 = st.columns([1, 1])
    
    with col1:
        if current_language != "Español":
            st.markdown(f'<a href="?lang=es" target="_self" style="text-decoration: none; color: #4682B4;">🇪🇸 Español</a>', unsafe_allow_html=True)
        else:
            st.markdown(f'<span style="font-weight: bold;">🇪🇸 Español</span>', unsafe_allow_html=True)
    
    with col2:
        if current_language != "English":
            st.markdown(f'<a href="?lang=en" target="_self" style="text-decoration: none; color: #4682B4;">🇬🇧 English</a>', unsafe_allow_html=True)
        else:
            st.markdown(f'<span style="font-weight: bold;">🇬🇧 English</span>', unsafe_allow_html=True)
    
    # Instructions in an expander at the right column's end
    with st.expander(lang['instructions_title']):
        for instruction in lang['instructions']:
            st.write(instruction)
    with st.expander(lang['about_title']):
        st.write(lang['about_text'])
        st.write(lang['author'])

st.divider()

# Divide the app into two columns
col_data, col_modelo, col_score = st.columns([3, 3, 2])

# In the left column (col_data) we place the logo, file upload, and selectors
with col_data:

    try:

        # Subir el archivo CSV
        st.subheader(lang['upload_title'], help=lang['upload_help'])
        uploaded_file = st.file_uploader(lang['upload_prompt'], type=['csv', 'xlsx', 'xls', 'txt'])

        # Reset session state si un nuevo archivo es cargado
        if uploaded_file:
            
            if 'df' not in st.session_state or st.session_state['last_uploaded_file'] != uploaded_file:
                st.session_state.df = load_data(uploaded_file)
                st.session_state.last_uploaded_file = uploaded_file

            # Asignar el dataframe desde la session state
            df = st.session_state.df

            # Mostrar una vista previa de los datos
            with st.expander(f"{lang['file_contents']} {len(df)} {lang['records_loaded']} {lang['and']} {len(df.columns)} {lang['columns_loaded']}"):
                st.write(df.head())
                
            # Preselect the target column if 'target', 'y', or a binary column exists
            possible_target_columns = ['target', 'y', 'Outcome', 'objective', 'class', 'diagnosis', 'result']
            preselected_target = None
            
            for col in df.columns:
                if col.lower() in [target.lower() for target in possible_target_columns]:
                    preselected_target = col
                    break
                elif df[col].nunique() == 2:  # Check if the column is binary
                    preselected_target = col
                    break

            # Show the target column selector, preselecting it if available
            target_column = st.selectbox(lang['select_target_variable'], df.columns, index=df.columns.get_loc(preselected_target) if preselected_target else 0, help=lang['select_target_help']) if preselected_target else st.selectbox(lang['select_target_column'], df.columns, help=lang['select_target_help'])

            # Column selection (except for the target variable) inside an expander
            with st.expander(lang['select_prediction_vars']):
                selected_columns = st.multiselect(lang['select_vars_prompt'], [col for col in df.columns if col != target_column], default=[col for col in df.columns if col != target_column], help=lang['select_vars_help'])

            if 'df' not in st.session_state:
                st.session_state.df = load_data(uploaded_file)

            df = st.session_state.df  # Use data stored in the session

            st.subheader(lang['exploratory_analysis_title'], help=lang['exploratory_help'])
            tab1, tab2, tab3 = st.tabs([lang['distributions_tab'], lang['correlations_tab'], lang['observations_tab']])
            with tab1:
                st.write(lang['stat_distributions'])
                st.write(df[selected_columns].describe())
            with tab2:
                # Show correlations between variables
                st.write(lang['correlation_title'])
                corr = df[selected_columns].corr()
                fig, ax = plt.subplots()
                sns.heatmap(corr, annot=True, cmap='coolwarm', ax=ax, annot_kws={"size": 6})
                ax.tick_params(axis='both', which='major', labelsize=6)
                st.pyplot(fig)
            with tab3:
                # Button to generate observations
                if st.button(lang['generate_observations_button'], key='centered_button'):
                    with st.spinner(lang['generating_observations']):
                        # Pairplot with target variable as hue
                        fig = sns.pairplot(df[selected_columns + [target_column]], hue=target_column)
                        fig.add_legend()  # Add legend for hue

                        # Add a single legend to the whole figure
                        fig._legend.set_bbox_to_anchor((0.5, 1.05))
                        fig._legend.set_loc('upper center')
                        fig._legend.set_title(target_column)
                        fig._legend.set_frame_on(True)

                        st.pyplot(fig)
                
    except Exception as e:
        st.error(f"{lang['error']} {e}")
    
# In the right column (col_modelo) show the model selector, results, ROC curve, and metrics table
with col_modelo:
    
    try:
        if uploaded_file:
        # if 'df' in locals():
            
            st.subheader(lang['data_preparation_title'], help=lang['data_preparation_help'])
            # Prepare the data
            features = df[selected_columns]
            target = df[target_column]
            
            # Identificar automáticamente los tipos de columnas
            column_types = identify_column_types(df)
            numeric_cols = [col for col in selected_columns if col in column_types['numeric']]
            binary_cols = [col for col in selected_columns if col in column_types['binary']]
            categorical_cols = [col for col in selected_columns if col in column_types['categorical']]
            date_cols = [col for col in selected_columns if col in column_types['date']]
            text_cols = [col for col in selected_columns if col in column_types['text']]
            
            # Mostrar información sobre los tipos de columnas detectados
            st.write(f"🔢 {lang.get('numeric_cols', 'Numeric columns')}: {len(numeric_cols)}")
            st.write(f"⚖️ {lang.get('binary_cols', 'Binary columns')}: {len(binary_cols)}")
            st.write(f"🔤 {lang.get('categorical_cols', 'Categorical columns')}: {len(categorical_cols)}")
            
            if date_cols:
                st.write(f"📅 {lang.get('date_cols', 'Date columns')}: {len(date_cols)}")
            if text_cols:
                st.write(f"📝 {lang.get('text_cols', 'Text columns')}: {len(text_cols)}")

            # Procesar variables binarias categóricas (con textos como YES/NO o F/M)
            processed_columns = []
            for col in binary_cols + categorical_cols:
                if df[col].dtype == object:  # Si es un tipo texto
                    try:
                        # Verificar si contiene valores como 'YES'/'NO' o 'F'/'M'
                        values = df[col].astype(str).str.lower().unique()
                        yes_no_values = {'yes', 'no', 'y', 'n', 'true', 'false', 't', 'f'}
                        male_female_values = {'f', 'm', 'female', 'male', 'mujer', 'hombre', 'h', 'woman', 'man'}
                        
                        contains_yes_no = any(val in yes_no_values for val in values)
                        contains_gender = any(val in male_female_values for val in values)
                        
                        if contains_yes_no:
                            st.info(f"🔄 {lang.get('yes_no_conversion', 'Convirtiendo valores SÍ/NO')}: {col}")
                        elif contains_gender:
                            st.info(f"🔄 {lang.get('gender_conversion', 'Convirtiendo valores de género')}: {col}")
                        
                        # Convertir a string primero para asegurar que LabelEncoder funcione correctamente
                        df[col] = df[col].astype(str)
                        
                        # Codificar variables binarias y categóricas de texto
                        label_encoder = LabelEncoder()
                        df[col] = label_encoder.fit_transform(df[col])
                        processed_columns.append(col)
                        
                        # Mostrar mapeo de valores para referencia del usuario
                        mapping = dict(zip(label_encoder.classes_, label_encoder.transform(label_encoder.classes_)))
                        st.write(f"✅ {lang.get('encoded_column', 'Columna codificada')}: {col} → {mapping}")
                        
                    except Exception as e:
                        st.warning(f"⚠️ {lang.get('conversion_error', 'Error al convertir')}: {col} - {str(e)}")
                        
                        # Plan alternativo: mapeo manual
                        try:
                            # Para sexo/género
                            if col.lower() in ['sex', 'gender', 'sexo', 'genero', 'género']:
                                # Crear un diccionario de mapeo
                                gender_map = {
                                    'f': 0, 'female': 0, 'mujer': 0, 'm': 1, 'male': 1, 'hombre': 1,
                                    'woman': 0, 'man': 1, 'F': 0, 'M': 1, 'FEMALE': 0, 'MALE': 1
                                }
                                df[col] = df[col].map(gender_map)
                                if df[col].isnull().sum() > 0:
                                    # Si hay valores no mapeados, rellenar con la moda
                                    df[col] = df[col].fillna(df[col].mode()[0])
                                st.success(f"✅ {lang.get('manual_gender_conversion', 'Valores de género convertidos manualmente')}: {col}")
                            
                            # Para yes/no
                            elif any(val.lower() in yes_no_values for val in df[col].dropna().astype(str).unique()):
                                yes_no_map = {
                                    'yes': 1, 'y': 1, 'true': 1, 't': 1, '1': 1, 'si': 1, 'sí': 1,
                                    'no': 0, 'n': 0, 'false': 0, 'f': 0, '0': 0,
                                    'YES': 1, 'Y': 1, 'TRUE': 1, 'T': 1, 'SI': 1, 'SÍ': 1,
                                    'NO': 0, 'N': 0, 'FALSE': 0, 'F': 0
                                }
                                df[col] = df[col].map(yes_no_map)
                                if df[col].isnull().sum() > 0:
                                    df[col] = df[col].fillna(df[col].mode()[0])
                                st.success(f"✅ {lang.get('manual_yes_no_conversion', 'Valores SÍ/NO convertidos manualmente')}: {col}")
                            
                            # Último intento: convertir a números si es posible
                            else:
                                df[col] = pd.to_numeric(df[col], errors='coerce')
                                if df[col].isnull().sum() > 0:
                                    df[col] = df[col].fillna(df[col].mode()[0])
                                st.info(f"ℹ️ {lang.get('numeric_conversion', 'Columna convertida a numérica')}: {col}")
                            
                            processed_columns.append(col)
                            
                        except Exception as e2:
                            # Si todo falla, eliminar la columna del análisis
                            if col in selected_columns:
                                selected_columns.remove(col)
                                st.error(f"❌ {lang.get('column_removed', 'Columna eliminada')}: {col} - {str(e2)}")

            if processed_columns:
                st.success(f"✅ {len(processed_columns)} {lang.get('columns_processed', 'columnas procesadas')} correctamente.")

            # Actualizar los datos seleccionados después del preprocesamiento
            features = df[selected_columns]
            target = df[target_column]
            
            if df.isnull().sum().sum() > 0:                
                # Handle missing values
                handle_missing = st.selectbox(lang['handle_missing'], ["None", "Delete", "Impute (Mean/Mode)"])
                if handle_missing == "Delete":
                    # Eliminar filas con valores nulos
                    df = df.dropna()
                    features = df[selected_columns]
                    target = df[target_column]
                elif handle_missing == "Impute (Mean/Mode)":
                    # Imputar valores numéricos con la media y categóricos con la moda
                    for col in numeric_cols:
                        if df[col].isnull().sum() > 0:
                            df[col].fillna(df[col].mean(), inplace=True)
                    
                    for col in categorical_cols + binary_cols:
                        if df[col].isnull().sum() > 0:
                            df[col].fillna(df[col].mode()[0], inplace=True)
                    
                    # Actualizar features y target después de la imputación
                    features = df[selected_columns]
                    target = df[target_column]
            else:
                st.text(lang['no_missing_values'], help=lang['no_missing_help'])
            
            normalize = st.toggle(lang['normalize_values'], help=lang['normalize_help'])
            lasso = st.toggle(lang['lasso_regularization'], help=lang['lasso_help'])
            outlier_detection = st.toggle(lang['remove_outliers'], help=lang['remove_outliers_help'])

            # Set up configurations on the left side
            st.text(lang['sample_splitting'], help=lang['sample_splitting_help'])
                        
            train_size = st.slider(lang['training_sample_size'], 0, 100, 80, format="%d%%", help=lang['training_sample_help']) / 100.0
            test_size = 1 - train_size

            # Assuming 'df' is your dataset and 'test_size' is already defined
            sample_counts = {
                lang['training']: int(len(df) * (1 - test_size)),
                lang['testing']: int(len(df) * test_size)
            }

            # Calculate total samples
            total_samples = sum(sample_counts.values())

            fig, ax = plt.subplots(figsize=(6, 1))  # Adjust the figure size as needed

            # Definimos colores OMINIS 
            ominis_blue = "#4682B4"  # Azul principal de OMINIS
            ominis_lightblue = "#6699CC"  # Azul claro
            
            # Horizontal stacked bar con los colores de OMINIS
            ax.barh('Samples', sample_counts[lang['training']], color=ominis_blue, label=f"{lang['training']}: {sample_counts[lang['training']]}")
            ax.barh('Samples', sample_counts[lang['testing']], left=sample_counts[lang['training']], color=ominis_lightblue, label=f"{lang['testing']}: {sample_counts[lang['testing']]}")

            # Adding labels to each section
            ax.text(sample_counts[lang['training']] / 2, 0, f"{lang['training']}: {sample_counts[lang['training']]}", va='center', ha='center', color='white')
            ax.text(sample_counts[lang['training']] + sample_counts[lang['testing']] / 2, 0, f"{lang['testing']}: {sample_counts[lang['testing']]}", va='center', ha='center', color='white')

            # Remove axes
            ax.axis('off')

            # Show plot in Streamlit
            st.pyplot(fig)

            n_splits = st.slider(lang['num_partitions'], 2, 10, 5, help=lang['num_partitions_help'])

            # Crear preprocesador para manejar columnas categóricas y numéricas
            preprocessor = create_preprocessor(features, 
                                          categorical_cols=categorical_cols,
                                          numeric_cols=numeric_cols,
                                          binary_cols=binary_cols)
            
            # Aplicar preprocesamiento si hay un preprocesador
            if preprocessor:
                try:
                    # Hacer una copia del DataFrame antes de transformarlo
                    features_copy = features.copy()
                    
                    # Aplicar el preprocesador
                    features_processed = preprocessor.fit_transform(features_copy)
                    
                    # Convertir a DataFrame si no lo es ya
                    if not isinstance(features_processed, pd.DataFrame):
                        # Generar nombres de columnas para las variables transformadas
                        processed_cols = []
                        
                        # Para columnas numéricas, mantener el nombre original
                        for col in numeric_cols:
                            processed_cols.append(col)
                        
                        # Para columnas categóricas, generar nombres para las variables dummy
                        for col in categorical_cols:
                            unique_values = features[col].unique()
                            for val in unique_values:
                                processed_cols.append(f"{col}_{val}")
                        
                        # Crear DataFrame con los nombres de columnas generados
                        # Si hay más columnas transformadas que nombres generados, usar nombres genéricos
                        if features_processed.shape[1] > len(processed_cols):
                            # Añadir nombres genéricos para las columnas adicionales
                            for i in range(len(processed_cols), features_processed.shape[1]):
                                processed_cols.append(f"feature_{i}")
                        
                        # Limitar a las columnas reales generadas
                        processed_cols = processed_cols[:features_processed.shape[1]]
                        
                        # Crear el DataFrame final
                        features_processed = pd.DataFrame(features_processed, 
                                                     columns=processed_cols,
                                                     index=features.index)
                except Exception as e:
                    st.error(f"Error al procesar características: {e}")
                    # En caso de error, usar las características originales
                    features_processed = features
            else:
                features_processed = features
            
            # Split the data into training and test sets
            X_train, X_test, y_train, y_test = train_test_split(features_processed, target, test_size=test_size, random_state=42)
            
            # Verificar si la variable objetivo es categórica
            is_classification = False
            target_classes = None
            
            if target_column in column_types['categorical'] or target_column in column_types['binary']:
                # Codificar la variable objetivo si es categórica
                is_classification = True
                
                # Si la variable objetivo es categórica de texto (no numérica)
                if df[target_column].dtype == object:
                    label_encoder = LabelEncoder()
                    y_train_encoded = label_encoder.fit_transform(y_train)
                    y_test_encoded = label_encoder.transform(y_test)
                    # Guardar las clases originales para interpretar predicciones
                    target_classes = label_encoder.classes_
                    # Reemplazar los valores originales con los codificados
                    y_train = y_train_encoded
                    y_test = y_test_encoded
                
                st.write(f"🎯 {lang.get('classification_problem', 'Classification problem')}: {np.unique(y_train).tolist()}")
            else:
                st.write(f"📊 {lang.get('regression_problem', 'Regression problem')}")
            
            st.subheader(lang['model_training_title'], help=lang['model_training_help'])

            model_descriptions = {
                'Logistic Regression': lang['logistic_regression_desc'],
                'KNN': lang['knn_desc'],
                'Naive Bayes': lang['naive_bayes_desc'],
                'SVM': lang['svm_desc'],
                'Decision Tree': lang['decision_tree_desc'],
                'Random Forest': lang['random_forest_desc'],
                'XGBoost': lang['xgboost_desc'],
                'Gradient Boosting': lang['gradient_boosting_desc'],
                'AdaBoost': lang['adaboost_desc'],
                'LightGBM': lang['lightgbm_desc']
            }

            # Select the model
            model_name = st.selectbox(lang['select_model'], list(models.keys()), help=lang['select_model_help'])
            ModelClass = models.get(model_name)
            
            # Model description
            st.write(model_descriptions.get(model_name))
            
            # Preprocesamiento específico por modelo
            preprocessing_message = ""
            if model_name == 'Logistic Regression':
                preprocessing_message += "• La regresión logística funciona mejor con datos normalizados.\n"
                if df.isnull().sum().sum() > 0:
                    preprocessing_message += "• Se recomienda imputar valores faltantes antes de entrenar.\n"
                    
            elif model_name == 'SVM':
                preprocessing_message += "• SVM es sensible a la escala, datos normalizados son recomendados.\n"
                if X_train.shape[1] > 100:
                    preprocessing_message += "• Muchas características podrían ralentizar el entrenamiento de SVM.\n"
                
            elif model_name == 'Naive Bayes':
                preprocessing_message += "• Naive Bayes asume independencia entre características.\n"
                preprocessing_message += "• Funciona bien con datos categóricos y textuales.\n"
                
            elif model_name == 'Decision Tree':
                preprocessing_message += "• Los árboles de decisión no requieren normalización.\n"
                preprocessing_message += "• Son robustos ante valores atípicos y faltantes.\n"
                
            elif model_name == 'Random Forest':
                preprocessing_message += "• Random Forest no requiere normalización de datos.\n"
                preprocessing_message += "• Funciona bien con características de diferentes escalas.\n"
                
            elif model_name == 'XGBoost' or model_name == 'Gradient Boosting' or model_name == 'LightGBM':
                preprocessing_message += "• Los algoritmos de boosting manejan bien valores faltantes.\n"
                preprocessing_message += "• No requieren normalización de características.\n"
                
            elif model_name == 'KNN':
                preprocessing_message += "• KNN es muy sensible a la escala de las características.\n"
                preprocessing_message += "• La normalización de datos es altamente recomendada.\n"
                if X_train.shape[0] > 10000:
                    preprocessing_message += "• Gran cantidad de muestras puede ralentizar KNN.\n"
                    
            elif model_name == 'AdaBoost':
                preprocessing_message += "• AdaBoost es sensible a valores atípicos.\n"
                preprocessing_message += "• Funciona mejor con modelos base simples.\n"
            
            # Mostrar mensaje de preprocesamiento si hay algún mensaje
            if preprocessing_message:
                with st.expander(lang.get('preprocessing_tips', 'Consejos de preprocesamiento')):
                    st.markdown(preprocessing_message)
            
            st.write(lang['hyperparameters'])
                      
            # Initialize the model based on the user's selection
            if model_name == 'Logistic Regression':
                C = st.slider(lang['logreg_c'], 0.01, 10.0, 1.0, help=lang['logreg_c_help'])
                if lasso:
                    model = ModelClass(C=C, penalty='l1', solver='saga')
                else:
                    model = ModelClass(C=C)

            elif model_name == 'KNN':
                n_neighbors = st.slider(lang['knn_neighbors'], 1, 15, 5, help=lang['knn_neighbors_help'])
                model = ModelClass(n_neighbors=n_neighbors)

            elif model_name == 'SVM':
                C = st.slider(lang['svm_c'], 0.01, 10.0, 1.0, help=lang['svm_c_help'])
                kernel = st.selectbox(lang['svm_kernel'], ['linear', 'poly', 'rbf', 'sigmoid'], index=2)
                model = ModelClass(C=C, kernel=kernel, probability=True)

            elif model_name == 'Decision Tree':
                max_depth = st.slider(lang['tree_depth'], 1, 20, 5, help=lang['tree_depth_help'])
                model = ModelClass(max_depth=max_depth)

            elif model_name == 'Random Forest':
                n_estimators = st.slider(lang['rf_estimators'], 10, 200, 100, help=lang['rf_estimators_help'])
                max_depth = st.slider(lang['tree_depth'], 1, 20, 5, help=lang['tree_depth_help'])
                model = ModelClass(n_estimators=n_estimators, max_depth=max_depth)

            elif model_name == 'XGBoost':
                learning_rate = st.slider(lang['xgb_learning_rate'], 0.001, 0.3, 0.01, help=lang['xgb_learning_rate_help'])
                n_estimators = st.slider(lang['xgb_estimators'], 10, 200, 10, help=lang['xgb_estimators_help'])
                max_depth = st.slider(lang['tree_depth'], 1, 20, 5, help=lang['tree_depth_help'])
                model = ModelClass(learning_rate=learning_rate, n_estimators=n_estimators, max_depth=max_depth)

            elif model_name == 'Gradient Boosting':
                learning_rate = st.slider(lang['xgb_learning_rate'], 0.01, 0.3, 0.1, help=lang['xgb_learning_rate_help'])
                n_estimators = st.slider(lang['xgb_estimators'], 10, 200, 100, help=lang['xgb_estimators_help'])
                max_depth = st.slider(lang['tree_depth'], 1, 20, 5, help=lang['tree_depth_help'])
                model = ModelClass(learning_rate=learning_rate, n_estimators=n_estimators, max_depth=max_depth)

            elif model_name == 'AdaBoost':
                n_estimators = st.slider(lang['adaboost_estimators'], 10, 200, 50, help=lang['adaboost_estimators_help'])
                learning_rate = st.slider(lang['xgb_learning_rate'], 0.01, 1.0, 1.0, help=lang['xgb_learning_rate_help'])
                model = ModelClass(n_estimators=n_estimators, learning_rate=learning_rate)

            elif model_name == 'Naive Bayes':
                model = ModelClass()  # No additional parameters required
                st.write(lang['naive_bayes_no_hyperparameters'])

            elif model_name == 'LightGBM':
                learning_rate = st.slider(lang['xgb_learning_rate'], 0.01, 0.3, 0.1, help=lang['xgb_learning_rate_help'])
                n_estimators = st.slider(lang['xgb_estimators'], 10, 200, 100, help=lang['xgb_estimators_help'])
                max_depth = st.slider(lang['tree_depth'], -1, 20, -1, help=lang['tree_depth_help'])
                model = ModelClass(learning_rate=learning_rate, n_estimators=n_estimators, max_depth=max_depth)

            else:
                st.error(lang['model_selection_error'])

            # Normalization
            auto_preprocessing = st.toggle(lang.get('auto_preprocessing', 'Preprocesamiento automático'), 
                                         value=True, 
                                         help=lang.get('auto_preprocessing_help', 'Aplicar automáticamente el preprocesamiento óptimo según el modelo seleccionado'))
            
            # Si el preprocesamiento automático está activado, realizamos el preprocesamiento según el modelo
            preprocessing_actions = []
            
            if auto_preprocessing:
                # Preprocesamiento específico para cada modelo
                if model_name in ['Logistic Regression', 'SVM', 'KNN']:
                    # Estos modelos se benefician de la normalización
                    normalizer = Normalizer()
                    X_train = normalizer.fit_transform(X_train)
                    X_test = normalizer.transform(X_test)
                    preprocessing_actions.append(lang.get('normalized_data', 'Datos normalizados'))
                
                # Imputación de valores faltantes para modelos sensibles a NaNs
                if model_name in ['Logistic Regression', 'SVM', 'KNN'] and df.isnull().sum().sum() > 0:
                    preprocessing_actions.append(lang.get('imputed_missing', 'Valores faltantes imputados'))
                
                # Para KNN, si hay muchas muestras, podría ser útil reducir dimensionalidad
                if model_name == 'KNN' and X_train.shape[0] > 10000:
                    preprocessing_actions.append(lang.get('knn_large_warning', 'Advertencia: Gran cantidad de muestras puede ralentizar KNN'))
                
                # Para árboles y boosting, no se necesita normalización pero sí manejo de valores NaN
                if model_name in ['Decision Tree', 'Random Forest', 'XGBoost', 'Gradient Boosting', 'LightGBM', 'AdaBoost'] and df.isnull().sum().sum() > 0:
                    preprocessing_actions.append(lang.get('trees_handle_missing', 'Los árboles manejan valores faltantes internamente'))
            else:
                # Si el preprocesamiento automático está desactivado, aplicar solo la normalización manual
                if normalize:
                    normalizer = Normalizer()
                    X_train = normalizer.fit_transform(X_train)
                    X_test = normalizer.transform(X_test)
                    preprocessing_actions.append(lang.get('manual_normalization', 'Normalización manual aplicada'))
            
            # Mostrar las acciones de preprocesamiento realizadas
            if preprocessing_actions:
                st.markdown("### " + lang.get('preprocessing_applied', 'Preprocesamiento aplicado:'))
                for action in preprocessing_actions:
                    st.markdown(f"✅ {action}")
            
            # Train the model
            model.fit(X_train, y_train)

            # Predictions
            try:
                if hasattr(model, "predict_proba"):
                    y_pred_proba = model.predict_proba(X_test)
                else:
                    # Para modelos sin predict_proba, usar decision_function si existe o crear una probabilidad simulada
                    if hasattr(model, "decision_function"):
                        decision_scores = model.decision_function(X_test)
                        # Normalizar los scores a un rango [0,1]
                        decision_scores = (decision_scores - decision_scores.min()) / (decision_scores.max() - decision_scores.min() + 1e-8)
                        # Convertir a formato similar a predict_proba (dos columnas para clasificación binaria)
                        if len(decision_scores.shape) == 1:  # Si es unidimensional
                            y_pred_proba = np.vstack([1-decision_scores, decision_scores]).T
                        else:
                            y_pred_proba = decision_scores
                    else:
                        # Si no tiene ninguna función de probabilidad o puntaje, usar predicciones directas
                        y_pred = model.predict(X_test)
                        # Convertir las predicciones a una matriz de probabilidades simuladas
                        y_pred_proba = np.zeros((len(y_pred), 2))
                        for i, pred in enumerate(y_pred):
                            y_pred_proba[i, int(pred)] = 1.0
                
                # Asegurarse de que y_pred_proba tenga la forma correcta para problemas binarios
                if y_pred_proba.shape[1] == 2:
                    # Predictions adjusted for the default threshold (0.5)
                    y_pred_adjusted = (y_pred_proba[:, 1] >= 0.5).astype(int)
                else:
                    # Si es multiclase, tomar la clase con mayor probabilidad
                    y_pred_adjusted = np.argmax(y_pred_proba, axis=1)
            except Exception as e:
                st.error(f"Error al generar predicciones: {e}")
                # Intentar una alternativa más simple
                y_pred_adjusted = model.predict(X_test)
                # Crear una matriz de probabilidades simuladas para seguir con el análisis
                y_pred_proba = np.zeros((len(y_pred_adjusted), len(np.unique(y_test))))
                for i, pred in enumerate(y_pred_adjusted):
                    y_pred_proba[i, int(pred)] = 1.0

            # Show feature importance chart
            if model_name in ['Decision Tree', 'Random Forest', 'XGBoost', 'Gradient Boosting', 'AdaBoost', 'LightGBM']:
                feature_importances = model.feature_importances_
                if feature_importances is not None:
                    st.text(lang['feature_importance_title'], help=lang['feature_importance_help'])
                    fig, ax = plt.subplots()
                    # Sort features by importance and select the top 10
                    sorted_idx = np.argsort(feature_importances)[-10:]
                    top_features = [selected_columns[i] for i in sorted_idx]
                    top_importances = feature_importances[sorted_idx]

                    # Usar color OMINIS para las barras
                    ax.barh(top_features, top_importances, color="#4682B4")
                    ax.set_xlabel(lang['importance'])
                    fig.set_size_inches(3, 2)
                    
                    # Reduce font size
                    for item in ([ax.title, ax.xaxis.label, ax.yaxis.label] +
                                 ax.get_xticklabels() + ax.get_yticklabels()):
                        item.set_fontsize('xx-small')
                    
                    st.pyplot(fig)
                             
    except Exception as e:
        st.error(f"{lang['error']} {e}")



with col_score:
    if 'df' in locals():
        st.subheader(lang['performance_title'], help=lang['performance_help'])

        # ROC Curve and AUC
        n_classes = len(np.unique(y_test))

        if n_classes == 2:
            fpr, tpr, thresholds = roc_curve(y_test, y_pred_proba[:, 1])  # Use only the second column for binary problems
            roc_auc = roc_auc_score(y_test, y_pred_proba[:, 1])

            # Plot the ROC curve
            fig, ax = plt.subplots()
            ax.plot(fpr, tpr, label=f"{lang['roc_curve']} (area = {roc_auc:.2f})", color="#4682B4")  # Color OMINIS
            ax.plot([0, 1], [0, 1], linestyle='--', color='gray')
            ax.set_xlabel(lang['fpr_label'])
            ax.set_ylabel(lang['tpr_label'])
            ax.set_title(f"{lang['roc_curve']} (AUC = {roc_auc:.2f})")  # Show AUC in the title
            fig.set_size_inches(4, 3)  # Adjust figure size
            st.pyplot(fig)

            # Calculate the optimal threshold
            optimal_idx = np.argmax(tpr - fpr)
            optimal_threshold = thresholds[optimal_idx]
            
            # Threshold slider
            threshold = st.slider(f"{lang['threshold_slider']} (optimal: {optimal_threshold:.2f})", 0.0, 1.0, optimal_threshold, 0.01, help=lang['threshold_help'])

            # Final predictions based on the adjusted threshold
            y_pred_adjusted = (y_pred_proba[:, 1] >= threshold).astype(int)

            # Confusion matrix
            cm = confusion_matrix(y_test, y_pred_adjusted)
            st.text(lang['confusion_matrix'], help=lang['confusion_matrix_help'])
            fig, ax = plt.subplots()
            sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax)  # Blues se mantiene, coincide con la paleta OMINIS
            ax.set_xlabel(lang['predicted_label'])
            ax.set_ylabel(lang['actual_label'])
            fig.set_size_inches(3, 2)  # Adjust figure size
            st.pyplot(fig)

        else:
            # Multiclass (one-vs-rest)
            y_test_bin = label_binarize(y_test, classes=np.unique(y_test))
            if y_test_bin.shape[1] != y_pred_proba.shape[1]:
                st.error(f"{lang['dimension_mismatch_error']}: y_test_bin {lang['has']} {y_test_bin.shape[1]} {lang['columns']}, {lang['but']} y_pred_proba {lang['has']} {y_pred_proba.shape[1]} {lang['columns']}.")
            else:
                roc_auc = roc_auc_score(y_test_bin, y_pred_proba, multi_class="ovr")
                st.write(f"{lang['roc_auc_multiclass']}: {roc_auc:.2f}")

        # Calculate Sensitivity and Specificity:
        sensitivity = recall_score(y_test, y_pred_adjusted)
        specificity = recall_score(y_test, y_pred_adjusted, pos_label=0)
        st.write(f"{lang['sensitivity_label']}: {sensitivity:.2f}")
        st.write(f"{lang['specificity_label']}: {specificity:.2f}")
        
        
        # Calculate evaluation metrics
        report = classification_report(y_test, y_pred_adjusted, output_dict=True, zero_division=0)
        report_df = pd.DataFrame(report).transpose()

        # Filter most important metrics if available
        if 'precision' in report_df.columns and 'recall' in report_df.columns:
            performance_metrics = report_df[['precision', 'recall', 'f1-score', 'support']]
        else:
            st.error(lang['metrics_error'])

        # Rename metrics for display in the table
        performance_metrics = performance_metrics.rename(index={'precision': lang['precision_label'], 'recall': lang['sensitivity_label'], 'f1-score': 'F1', 'support': lang['support_label']})

        st.text(lang['performance_metrics_table'], help=lang['performance_metrics_help'])
        st.dataframe(performance_metrics)

# Footer
st.divider()
st.write(lang["footer"])
# Modal for displaying LICENSE file
with open('LICENSE', 'r') as license_file:
    license_text = license_file.read()

if st.button(lang['show_license']):
    st.text_area(lang['license_title'], license_text, height=300)

