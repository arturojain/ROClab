import streamlit as st
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, Normalizer, label_binarize
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

st.set_page_config(page_title="Didactiva AI Lab", layout="wide", initial_sidebar_state="expanded")

# Definir los modelos disponibles
models = {
    'Regresión Logística': LogisticRegression,
    'KNN': KNeighborsClassifier,
    'Naive Bayes': GaussianNB,
    'SVM': SVC,
    'Árbol de decisión': DecisionTreeClassifier,
    'Random Forest': RandomForestClassifier,
    'XGBoost': XGBClassifier,
    'Gradient Boosting': GradientBoostingClassifier,
    'AdaBoost': AdaBoostClassifier,
    'LightGBM': lgb.LGBMClassifier
}

# Cargar datos
@st.cache_data
def load_data(file):
    return pd.read_csv(file)

header1, header2, header3 = st.columns([2, 3, 2])
with header1:
    st.image('https://www.didactiva.com/wp-content/uploads/2024/06/logo-didactiva-20.png', width=200)
with header2:
    st.title('Lab de optimización')
    st.write('Clasificadores para datos numéricos con ML - Módulo 6 - Diplomado IA en Salud')
with header3:
    # Instrucciones en un expander al final de la columna derecha
    with st.expander("Instrucciones"):
        st.write('1. Carga un archivo CSV con datos estructurados y selecciona la variable objetivo y variables a usar.')
        st.write('2. Explora los datos, observa correlaciones y distribuciones.')
        st.write('3. Decide cómo preparar los datos para el entrenamiento.')
        st.write('4. Entrena diferentes modelos y ajusta sus parámetros para optimizar su rendimiento.')
        st.write('5. Visualiza la curva ROC, ajusta el umbral y observa la matriz de confusión.')
    with st.expander("Acerca de"):
        st.write('Experimental: para el Diplomado de IA en Salud (https://didactiva.com/funsalud) con fines didácticos. No debe ser usado para fines médicos. Hay mucho trabajo pendiente, gracias por enviar tus comentarios y sugerencias:')
        st.write('Autor: Gustavo Ross - gross@funsalud.org.mx')

st.divider()

# División de la app en dos columnas
col_data, col_modelo, col_score = st.columns([3, 3, 2])

# En la columna izquierda (col_data) colocamos el logo, la carga de archivos y los selectores
with col_data:

    try:
        # Cargar el archivo CSV
        st.subheader('1. Carga de datos', help="Carga un archivo CSV que contenga datos estructurados con variables numéricas para el entrenamiento de los modelos de clasificación. El archivo debe incluir una columna objetivo (target), que es la variable que deseas predecir (como diagnóstico, resultado, target, clase, entre otras). La columna objetivo puede ser binaria (con dos valores, por ejemplo, 0 y 1) o multiclase (varias categorías). Además, el archivo debe tener suficientes variables predictoras (características o atributos numéricos) que ayuden a predecir la columna objetivo. Asegúrate de que los datos estén limpios y estructurados correctamente, sin valores faltantes en las columnas importantes. Si es necesario, el laboratorio ofrece opciones para manejar valores faltantes y normalizar los datos.")
        uploaded_file = st.file_uploader('Carga un archivo CSV con datos estructurados', type='csv')
        if uploaded_file:
            df = load_data(uploaded_file)


            # Mostrar una vista previa de los datos
            with st.expander(f"Se cargaron {len(df)} registros con {len(df.columns)} columnas (Ver 5 ejemplos)"):
                st.write(df.head())
            
            # Preseleccionar la columna target si existe "target", "y" o una columna binaria
            possible_target_columns = ['target', 'y', 'Outcome', 'objetivo', 'clase', 'diagnóstico', 'resultado']
            preselected_target = None
            
            for col in df.columns:
                if col.lower() in [target.lower() for target in possible_target_columns]:
                    preselected_target = col
                    break
                elif df[col].nunique() == 2:  # Verificar si la columna es binaria
                    preselected_target = col
                    break

            # Mostrar el selector de columna objetivo, preseleccionando la columna si existe
            target_column = st.selectbox("Selecciona la variable objetivo", df.columns, index=df.columns.get_loc(preselected_target) if preselected_target else 0, help="La variable objetivo que deseas predecir.") if preselected_target else st.selectbox("Selecciona la columna objetivo", df.columns, help="La variable objetivo que deseas predecir.")


            # Selección de columnas (excepto la variable objetivo) dentro de un expander
            with st.expander(f"Selecciona variables usadas para la predicción"):
                selected_columns = st.multiselect('Selecciona datos a incluir', [col for col in df.columns if col != target_column], default=[col for col in df.columns if col != target_column], help="Columnas a incluir en el modelo de clasificación. Observa cómo cambian las métricas al incluir o excluir columnas.")


            if 'df' not in st.session_state:
                st.session_state.df = load_data(uploaded_file)

            df = st.session_state.df  # Usar los datos almacenados en la sesión


            st.subheader('2. Análisis exploratorio', help="Explora los datos cargados para comprender mejor las distribuciones, correlaciones y observaciones. Observa cómo las variables se relacionan entre sí y con la variable objetivo.")
            tab1, tab2, tab3 = st.tabs(["Distribuciones", "Correlaciones", "Observaciones"])
            with tab1:
                st.write('Distribuciones estadísticas de las variables')
                st.write(df[selected_columns].describe())
            with tab2:
                # Mostrar la correlación entre las variables
                st.write('Correlación entre variables')
                corr = df[selected_columns].corr()
                fig, ax = plt.subplots()
                sns.heatmap(corr, annot=True, cmap='coolwarm', ax=ax, annot_kws={"size": 6})
                ax.tick_params(axis='both', which='major', labelsize=6)
                st.pyplot(fig)
            with tab3:
                # Botón para generar observaciones
                if st.button('Generar observaciones', key='centered_button'):
                    with st.spinner('Generando observaciones...'):
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
        st.error(f"Error: {e}")
    
# En la columna derecha (col_modelo) se muestra el selector de modelo, resultados, curva ROC, y la tabla de métricas
with col_modelo:
    
    try:
        if 'df' in locals():
            
            st.subheader('3. Preparación de datos', help="Prepara los datos para el entrenamiento del modelo de clasificación. Asegúrate de que los datos estén limpios y estructurados correctamente, sin valores faltantes en las columnas importantes. Si es necesario, el laboratorio ofrece opciones para manejar valores faltantes y normalizar los datos.")
            # Preparar los datos
            features = df[selected_columns]
            target = df[target_column]
            
            if df.isnull().sum().sum() > 0:                
                # Procesamiento de valores faltantes
                handle_missing = st.selectbox("Manejo de valores faltantes", ["Ninguno", "Eliminar", "Imputar (Promedios)"])
                if handle_missing == "Eliminar":
                    df = df.dropna()
                    features = df[selected_columns]
                    target = df[target_column]
                elif handle_missing == "Imputar (Promedios)":
                    imputer = SimpleImputer(strategy='mean')
                    features = imputer.fit_transform(features)
            else:
                st.text('No se encontraron valores faltantes en los datos.', help="No es necesario realizar imputación de valores faltantes.")
            

            normalize = st.checkbox('Normalización de valores', help="Normaliza los datos antes de entrenar el modelo")
            lasso = st.checkbox('Regularización Lasso', help="La regularización Lasso es una técnica utilizada en modelos de regresión que añade una penalización sobre los coeficientes de las variables, forzando a algunos de ellos a ser exactamente cero, lo que resulta en la selección automática de características. Esto ayuda a simplificar el modelo y prevenir el sobreajuste, especialmente cuando se tienen muchas variables, ya que elimina aquellas que no contribuyen significativamente a la predicción. Se utiliza cuando se desea mejorar la interpretabilidad del modelo y reducir el riesgo de ajustar demasiado los datos de entrenamiento.")
            outlier_detection = st.checkbox('Eliminar outliers (valores atípicos)', help="Los outliers son valores atípicos que se desvían significativamente del resto de los datos y pueden afectar negativamente el rendimiento de los modelos de machine learning, especialmente en algoritmos sensibles como la regresión lineal. Estos puntos extremos pueden ser causados por errores de medición o representar casos raros, y al no manejarlos adecuadamente, pueden distorsionar las predicciones, sesgar métricas como la media o desviación estándar, y llevar al sobreajuste del modelo. Para discriminar outliers, se utilizan métodos como la detección de valores que están fuera de un rango esperado (por ejemplo, mediante el uso del rango intercuartílico o Z-score) o mediante técnicas más avanzadas como el aislamiento de bosque (isolation forest) o el clustering.")

            # Crear las configuraciones en el lado izquierdo
            st.write('Separación de muestras', help="Configuración del tamaño de la muestra de prueba y el número de particiones para la validación cruzada.")
                        
            col1, col2 = st.columns([1, 1])
            with col2:
                test_size = st.slider('Tamaño de muestra de prueba', 0.1, 0.5, 0.2, help="Proporción de los datos que se utilizarán para la prueba")
            with col1:
                # Mostrar el número de muestras de entrenamiento y de prueba en un gráfico de pastel
                sample_counts = {
                    'Entrenamiento': int(len(df) * (1 - test_size)),
                    'Prueba': int(len(df) * test_size)
                }
                fig, ax = plt.subplots(figsize=(2, 2))  # Hacer el gráfico más pequeño
                ax.pie(sample_counts.values(), labels=sample_counts.keys(), autopct=lambda p: f'{int(p * sum(sample_counts.values()) / 100)}', startangle=90)
                ax.axis('equal')  # Para asegurar que el gráfico sea un círculo
                st.pyplot(fig)
            
            n_splits = st.slider('Número de particiones (GroupKFold)', 2, 10, 5, help="Número de particiones para la validación cruzada.")


            # División de los datos en conjuntos de entrenamiento y prueba
            X_train, X_test, y_train, y_test = train_test_split(features, target, test_size=test_size, random_state=42)
            
            
            
            
            st.subheader('4. Entrenamiento del modelo', help="Entrena un modelo de clasificación con los datos seleccionados. Ajusta los parámetros del modelo para mejorar su rendimiento y evalúa su precisión con las métricas de evaluación.")

            descripciones_modelos = {
                'Regresión Logística': 'Este modelo predice resultados binarios, estimando la probabilidad de que una observación pertenezca a una clase específica. Es ideal para problemas donde las salidas son "sí" o "no". [Más información](https://es.wikipedia.org/wiki/Regresi%C3%B3n_log%C3%ADstica)',
                'KNN': 'El algoritmo de los k vecinos más cercanos clasifica un dato nuevo basándose en los datos más cercanos en el conjunto de entrenamiento, asignando la clase mayoritaria de sus vecinos. [Más información](https://es.wikipedia.org/wiki/K_vecinos_m%C3%A1s_cercanos)',
                'Naive Bayes': 'Este modelo utiliza el teorema de Bayes y asume que todas las características son independientes entre sí, siendo útil para problemas como la clasificación de texto. [Más información](https://es.wikipedia.org/wiki/Clasificador_bayesiano_ingenuo)',
                'SVM': 'Las máquinas de vectores de soporte buscan la mejor línea o frontera que separa diferentes clases de datos, maximizando la distancia entre las clases para mejorar la clasificación. [Más información](https://es.wikipedia.org/wiki/M%C3%A1quina_de_vectores_de_soporte)',
                'Árbol de decisión': 'Este modelo crea un árbol de decisiones donde cada nodo representa una decisión basada en una característica, lo que facilita clasificar los datos en una categoría. [Más información](https://es.wikipedia.org/wiki/%C3%81rbol_de_decisi%C3%B3n)',
                'Random Forest': 'Combina muchos árboles de decisión, donde cada árbol vota y la clasificación final es el promedio o voto mayoritario, haciéndolo más preciso y robusto. [Más información](https://es.wikipedia.org/wiki/Random_forest)',
                'XGBoost': 'Un modelo basado en árboles de decisión que aplica la técnica de "boosting", donde los árboles se construyen secuencialmente, corrigiendo los errores de los anteriores. [Más información](https://es.wikipedia.org/wiki/XGBoost)',
                'Gradient Boosting': 'Similar a XGBoost, pero optimiza el rendimiento del modelo ajustando cuidadosamente las predicciones para mejorar su precisión. [Más información](https://es.wikipedia.org/wiki/Gradient_boosting)',
                'AdaBoost': 'Funciona ponderando más los ejemplos mal clasificados por los modelos anteriores, concentrando los esfuerzos en corregir esos errores en los árboles posteriores. [Más información](https://es.wikipedia.org/wiki/Boosting)',
                'LightGBM': 'Optimizado para manejar grandes volúmenes de datos de manera rápida y eficiente, LightGBM construye árboles de decisión de manera más rápida sin perder precisión. [Más información](https://en.wikipedia.org/wiki/LightGBM)'
            }

            # Selección del modelo
            model_name = st.selectbox('Selecciona un modelo de IA', list(models.keys()), help="Selecciona diferentes modelos de Machine Learning y ajusta sus parámetros hasta encontrar la mejor curva ROC.")
            ModelClass = models.get(model_name)
            
            # Descripción del modelo
            st.write(descripciones_modelos.get(model_name))
                      
            # Inicializa el modelo según la selección del usuario
            if model_name == 'Regresión Logística':
                C = st.slider('C (Inverso de intensidad de regulación)', 0.01, 10.0, 1.0, help="Valor más bajo para mayor regularización")
                if lasso:
                    model = ModelClass(C=C, penalty='l1', solver='saga')
                else:
                    model = ModelClass(C=C)

            elif model_name == 'KNN':
                n_neighbors = st.slider('Número de vecinos', 1, 15, 5, help="Número de vecinos a considerar para la clasificación")
                model = ModelClass(n_neighbors=n_neighbors)

            elif model_name == 'SVM':
                C = st.slider('C (Inverso de intensidad de regulación)', 0.01, 10.0, 1.0, help="Valor más bajo para mayor regularización")
                kernel = st.selectbox('Kernel', ['linear', 'poly', 'rbf', 'sigmoid'], index=2)
                model = ModelClass(C=C, kernel=kernel, probability=True)

            elif model_name == 'Árbol de decisión':
                max_depth = st.slider('Profundidad', 1, 20, 5, help="Profundidad máxima del árbol de decisión")
                model = ModelClass(max_depth=max_depth)

            elif model_name == 'Random Forest':
                n_estimators = st.slider('Número de estimadores', 10, 200, 100, help="Número de árboles a construir")
                max_depth = st.slider('Profundidad', 1, 20, 5, help="Profundidad máxima de cada árbol")
                model = ModelClass(n_estimators=n_estimators, max_depth=max_depth)

            elif model_name == 'XGBoost':
                learning_rate = st.slider('Vel aprendizaje', 0.001, 0.3, 0.01, help="Tasa de aprendizaje para cada iteración")
                n_estimators = st.slider('Número de estimadores', 10, 200, 10, help="Número de árboles a construir")
                max_depth = st.slider('Profundidad', 1, 20, 5, help="Profundidad máxima de cada árbol")
                model = ModelClass(learning_rate=learning_rate, n_estimators=n_estimators, max_depth=max_depth)

            elif model_name == 'Gradient Boosting':
                learning_rate = st.slider('Vel aprendizaje', 0.01, 0.3, 0.1, help="Tasa de aprendizaje para cada iteración")
                n_estimators = st.slider('Número de estimadores', 10, 200, 100, help="Número de árboles a construir")
                max_depth = st.slider('Profundidad', 1, 20, 5, help="Profundidad máxima de cada árbol")
                model = ModelClass(learning_rate=learning_rate, n_estimators=n_estimators, max_depth=max_depth)

            elif model_name == 'AdaBoost':
                n_estimators = st.slider('Número de estimadores', 10, 200, 50, help="Número de árboles a construir")
                learning_rate = st.slider('Vel aprendizaje', 0.01, 1.0, 1.0, help="Tasa de aprendizaje para cada iteración")
                model = ModelClass(n_estimators=n_estimators, learning_rate=learning_rate)

            elif model_name == 'Naive Bayes':
                model = ModelClass()  # No requiere parámetros adicionales

            elif model_name == 'LightGBM':
                learning_rate = st.slider('Vel aprendizaje', 0.01, 0.3, 0.1, help="Tasa de aprendizaje para cada iteración")
                n_estimators = st.slider('Número de estimadores', 10, 200, 100, help="Número de árboles a construir")
                max_depth = st.slider('Profundidad', -1, 20, -1, help="Profundidad máxima de cada árbol")
                model = ModelClass(learning_rate=learning_rate, n_estimators=n_estimators, max_depth=max_depth)

            else:
                st.error("Modelo no seleccionado correctamente")

            # Normalización
            if normalize:
                normalizer = Normalizer()
                X_train = normalizer.fit_transform(X_train)
                X_test = normalizer.transform(X_test)

            # Entrenar el modelo
            model.fit(X_train, y_train)

            # Predicciones
            if hasattr(model, "predict_proba"):
                y_pred_proba = model.predict_proba(X_test)
            else:
                y_pred_proba = model.decision_function(X_test)
                y_pred_proba = (y_pred_proba - y_pred_proba.min()) / (y_pred_proba.max() - y_pred_proba.min())

            # Predicciones ajustadas para el threshold predeterminado (0.5)
            y_pred_adjusted = (y_pred_proba[:, 1] >= 0.5).astype(int)

            # Mostrar gráfico de importancia de características
            if model_name in ['Árbol de decisión', 'Random Forest', 'XGBoost', 'Gradient Boosting', 'AdaBoost', 'LightGBM']:
                feature_importances = model.feature_importances_
                if feature_importances is not None:
                    st.text('Importancia de características', help="Muestra la importancia de las características o variables para generar la predicción con el modelo entrenado.")
                    fig, ax = plt.subplots()
                    # Ordenar las características por importancia y seleccionar las top 10
                    sorted_idx = np.argsort(feature_importances)[-10:]
                    top_features = [selected_columns[i] for i in sorted_idx]
                    top_importances = feature_importances[sorted_idx]

                    ax.barh(top_features, top_importances)
                    ax.set_xlabel('Importancia')
                    fig.set_size_inches(3, 2)
                    
                    # Reducir el tamaño de la fuente
                    for item in ([ax.title, ax.xaxis.label, ax.yaxis.label] +
                                 ax.get_xticklabels() + ax.get_yticklabels()):
                        item.set_fontsize('xx-small')
                    
                    st.pyplot(fig)
                        

                    
    except Exception as e:
        st.error(f"Error: {e}")



with col_score:
    if 'df' in locals():
        st.subheader('5. Rendimiento', help="Evalúa el rendimiento del modelo de clasificación con las métricas de evaluación y visualiza la curva ROC. Ajusta el umbral para clasificar las predicciones como positivas o negativas y observa la matriz de confusión.")

        # Curva ROC y AUC
        n_classes = len(np.unique(y_test))

        if n_classes == 2:
            fpr, tpr, thresholds = roc_curve(y_test, y_pred_proba[:, 1])  # Usar solo la segunda columna para problemas binarios
            roc_auc = roc_auc_score(y_test, y_pred_proba[:, 1])

            # Graficar la curva ROC
            fig, ax = plt.subplots()
            ax.plot(fpr, tpr, label=f'ROC curve (area = {roc_auc:.2f})')
            ax.plot([0, 1], [0, 1], linestyle='--', color='gray')
            ax.set_xlabel('Tasa de Falsos Positivos')
            ax.set_ylabel('Tasa de Verdaderos Positivos')
            ax.set_title('Curva ROC')
            fig.set_size_inches(4, 3)  # Ajustar el tamaño de la figura
            st.pyplot(fig)

            # Calcular Threshold ideal
            optimal_idx = np.argmax(tpr - fpr)
            optimal_threshold = thresholds[optimal_idx]
            
            # Threshold slider
            threshold = st.slider(f"Umbral/Threshold (óptimo: {optimal_threshold:.2f})", 0.0, 1.0, optimal_threshold, 0.01, help="Ajusta el umbral para ser más permisivo o más exigente. Un umbral bajo genera falsos positivos, un umbral alto genera falsos negativos.")

            # Predicciones finales según el umbral ajustado
            y_pred_adjusted = (y_pred_proba[:, 1] >= threshold).astype(int)

            # Matriz de confusión
            cm = confusion_matrix(y_test, y_pred_adjusted)
            st.text('Matriz de Confusión', help="Busca que la diagnoal invertida sea más obscura (menos falsos positivos/negativos).")
            fig, ax = plt.subplots()
            sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax)
            ax.set_xlabel('Predicción')
            ax.set_ylabel('Real')
            fig.set_size_inches(3, 2)  # Ajustar el tamaño de la figura
            st.pyplot(fig)

        else:
            # Multiclase (one-vs-rest)
            y_test_bin = label_binarize(y_test, classes=np.unique(y_test))
            if y_test_bin.shape[1] != y_pred_proba.shape[1]:
                st.error(f"Dimension mismatch: y_test_bin tiene {y_test_bin.shape[1]} columnas, pero y_pred_proba tiene {y_pred_proba.shape[1]} columnas.")
            else:
                roc_auc = roc_auc_score(y_test_bin, y_pred_proba, multi_class="ovr")
                st.write(f'ROC AUC (multiclase): {roc_auc:.2f}')

        # Calcular sensibilidad y especificidad:
        sensitivity = recall_score(y_test, y_pred_adjusted)
        specificity = recall_score(y_test, y_pred_adjusted, pos_label=0)
        st.write(f'Sensibilidad (Recall): {sensitivity:.2f}')
        st.write(f'Especificidad: {specificity:.2f}')
        
        
        # Calcular las métricas de evaluación
        report = classification_report(y_test, y_pred_adjusted, output_dict=True, zero_division=0)
        report_df = pd.DataFrame(report).transpose()

        # Filtrar las métricas más importantes si existen
        if 'precision' in report_df.columns and 'recall' in report_df.columns:
            performance_metrics = report_df[['precision', 'recall', 'f1-score', 'support']]
        else:
            st.error("Las métricas 'precision', 'recall', 'f1-score' y 'support' no se encontraron en el reporte. Verifica los datos.")

        # Renombrar las métricas en español para mostrar en la tabla
        performance_metrics = performance_metrics.rename(index={'precision': 'Precisión', 'recall': 'Sensibilidad', 'f1-score': 'F1', 'support': 'Soporte'})

        st.write("Tabla de métricas de rendimiento:", help="Métricas de evaluación del modelo de clasificación.")
        st.dataframe(performance_metrics)


