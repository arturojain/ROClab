# ROC Lab

Una aplicación interactiva para la optimización de modelos de clasificación.

## Descripción

ROC Lab es un laboratorio interactivo para entender y optimizar modelos de clasificación utilizando curvas ROC. Esta herramienta te permite:

- Cargar tus propios datos CSV con variables numéricas
- Explorar visualmente los datos, correlaciones y distribuciones
- Probar diferentes modelos de machine learning (Regresión Logística, SVM, Random Forest, etc.)
- Ajustar hiperparámetros y ver cómo afectan al rendimiento
- Visualizar curvas ROC y matrices de confusión
- Optimizar el umbral de clasificación

![image](https://github.com/user-attachments/assets/0b62dab4-7e3a-42a4-9340-9e2f80112434)

## Conjuntos de datos de ejemplo

Puedes utilizar estos conjuntos de datos públicos para probar la aplicación:

1. **[Diabetes de Pima Indians](https://www.kaggle.com/datasets/uciml/pima-indians-diabetes-database)** - Dataset para predecir si un paciente tiene diabetes basado en medidas diagnósticas.
   - Variable objetivo: `Outcome` (0 o 1)
   - 768 instancias, 8 variables numéricas

2. **[Enfermedad Cardiovascular](https://www.kaggle.com/datasets/sulianova/cardiovascular-disease-dataset)** - Dataset para predecir enfermedad cardiovascular.
   - Variable objetivo: `cardio` (0 o 1)
   - 70,000 instancias, 11 variables

3. **[Cáncer de Mama](https://www.kaggle.com/datasets/uciml/breast-cancer-wisconsin-data)** - Dataset para clasificar tumores de mama como malignos o benignos.
   - Variable objetivo: `diagnosis` (M o B)
   - 569 instancias, 30 características
  
4. **[Riesgo Cardiovascular a 10 años](https://www.kaggle.com/code/bansodesandeep/cardiovascular-risk-prediction/input)**
   - Variables objetivo: `BPMeds`, `prevalentStroke`, `prevalentHyp`, `diabetes` (1 o 0)
   - 3390 instancias, 16 características

## Requisitos

- Python 3.7 o superior
- Dependencias listadas en `requirements.txt`

## Instalación y ejecución

Tienes dos opciones para instalar y ejecutar ROClab:

### Método automático (recomendado)

Este método automatiza todo el proceso de configuración y es la forma más sencilla de comenzar.

1. Clona o descarga este repositorio:
   ```bash
   git clone https://github.com/mcquaas/ROClab.git
   cd ROClab
   ```

2. Dale permisos de ejecución al script de instalación:
   ```bash
   chmod +x setup.sh
   ```

3. Ejecuta el script de instalación:
   ```bash
   ./setup.sh
   ```

El script de instalación realizará automáticamente las siguientes tareas:
- Detectará tu sistema operativo (macOS, Linux, Windows)
- Verificará que Python esté instalado
- Creará un entorno virtual para aislar las dependencias
- Instalará todas las bibliotecas necesarias
- Creará un script de ejecución personalizado para tu sistema
- Te ofrecerá la opción de iniciar la aplicación inmediatamente

> **Nota para usuarios con rutas especiales**: Si la ruta de tu proyecto contiene caracteres especiales como dos puntos (:), el script creará el entorno virtual en tu directorio home en lugar del directorio del proyecto.

### Método manual

Si prefieres configurar manualmente la aplicación, sigue estos pasos:

1. Clona o descarga este repositorio:
   ```bash
   git clone https://github.com/mcquaas/ROClab.git
   cd ROClab
   ```

2. Crea un entorno virtual:
   ```bash
   # En sistemas con rutas normales (sin caracteres especiales)
   python3 -m venv .venv
   
   # En sistemas con rutas que contienen caracteres especiales
   python3 -m venv ~/roclab_venv
   ```

3. Activa el entorno virtual:
   ```bash
   # En macOS/Linux para .venv local
   source .venv/bin/activate
   
   # En macOS/Linux para entorno en el directorio home
   source ~/roclab_venv/bin/activate
   
   # En Windows para .venv local
   .venv\Scripts\activate
   
   # En Windows para entorno en el directorio home
   %USERPROFILE%\roclab_venv\Scripts\activate
   ```

4. Instala las dependencias:
   ```bash
   pip install -r requirements.txt
   ```

5. Ejecuta la aplicación:
   ```bash
   streamlit run ROClab.py
   ```

## Uso

1. Una vez iniciada la aplicación, abre tu navegador en la dirección que indica Streamlit (normalmente http://localhost:8501)
2. Carga un archivo CSV con datos estructurados
3. Selecciona la variable objetivo y las variables predictivas
4. Explora los datos y entrena los diferentes modelos
5. Analiza el rendimiento y ajusta los parámetros según sea necesario

## Solución de problemas comunes

### Error al crear el entorno virtual

Si encuentras un error como `Error: Refusing to create a venv in ... because it contains the PATH separator :`, significa que la ruta de tu directorio contiene caracteres especiales. Intenta:

1. Usar el script automático `setup.sh` que detectará este problema y creará el entorno en tu directorio home
2. O crear manualmente el entorno virtual en tu directorio home:
   ```bash
   python3 -m venv ~/roclab_venv
   source ~/roclab_venv/bin/activate
   ```

### Módulos no encontrados

Si recibes errores del tipo `ModuleNotFoundError: No module named 'xgboost'` u otros módulos, ejecuta:

```bash
pip install -r requirements.txt
```

### Error en el archivo JSON de idioma

Si ves un error como `json.decoder.JSONDecodeError`, puede haber un problema con los archivos de idioma. Verifica que los archivos `lang_es.json` y `lang_en.json` estén correctamente formateados. Son archivos JSON válidos que no deben contener errores de sintaxis.

### Cambio de idioma

- Para cambiar el idioma a través de la interfaz: usa el selector en la esquina superior derecha
- Para cambiar el idioma a través de la URL: añade `?lang=en` o `?lang=es` al final de la URL

### Watchdog no instalado

Si ves una advertencia sobre instalar Watchdog para un mejor rendimiento, puedes seguir las instrucciones:

```bash
xcode-select --install  # Solo en macOS
pip install watchdog
```

## Características

- **Multilingüe**: Puedes cambiar entre español e inglés con el selector en la parte superior o mediante el parámetro en la URL: `?lang=en` o `?lang=es`.
- **Visualización avanzada**: Incluye matrices de correlación, distribuciones, curvas ROC y matrices de confusión.
- **Múltiples modelos**: Logistic Regression, KNN, Naive Bayes, SVM, Decision Tree, Random Forest, XGBoost, Gradient Boosting, AdaBoost, LightGBM.
- **Métricas de rendimiento**: Accuracy, precision, recall, F1-score, AUC, sensibilidad, especificidad.

## Licencia

Este proyecto está licenciado bajo GNU General Public License v3.0 - ver el archivo `LICENSE` para más detalles.

## Contacto

Gustavo Ross - gross@funsalud.org.mx
