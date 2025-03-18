# ROC Lab

An interactive application for optimizing classification models.

## Description

ROC Lab is an interactive laboratory for understanding and optimizing classification models using ROC curves. This tool allows you to:

- Upload your own CSV files with numerical variables
- Visually explore data, correlations, and distributions
- Test different machine learning models (Logistic Regression, SVM, Random Forest, etc.)
- Adjust hyperparameters and see how they affect performance
- Visualize ROC curves and confusion matrices
- Optimize the classification threshold

## Sample Datasets

You can use these public datasets to test the application:

1. **[Pima Indians Diabetes](https://www.kaggle.com/datasets/uciml/pima-indians-diabetes-database)** - Dataset to predict whether a patient has diabetes based on diagnostic measurements.
   - Target variable: `Outcome` (0 or 1)
   - 768 instances, 8 numerical variables

2. **[Cardiovascular Disease](https://www.kaggle.com/datasets/sulianova/cardiovascular-disease-dataset)** - Dataset to predict cardiovascular disease.
   - Target variable: `cardio` (0 or 1)
   - 70,000 instances, 11 variables

3. **[Breast Cancer](https://www.kaggle.com/datasets/uciml/breast-cancer-wisconsin-data)** - Dataset to classify breast tumors as malignant or benign.
   - Target variable: `diagnosis` (M or B)
   - 569 instances, 30 features

> **Note**: The application is designed to work with numerical variables. Categorical variables should be pre-encoded (one-hot encoding, label encoding, etc.).

## Requirements

- Python 3.7 or higher
- Dependencies listed in `requirements.txt`

## Installation and Execution

You have two options for installing and running ROClab:

### Automatic Method (recommended)

This method automates the entire setup process and is the easiest way to get started.

1. Clone or download this repository:
   ```bash
   git clone https://github.com/mcquaas/ROClab.git
   cd ROClab
   ```

2. Give execution permissions to the installation script:
   ```bash
   chmod +x setup.sh
   ```

3. Run the installation script:
   ```bash
   ./setup.sh
   ```

The installation script will automatically perform the following tasks:
- Detect your operating system (macOS, Linux, Windows)
- Verify that Python is installed
- Create a virtual environment to isolate dependencies
- Install all necessary libraries
- Create a custom execution script for your system
- Offer you the option to start the application immediately

> **Note for users with special paths**: If your project path contains special characters such as colons (:), the script will create the virtual environment in your home directory instead of the project directory.

### Manual Method

If you prefer to manually configure the application, follow these steps:

1. Clone or download this repository:
   ```bash
   git clone https://github.com/mcquaas/ROClab.git
   cd ROClab
   ```

2. Create a virtual environment:
   ```bash
   # On systems with normal paths (without special characters)
   python3 -m venv .venv
   
   # On systems with paths containing special characters
   python3 -m venv ~/roclab_venv
   ```

3. Activate the virtual environment:
   ```bash
   # On macOS/Linux for local .venv
   source .venv/bin/activate
   
   # On macOS/Linux for environment in home directory
   source ~/roclab_venv/bin/activate
   
   # On Windows for local .venv
   .venv\Scripts\activate
   
   # On Windows for environment in home directory
   %USERPROFILE%\roclab_venv\Scripts\activate
   ```

4. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

5. Run the application:
   ```bash
   streamlit run ROClab.py
   ```

## Usage

1. Once the application is started, open your browser at the address indicated by Streamlit (usually http://localhost:8501)
2. Upload a CSV file with structured data
3. Select the target variable and predictive variables
4. Explore the data and train different models
5. Analyze performance and adjust parameters as needed

## Troubleshooting Common Issues

### Error Creating Virtual Environment

If you encounter an error like `Error: Refusing to create a venv in ... because it contains the PATH separator :`, it means your directory path contains special characters. Try:

1. Using the automatic `setup.sh` script which will detect this problem and create the environment in your home directory
2. Or manually create the virtual environment in your home directory:
   ```bash
   python3 -m venv ~/roclab_venv
   source ~/roclab_venv/bin/activate
   ```

### Modules Not Found

If you receive errors like `ModuleNotFoundError: No module named 'xgboost'` or other modules, run:

```bash
pip install -r requirements.txt
```

### Error in Language JSON File

If you see an error like `json.decoder.JSONDecodeError`, there may be a problem with the language files. Verify that the `lang_es.json` and `lang_en.json` files are correctly formatted. They are valid JSON files that should not contain syntax errors.

### Language Change

- To change the language through the interface: use the selector in the top-right corner
- To change the language through the URL: add `?lang=en` or `?lang=es` to the end of the URL

### Watchdog Not Installed

If you see a warning about installing Watchdog for better performance, you can follow the instructions:

```bash
xcode-select --install  # Only on macOS
pip install watchdog
```

## Features

- **Multilingual**: You can switch between Spanish and English with the selector at the top or using the URL parameter: `?lang=en` or `?lang=es`.
- **Advanced Visualization**: Includes correlation matrices, distributions, ROC curves, and confusion matrices.
- **Multiple Models**: Logistic Regression, KNN, Naive Bayes, SVM, Decision Tree, Random Forest, XGBoost, Gradient Boosting, AdaBoost, LightGBM.
- **Performance Metrics**: Accuracy, precision, recall, F1-score, AUC, sensitivity, specificity.

## License

This project is licensed under GNU General Public License v3.0 - see the `LICENSE` file for more details.

## Contact

Gustavo Ross - gross@funsalud.org.mx 