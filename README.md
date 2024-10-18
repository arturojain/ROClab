# README for Didactiva AI Lab - ML Classifiers App

## Overview

This project provides a web-based application built using **Streamlit** to train and evaluate machine learning classification models. The app is designed as part of **Module 6 of the AI in Health Diploma**, aimed at teaching students how to handle structured numerical data for classification tasks, evaluate performance, and optimize model parameters.

## Features

1. **Data Upload**: Users can upload structured CSV files containing features and a target column for classification.
2. **Exploratory Data Analysis**: Users can explore the uploaded data, view statistical summaries, correlations, and generate visualizations.
3. **Data Preprocessing**: Options to handle missing values, normalize data, or apply outlier detection.
4. **Model Training**: Users can select from a wide variety of machine learning classifiers, adjust model hyperparameters, and train the model on the uploaded dataset.
5. **Model Evaluation**: After training, the app provides evaluation metrics such as ROC curve, confusion matrix, accuracy, precision, recall, F1-score, and other relevant metrics.

## Example Datasets

You can use the following datasets to practice with this app:

1. **[Cardiovascular Disease Dataset](https://www.kaggle.com/datasets/jocelyndumlao/cardiovascular-disease-dataset?resource=download)**:
   - This dataset contains cardiovascular disease data. It can be used to predict the presence of heart-related diseases based on features like age, cholesterol levels, and blood pressure.
   
2. **[Pima Indians Diabetes Dataset](https://www.kaggle.com/datasets/uciml/pima-indians-diabetes-database)**:
   - This dataset is used for diabetes prediction and contains health metrics of Pima Indians. Features include glucose levels, BMI, and family history.
   
3. **[Breast Cancer Prediction Dataset](https://www.kaggle.com/datasets/merishnasuwal/breast-cancer-prediction-dataset)**:
   - This dataset contains data for breast cancer prediction, which can be used to classify whether a tumor is malignant or benign based on features like the size and texture of cell nuclei.

## Dependencies

The following Python libraries are used in this project:

```bash
streamlit
pandas
numpy
scikit-learn
xgboost
lightgbm
matplotlib
seaborn
```


## Setting Up a Virtual Environment (Optional)

To create a virtual environment (venv) for your project, follow these steps:

1. **Navigate to your project directory**:
   ```bash
   cd /path/to/your/project
   ```

2. **Create a virtual environment**:
   ```bash
   python3 -m venv venv
   ```

3. **Activate the virtual environment**:

   - On **Linux/macOS**:
     ```bash
     source venv/bin/activate
     ```
   - On **Windows**:
     ```bash
     .\venv\Scripts\activate
     ```

4. **Install the required dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

5. **Deactivate the environment** when done:
   ```bash
   deactivate
   ```

Using a virtual environment ensures that your project's dependencies are isolated and do not affect system-wide packages.


## Installation without venv

1. Clone the repository:

```bash
git clone https://github.com/didactiva-ai-lab/ml-classifiers-app.git
cd ml-classifiers-app
```

2. Install the required dependencies:

```bash
pip install -r requirements.txt
```

3. Run the application:

```bash
streamlit run app.py
```

## Instructions for Use

1. **Upload Data**: Load a CSV file that includes a target column for classification and the features.
2. **Exploratory Data Analysis**: 
   - Explore variable distributions.
   - View correlation heatmaps.
   - Generate pair plots for visual insights.
3. **Data Preparation**:
   - Choose how to handle missing values (drop or impute).
   - Normalize data if needed.
   - Select relevant features for the model.
4. **Train Models**: Choose from various classification models such as Logistic Regression, Random Forest, XGBoost, and more.
5. **Model Evaluation**:
   - Visualize the ROC curve.
   - Adjust the decision threshold.
   - View the confusion matrix and performance metrics.

## Available Models

The following classification algorithms are available in the app:

- Logistic Regression
- K-Nearest Neighbors (KNN)
- Naive Bayes
- Support Vector Machines (SVM)
- Decision Tree
- Random Forest
- XGBoost
- Gradient Boosting
- AdaBoost
- LightGBM

## Author

This application was developed by **Gustavo Ross** for the **AI in Health Diploma** hosted by **FUNSALUD** and **Didactiva**.

For any feedback or inquiries, feel free to contact:
- Email: gross@funsalud.org.mx

---

This application is for **educational purposes only** and should not be used for medical decision-making.
