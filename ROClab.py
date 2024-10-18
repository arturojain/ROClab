import streamlit as st
import pandas as pd
import numpy as np
import json
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

# Load language files based on user selection
def load_lang(lang_code):
    with open(f'lang_{lang_code}.json', 'r') as lang_file:
        return json.load(lang_file)

# Radio button for language selection
lang_code = st.radio("Choose language", ['es', 'en'], index=0)
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
    return pd.read_csv(file)

header1, header2, header3 = st.columns([2, 3, 2])
with header1:
    st.image('https://www.didactiva.com/wp-content/uploads/2024/06/logo-didactiva-20.png', width=200)
with header2:
    st.title(lang['title'])
    st.write(lang['subtitle'])
with header3:
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
        # Load the CSV file
        st.subheader(lang['upload_title'], help=lang['upload_help'])
        uploaded_file = st.file_uploader(lang['upload_prompt'], type='csv')
        if uploaded_file:
            df = load_data(uploaded_file)

            # Show a preview of the data
            with st.expander(f"{lang['file_contents']} {len(df)} {lang['records_loaded']} {len(df.columns)} {lang['and']} {lang['columns_loaded']}"):
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
        if 'df' in locals():
            
            st.subheader(lang['data_preparation_title'], help=lang['data_preparation_help'])
            # Prepare the data
            features = df[selected_columns]
            target = df[target_column]
            
            if df.isnull().sum().sum() > 0:                
                # Handle missing values
                handle_missing = st.selectbox(lang['handle_missing'], ["None", "Delete", "Impute (Mean)"])
                if handle_missing == "Delete":
                    df = df.dropna()
                    features = df[selected_columns]
                    target = df[target_column]
                elif handle_missing == "Impute (Mean)":
                    imputer = SimpleImputer(strategy='mean')
                    features = imputer.fit_transform(features)
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

            # Horizontal stacked bar with blue and green colors
            ax.barh('Samples', sample_counts[lang['training']], color='blue', label=f"{lang['training']}: {sample_counts[lang['training']]}")
            ax.barh('Samples', sample_counts[lang['testing']], left=sample_counts[lang['training']], color='green', label=f"{lang['testing']}: {sample_counts[lang['testing']]}")

            # Adding labels to each section
            ax.text(sample_counts[lang['training']] / 2, 0, f"{lang['training']}: {sample_counts[lang['training']]}", va='center', ha='center', color='white')
            ax.text(sample_counts[lang['training']] + sample_counts[lang['testing']] / 2, 0, f"{lang['testing']}: {sample_counts[lang['testing']]}", va='center', ha='center', color='white')

            # Remove axes
            ax.axis('off')

            # Show plot in Streamlit
            st.pyplot(fig)



            n_splits = st.slider(lang['num_partitions'], 2, 10, 5, help=lang['num_partitions_help'])

            # Split the data into training and test sets
            X_train, X_test, y_train, y_test = train_test_split(features, target, test_size=test_size, random_state=42)
            
            print("X_train, X_test, y_train, y_test")
            print (X_train.shape, X_test.shape, y_train.shape, y_test.shape)
            
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
            if normalize:
                normalizer = Normalizer()
                X_train = normalizer.fit_transform(X_train)
                X_test = normalizer.transform(X_test)

            # Train the model
            model.fit(X_train, y_train)

            # Predictions
            if hasattr(model, "predict_proba"):
                y_pred_proba = model.predict_proba(X_test)
            else:
                y_pred_proba = model.decision_function(X_test)
                y_pred_proba = (y_pred_proba - y_pred_proba.min()) / (y_pred_proba.max() - y_pred_proba.min())

            # Predictions adjusted for the default threshold (0.5)
            y_pred_adjusted = (y_pred_proba[:, 1] >= 0.5).astype(int)

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

                    ax.barh(top_features, top_importances)
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
            ax.plot(fpr, tpr, label=f"{lang['roc_curve']} (area = {roc_auc:.2f})")
            ax.plot([0, 1], [0, 1], linestyle='--', color='gray')
            ax.set_xlabel(lang['fpr_label'])
            ax.set_ylabel(lang['tpr_label'])
            ax.set_title(lang['roc_curve'])
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
            sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax)
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
