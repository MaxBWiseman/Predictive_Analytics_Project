from sklearn.metrics import classification_report, confusion_matrix
import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import pickle
import sys
import os
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.feature_selection import SelectFromModel
from feature_engine.discretisation import EqualWidthDiscretiser
import seaborn as sns

# Add the src directory to the Python path
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'src'))
import data_management as dm

def confusion_matrix_and_report(X, y, pipeline, label_map):
    prediction = pipeline.predict(X)
    
    # Ensure labels are strings
    y = y.astype(str)
    prediction = pd.Series(prediction).astype(str)
    
    # Confusion Matrix
    st.write('---  Confusion Matrix  ---')
    cm = confusion_matrix(y_true=y, y_pred=prediction)
    cm_df = pd.DataFrame(cm, 
                         columns=[f"Actual {label_map[n]}" for n in range(len(label_map))],
                         index=[f"Predicted {label_map[n]}" for n in range(len(label_map))])
    st.dataframe(cm_df)  # Use st.dataframe for better rendering in Streamlit
    
    # Classification Report
    st.write('---  Classification Report  ---')
    report = classification_report(y, prediction, target_names=label_map, output_dict=True)
    report_df = pd.DataFrame(report).transpose()
    st.dataframe(report_df)  # Show the classification report as a DataFrame

def clf_performance(X_train, y_train, X_test, y_test, pipeline, label_map):
    st.write("#### Train Set ####")
    confusion_matrix_and_report(X_train, y_train, pipeline, label_map)

    st.write("#### Test Set ####")
    confusion_matrix_and_report(X_test, y_test, pipeline, label_map)


def FullDiscretPipelineFinal():
    best_pipeline = Pipeline([
        ('data_type_transformer', dm.DataTypeTransformer()),
        ('knn_imputer', dm.KNNImputerTransformer(n_neighbors=3)),
        ('winsorizer_transformer', dm.WinsorizerTransformer()),
        ('categorical_imputer', dm.CategoricalImputer()),
        ('category_corrector', dm.CategoryCorrector()),
        ('float_to_int', dm.FloatToInt()),
        ('data_smoother', dm.DataSmoother(k=3)),
        ('outlier_trimmer_transformer', dm.OutlierTrimmerTransformer()),
        ('encoder', dm.OrdinalEncoder(encoding_method='arbitrary', variables=["Activity Level"])),
        ('minmax', dm.DataFrameScaler(exclude_columns=[])),
    ])
    return best_pipeline


def DiscretPipelineModel():
    pipeline = Pipeline([
        ('feature_selection', SelectFromModel(estimator=ExtraTreesClassifier())),
        ('model',  ExtraTreesClassifier())
    ])
    return pipeline


def predict_step_count_task():
    st.title("Smartwatch Health Data Step Count Discretization and Prediction")

    # Initialize session state for the checkbox
    if 'use_inbuilt_data' not in st.session_state:
        st.session_state.use_inbuilt_data = False
    
    # Upload file
    uploaded_file = st.file_uploader(
        "Upload your dataset (CSV format)", type=["csv"])

    st.write("Or select here to use the in-built dataset:")
    use_inbuilt_data = st.checkbox("Use in-built dataset")
    
    if use_inbuilt_data:
        df = pd.read_csv("src/unclean_smartwatch_health_data.csv")
        st.write("In-built Dataset:")
        st.dataframe(df.head())
        uploaded_file = df
    else:
        st.session_state.use_inbuilt_data = False

    if uploaded_file is not None:
        # Read dataset
        if isinstance(uploaded_file, pd.DataFrame):
            df = uploaded_file
        else:
            df = pd.read_csv(uploaded_file)

        df.drop("User ID", axis=1, inplace=True)

        st.write("Uploaded Dataset:")
        st.dataframe(df.head())

        # placeholder
        placeholder = st.empty()
        placeholder.write("### Please wait for the model to be trained and predictions to be made...")

        best_pipeline = FullDiscretPipelineFinal()
        df_discret = best_pipeline.fit_transform(df)

        # Discretize 'Step Count'
        disc = EqualWidthDiscretiser(bins=7, variables=['Step Count'])
        df_clf = disc.fit_transform(df_discret)

        label_names = ['<3524', '3525-7048', '7049-10572', '10573-14096', '14097-17620', '17621-20000', '20000+']

        st.write(f"* The classes represent the following ranges: \n{label_names} \n")

        plt.figure(figsize=(10, 6))
        sns.countplot(data=df_clf, x='Step Count')
        plt.xticks(ticks=range(7), labels=label_names, rotation=45)
        st.pyplot(plt)

        df_discret["Step Count"] = df_clf["Step Count"]

        X_train, X_test, y_train, y_test = train_test_split(df_discret.drop(columns=['Step Count']),
                                                            df_discret['Step Count'],
                                                            test_size=0.2,
                                                            random_state=42)

        st.write("* Train set:", X_train.shape, y_train.shape,
              "\n* Test set:",  X_test.shape, y_test.shape)

        placeholder.empty()
            
        best_pipeline_modeler = DiscretPipelineModel()
        best_pipeline_modeler.fit(X_train, y_train)

        predictions = best_pipeline_modeler.predict(X_test)
        label_map = ['<3524', '3525-7048', '7049-10572', '10573-14096', '14097-17620', '17621-20000', '20000+']
        predictions = pd.Series(predictions).astype(str)

        st.write("Prediction Results:")
        results_df = X_test.copy()
        results_df["Predicted Step Count"] = predictions
        st.dataframe(results_df)

        # Save the model as pkl
        if not os.path.exists('outputs/predict_step_count'):
            os.makedirs('outputs/predict_step_count')
        with open("outputs/predict_step_count/discretized_smartwatch_health_data_model.pkl", "wb") as file:
            pickle.dump(best_pipeline_modeler, file)

        # Download results
        st.write("Download processed data:")
        processed_file = results_df.to_csv(index=False).encode('utf-8')
        st.download_button("Download CSV", data=processed_file,
                           file_name="processed_data.csv")

        y_train = y_train.astype(str)
        y_test = y_test.astype(str)
    
        clf_performance(X_train=X_train, y_train=y_train.values.ravel(),
                        X_test=X_test, y_test=y_test.values.ravel(),
                        pipeline=best_pipeline_modeler,
                        label_map=label_map
                        )
        
        columns_after_data_cleaning_feat_eng = df_discret.columns.to_list()
        st.write(f"* The pipeline has {len(columns_after_data_cleaning_feat_eng)} features after data cleaning and feature engineering.")

        # Ensure the indices are within bounds
        support = best_pipeline_modeler.named_steps['feature_selection'].get_support()
        best_features = [columns_after_data_cleaning_feat_eng[i] for i in range(len(support)) if support[i]]

        # create DataFrame to display feature importance
        df_feature_importance = (pd.DataFrame(data={
            'Feature': best_features,
            'Importance': best_pipeline_modeler['model'].feature_importances_})
            .sort_values(by='Importance', ascending=False)
        )

        st.write(f"* These are the {len(best_features)} most important features in descending order. "
            f"The model was trained on them: \n{df_feature_importance['Feature'].to_list()}")

        fig, ax = plt.subplots()
        df_feature_importance.plot(kind='bar', x='Feature', y='Importance', ax=ax)
        plt.title('Feature Importance')
        plt.tight_layout()
        st.pyplot(fig)