import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
from src.data_management import load_pkl_file
import pickle
from src.machine_learning.evaluate_clf import clf_performance


def page_predict_stress_level():
    
    # Load pre-trained model
    def load_pkl_file(file_path):
        with open(file_path, "rb") as file:
            return pickle.load(file)

    classification_model = load_pkl_file("src/predict_stress_level/discretized_smartwatch_health_data_model.pkl")

    # File input for the model
    st.sidebar.title("Model File Input")
    uploaded_file = st.sidebar.file_uploader("Upload smartwatch health data", type=["pkl"])

    if uploaded_file is not None:
        # Load and preprocess the uploaded file
        try:
            data = pickle.load(uploaded_file)
            st.success("Best Model File Uploaded Successfully!")
            
            # Example: Use the model for prediction
            predictions = classification_model.predict(data)  # Ensure 'data' is formatted correctly
            st.write("Predictions:", predictions)
        except Exception as e:
            st.error(f"Error processing uploaded file: {e}")


    # Checkbox to view best_model_data
    view_data = st.sidebar.checkbox("View the Best Model Data")

    if view_data:
        best_model_data = pd.read_csv("src/predict_stress_level/discretized_smartwatch_health_data.csv")
        st.write("### Best Model Data")
        st.dataframe(best_model_data)

    # Display classification report
    st.write("### Classification Report")
    with open("src/predict_stress_level/classification_report.txt", "r") as file:
        classification_report = file.read()
    st.text(classification_report)
    
    st.write("### Classification Model Performance")
    classification_performance = plt.imread("src/predict_stress_level/classification_performance_testset.png")
    st.image(classification_performance, caption="Classification Performance Test Set")

    # Layout the images in sections
    st.write("### Feature Importances")
    feat_import_class = plt.imread("src/predict_stress_level/feature_importance_classification.png")
    feat_import_reg = plt.imread("src/predict_stress_level/feature_importance_regression.png")
    st.image(feat_import_class, caption="Feature Importance - Classification")
    st.image(feat_import_reg, caption="Feature Importance - Regression")

    st.write("### PCA Analysis")
    feat_import_pca = plt.imread("src/predict_stress_level/pca_components_analysis.png")
    pca_performance = plt.imread("src/predict_stress_level/pca_performance.png")
    st.image(feat_import_pca, caption="PCA Components Analysis")
    st.image(pca_performance, caption="PCA Performance")

    st.write("### Regression Model Performance")
    regression_model_plot = plt.imread("src/predict_stress_level/regression_evaluation_plots.png")
    regression_performance = plt.imread("src/predict_stress_level/regression_performance.png")
    st.image(regression_model_plot, caption="Regression Evaluation Plots")
    st.image(regression_performance, caption="Regression Performance")


