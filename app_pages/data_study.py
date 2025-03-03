import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import numpy as np
import pandas as pd
import streamlit as st
from src.data_management import load_data_study
from feature_engine.encoding import OneHotEncoder
from feature_engine.discretisation import ArbitraryDiscretiser
import ppscore as pps


def heatmap_corr(data, threshold, figsize=(12, 12), annot_size=6):
    # Create the mask for the upper diagonal and
    # show only values greater than the threshold
    mask = np.zeros_like(data, dtype=bool)
    mask[np.triu_indices_from(mask)] = True
    mask[abs(data) < threshold] = True

    # Plot the heatmap
    fig, axes = plt.subplots(figsize=figsize)
    sns.heatmap(data=data, annot=True, xticklabels=True, yticklabels=True,
                mask=mask, cmap='viridis',
                annot_kws={"size": annot_size}, ax=axes, linewidth=0.5)
    plt.ylim(len(data.columns), 0)
    st.pyplot(fig)


def heatmap_pps(df, threshold, figsize=(8, 8), font_annot=10):
    if len(df.columns) > 1:
        mask = np.zeros_like(df, dtype=bool)
        mask[abs(df) < threshold] = True

        fig, ax = plt.subplots(figsize=figsize)
        sns.heatmap(df, annot=True, annot_kws={"size": font_annot},
                    mask=mask, cmap='rocket_r', linewidth=0.05,
                    linecolor='lightgrey')
        plt.ylim(len(df.columns), 0)
        st.pyplot(fig)


def parallel_plot(df, columns):
    # Define the bin edges and labels
    # Define custom bins for 'Step Count'
    bin_edges = [0, 3524, 7048, 10572, 14096, 17620, float('inf')]
    bin_labels = ['<3524', '3525-7048', '7049-10572',
                  '10573-14096', '14097-17620', '20000+']

    # Define custom bins for 'Stress Level'
    bin_edges2 = [0, 3, 5, 7, 10]
    bin_labels2 = ['Very Low', 'Low', 'Moderate', 'High']

    # Define Custom bins for 'Sleep Duration (hours)'
    # sleep duration is between 2 and 10 hours
    bin_edges3 = [2, 4, 6, 8, 11]
    bin_labels3 = ['2-4', '4-6', '6-8', '8-11']

    # Define custom bins for Heart Rate (BPM)
    # heart rate is between 40 and 114
    bin_edges4 = [39, 60, 80, 100, 114]
    bin_labels4 = ['39-60', '60-80', '80-100', '100-114']

    # Define custom bins for Blood Oxygen Level (%)
    # blood oxygen level is between 92 and 100
    bin_edges5 = [92, 94, 96, 98, 100]
    bin_labels5 = ['92-94', '94-96', '96-98', '98-100']

    # Initialize the ArbitraryDiscretiser with custom bins
    discretizer = ArbitraryDiscretiser(
        binning_dict={'Step Count': bin_edges, 'Stress Level': bin_edges2,
                      'Sleep Duration (hours)': bin_edges3,
                      'Heart Rate (BPM)': bin_edges4,
                      'Blood Oxygen Level (%)': bin_edges5}, return_object=True)

    # Fit and transform the data
    df_discretized = discretizer.fit_transform(df)

    # Map the bin labels to the discretized columns
    df_discretized['Step Count'] = pd.cut(
        df['Step Count'], bins=bin_edges, labels=bin_labels,
        include_lowest=True)
    df_discretized['Stress Level'] = pd.cut(
        df['Stress Level'], bins=bin_edges2, labels=bin_labels2,
        include_lowest=True)
    df_discretized['Sleep Duration (hours)'] = pd.cut(
        df['Sleep Duration (hours)'], bins=bin_edges3, labels=bin_labels3,
        include_lowest=True)
    df_discretized['Heart Rate (BPM)'] = pd.cut(
        df['Heart Rate (BPM)'], bins=bin_edges4, labels=bin_labels4,
        include_lowest=True)
    df_discretized['Blood Oxygen Level (%)'] = pd.cut(
        df['Blood Oxygen Level (%)'], bins=bin_edges5, labels=bin_labels5,
        include_lowest=True)

    # Convert columns to categorical
    for column in columns:
        df_discretized[column] = df_discretized[column].astype('category')

    # Plot parallel categories chart
    fig = px.parallel_categories(df_discretized, dimensions=columns)
    st.plotly_chart(fig)

# Define the Streamlit page function


def page_study_body():
    st.title("Page Study Body")

    df_unclean = pd.read_csv("src/unclean_smartwatch_health_data.csv")
    st.write("Uncleaned Data:")
    st.write(df_unclean.head())

    df_clean = load_data_study()
    st.write("Cleaned Data:")
    st.write(df_clean.head())

    # Ensure 'Activity Level' is categorical
    if 'Activity Level' in df_clean.columns:
        df_clean['Activity Level'] = df_clean['Activity Level'].astype(
            'category')

    # Apply one-hot encoding
    encoder = OneHotEncoder(variables=["Activity Level"])
    df_encoded = encoder.fit_transform(df_clean.copy())
    st.write("Encoded Data:")
    st.write(df_encoded.head())

    # View the parallel plot
    columns_to_parallel = [
        'Step Count', 'Sleep Duration (hours)',
        'Stress Level', 'Activity Level']
    st.write("Parallel Plot:")
    parallel_plot(df_clean, columns_to_parallel)

    # Calculate the correlation matrix
    corr_matrix = df_encoded.corr(method='spearman')

    if st.checkbox("Show correlation heatmap"):
        heatmap_corr(data=corr_matrix, threshold=0.0)

    if st.checkbox("Show density plot montage"):
        density_plot = plt.imread("src/density_plots_montage.png")
        st.image(density_plot)

    pps_matrix_raw = pps.matrix(df_encoded)
    pps_matrix = pps_matrix_raw.filter(['x', 'y', 'ppscore']).pivot(
        columns='x', index='y', values='ppscore')

    if st.checkbox("Show PPS correlations"):
        heatmap_pps(df=pps_matrix, threshold=0)


if __name__ == '__main__':
    page_study_body()
