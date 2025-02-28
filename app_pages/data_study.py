import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import numpy as np
import streamlit as st
from src.data_management import load_data
from feature_engine.encoding import OneHotEncoder
from feature_engine.discretisation import ArbitraryDiscretiser
import ppscore as pps

def heatmap_corr(data, threshold, figsize=(12, 12), annot_size=6):
    # Create the mask for the upper diagonal and show only values greater than the threshold
    mask = np.zeros_like(data, dtype=bool)
    mask[np.triu_indices_from(mask)] = True
    mask[abs(data) < threshold] = True

    # Plot the heatmap
    fig, axes = plt.subplots(figsize=figsize)
    sns.heatmap(data=data, annot=True, xticklabels=True, yticklabels=True,
                mask=mask, cmap='viridis', annot_kws={"size": annot_size}, ax=axes,
                linewidth=0.5)
    plt.ylim(len(data.columns), 0)
    plt.show()

def density_plot_montage(df, columns):
    fig, axes = plt.subplots(len(columns), len(columns), figsize=(20, 20))
    fig.suptitle('Density Plots Montage', fontsize=20)

    for i, col1 in enumerate(columns):
        for j, col2 in enumerate(columns):
            if i > j:
                sns.kdeplot(
                    x=df[col2], y=df[col1],
                    cmap='viridis', fill=True,
                    ax=axes[i, j]
                )
                axes[i, j].set_xlabel(col2)
                axes[i, j].set_ylabel(col1)
            else:
                axes[i, j].axis('off')
    
    plt.tight_layout(rect=[0, 0, 1, 0.96])
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
        plt.show()

def parallel_plot(df, columns):
    # Define the bin edges and labels
    bin_edges = [0, 1000, 5000, 10000, 20000]
    bin_labels = ['Very Low', 'Low', 'Moderate', 'High', 'Very High']
    bin_edges2 = [0, 2, 4, 6, 8]
    bin_labels2 = ['Very Low', 'Low', 'Moderate', 'High', 'Very High']
    bin_edges3 = [0, 3, 5, 7, 9]
    bin_labels3 = ['Very Short', 'Short', 'Adequate', 'Long', 'Very Long']
    bin_edges4 = [30, 60, 90, 120, 150]
    bin_labels4 = ['Very Low', 'Low', 'Moderate', 'High', 'Very High']
    bin_edges5 = [80, 85, 90, 95, 100]
    bin_labels5 = ['Low', 'Below Normal', 'Normal', 'Above Normal', 'High']
    
    # Initialize the ArbitraryDiscretiser with custom bins
    discretizer = ArbitraryDiscretiser(
        binning_dict={'Step Count': bin_edges, 'Stress Level': bin_edges2,
                      'Sleep Duration (hours)': bin_edges3,
                      'Heart Rate (BPM)': bin_edges4,
                      'Blood Oxygen Level (%)': bin_edges5}, return_object=True)

    # Fit and transform the data
    df_discretized = discretizer.fit_transform(df)

    # Map the bin labels to the discretized columns
    df_discretized['Step Count'] = pd.cut(df['Step Count'], bins=bin_edges, labels=bin_labels, include_lowest=True)
    df_discretized['Stress Level'] = pd.cut(df['Stress Level'], bins=bin_edges2, labels=bin_labels2, include_lowest=True)
    df_discretized['Sleep Duration (hours)'] = pd.cut(df['Sleep Duration (hours)'], bins=bin_edges3, labels=bin_labels3, include_lowest=True)
    df_discretized['Heart Rate (BPM)'] = pd.cut(df['Heart Rate (BPM)'], bins=bin_edges4, labels=bin_labels4, include_lowest=True)
    df_discretized['Blood Oxygen Level (%)'] = pd.cut(df['Blood Oxygen Level (%)'], bins=bin_edges5, labels=bin_labels5, include_lowest=True)

    # Plot parallel categories chart
    fig = px.parallel_categories(df_discretized, dimensions=columns)
    return fig

# Define the Streamlit page function
def page_study_body():
    st.title("Page Study Body")

    # Load data
    df = load_data()
    st.write("Original Data:")
    st.write(df)
    
    # Apply one-hot encoding
    encoder = OneHotEncoder(variables=["Activity Level"])
    df_encoded = encoder.fit_transform(df)
    st.write("Encoded Data:")
    st.write(df_encoded)
    
    # Calculate the correlation matrix
    corr_matrix = df_encoded.corr(method='spearman')
    
    if st.checkbox("Show correlation heatmap"):
        heatmap_corr(data=corr_matrix, threshold=0.0)
    
    columns = ['Heart Rate (BPM)', 'Blood Oxygen Level (%)', 'Step Count',
               'Sleep Duration (hours)', 'Stress Level',
               'Activity Level_Highly Active',
               'Activity Level_Active', 'Activity Level_Sedentary']
    
    if st.checkbox("Show density plot montage (will take about 3 minutes)"):
        density_plot_montage(df_encoded, columns)
    
    pps_matrix_raw = pps.matrix(df_encoded)
    pps_matrix = pps_matrix_raw.filter(['x', 'y', 'ppscore']).pivot(columns='x', index='y', values='ppscore')
    
    if st.checkbox("Show PPS correlations"):
        heatmap_pps(df=pps_matrix, threshold=0)
    
    columns_to_parallel = ['Step Count', 'Sleep Duration (hours)', 'Stress Level', 'Activity Level']
    
    if st.checkbox("Show parallel plot"):
        fig = parallel_plot(df_encoded, columns_to_parallel)
        st.plotly_chart(fig)

if __name__ == "__main__":
    page_study_body()


