from sklearn.metrics import classification_report, confusion_matrix
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.feature_selection import SelectFromModel
from sklearn.cluster import KMeans
from sklearn.model_selection import train_test_split
from sklearn.decomposition import PCA
from sklearn.pipeline import Pipeline
import src.data_management as dm
import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import plotly.express as px
import os
import pickle
import sys

# Add the src directory to the Python path
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))


RANDOM_STATE = 42


def cluster_distribution_per_variable(df, target):
    df_bar_plot = df.groupby(
        ['Clusters', target]).size().reset_index(name='Count')
    df_bar_plot.columns = ['Clusters', target, 'Count']
    df_bar_plot[target] = df_bar_plot[target].astype('object')

    print(f"Clusters distribution across {target} levels")
    fig = px.bar(df_bar_plot, x='Clusters', y='Count',
                 color=target, width=800, height=500)
    fig.update_layout(xaxis=dict(tickmode='array',
                      tickvals=df['Clusters'].unique()))
    st.plotly_chart(fig)

    df_relative = (df.groupby(
        ["Clusters", target]).size().unstack(fill_value=0)
        .apply(lambda x: 100 * x / x.sum(), axis=1).stack()
        .reset_index(name='Relative Percentage (%)')
        .sort_values(by=['Clusters', target]))

    print(f"Relative Percentage (%) of {target} in each cluster")
    fig = px.line(df_relative, x='Clusters', y='Relative Percentage (%)',
                  color=target, width=800, height=500)
    fig.update_layout(xaxis=dict(tickmode='array',
                      tickvals=df['Clusters'].unique()))
    fig.update_traces(mode='markers+lines')
    st.plotly_chart(fig)


def plot_feature_importance(df_feature_importance):
    fig, ax = plt.subplots()
    df_feature_importance.plot(
        kind='bar', x='Feature', y='Importance', legend=False, ax=ax)
    plt.xlabel("Features")
    plt.ylabel("Importance")

    for idx, row in df_feature_importance.iterrows():
        ax.text(idx, row['Importance'] + 0.01,
                row['Feature'], ha='center', va='bottom')

    st.pyplot(fig)
    plt.close(fig)


def generate_and_save_model(df):
    def PipelineCluster():
        pipeline_base = Pipeline([
            ('pca', PCA(n_components=3, random_state=RANDOM_STATE)),
            ('cluster', KMeans(n_clusters=6, random_state=RANDOM_STATE)),
        ])
        return pipeline_base

    def PipelineClassify():
        pipeline_final = Pipeline([
            ("feat_selection", SelectFromModel(
                GradientBoostingClassifier(random_state=RANDOM_STATE))),
            ("model", GradientBoostingClassifier(random_state=RANDOM_STATE)),
        ])
        return pipeline_final

    X = df.copy()

    pipeline_cluster = PipelineCluster()

    pipeline_cluster.fit(X)

    X['Clusters'] = pipeline_cluster.named_steps['cluster'].labels_

    X_train, X_test, y_train, y_test = train_test_split(
        X.drop(['Clusters'], axis=1),
        X['Clusters'],
        test_size=0.2,
        random_state=RANDOM_STATE
    )

    pipeline_classify = PipelineClassify()
    pipeline_classify.fit(X_train, y_train)

    y_train_pred = pipeline_classify.predict(X_train)
    y_test_pred = pipeline_classify.predict(X_test)

    st.subheader("Model Evaluation")
    st.text("Confusion Matrix Train:")
    st.write(confusion_matrix(y_train, y_train_pred))
    st.text("Classification Report Train:")
    st.write(classification_report(y_train, y_train_pred))
    st.text("Confusion Matrix Test:")
    st.write(confusion_matrix(y_test, y_test_pred))
    st.text("Classification Report Test:")
    st.write(classification_report(y_test, y_test_pred))

    pipeline_classify.fit(X_train, y_train)

    pipeline_classify.named_steps['feat_selection'].transform(X_train)

    columns_after_data_cleaning_feat_eng = X_train.columns[
        pipeline_classify.named_steps['feat_selection'].get_support()]

    df_feature_importance = pd.DataFrame({
        'Feature': columns_after_data_cleaning_feat_eng,
        'Importance': pipeline_classify.named_steps[
            'model'].feature_importances_
    }).sort_values(by='Importance', ascending=False)

    df_feature_importance.reset_index(drop=True, inplace=True)

    best_features = df_feature_importance['Feature'].to_list()

    st.write(
        f"* These are the {len(best_features)} most important"
        f" features in descending order. ")

    st.write(f"The model was trained on them: \n{best_features} \n")

    plot_feature_importance(df_feature_importance)

    if not os.path.exists('outputs/streamlit_outputs'):
        os.makedirs('outputs/streamlit_outputs')
    with open('outputs/streamlit_outputs/model.pkl', 'wb') as f:
        pickle.dump(pipeline_cluster, f)

    X.to_csv('outputs/streamlit_outputs/clustered_data.csv', index=False)

    st.success(
        "Model has been trained and"
        " saved as 'outputs/streamlit_outputs/model.pkl'.")


def page_cluster_body():
    df = pd.read_csv('outputs/streamlit_outputs/clustered_data.csv')
    cluster_pipe = dm.load_pkl_file("outputs/streamlit_outputs/model.pkl")
    pca_component_plot = plt.imread(
        "src/cluster_perm/pca_component_plot_cluster.png")
    cluster_profile = df["Clusters"].value_counts().reset_index()
    cluster_final_df = df

    if st.checkbox("Inspect cleaned clustering data"):
        st.write(cluster_final_df)

    st.write("### ML Pipeline: Cluster Analysis")
    st.info(
        "* The pipeline was tested with multiple variations"
        " of cluster sizes between 2 and 10.\n"
        "* The pipeline was fitted with a cluster size of 6,"
        " which was one of the best silhouette scores.\n"
        "* The pipeline average silhouette score is 0.4"
    )
    st.write("---")

    st.write("#### Cluster ML Pipeline steps")
    st.write(cluster_pipe)

    import numpy as np

    def AnalysisPipeline():
        clustering_pipeline = Pipeline([
            ('pca', PCA(n_components=3, random_state=RANDOM_STATE)),
            ('model', KMeans(n_clusters=6, random_state=RANDOM_STATE))
        ])
        return clustering_pipeline

    analysis_pipeline = AnalysisPipeline()
    df_analysis = analysis_pipeline.fit_transform(
        cluster_final_df.drop(['Clusters'], axis=1))

    st.write("#### Clusters Silhouette Plot")
    from yellowbrick.cluster import SilhouetteVisualizer
    from yellowbrick.cluster import KElbowVisualizer
    from matplotlib import rcParams
    rcParams['font.family'] = ['DejaVu Sans']

    print("=== Average Silhouette Score for different number of clusters ===")
    fig, ax = plt.subplots()
    visualizer = KElbowVisualizer(
        KMeans(random_state=42), k=(2, 7), metric='silhouette')
    visualizer.fit(df_analysis)
    visualizer.show()
    st.pyplot(fig)
    plt.close(fig)
    print("\n")

    for n_clusters in np.arange(start=2, stop=7):
        print(f"=== Silhouette plot for {n_clusters} Clusters ===")
        fig, ax = plt.subplots()
        visualizer = SilhouetteVisualizer(
            estimator=KMeans(n_clusters=n_clusters, random_state=RANDOM_STATE),
            colors='yellowbrick', ax=ax)
        visualizer.fit(df_analysis)
        st.pyplot(fig)
        plt.close(fig)
        print("\n")

    st.write(
        "Activity Level and Stress Level are the most"
        " important features for the clustering model")
    st.write("The 3rd most important feature is Blood Oxygen Level,"
             " although its values are very dense and only"
             " about 3-5% of the variable actually goes towards the"
             " end clustering, so it doesn't plot well.")

    st.write(
        "Activity Level and Stress Level are the most "
        "important features for the clustering model."
    )

    st.write(
        "The 3rd most important feature is Blood Oxygen Level, "
        "although its values are very dense and only about 3-5% "
        "of the variable actually goes towards the end clustering, "
        "so it doesn't plot well."
    )

    st.write("#### PCA Component Plot")
    st.image(pca_component_plot)

    st.write("### Insights from the PCA Scatter Plot")
    st.write(
        "1. **Cluster Structure**: The scatter plot shows data points "
        "clustered into six groups after applying PCA. Each cluster, "
        "numbered from 0 to 5, is represented by unique colors for easy "
        "distinction."
    )
    st.write(
        "2. **Cluster Centers**: The black crosses mark the centroids of "
        "the clusters, indicating the average position of each group in "
        "the reduced dimensional space."
    )
    st.write(
        "3. **PCA Components**: The x-axis represents PCA Component 0, "
        "while the y-axis represents PCA Component 1. These components "
        "capture the major variations in the dataset."
    )
    st.write(
        "4. **Distinct Groupings**: The visualization highlights clear "
        "separation between most clusters, suggesting good clustering "
        "performance."
    )
    st.write(
        "5. **Dimensionality Reduction**: PCA effectively reduced the "
        "dataset's dimensionality, while preserving its structure."
    )

    st.write("#### Cluster Distribution per Activity Level and Stress Level")
    df_cluster_vs1 = cluster_final_df.copy()
    cluster_distribution_per_variable(df=df_cluster_vs1, target='Stress Level')

    df_cluster_vs2 = cluster_final_df.copy()
    cluster_distribution_per_variable(
        df=df_cluster_vs2, target='Activity Level')

    st.write("#### Cluster Profile")
    statement = """
    * Overall, the best way to segment individuals with"
    " smartwatch health data is by their activity level and stress level.
    * The cluster profile interpretation allowed us to"
    " group the individuals in the following fashion:

    **Cluster 0:**
    *Stress Levels:* 4, 3, 2
    *Description:* This cluster shows a concentration of individuals
    with stress levels 4, 4 and 2, indicative of very low to low stress levels.
    Tailored advertisements could include:
    - Relaxation programs or mindfulness apps
    - Leisure activities such as yoga retreats or hobbies
    - Low-stress lifestyle products like fitness gadgets or
    time-management tools
    - Personalized wellness approaches for varying stress intensities
    - Supplements or vitamins targeting overall mental well-being
    - A combination of stress management and relaxation techniques
    - Incentives for healthy habits, such as wellness program memberships
    - Wearables with comprehensive health tracking features

    **Cluster 1:**
    *Stress Levels:* 7, 6
    *Description:* Predominantly higher stress levels in this cluster.
    Campaigns could focus on:
    - Wellness programs for proactive stress prevention
    - Fitness trackers with stress management features
    - Calming teas, aromatherapy, or stress-relief supplements

    **Cluster 2:**
    *Stress Levels:* 5, 7, 6
    *Description:* Represents the highest stress levels.
    Suitable advertising includes:
    - Stress-reduction workshops and webinars
    - Fitness programs integrating stress relief
    - Products like ergonomic office equipment to reduce stress

    **Cluster 3:**
    *Stress Levels:* 5, 4
    *Description:* Mid-level stress is prevalent. Suggested advertising:
    - General stress-relief products such as work-life balance courses
    - Items enhancing productivity or mental clarity, like ergonomic tools
    - Sports and outdoor activities to reduce stress actively

    **Cluster 4:**
    *Stress Levels:* 10, 9, 8
    *Description:* A mix of moderate and low stress levels.
    Campaign ideas include:
    - Intensive stress management solutions, e.g., therapy apps or counseling
    - Premium wellness services tailored to high-stress individuals
    - Tools like guided meditation, sleep aids, or personalized coaching

    **Cluster 5:**
    *Stress Levels:* 1, 2, 3
    *Description:* A low range of stress levels, from very low to low.
    Campaigns could highlight:
    - Relaxation programs or mindfulness apps
    - Leisure activities such as yoga retreats or hobbies
    - Low-stress lifestyle products like
    fitness gadgets or time-management tools
    - Solutions for low stress, such yoga or meditation classes
    - Supplements or vitamins targeting overall mental well-being

    ### Summary of Strategy:
    By segmenting users into these clusters, advertisers can create
    tailored campaigns to meet the specific needs of each stress group:
    - **High Stress Levels (e.g., 7, 8, 9):** Intensive stress-relief
    solutions or premium health services
    - **Moderate Stress Levels (e.g., 4, 5, 6):** General wellness and
    balance-focused campaigns
    - **Low Stress Levels (e.g., 1, 2, 3):** Proactive lifestyle programs
    and relaxation activities
    """
    st.write(statement)

    cluster_profile.columns = ["Cluster", "Count"]
    st.table(cluster_profile)


def cluster_task_start():
    st.title("Smartwatch Health Data Clustering")

    def PipelineCluster():
        pipeline_base = Pipeline([
            ('ordinal_encoder', dm.OrdinalEncoder(
                encoding_method='arbitrary', variables=["Activity Level"])),
            ('pca', PCA(n_components=3, random_state=RANDOM_STATE)),
            ('cluster', KMeans(n_clusters=6, random_state=RANDOM_STATE)),
        ])
        return pipeline_base

    def PipelineClassify():
        pipeline_final = Pipeline([
            ("feat_selection", SelectFromModel(
                GradientBoostingClassifier(random_state=RANDOM_STATE))),
            ("model", GradientBoostingClassifier(random_state=RANDOM_STATE)),
        ])
        return pipeline_final

    df = dm.load_data()

    generate_and_save_model(df)

    st.sidebar.title("Data File Input")
    uploaded_file = st.sidebar.file_uploader(
        "Upload smartwatch health data", type=["csv"])

    if uploaded_file is not None:
        try:
            data = pd.read_csv(uploaded_file)
            st.success("File Uploaded Successfully!")

            pipeline_cluster = PipelineCluster()
            df_final = pipeline_cluster.fit_transform(data.copy())
            df_final = pd.DataFrame(df_final, columns=data.columns)
            df_final['Clusters'] = pipeline_cluster.named_steps[
                'cluster'].labels_

            X_train, X_test, y_train, y_test = train_test_split(
                df_final.drop(['Clusters'], axis=1),
                df_final['Clusters'],
                test_size=0.2,
                random_state=RANDOM_STATE
            )
            cluster_predictions = PipelineClassify()
            cluster_predictions.fit(X_train, y_train)
            predictions = cluster_predictions.predict(X_test)
            st.write(confusion_matrix(y_test, predictions))
            st.write(classification_report(y_test, predictions))
            st.write("Predictions:", predictions)
        except Exception as e:
            st.error(f"Error processing uploaded file: {e}")

    page_cluster_body()
