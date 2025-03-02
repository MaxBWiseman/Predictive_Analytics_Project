import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import plotly.express as px
import src.data_management as dm


def page_cluster_body():


    cluster_pipe = dm.load_pkl_file(
        "src/cluster_perm/final_cluster_model.pkl")
    cluster_silhouette = plt.imread(
        "src/cluster_perm/silhouette_plot_6_clusters.png")
    important_features = plt.imread(
        "src/cluster_perm/feature_importance.png")
    cluster_profile = pd.read_csv(
        "src/cluster_perm/df_clusters_profile.csv")
    cluster_final_df = (pd.read_csv("src/cluster_perm/final_cluster_data.csv"))
    final_model = dm.load_pkl_file("src/cluster_perm/final_cluster_model.pkl")
    cluster_features = final_model['preprocessor'].transformers_[0][1].get_feature_names_out()
    
    if st.checkbox("Inspect cleaned clustering data"):
        st.write(cluster_final_df)                



    st.write("### ML Pipeline: Cluster Analysis")
    # display pipeline training summary conclusions
    st.info(
        "* The pipeline was tested with multiple variations of cluster sizes between 2 and 10.\n"
        "* The pipeline was fitted with a cluster size of 6, which was one of the best silhouette scores.\n"
        "* The pipeline average silhouette score is 0.6"
    )
    st.write("---")

    st.write("#### Cluster ML Pipeline steps")
    st.write(cluster_pipe)

    st.write("#### The features the model was trained with")
    st.write(cluster_features)

    st.write("#### Clusters Silhouette Plot")
    st.image(cluster_silhouette)

    st.write("#### Cluster Distribution per Activity Level and Stress Level")
    dm.cluster_distribution_per_variable(df=cluster_final_df, target='Activity Level')
    
    dm.cluster_distribution_per_variable(df=cluster_final_df, target='Stress Level')
    st.write("Activity Level and Stress Level are the most important features for the clustering model")
    st.write("The 3rd most important feature is Blood Oxygen Level, although its values are very dense, so it doesnt plot well.")
    st.write("I would have liked time to improve this, but I ran out of time.")
    

    # Displaying Cluster Profile Section
    st.write("#### Cluster Profile")

    # Defining the cluster profile statement
    statement = """
    * Overall, the best way to segment individuals with smartwatch health data is by their activity level and stress level.
    * The cluster profile interpretation allowed us to group the individuals in the following fashion:

    **Cluster 0:**  
    *Stress Levels:* 7, 6
    *Description:* This cluster shows a high concentration of individuals with stress levels 7 and 6, indicative of moderate to high stress. Tailored advertisements could include:
    - Relaxation programs or mindfulness apps
    - Fitness trackers with stress management features
    - Calming teas, aromatherapy, or stress-relief supplements

    **Cluster 1:**  
    *Stress Levels:* 3, 2, 1  
    *Description:* Predominantly lower stress levels in this cluster. Campaigns could focus on:
    - Wellness programs for proactive stress prevention
    - Leisure activities such as yoga retreats or hobbies
    - Low-stress lifestyle products like fitness gadgets or time-management tools

    **Cluster 2:**  
    *Stress Levels:* 8, 9, 10  
    *Description:* Represents the highest stress levels. Suitable advertising includes:
    - Intensive stress management solutions, e.g., therapy apps or counseling
    - Premium wellness services tailored to high-stress individuals
    - Tools like guided meditation, sleep aids, or personalized coaching

    **Cluster 3:**  
    *Stress Levels:* 5, 4
    *Description:* Mid-level stress is prevalent. Suggested advertising:
    - General stress-relief products such as work-life balance courses
    - Items enhancing productivity or mental clarity, like ergonomic tools
    - Sports and outdoor activities to reduce stress actively

    **Cluster 4:**  
    *Stress Levels:* 6, 7
    *Description:* A mix of moderate and low stress levels. Campaign ideas include:
    - A combination of stress management and relaxation techniques
    - Incentives for healthy habits, such as wellness program memberships
    - Wearables with comprehensive health tracking features

    **Cluster 5:**  
    *Stress Levels:* 4, 3, 5  
    *Description:* A broad range of stress levels, from low (3) to mid (5). Campaigns could highlight:
    - Personalized wellness approaches for varying stress intensities
    - Solutions for extreme stress, such as intensive yoga or stress alarms
    - Supplements or vitamins targeting overall mental well-being

    ### Summary of Strategy:
    By segmenting users into these clusters, advertisers can create tailored campaigns to meet the specific needs of each stress group:
    - **High Stress Levels (e.g., 7, 8, 9):** Intensive stress-relief solutions or premium health services
    - **Moderate Stress Levels (e.g., 4, 5, 6):** General wellness and balance-focused campaigns
    - **Low Stress Levels (e.g., 1, 2, 3):** Proactive lifestyle programs and relaxation activities
    """
    st.write(statement)
    
    # hack to not display the index in st.table() or st.write()
    cluster_profile.index = [" "] * len(cluster_profile)
    st.table(cluster_profile)
