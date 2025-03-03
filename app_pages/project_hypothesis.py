import streamlit as st

def page_project_hypothesis_body():
    # Title
    st.write("### Project Hypothesis and Validation")

    # Business Requirement 1
    st.success("## Business Requirement 1: Segment users into different groups based on health and activity data")
    st.info("### Rationale:")
    st.write("""
    * **Data Visualizations**:
        * Heatmaps: To visualize the correlation between different health metrics. This will help in understanding how variables like heart rate, activity level, and stress level interact with each other.
        * Density Plots: To identify relationships between pairs of variables such as heart rate vs. step count or sleep duration vs. stress level etc.
        * Box Plots: To compare the distribution of health metrics across different user segments.
    * **ML Tasks**:
        * Clustering (e.g., K-means): To segment users into distinct groups based on their health and activity data. Clustering will help in identifying natural groupings in the data that can be targeted with specific marketing strategies.
        * Dimensionality Reduction (e.g., PCA): To reduce the number of features while retaining most of the variance in the data. This helps in visualizing high-dimensional data and understanding the key factors that differentiate user segments.
    """)
    st.warning("### Hypothesis 1:")
    st.write("""
    * **Hypothesis**: Users can be segmented into distinct groups based on their health and activity data.
    * **Significance Level/Alpha**: 5%
    * **Null Hypothesis**: Users cannot be segmented into groups based on their smart-watch health and activity data.
    * **Validation Approach**:
        * Data cleaning and feature engineering
        * Clustering methods
        * Clustering evaluation
        * Predict what marketable group the customers belong to
    """)

    # Business Requirement 2
    st.success("## Business Requirement 2: Identify correlations between the health metric Stress Level, and if applicable predict these values")
    st.info("### Rationale:")
    st.write("""
    * **Data Visualizations**:
        * Bar Charts: To show the frequency of different health metrics and activities among user segments, helping in the identification of product preferences.
        * Cluster Profiles: To summarize the characteristics of each user segment identified through clustering.
    * **ML Tasks**:
        * Recommendation Algorithms (e.g., Collaborative Filtering): To suggest health products and wellness programs based on user preferences and health metrics.
        * Classification Models: To predict the likelihood of users being interested in specific products or programs based on their health and activity data.
        * Association Rule Learning: To identify common patterns and associations between different health metrics and product preferences.
    """)
    st.warning("### Hypothesis 2:")
    st.write("""
    * **Hypothesis**: Users with varying stress levels are expected to find either positive or negative correlations with other smart-watch health variables. Try and predict stress levels if applicable.
    * **Significance Level/Alpha**: 5%
    * **Target Segment**: High-intensity fitness products and workout programs.
    * **Null Hypothesis**: There won’t be a correlation between activity level and stress level.
    * **Validation Approach**:
        * Data cleaning and feature engineering
        * Data visualization
        * Correlation/PPS analysis
        * Discover marketable correlations/relationships with an ML solution if applicable
    """)

    # Business Requirement 3
    st.success("## Business Requirement 3: Identify correlations between the health metric Step Count, and if applicable predict these values")
    st.info("### Rationale:")
    st.write("""
    * **Data Visualizations**:
        * Scatter Plots: To visualize the relationships between pairs of variables, such as heart rate vs. step count, and identify patterns or trends.
        * Correlation Matrices: To display the strength and direction of relationships between multiple health metrics.
        * Line Charts: To show trends in health metrics over time, such as how step count changes with varying heart rates.
    * **ML Tasks**:
        * Regression Analysis: To predict step counts based on other health metrics such as heart rate, sleep duration, and activity levels.
        * Correlation Analysis: To assess the strength and direction of relationships between health metrics.
        * Predictive Modeling: To develop models that can predict health outcomes like stress levels or step counts based on other variables. This can help in targeting specific products to users with certain health characteristics.
    """)
    st.warning("### Hypothesis 3:")
    st.write("""
    * **Hypothesis**: Users with a high or low step count are expected to find either positive or negative correlations with other smart-watch health variables. Try and predict step counts if applicable.
    * **Significance Level/Alpha**: 5%
    * **Target Segment**: Running-related products.
    * **Null Hypothesis**: There won’t be a correlation between heart rate and step count.
    * **Validation Approach**:
        * Data cleaning and feature engineering
        * Data visualization
        * Correlation/PPS analysis
        * Discover marketable correlations/relationships with an ML solution if applicable
    """)

