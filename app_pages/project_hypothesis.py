import streamlit as st

def page_project_hypothesis_body():
    st.write("### Project Hypothesis and Validation")

    st.success(
        f"## The rationale to map the business requirements to the Data Visualizations and ML tasks\n\n"
        
        f"### Business Requirement 1: Segment users into different groups based on health and activity data\n"
        f"#### Rationale:\n"
        f"* **Data Visualizations**:\n"
        f"  * Heatmaps: To visualize the correlation between different health metrics. This will help in understanding how variables like heart rate, activity level, and stress level interact with each other.\n"
        f"  * Density Plots: To identify relationships between pairs of variables such as heart rate vs. step count or sleep duration vs. stress level etc.\n"
        f"  * Box Plots: To compare the distribution of health metrics across different user segments.\n"
        f"* **ML Tasks**:\n"
        f"  * Clustering (e.g., K-means): To segment users into distinct groups based on their health and activity data. Clustering will help in identifying natural groupings in the data that can be targeted with specific marketing strategies.\n"
        f"  * Dimensionality Reduction (e.g., PCA): To reduce the number of features while retaining most of the variance in the data. This helps in visualizing high-dimensional data and understanding the key factors that differentiate user segments.\n\n"
        
        f"### Business Requirement 2: Conduct research for a more tailored recommender system to suggest health products and wellness programs by finding relationships from smart watch health and activity data\n"
        f"#### Rationale:\n"
        f"* **Data Visualizations**:\n"
        f"  * Bar Charts: To show the frequency of different health metrics and activities among user segments, helping in the identification of product preferences.\n"
        f"  * Cluster Profiles: To summarize the characteristics of each user segment identified through clustering.\n"
        f"* **ML Tasks**:\n"
        f"  * Recommendation Algorithms (e.g., Collaborative Filtering): To suggest health products and wellness programs based on user preferences and health metrics.\n"
        f"  * Classification Models: To predict the likelihood of users being interested in specific products or programs based on their health and activity data.\n"
        f"  * Association Rule Learning: To identify common patterns and associations between different health metrics and product preferences.\n\n"
        
        f"### Business Requirement 3: Identify correlations between health metrics (e.g., heart rate, step count) and predict outcomes\n"
        f"#### Rationale:\n"
        f"* **Data Visualizations**:\n"
        f"  * Scatter Plots: To visualize the relationships between pairs of variables, such as heart rate vs. step count, and identify patterns or trends.\n"
        f"  * Correlation Matrices: To display the strength and direction of relationships between multiple health metrics.\n"
        f"  * Line Charts: To show trends in health metrics over time, such as how step count changes with varying heart rates.\n\n"
        f"* **ML Tasks**:\n"
        f"  * Regression Analysis: To predict step counts based on other health metrics such as heart rate, sleep duration, and activity levels.\n"
        f"  * Correlation Analysis: To assess the strength and direction of relationships between health metrics.\n"
        f"  * Predictive Modeling: To develop models that can predict health outcomes like stress levels or step counts based on other variables. This can help in targeting specific products to users with certain health characteristics.\n\n"
        
        f"#### Hypothesis 3:\n"
        f"  * Users with high or low step counts are expected to find either positive or negative correlations with other smart-watch health variables. Try and predict step counts if applicable.\n"
        f"  * Significance Level/Alpha: 5%\n"
        f"  * This segment could be targeted with more running-related products.\n"
        f"  * Null Hypothesis: There won't be a correlation between heart rate and step count.\n"
        f"    * Validation approach:\n"
        f"      * Data cleaning and feature engineering\n"
        f"      * Data visualization\n"
        f"      * Correlation/PPS analysis\n"
        f"      * Discover marketable correlations/relationships with an ML solution if applicable.\n"
    )

if __name__ == "__main__":
    page_project_hypothesis_body()
