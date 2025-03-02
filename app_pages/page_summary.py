import streamlit as st

def page_summary_body():
    st.write("### Quick Project Summary")

    st.info(
        f"**Dataset Content**\n"
        f"* The dataset is public and posted to Kaggle.com by user 'Mohammed Arfath R'.\n"
        f"* The dataset includes 6 continuous number columns (integer/float) and one categorical column (object).\n"
        f"* The columns describe data about Heart Rate, Blood Oxygen, Step Count, Sleep Duration, Activity Levels, and Stress Level.\n\n"
        
        f"**Project Dataset**\n"
        f"* The dataset is uncleaned and includes raw data from smart watch sensors.\n"
        f"* The data is used for segmenting users into different groups based on their health and activity data.\n\n"
    )
    
    st.info(
        f"**ML Business Case**\n\n"
        f"* **User Segmentation**: Develop a machine learning model to segment users based on their health and activity data collected from smart watch sensors. This segmentation will enable the company to target users with specific market advertisements for fitness products, health supplements, or health/wellness classes/programs.\n"
        f"* **Correlation Research**: Conduct research on the correlation between certain health metrics collected from smart watch sensors to contribute to the development of better advertisement of health and sport products.\n"
        f"* If correlation exists, attempt to predict (Stress Level, Step Count).\n"
    )

    st.write(
        f"* For additional information, please visit and **read** the "
    

    st.success(
        f"**Business Requirements**\n"
        f"A company wishes to:\n"
        f"* Segment users into different groups based on health and activity data collected from smart watch sensors.\n"
        f"* Target these segments with specific market advertisements for fitness products, health supplements, or health/wellness classes/programs.\n"
        f"* Conduct research on the correlation between various health metrics (e.g., heart rate, activity level, stress level) to improve customer experience with targeted advertisements and product suggestions.\n\n"
        
        f"**Hypothesis and Validation**\n"
        
        f"* **Hypothesis 1**\n"
        f"  * Users can be segmented into distinct groups based on their health and activity data.\n"
        f"  * Significance Level/Alpha: 5%\n"
        f"  * Null Hypothesis: Users cannot be segmented into groups based on their smart watch health and activity data.\n"
        f"  * Validation Approach:\n"
        f"    * Data cleaning and feature engineering\n"
        f"    * Clustering methods\n"
        f"    * Clustering evaluation\n"
        f"    * Predict what marketable group the customers belong to\n\n"
        
        f"* **Hypothesis 2**\n"
        f"  * Users with varying stress levels are expected to find either positive or negative correlations with other smart-watch health variables. Try and predict stress levels if applicable.\n"
        f"  * Significance Level/Alpha: 5%\n"
        f"  * Null Hypothesis: There won't be a correlation between activity level and stress level.\n"
        f"  * Validation Approach:\n"
        f"    * Data cleaning and feature engineering\n"
        f"    * Data visualization\n"
        f"    * Correlation/PPS analysis\n"
        f"    * Discover marketable correlations/relationships with an ML solution if applicable\n\n"
        
        f"* **Hypothesis 3**\n"
        f"  * Users with high or low step counts are expected to find either positive or negative correlations with other smart-watch health variables. Try and predict step counts if applicable.\n"
        f"  * Significance Level/Alpha: 5%\n"
        f"  * Null Hypothesis: There won't be a correlation between heart rate and step count.\n"
        f"  * Validation Approach:\n"
        f"    * Data cleaning and feature engineering\n"
        f"    * Data visualization\n"
        f"    * Correlation/PPS analysis\n"
        f"    * Discover marketable correlations/relationships with an ML solution if applicable\n"
    )

if __name__ == "__main__":
    page_summary_body()
