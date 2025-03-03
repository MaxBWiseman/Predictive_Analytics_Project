import streamlit as st


def page_summary_body():
    st.write("### Quick Project Summary")

    st.info(
        "**Dataset Content**\n"
        "* The dataset is public and posted to Kaggle.com"
        " by user 'Mohammed Arfath R'.\n"
        "* The dataset includes 6 continuous number columns"
        " (integer/float) and one categorical column (object).\n"
        "* The columns describe data about Heart Rate, Blood Oxygen,"
        " Step Count, Sleep Duration, Activity Levels, and Stress Level.\n\n"

        "**Project Dataset**\n"
        "* The dataset is uncleaned and includes raw"
        " data from smart watch sensors.\n"
        "* The data is used for segmenting users into different"
        " groups based on their health and activity data.\n\n"
    )

    st.info(
        "**ML Business Case**\n\n"
        "* **User Segmentation**: Develop a machine learning model"
        " to segment users based on their health and activity data collected"
        " from smart watch sensors. This segmentation will enable the company"
        " to target users with specific market advertisements for fitness"
        " products, health supplements, or health/wellness classes/programs.\n"
        "* **Correlation Research**: Conduct research on the correlation"
        " between certain health metrics collected from smart watch sensors"
        " to contribute to the development of"
        " better advertisement of health and sport products.\n"
        "* If correlation exists, attempt to"
        " predict (Stress Level, Step Count).\n"
    )

    st.write(
        "* For additional information, please visit and **read**"
        " the README.md file in the project's GitHub repository."
        "* https://github.com/MaxBWiseman/Predictive_Analytics_Project "
    )

    # Business Requirements
    st.success(
        "**Business Requirements**\n"
        "A company wishes to:\n"
        "* Segment users into different groups based on health and"
        " activity data collected from smart watch sensors.\n"
        "* Target these segments with specific market"
        " advertisements for fitness products, health supplements,"
        " or health/wellness classes/programs.\n"
        "* Conduct research on the correlation between various"
        " health metrics (e.g., heart rate, activity level, stress level)"
        " to improve customer experience with targeted"
        " advertisements and product suggestions."
    )

    # Hypothesis 1
    st.success(
        "**Hypothesis 1**\n"
        "* Users can be segmented into distinct groups"
        " based on their health and activity data.\n"
        "* Significance Level/Alpha: 5%\n"
        "* Null Hypothesis: Users cannot be segmented into"
        " groups based on their smart watch health and activity data.\n"
        "* Validation Approach:\n"
        "    * Data cleaning and feature engineering\n"
        "    * Clustering methods\n"
        "    * Clustering evaluation\n"
        "    * Predict what marketable group the customers belong to"
    )

    # Hypothesis 2
    st.success(
        "**Hypothesis 2**\n"
        "* Users with varying stress levels are expected to find either"
        " positive or negative correlations with other smart-watch health"
        " variables. Try and predict stress levels if applicable.\n"
        "* Significance Level/Alpha: 5%\n"
        "* Null Hypothesis: There won't be a correlation"
        " between activity level and stress level.\n"
        "* Validation Approach:\n"
        "    * Data cleaning and feature engineering\n"
        "    * Data visualization\n"
        "    * Correlation/PPS analysis\n"
        "    * Discover marketable correlations/relationships with"
        " an ML solution if applicable"
    )

    # Hypothesis 3
    st.success(
        "**Hypothesis 3**\n"
        "* Users with high or low step counts are expected to find"
        " either positive or negative correlations with other"
        " smart-watch health variables."
        " Try and predict step counts if applicable.\n"
        "* Significance Level/Alpha: 5%\n"
        "* Null Hypothesis: There won't be a correlation"
        " between heart rate and step count.\n"
        "* Validation Approach:\n"
        "    * Data cleaning and feature engineering\n"
        "    * Data visualization\n"
        "    * Correlation/PPS analysis\n"
        "    * Discover marketable correlations/relationships"
        " with an ML solution if applicable"
    )
