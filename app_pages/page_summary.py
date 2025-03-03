import streamlit as st


def page_summary_body():
    st.write("### Quick Project Summary")

    st.write("## Dataset Content")
    st.write(
        "The dataset is public and was posted on Kaggle.com by user "
        "**Mohammed Arfath R**. For the purpose of this project, an "
        "uncleaned dataset was selected. Below are the key details:\n\n"
        "- **Data Types**: The dataset contains 6 continuous numeric "
        "columns (integer/float) and 1 categorical column (object).\n"
        "- **Columns**: The columns describe data about the following:\n"
        "  - Heart Rate\n"
        "  - Blood Oxygen\n"
        "  - Step Count\n"
        "  - Sleep Duration\n"
        "  - Activity Levels\n"
        "  - Stress Level"
    )
    st.write(
        "For more information or to access the dataset, visit the Kaggle "
        "link below:"
    )
    st.write("[Kaggle Dataset Link](https://www.kaggle.com/datasets/mohamm"
             "edarfathr/smartwatch-health-data-uncleaned/data)")

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
