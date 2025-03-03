import streamlit as st
from app_pages.multipage import MultiPage

# load pages scripts
from app_pages.page_summary import page_summary_body
from app_pages.project_hypothesis import page_project_hypothesis_body
from app_pages.data_study import page_study_body
from app_pages.clustering_task import cluster_task_start
from app_pages.predict_step_count_task import predict_step_count_task
from app_pages.predict_stress_level_task import predict_stress_level_start

app = MultiPage(app_name= "Smartwatch Health Data ML Project")

app.add_page("Quick Project Summary", page_summary_body)
app.add_page("Project Hypothesis and Validation", page_project_hypothesis_body)
app.add_page("Smartwatch Health Data Study", page_study_body)
app.add_page("ML: Clustering task", cluster_task_start)
app.add_page("ML: Predict Step Count", predict_step_count_task)
app.add_page("ML: Predict Stress Level", predict_stress_level_start)

app.run() # Run the  app