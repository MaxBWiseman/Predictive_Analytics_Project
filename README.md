
[![wakatime](https://wakatime.com/badge/user/d85da0fd-b442-4c33-98af-3ef622520fc1/project/57b25889-4ebb-44ea-b083-b0d3fb3c5d16.svg)](https://wakatime.com/badge/user/d85da0fd-b442-4c33-98af-3ef622520fc1/project/57b25889-4ebb-44ea-b083-b0d3fb3c5d16)

## Dataset Content

* The dataset is public and posted to Kaggle.com by user "Mohammed Arfath R", I made sure to pick an uncleaned dataset for my project. The data types include 6 continuous number columns (interger/float) and one categorical column (object). The columns describe data about Heart Rate, Blood Oxygen, Step Count, Sleep Duration, Activity Levels and Stress level.


## Business Requirements

* A company wishes to segment users into different groups based on health and activity data collected from smart watch sensors to target them with specific market advertisements for things like fitness products, health supplements or health/wellness classes/programs. The company also wants to conduct research on the correlation between various health metrics (e.g., heart rate, activity level, stress level), contributing to a better customer experience with targetted advertisments and product suggestions.

## Hypothesis and how to validate?

* Hypothesis 1 - 
  * Users can be segmented into distinct groups based on their health and acitivity data
  * Significance Level/Alpha: 5%
  * This is directley inline with the business requirement
  * This will be a Null Hypothesis, that users cannot be segmented into groups based on their smart watch health and acitivity data
    * Validation approach:
      * Data cleaning and feature engineering
      * Clustering methods
      * Custering evaluation
      * Predict what marketable group the customers belongs to
* Hypothesis 2 -
  * Users with varying stress levels are expected to find either positive or negative correlations with other smart-watch health variables. Try and predict stress levels if applicable
  * Significance Level/Alpha: 5%
  * This segment could be targeted with high-intensity fitness products and workout classes/programs.
  * This will be a Null Hypothesis, there wont be a correlation between activity level and stress level
    * Validation approach:
      * Data cleaning and feature engineering
      * Data visualization
      * Correlation/PPS analysis
      * Discover marketable correlations/relationships with an ML solution if applicable
* Hypothesis 3 -
  * Users with a high or low step counts are expected to find either positive or negative correlations with other smart-watch health variables. Try and predict step counts if applicable
  * Significance Level/Alpha: 5%
  * This segment could be targeted with more running related products
  * This will be a Null Hypothesis, there wont be a correlation between heart rate and step count
    * Validation approach:
      * Data cleaning and feature engineering
      * Data visualization
      * Correlation/PPS analysis
      * Discover marketable correlations/relationships with an ML solution if applicable


## The rationale to map the business requirements to the Data Visualizations and ML tasks

### Business Requirement 1: Segment users into different groups based on health and activity data

#### Rationale:

* Data Visualizations:
  * Heatmaps: To visualize the correlation between different health metrics. This will help in understanding how variables like heart rate, activity level, and stress level interact with each other.
  * Density Plots: To identify relationships between pairs of variables such as heart rate vs. step count or sleep duration vs. stress level etc.
  * Box Plots: To compare the distribution of health metrics across different user segments.
* ML Tasks:
  * Clustering (e.g., K-means): To segment users into distinct groups based on their health and activity data. Clustering will help in identifying natural groupings in the data that can be targeted with specific marketing strategies.
  * Dimensionality Reduction (e.g., PCA): To reduce the number of features while retaining most of the variance in the data. This helps in visualizing high-dimensional data and understanding the key factors that differentiate user segments.


### Business Requirement 2: Conduct research for a more tailored recommender system to suggest health products and wellness programs by finding relationships from smart watch health and activity data

#### Rationale:

* Data Visualizations:
  * Bar Charts: To show the frequency of different health metrics and activities among user segments, helping in the identification of product preferences.
  * Cluster Profiles: To summarize the characteristics of each user segment identified through clustering.
* ML Tasks:
  * Recommendation Algorithms (e.g., Collaborative Filtering): To suggest health products and wellness programs based on user preferences and health metrics.
  * Classification Models: To predict the likelihood of users being interested in specific products or programs based on their health and activity data.
  * Association Rule Learning: To identify common patterns and associations between different health metrics and product preferences.


## ML Business Case

User Segmentation: Develop a machine learning model to segment users based on their health and activity data collected from smart watch sensors. This segmentation will enable the company to target users with specific market advertisements for fitness products, health supplements, or health/wellness classes/programs.
Correlation Research: Conduct research on the correlation between certain health metrics collected from smart watch sensors to contribute to the development of better advertisement of health and sport products.
If correlation exists attempt to predict (Stress Level, Step Count).


## Dashboard Design

* List all dashboard pages and their content, either blocks of information or widgets, like buttons, checkboxes, images, or any other item that your dashboard library supports.
* Later, during the project development, you may revisit your dashboard plan to update a given feature (for example, at the beginning of the project you were confident you would use a given plot to display an insight but subsequently you used another plot type).



## Unfixed Bugs

* None I am aware of.

## Deployment

### Heroku

* The App live link is: https://YOUR_APP_NAME.herokuapp.com/ 
* Set the runtime.txt Python version to a [Heroku-24](https://devcenter.heroku.com/articles/python-support#supported-runtimes) stack currently supported version.
* The project was deployed to Heroku using the following steps.

1. Log in to Heroku and create an App
2. At the Deploy tab, select GitHub as the deployment method.
3. Select your repository name and click Search. Once it is found, click Connect.
4. Select the branch you want to deploy, then click Deploy Branch.
5. The deployment process should happen smoothly if all deployment files are fully functional. Click now the button Open App on the top of the page to access your App.
6. If the slug size is too large then add large files not required for the app to the .slugignore file.


## Main Data Analysis and Machine Learning Libraries

* Pandas - Used to load and preprocess the data
* Numpy - For efficient numerical computations
* SciKit-Learn - For machine learning modelling, including splitting, scaling, training etc
* Seaborn - For data visuals
* Matplotlib.pyplot - Also for visuals
* Pyplot - Interactive visuals
* xgboost - Used to test the three tasks, to see if suitable algorithm.
* ydata-profiling - For quick and insightful EDA's
* Yellowbrick - Helps visualise model related things like feature importance or silhouette/Kelbow plots
* Feature-Engine - Provides various feature cleaning and engineering solutions
* ppscore - To aid in finding correlations


## Credits 

* A big thankyou to Code Institute for providing me with the learning material and base functions for building software for predictive analytics.
* Another big thankyou to Kaggle.com and user "Mohammed Arfath R" for uploading the dataset used in this project. 

### Content 

Everything in this project apart from the dataset and a few select functions provided by code institute, is all made by myself.

### Media

- No third part media included in project


## Acknowledgements (optional)

My journey through Code Institute has been nothing short of a rollercoaster, an exciting transformative experience that has reshaped my life. I am forever grateful to Code Institute for igniting a passion for coding that started as a small hobby and has since growb into a life-changing career.

At the beginning of the course, I was filled with confidence and flew through the material. However, the journey took a challenging turn during Project 3, when I began receiving my results for Projects 1 and 2. I hadn’t fully understood the importance of the README file, and, as a result, my submissions failed. Determined to improve, I reworked my projects and eventually earned a merit for Project 3—though it was capped at a pass. Without the cap, it would have achieved a merit.

When it came to Project 4, I saw an opportunity to redeem myself, and I came incredibly close—just two marks shy of a merit. Unfortunately, during the middle of this portfolio project, I experienced the devastating loss of my Nan, Carol Major, who passed away on 2nd November 2024 and was laid to rest at Lympstone Church in early December. Despite my grief, I chose not to take time off, believing that staying focused would help me cope. I pushed through and submitted Project 4, only to discover hours later that one of my site features had a 505 error. This oversight meant that my project received a pass, falling just shy of a merit by only two marks.

I poured my heart and soul into this project, dedicating over 70 hours in just nine days. My timer app—which pauses if the IDE is inactive for more than 10 minutes—accurately tracked the time I invested. For the first time in my life, I found myself wholeheartedly committed to something I genuinely enjoy. I never imagined returning to education, but everything changed the day I stumbled across a YouTube ad for Code Institute. I took on the 5-day coding challenge, and I haven’t looked back since.

This journey has given me something I deeply lacked before: purpose. Despite the ups and downs, I have discovered a true passion that has transformed my life. I can’t wait to share my achievements with others, and I know in my heart that my Nan would be so proud.

Thank you, Code Institute.

