
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
  * Users with varying stress levels are expected to find either positive or negative correlations with other smart-watch health variables
  * Significance Level/Alpha: 5%
  * This segment could be targeted with high-intensity fitness products and workout classes/programs.
  * This will be a Null Hypothesis, there wont be a correlation between activity level and stress level
    * Validation approach:
      * Data cleaning and feature engineering
      * Data visualization
      * Correlation/PPS analysis
      * Discover marketable correlations/relationships with an ML solution if applicable
* Hypothesis 3 -
  * Users with a high or low step counts are expected to find either positive or negative correlations with other smart-watch health variables
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


## Dashboard Design

* List all dashboard pages and their content, either blocks of information or widgets, like buttons, checkboxes, images, or any other item that your dashboard library supports.
* Later, during the project development, you may revisit your dashboard plan to update a given feature (for example, at the beginning of the project you were confident you would use a given plot to display an insight but subsequently you used another plot type).



## Unfixed Bugs

* You will need to mention unfixed bugs and why they were not fixed. This section should include shortcomings of the frameworks or technologies used. Although time can be a significant variable to consider, paucity of time and difficulty understanding implementation is not a valid reason to leave bugs unfixed.

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

* Here you should list the libraries you used in the project and provide an example(s) of how you used these libraries.


## Credits 

* In this section, you need to reference where you got your content, media and extra help from. It is common practice to use code from other repositories and tutorials, however, it is important to be very specific about these sources to avoid plagiarism. 
* You can break the credits section up into Content and Media, depending on what you have included in your project. 

### Content 

- The text for the Home page was taken from Wikipedia Article A
- Instructions on how to implement form validation on the Sign-Up page were taken from [Specific YouTube Tutorial](https://www.youtube.com/)
- The icons in the footer were taken from [Font Awesome](https://fontawesome.com/)

### Media

- The photos used on the home and sign-up page are from This Open-Source site
- The images used for the gallery page were taken from this other open-source site



## Acknowledgements (optional)

* Thank the people who provided support through this project.

