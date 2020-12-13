# KobeBryant-ShotSelection
## 1	Project Description, Details, and Goals
This project was found on Kaggle, which describes every shot Kobe has taken during his career. Our goal for this project is to determine what type of shot it was, which specific game it was, the location of the shot (loc_x, loc_y), predict whether the basket went in (shot_made_flag), how many shots did Kobe sink, game date, season number, the probability of Kobe's shot and many other features in the data set. Another goal for this project is to portray all data entries and separate them by multiple categories for each game, each season, and or the whole career of Kobe Bryant. Link to project: https://www.kaggle.com/c/kobe-bryant-shot-selection/overview
## 2	Data
The data set includes over 30,000 entries and 24 features, with data involving the location and circumstances of every field goal attempted by Kobe Bryant during his 20-year career. The features are listed below, with our target label indicated in bold.
#### action_type
#### combined_shot_type
#### game_event_id
#### game_id
#### lat
#### loc_x
#### loc_y
#### lon
#### minutes_remaining
#### period
#### playoffs
#### season 
#### seconds_remaining
#### shot_distance
#### shot_made_flag (Target Label)
#### shot_type
#### shot_zone_area
#### shot_zone_basic
#### shot_zone_range
#### team_id
#### team_name
#### game_date
#### matchup
#### opponent
#### Shot_id
 
### 2.1	Processing
Before using any kind of predictive model on the dataset, we examined the data to search for any missing, redundant, or useless information. We discovered that there were 5000 labels missing from random rows, redundant features that give us no new information, and useless features that had no significant value at all. Once we have a clean dataset, we want to make sure that there are no issues when we run it through our machine learning models. We were able to identify some categorical features, "action_type", "combined_shot_type","season","game_date", "shot_type", "shot_zone_area", "shot_zone_basic", "shot_zone_range", and "opponent", which do not work well for classifiers such as decision tree or logistic regression. To solve this issue, we implemented One-Hot encoding which allowed us to eliminate our categorical data and fit our models appropriately.
After cleaning our data and implementing One-Hot encoding, we decided to use all of the features remaining for our models.
	
## 3	Developed Methods, Algorithms, and Tools
Once the data was processed, we split the data into training and testing sets with various test sizes ranging from 0.2 to 0.8. We then trained multiple supervised learning algorithms and tested them with the corresponding datasets. The classifiers that we chose to use were K-neighbors, decision tree, and logistic regression. Sections 3.1-3.3 details the methods, algorithms, and tools used in our analysis.

### 3.1	The KNN Classifier
We began our KNN classifier quite simply with a train/test split of test size 0.4 and k=3. However, we realized that this value of k was not the most optimal, so we decided to test the accuracy of more KNN classifiers. We were able to loop through values of k from 1 to 20, but anything that went higher resulted in major performance issues. However, we noticed significant changes in our prediction accuracy with different values of K. 
	Another approach was that we split up the original dataset into multiple smaller datasets by each of the 20 seasons. For each of these datasets, we processed the data similarly to our original KNN classifier, then tested our models for each season. To optimize our most accurate value for k, we tested k for 1 - 30 and found the best values of k to choose our accuracy.

### 3.2	Logistic Regression Classifier
After processing our data by removing unlabeled data, we decided to try only using numerical data for our logistic regression classifier. We did not use One-Hot encoding. We only used the default numerical features for the classifier. The approach was straight-forward in that we trained with our training set and tested with our testing set.
The second approach was to use One-Hot encoding to include the categorical features and normalizing the data after. We then used LBFGS and SAG solvers within our logistic regression and plotted the AUC and ROC curve.

### 3.3	Decision Tree Classifier
For our decision tree classifier, we used a test size of 0.3 in our train/test split. We initially went for a straight-forward approach (training our decision tree with training data, then predicting on test data). However, we ended up with an unsatisfactory result which led us to pursue different ways of increasing our accuracy. We tried 10-fold cross validation, which gave us a more accurate accuracy (no improvement). Then we tried to implement ensemble learning and “bagging” on 20 decision tree classifiers. 
Using a n_sample size of 0.8 times the size of the original dataset, we found out the prediction results for all 20 decision trees and used voting to predict the final accuracy. With the unexpectedness of our outcomes, we decided to use a random forest classifier with 100 estimators and bootstrapping enabled. Our results stayed roughly the same using the prior methods.
### 3.4	Tools
The tools that we used were numpy, pandas, sklearn, matplotlib, and seaborn. We used numpy, pandas, and sklearn to process and apply our classifiers. We also used Jupyter notebook to organize and visualize our data with matplotlib and seaborn.

## 4	Results
### 4.1	KNN Accuracy
	Using a simple KNN classifier with k=3, we were able to obtain an accuracy score of 0.55. However, Figure 4.1.1 demonstrates the accuracy of our classifier increasing as our value of K increases. In Figure 4.1.2, we can see that the accuracy only merely increases after K=25. This shows that a good value for K would be either 10 or 25 due to the increase in accuracy.
	When separating our seasons and using KNN on each one, we found accuracy scores ranging from 0.55 to 0.72. We were able to search for a more optimal k from 1-30 and picked the best k with the best accuracy.
### 4.2	Logistic Regression Accuracy, AUC, and ROC Curve
	Using our standard logistic regression classifier with just default numerical data, we achieved an accuracy score of 0.595. However, after implementing One-Hot encoding to include our categorical data and normalizing the result, we were able to achieve a significant increase in accuracy at 0.674 using the LBFGS solver. Using a SAG solver did not do much to our result (still 0.674), but it was much faster in performance.
### 4.3	Decision Tree Accuracy
	Using a standard decision tree classifier, we were able to obtain an accuracy score of 0.63. We also decided to use 10-fold cross validation, where we actually ended up with a less accurate score of 0.587. At this point, we started to see a trend where our accuracy would go down the more we test, which leads us to believe that the accuracy is approaching its true value. However, when applying bagging using 20 decision tree classifiers and then using voting to make the final accuracy, we went back up to a 0.645 accuracy.  It leads us to believe that 10-fold cross validation may not be as accurate as we thought. Instead, we used random forest with n_estimators = 100 and were able to achieve our highest score of 0.648.
### 4.4	Final Results
	By comparing each classifier we used, the logistic regression classifier came up on top with the highest accuracy score of 0.674. We believe that this is due to the normalization that the classifier does for us and gives us better “weights” for the importance of each feature. While receiving a higher score would be much more desirable, we can attribute these accuracies to having smaller data sets varying by season. It is much more difficult to predict a shot if there is not much data to learn from and when the probability of shots made is barely over 60%. With seasons where there is more data available, such as Kobe Bryant’s prime years (where he performs the best), we would be able to use this classifier in real practice. Due to Kobe’s legendary status in the NBA, the data can be used to compare him with rising stars and with other players for performance statistics.
