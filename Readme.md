-- Originally a  Colab Notebook

A short-case study of application of Supervised, Un-supervised, and semi-supervised models on structured data for a classical classification and clustering problem.


## Data Source:
Online Shoppers Purchasing Intention Dataset
Data Set https://www.kaggle.com/henrysue/online-shoppers-intention


# Description of data set:

This data set represents skewed data, such that 84.5% of user journeys did NOT result in a purchase (Revenue=False)
```
a. The dataset consists of 10 numerical and 8 categorical attributes.
b. The 'Revenue' attribute can be used as the class label.
c. "Administrative", "Administrative Duration", "Informational", "Informational Duration", "Product Related" and "Product Related Duration" represent the number of different types of pages visited by the visitor in that session and total time spent in each of these page categories.
d. The values of these features are derived from the URL information of the pages visited by the user and updated in real time when a user takes an action, e.g. moving from one page to another.
e. The "Bounce Rate", "Exit Rate" and "Page Value" features represent the metrics measured by "Google Analytics" for each page in the e-commerce site.
f. The value of "Bounce Rate" feature for a web page refers to the percentage of visitors who enter the site from that page and then leave ("bounce") without triggering any other requests to the analytics server during that session.
g. The value of "Exit Rate" feature for a specific web page is calculated as for all pageviews to the page and it represents the percentage that the page was seen in the last session.
h. The "Page Value" feature represents the average value for a web page that a user visited before completing an e-commerce transaction.
i. The "Special Day" feature indicates the closeness of the site visiting time to a specific special day (e.g. Mother’s Day, Valentine's Day) in which the sessions are more likely to be finalized with transaction. The value of this attribute is determined by considering the dynamics of e-commerce such as the duration between the order date and delivery date.
j. The dataset also includes operating system, browser, region, traffic type, visitor type as returning or new visitor, a Boolean value indicating whether the date of the visit is weekend, and month of the year.
```

## 1: Supervised Model
Build a predictive classification model (ensuring optimal features and classifier).

a. Fit a best Linear and non-linear classifier.
b. Hyper-parameter fitting process.
c. Comparison with T-POT model search

### Results
The training data after feature selection has m ~30 features and number of training records n ~ 7K. For this scale of data and features generally logistic regression and SVM are advised.

For the problem we would like to accurate in terms of prediction of right classes an would like a model which is reasonable in both precision and recall and hence f1-score is the choice of metric for the most optimal model.

So for our model search three set of Models are selected for this problems. For each we are ran 5 cross-validations sets across set of values of hyper-parameters to find an optimal model.

#### Model - 1 Logistic Regression:
Params = {{'C': 4.281332398719396, 'penalty': 'l1', 'solver': 'liblinear'}
Best F1 Score = 0.448
#### Model - 2 Random Forest Classifier
Params = {'class_weight': None, 'criterion': 'entropy', 'max_features': 10, 'n_estimators': 200}
Best F1 Score = 0.567
#### Model - 3 Linear SVM [BEST LINEAR MODEL ON CVs]
Params = {'C': 2, 'class_weight': 'balanced', 'penalty': 'l2'}
Best F1 Score = 0.585
Results of the best linear model on the test-set

Based on CV results Model 3 - Linear SVM was found as the best model for this problem. Result of that models on the test set are as follows:
Accuracy = 0.930
Precision = 0.597
Recall = 0.759
F1-score = 0.668
Confusion Matrix is: [[1796 100] [ 47 148]]

#### Model - 4 Non-Linear SVM [BEST NON-LINEAR MODEL ON CVs]
{'C': 1000, 'class_weight': 'balanced', 'gamma': 0.0001, 'kernel': 'rbf'}
Best F1 Score = 0.617
Results of the best non-linear SVM on the test-set
Based on CV results Model 4 - Linear SVM was found as the best model for this problem. Result of that models on the test set are as follows:

Accuracy = 0.945
Precision = 0.668
Recall = 0.826
F1-score = 0.739
Confusion Matrix is: [[1816 80] [34 161]]


#### Comparison with T-POT Grid Search
TPOT - Pipeline:

MinMaxScaler(),
RandomForestClassifier(bootstrap=True, criterion="entropy", max_features=0.6, min_samples_leaf=8, min_samples_split=12, n_estimators=100)
Results on Test-set:

Accuracy = 0.945,
Precision = 0.713,
Recall = 0.677,
F1-score = 0.695
Confusion Matrix: [[1843 53] [ 63 132]]

# 2: Unsuperivsed model
User-behavior clusters based on the purchasing behavior data for the complete dataset.

For the clustering task we will start from these features and will be using a combination of PCA and K-Means. PCA will reduce dimension of data and k-means will be executed on the reduced data.

# 3: Semi-supervised Learning
Can we Improve performance with un-labeled data in training?
For semi-supervised learning Label spreading was chosen as the choice of algorithm of choice.
