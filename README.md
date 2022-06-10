 <p align="center">
 <img src="https://user-images.githubusercontent.com/98360572/173111303-2afb6fbb-502b-49ff-b0a8-de95f88bbf8c.png" width="50%" height="50%">
</p>

# Module 17 Challenge: Supervised Machine Learning - Credit Risk Analysis
 
Because good loans outnumber risky loans, credit risk is an inherently unbalanced classification problem. As a result, it is neccesary to use a variety of techniques to train and evaluate models with unbalanced classes. For this purpose we will build and evaluate models with resampling using the imbalanced-learn and scikit-learn libraries.

We will also use the `RandomOverSampler` and `SMOTE` algorithms to oversample the credit card credit dataset from **LendingClub**, a peer-to-peer lending services company, and the `ClusterCentroids` algorithm to undersample the data. The `SMOTEENN` algorithm will then be used to perform a combinatorial approach of `over- and undersampling`. Then, to predict credit risk, we will compare two machine learning models that reduce bias, `BalancedRandomForestClassifier` and `EasyEnsembleClassifier`. After that, we will assess the performance of these models and conclude whether they should be used to predict credit risk.

---
# :one: Overview of the analysis: Explain the purpose of this analysis.

The objective of this analysis is to apply six different unbalanced machine learning models to the same dataset and compare the results obtained.  The dataset used contains the credit applications for the first quarter of 2019, obtained from a personal credit company called **LendingClub**.  The original data file contains information on `115,675` applications.

The study of the data is done with two Python code files, using Jupyter Notebooks:

[**Code File #1**: credit_risk_resampling.ipynb](https://github.com/Peteresis/Credit_Risk_Analysis/blob/e08cba0cb4d441409df51a5e494665d156c13120/credit_risk_resampling.ipynb)

[**Code File #2**: credit_risk_ensemble.ipynb](https://github.com/Peteresis/Credit_Risk_Analysis/blob/e08cba0cb4d441409df51a5e494665d156c13120/credit_risk_ensemble.ipynb)

Both files are based on the starter code provided by the course.  At the beginning of the files there are a series of manipulations to clean the dataset.  These data cleaning operations are done using the Python Pandas library.

In the dataset there is a column named `loan_status` which contains the following values: `Charged Off`, `Current`, `Fully Paid`, `In Grace Period`, `Issued`, `Late (16-30 days)`, `Late (31-120 days)`.

For this analysis, the applications were classified into two groups: 

The **low risk** group, composed of the applications marked as `Current`

The **high risk** group which is composed of applications marked with these values (`In Grace Period`, `Late (16-30 days)`, `Late (31-120 days)`).

Applications marked as `Issued`, `Fully Paid` and `Charged Off` were ignored.

With these operations the dataset was reduced from `115,675` applications to `68,817`.

Once the dataset was cleaned and organized for our analysis, the data was split into `Training` and `Testing` and the analysis could start.

### New Functions Learned for creating `Training` and `Testing` datasets:

### get.dummies(X)
Since the Machine Learning models only work with numbers we need to convert all the cells with `low risk` and `high risk` to numbers.  The get_dummies() function is used to convert categorical variable into dummy/indicator variables.  In this case all the 'low risk' were changed to `0` and the `high risk` to `1`

### train_test_split(X, y, random_state=1)
This is a function from the `sklearn` Python library.  The train_test_split function is for splitting a single dataset for two different purposes: training and testing. The testing subset is for building the model. The testing subset is for using the model on unknown data to evaluate the performance of the model.

After the `Training` and `Testing` groups are created, we started testing the **Machine Learning Algorithms**.

## :one:.:one: Oversampling Algorithms

Over sampling and under sampling are techniques used in data mining and data analytics to modify unequal data classes to create balanced data sets. Over sampling and under sampling are also known as resampling.

When one class of data is the underrepresented minority class in the data sample (in our case this would be the `high risk` category), over sampling techniques maybe used to duplicate these results for a more balanced amount of positive results in training. Over sampling is used when the amount of data collected is insufficient. A couple of popular over sampling technique are the **Naive Random Oversampling** and **SMOTE (Synthetic Minority Over-sampling Technique)**, which creates synthetic samples by randomly sampling the characteristics from occurrences in the minority class.

## Naive Random Oversampling

### New Functions Learned for `Oversampling` datasets:

### RandomOverSampler(random_state=1) 
Random oversampling can be implemented using the RandomOverSampler class. The class can be defined and takes a sampling_strategy argument that can be set to “minority” to automatically balance the minority class with majority class or classes. This means that if the majority class had 1,000 examples and the minority class had 100, this strategy would oversampling the minority class so that it has 1,000 examples.

### fit_resample(X,y) 
The `fit_resample` method is used in conjunction with the `RandomOverSampler` to resample the data and targets into a dictionary with a key-value pair of data_resampled and targets_resampled.

### LogisticRegression(solver='lbfgs', random_state=1)
This is part of the Python library `sklearn`. The `solver` argument is set to `lbfgs`, which is the default setting. The `random_state` is specified so that anyone will be able to reproduce the same results as they run the code.

### model.predict(X_test) and balance_accuracy_score(y_test,y_pred)
This is the model used to create predictions and generate the accuracy score of the model

### confusion_matrix(y_test, y_pred)
This is another function from the `sklearn` Python Library.  It creates a 3x3 table (or a matrix), where all the information about false positives and false negatives is displayed.  It is used to evaluate the precision of the model.

### classification_report_imbalanced


















<br>


<p align="center">
    <img src="https://user-images.githubusercontent.com/98360572/172030002-e3fe19f0-f388-4959-b5de-c341b1803970.png" width="50%" height="50%">
</p>


<p align="center">
<div class="row">
  <div class="column">
    <img src="https://user-images.githubusercontent.com/98360572/172029042-a3207a29-51f1-4e8a-97aa-43b181cad713.png" width="40%" height="100%">
    <img src="https://user-images.githubusercontent.com/98360572/172029049-b052e759-27da-4f6d-9dac-8e731f4805a6.png" width="50%" height="110%">
  </div>
</div>
</p>



|   ⚠️ **NOTE: Please click on any image to zoom**     |
| ----------- |




---
# :two: Results: Using bulleted lists, describe the balanced accuracy scores and the precision and recall scores of all six machine learning models. Use screenshots of your outputs to support your results.


---

# :three: Summary: Summarize the results of the machine learning models, and include a recommendation on the model to use, if any. If you do not recommend any of the models, justify your reasoning.

---

## :four: References

**Module 17: Supervised Machine Learning**, https://courses.bootcampspot.com/courses/1145/pages/17-dot-0-1-introduction-to-machine-learning, :copyright: 2020-2021 Trilogy Education Services, Web 21 May 2022.

**Stack Overflow: scikit learn output metrics.classification_report into CSV/tab-delimited format**, https://stackoverflow.com/questions/39662398/scikit-learn-output-metrics-classification-report-into-csv-tab-delimited-format

**GitHub: [BUG] 'BalancedBaggingClassifier' and 'EasyEnsembleClassifier' objects have no attribute 'n_features_in_' #872**, [https://github.com/scikit-learn-contrib/imbalanced-learn/issues/872](https://github.com/scikit-learn-contrib/imbalanced-learn/issues/872#:~:text=You%20have%20to%20downgrade%20the%20scikit%2Dlearn%20package%20using%3A%20pip3%20install%20scikit%2Dlearn%3D%3D1.0%20%2DU%0AThe%20attribute%20n_features_in_%20is%20deprecated%20and%20its%20support%20was%20lost%20in%20sklearn%20version%201.2)

**W3Resource.com: Pandas: Data Manipulation - get_dummies() function**, https://www.w3resource.com/pandas/get_dummies.php

**BitDegree.org: Splitting Datasets With the Sklearn train_test_split Function**, https://www.bitdegree.org/learn/train-test-split

**TechTarget.com: over sampling and under sampling**, https://www.techtarget.com/whatis/definition/over-sampling-and-under-sampling

**MachineLearningMastery.com: Random Oversampling and Undersampling for Imbalanced Classification**, [https://machinelearningmastery.com/random-oversampling-and-undersampling-for-imbalanced-classification/](https://machinelearningmastery.com/random-oversampling-and-undersampling-for-imbalanced-classification/#:~:text=Random%20oversampling%20can,class%20or%20classes)

**GeeksForGeeks.com: Imbalanced-Learn module in Python**, [https://www.geeksforgeeks.org/imbalanced-learn-module-in-python/](https://www.geeksforgeeks.org/imbalanced-learn-module-in-python/#:~:text=The%20fit_resample%20method%20resample%20the%20data%20and%20targets%20into%20a%20dictionary%20with%20a%20key%2Dvalue%20pair%20of%20data_resampled%20and%20targets_resampled.)

**glemaitre.github.io: imblearn.metrics.classification_report_imbalanced**, http://glemaitre.github.io/imbalanced-learn/generated/imblearn.metrics.classification_report_imbalanced.html
