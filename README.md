 <p align="center">
 <img src="https://user-images.githubusercontent.com/98360572/173111303-2afb6fbb-502b-49ff-b0a8-de95f88bbf8c.png" width="50%" height="50%">
</p>

# Module 17 Challenge: Supervised Machine Learning - Credit Risk Analysis
 
Because good loans outnumber risky loans, credit risk is an inherently unbalanced classification problem. As a result, it is neccesary to use a variety of techniques to train and evaluate models with unbalanced classes. For this purpose we will build and evaluate models with resampling using the imbalanced-learn and scikit-learn libraries.

We will also use the `RandomOverSampler` and `SMOTE` algorithms to oversample the credit card credit dataset from **LendingClub**, a peer-to-peer lending services company, and the `ClusterCentroids` algorithm to undersample the data. The `SMOTEENN` algorithm will then be used to perform a combinatorial approach of `over- and undersampling`. Then, to predict credit risk, we will compare two machine learning models that reduce bias, `BalancedRandomForestClassifier` and `EasyEnsembleClassifier`. After that, we will assess the performance of these models and conclude whether they should be used to predict credit risk.

---
# :one: Overview of the analysis: Explain the purpose of this analysis.

Manufacturers and publishers can get product reviews through the Amazon Vine program. Small fees are paid to Amazon by companies like SellBy, which then offer Amazon Vine users with items in exchange for a product review.

As part of this project, we had access to roughly 50 datasets. From clothing to wireless devices, each one discusses the pros and cons of a single product. The dataset selected for this analysis was **Pet Products**. After selecting the dataset, we used the PySpark Python library to perform an `ETL process` (**E**xtract, **T**ransform and **L**oad) to extract the information in the dataset, perform the neccesary transformations and load the data into `pgAdmin` via an `AWS RDS instance`.

As a further step, we then used `Python's PySpark library` to see if Amazon's Vine members are more likely to give positive evaluations of the products.

## Deliverable 1

