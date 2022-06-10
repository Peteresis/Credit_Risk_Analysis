 <p align="center">
 <img src="https://user-images.githubusercontent.com/98360572/173111303-2afb6fbb-502b-49ff-b0a8-de95f88bbf8c.png" width="50%" height="50%">
</p>

# Module 17 Challenge: Supervised Machine Learning - Credit Risk Analysis
 
Because good loans outnumber risky loans, credit risk is an inherently unbalanced classification problem. As a result, it is neccesary to use a variety of techniques to train and evaluate models with unbalanced classes. For this purpose we will build and evaluate models with resampling using the imbalanced-learn and scikit-learn libraries.

We will also use the `RandomOverSampler` and `SMOTE` algorithms to oversample the credit card credit dataset from **LendingClub**, a peer-to-peer lending services company, and the `ClusterCentroids` algorithm to undersample the data. The `SMOTEENN` algorithm will then be used to perform a combinatorial approach of `over- and undersampling`. Then, to predict credit risk, we will compare two machine learning models that reduce bias, `BalancedRandomForestClassifier` and `EasyEnsembleClassifier`. After that, we will assess the performance of these models and conclude whether they should be used to predict credit risk.

---
# :one: Overview of the analysis: Explain the purpose of this analysis.

El objetivo de este analisis es aplicar seis modelos diferentes de machine learning a un mismo dataset y comparar los resultados obtenidos.  El dataset utilizado contiene las aplicaciones de crédito para el primer trimestre del 2019, obtenido de una empresa de créditos personales llamada LendingClub.  El archivo de datos original contiene la información de `115,675` aplicaciones.

El estudio de los datos se hace con dos archivos de codigo de Python, usando Jupyter Notebooks:

[credit_risk_resampling.ipynb](https://github.com/Peteresis/Credit_Risk_Analysis/blob/e08cba0cb4d441409df51a5e494665d156c13120/credit_risk_resampling.ipynb)
[credit_risk_ensemble.ipynb](https://github.com/Peteresis/Credit_Risk_Analysis/blob/e08cba0cb4d441409df51a5e494665d156c13120/credit_risk_ensemble.ipynb)




Usando la librería Pandas de Python

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


