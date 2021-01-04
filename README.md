# Optimizing an ML Pipeline in Azure

## Overview
This project is part of the Udacity Azure ML Nanodegree.
In this project, we build and optimize an Azure ML pipeline using the Python SDK and a provided Scikit-learn model.
This model is then compared to an Azure AutoML run. 

## Summary
**Problem statement:**  
This Dataset is regarding Bank-Marketing campain, which contains columns regarding age,marital status, contact type, type of job etc. We will use this dataset to predict if the customer will subscribe to the term deposite.  
**The Solution:**  
We are using Azure ML to predict the outcome. We are following two approaches to so that.  
1. Using HyperDrive to select Hyperparameters for Logistic Regression Algorithm of Scikit learn.  
2. Using AutoML to find the best performing model (I put fasle for Voting ensemble and stack ensemble in autoMLconfig)  
**The Best performing model was "SparseNormalizer, XGBoostClassifier" from AutoML with the accuracy 0.915 and accuracy of model using HyperDrive is 0.912**  
  
## Configuration Details  
We created workspace and cpu cluster to run the HyperDrive and AutoMl with vm_size='Standard_D2_V2' and max_nodes=4.  
## Scikit-learn Pipeline  
 
We used scikit learn Logistic Regrssion algorithm for the predicion as this task is about Classification. We created train.py script to do that.
***train.py***  
This script accepts two arguments.  
   1. C : Inverse of regularization [For More Details](https://stackoverflow.com/questions/22851316/what-is-the-inverse-of-regularization-strength-in-logistic-regression-how-shoul)  
   2. max_iter : Maximum number of iterations taken for the solvers to converge.  
  
Script contains a function named "clean_data" which cleans the data and one hot encode it.   
   e.g. changing month to integer value such as "jan" to 1  
  
Logistic Regression algorithm is used from the scikit learn.  

***Tuning the Hyperparameters and training the model***    
 If we tune the HyperParameters manually it will take log time. So, we used the Azure HyperDrive tool to find and tune the best Hyperparameters. We followed the below steps.  
   1.  Imported the required libraries.  
   2.  Used RandomParameterSampler as it is fast as compared to grid search and saves the budget to exhaustively search over the search space.  
       The parameter search spcae used for *C* and *max_iter* is choice(1,2,3,4,5) and choice(100,150,200,250) resp.  
       [For more Details on RandomSampling](https://docs.microsoft.com/en-us/azure/machine-learning/how-to-tune-hyperparameters)  
   3.  Specified the Early stopping policy as *BanditPolicy with slack_factor as 0.1, evaluation_interval as 1 and delay_evaluation as 5*   
       Bandit is an early termination policy based on slack factor/slack amount and evaluation interval.   
       The policy early terminates any runs where the primary metric is not within the specified slack factor/slack amount with respect to the best performing     training run.  
       [For more Details on Bandit Policy](https://azure.github.io/azureml-sdk-for-r/reference/bandit_policy.html)  
**The best HyperDrive model was found for "C = 2" and "max_iter = 150" with the accuracy of 0.912.**  

## AutoML  

It uses multiple algorithms and finds the best algorithms for the provided task. AutoML used LightGBM, XGBoostClassifier,RandomForest etc. I did not use Voting Ensemble and stack ensemble.  
So, provided false to these parameters in AutoMLconfig.  
**Best model with the highest accuracy 0.915 is XGBoostClassifier with SparseNormalizer.**  
[For More Details on XGBoost](https://towardsdatascience.com/a-beginners-guide-to-xgboost-87f5d4c30ed7)  

## Pipeline comparison  

**AutoMl and Hyperdrive achieved the accuracy over 0.90**. The accuracy achieved by the **AutoML** is slightly greater than the **HyperDrive** model which is **0.915.**  
Used below automl configuration for the automl    
```
automl_config = AutoMLConfig(  
                       experiment_timeout_minutes=30,    
                       task='classification',        # type of task    
                       primary_metric='accuracy',         
                       training_data=data_auto,
                       label_column_name='y',        # prediction column  
                       compute_target=cpu_cluster,      
                       enable_voting_ensemble=False, # I decided not to use votinfg and stack ensemble  
                       enable_stack_ensemble=False,  
                       enable_onnx_compatible_models=True,  
                       n_cross_validations=3)
 ```

To perform cross-validation, included the n_cross_validations parameter and set it to 3. This parameter sets how many cross validations to perform, based on the same number of folds.

In the above code, three folds for cross-validation are defined. Hence, three different trainings, each training using 2/3 of the data, and each validation using 1/3 of the data with a different holdout fold each time.
As a result, metrics are calculated with the average of the 3 validation metrics.
## Future work
We can use different primary metric to decide which is the best model. Also, we can consult to the person with the domain knowledge to decide which colums are irrelevant and which are relevant.
We can also gather data from different banks and check whether the data is same and if different hot it is affecting the outcome.
## Proof of cluster clean up  
**Image of cluster marked for deletion**  
![alt text](https://github.com/Bhosalenileshn/Optimizing-ML-Pipeline-Azure/blob/main/deleting%20_cluster.png)  
