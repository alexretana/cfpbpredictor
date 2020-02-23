# Predict from Financial Service Complaint Data How a Claim Will Resolve

The Consumer Financial Protection Bureau sends complaints from consumer to companies regarding issues with financial services, which updates daily, and can be retrieved from an API1. These issues range from car financing, to credit card issues, to banking problems. Claims can result in being closed with relief, explanation, non-monetary relief, monetary relief. 

An objective of interest  is making a predictive response for complaints still in progress. This could be used in focusing complaint that merit relief first, since there is legitimate compensations to be distributed, and there are significantly less complaints that received relief compared to how many are closed with an explanation. As a secondary objective, a classifier will be made that treats each type of company responses as a separate category. Scikit-learn will be utilized, as well as keras with a tensorflow backend, and a few Natural Language Processing (NLP) libraries to create my pipeline.

The strategy here is to start off by seeing which columns are redundant and which columns have too many missing values, and then remove those columns from the model. A Grid Search Cross Validation Method on a handful of different prediction models to see which model performs best. Models that will be trained through the GridSearch include Random Tree Classifier, Gradient Boosted Tree Classifier, Logistic Regression. From logistic regression and gradient boosted tree classifier, feature importance can be extracted to determine which factors influence the prediction most. 

To continue the investigation, a Deep Neural Network (DNN) model will be built and trained to see if it can out match the classic Machine Learning models. About 30% of the complaints have consumer submitted narrative. Using these text samples, basic NLP techniques will be used to check if using a narrative has more predictive power than the other information. 

After determining which model is best, I will build an updater that will check if new entries have been added and make a prediction for them, as well as to check if incomplete complaints have been resolved and create a score a validation score with them. This is easily possible because the data set has an API that updates everyday.

# API usage

Currently the API is being served from 

Following the link provides the swagger documentation.

There are three operations:

GET predition/{complaintId}: fetches a complaint submitted to the CFPB database and returns the outcome's prediction

GET train_model/{adminkey}*: Retrieves the model's test scores for all previously trained models

POST train_model/{adminkey}*: Commands server to train a new model and update working model if a better score is obtained

*These HTTP calls require knowing a password to trigger them

## Production Code

This folder includes two python scripts:
- TrainSaveModel.py: file that uses downloaded dataset to train a Logisitc Regession Model and saves it to lrmodelpipeline.save
- LoadModelPredict.py: file that accepts a complaint ID, calls the api for the complaint, loads the model in lrmodelpipeline.save, and predicts the outcome and probability of the outcome.

## TrainSaveModel.py

To run this file, the csv from the CFPB's website at the link http://files.consumerfinance.gov/ccdb/complaints.csv.zip must be downloaded and put in the folder outside of the production folder. After running, the model will create or overwrite the lrmodelpipeline.save file. Running the file requires no arguments

## LoadModelPredict.py

To run this file, the lrmodelpipeline.save must be in the same directory. It also requires one arguement, specified with the flag -i or --complaint, the complaint ID. Using this code, it connects to the CFPB's Api for database access and pulls all the submitted information. The file tansfroms the data, and pushes through the logistic regression model, then prints to screen the most likely outcome, along with the probability of being resolved with relief.

## test_LoadModelPredict.py & test_TrainSaveModel.py

This is a unittest file that ensures the functionality of the LoadModelPredict.py and TrainSaveModel.py including all the defined function in the file.

## flask_app.py

Python Flask File, uses flask and flask restplus to serve a basic api to call the methods of the other python scripts. The swagger documentation can be seen making a get request to the base url.