# Predict from Financial Service Complaint Data How a Claim Will Resolve

The Consumer Financial Protection Bureau (CFPB) sends complaints from consumer to companies regarding issues with financial services, which updates daily, and can be retrieved from an API1. These issues range from car financing, to credit card issues, to banking problems. Claims can result in being closed with relief, explanation, non-monetary relief, monetary relief. If one were to estimate the probibility of resolution, finacial firms could use this metric to prioritize their helpdesk task assignment, in turn leading to better customer satisfaction.

This project utilizes the full dataset provided by the CFPB, multiple preprocessing pipelines, such as data imputation, categorizing columns, and standard scaling, and a Logistic Regression Model to predict weather a consumer is entitled to compensation. The productionalize version of the code hosts on a server using the up-to-date best trained model. When a consumer calls the api, it returns their predicted outcome along with a probility value. The admin who host the server can always make a call to retrain the model, or return the log of all the scores.

Note that the design and model decisions were planned out through a trail error process that is documented at https://github.com/alexretana/Springboard/tree/master/Capstone-Finacial_Products_Complaints.

At that github repo, you'll see investigation into the data in question, various models tested included deep neural networks and natural language processing.

# API usage

Currently the API is being served from https://cfpbpredictorwebapi.azurewebsites.net/

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