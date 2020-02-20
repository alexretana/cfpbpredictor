# Production Code

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