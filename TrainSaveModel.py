import pandas as pd
import numpy as np
import re
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler,OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import LabelEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import precision_recall_fscore_support, roc_auc_score, accuracy_score
from sklearn.utils.testing import ignore_warnings
from sklearn.exceptions import ConvergenceWarning
import warnings
import joblib
import wget
import zipfile
import os
import logging
import datetime

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

formatter = logging.Formatter('%(asctime)s | %(name)s | %(levelname)s | %(message)s')

file_handler = logging.FileHandler('./LogsAndModels/ModelTrainingFailure.log')
file_handler.setLevel(logging.ERROR)
file_handler.setFormatter(formatter)

file_handler_score = logging.FileHandler('./LogsAndModels/ModelScore.log')
file_handler_score.setLevel(logging.INFO)
file_handler_score.setFormatter(formatter)

logger.addHandler(file_handler)
logger.addHandler(file_handler_score)

logbestscore = logging.getLogger("BestScoreLogger")
logbestscore.setLevel(logging.INFO)
bestFormatter = logging.Formatter('%(asctime)s | %(message)s')
best_score_handler = logging.FileHandler('./LogsAndModels/BestScore.log', 'w')
best_score_handler.setLevel(logging.INFO)
best_score_handler.setFormatter(bestFormatter)
logbestscore.addHandler(best_score_handler)

warnings.simplefilter(action='ignore', category=FutureWarning)

def readScore(entry = -1):
    try:
        with open("./LogsAndModels/ModelScore.log") as f:
            scoreString = f.readlines()[entry]
            precisionString = float(scoreString.split("|")[4].strip())
            recallString = float(scoreString.split("|")[5].strip())
            fscoreString = float(scoreString.split("|")[6].strip())
            accuracyString = float(scoreString.split("|")[7].strip())
            dateString = scoreString.split("|")[0].split(",")[0]
    except:
        TestScoreObj = {'date' : 'Null',
                        'precision': 0.0,
                        'recall': 0.0,
                        'fscore': 0.0,
                        'accuracy': 0.0}
    else:
        TestScoreObj = {'date' : dateString,
                        'precision': precisionString,
                        'recall': recallString,
                        'fscore': fscoreString,
                        'accuracy': accuracyString}
    finally:
        return TestScoreObj

def downloadCFPBDataset():
    url = 'http://files.consumerfinance.gov/ccdb/complaints.csv.zip'
    wget.download(url, './complaints.csv.zip')

    with zipfile.ZipFile('./complaints.csv.zip') as z:
        # extract /res/drawable/icon.png from apk to /temp/...
        z.extract('complaints.csv', '.')

    os.remove("complaints.csv.zip")

#function to apply to column to convert less common results to 'Other', as well as NaN
def convertToOther(value, keepList):
    if (value == ''):
        return "Other"
    else:
        return value if value in keepList else "Other"

#Lists top 23 value counts (allowed to exclude values), turns NaN to '' to others, converts to category dtype
def cleanReduceConvert(df, column, blackList=[]):
    keepList = []
    for category in df[column].value_counts().head(23).index.tolist():
        if (category.lower().split()[0] != "other"):
            keepList.append(category)
    for category in blackList:
        try:
            keepList.remove(category)
        except ValueError:
            pass

    df[column].fillna('', inplace=True)
    return pd.Series(df[column].apply(convertToOther, args=(keepList,)), dtype = 'category')

def entryOrNull(strVal):
    return 1.0 if strVal is not np.nan else 0.0

def dtToCols(df, dtcolumn):
    df["{} day".format(dtcolumn)] = df[dtcolumn].dt.day
    df["{} month".format(dtcolumn)] = df[dtcolumn].dt.month
    df["{} year".format(dtcolumn)] = df[dtcolumn].dt.year

def readCSVToDataFrame():
    #Defining what dtype to convert each column to
    #numberic columns are transformed after reading in
    dtype_dict = {'Product':"category",
                'Consumer consent provided?': "category",
                'Submitted via': "category",
                'Company response to consumer': "category",
                'Consumer disputed?': "category"}

    #read in .csv file, dates are parsed into datetime objects. 
    #The Complaint ID is Unique in every entry, so it can be used as index
    df = pd.read_csv('./complaints.csv',
                    index_col=['Complaint ID'],
                    parse_dates=["Date received","Date sent to company"],
                    dtype=dtype_dict)

    return df

def cleanDf(df):
    #This will replace ending '-' to 5 (average linespace of 10)
    regexReplaceDash = r"(\d+)(-)$"
    df['ZIP code'] = df['ZIP code'].str.replace(regexReplaceDash, r'\g<1>5')

    #This will change ending XX to 50 (average linespace of 100)
    regex_XX = r'(\d{3})(XX)'
    df['ZIP code'] = df['ZIP code'].str.replace(regex_XX, r'\g<1>50')

    #This will remove all other entries that are still not 5 digits
    regexRemove = r'\D+'
    df['ZIP code'] = df['ZIP code'].replace(regexRemove, np.nan, regex=True)

    #imputes the mean for nan 
    imputeMean = df['ZIP code'].astype(np.float).mean()
    df['ZIP code'] = df['ZIP code'].astype(np.float).fillna(imputeMean)

    #Transforming 2 unique valued col to float boolean
    booleanize = {'Yes': 1, 'No': 0}
    df['Timely response?'] = pd.Series(df['Timely response?'].map(booleanize), dtype = np.float)



    df['Sub-product'] = cleanReduceConvert(df, 'Sub-product', blackList= ['I do not know'])
    df['Issue'] = cleanReduceConvert(df, 'Issue')
    df['Sub-issue'] = cleanReduceConvert(df, 'Sub-issue')
    df['Company'] = cleanReduceConvert(df, 'Company')

    df['Consumer complaint narrative submitted?'] = df['Consumer complaint narrative'].apply(entryOrNull)
        
    dtToCols(df, "Date received")
    dtToCols(df, "Date sent to company")

    df["Consumer consent provided?"] = df["Consumer consent provided?"].cat.add_categories("Not recorded").fillna("Not recorded")

    df = df.drop(df[df["Company response to consumer"].isna()].index)

    dfInProgress = df[df["Company response to consumer"] == "In progress"]
    df = df[df["Company response to consumer"] != "In progress"]

    dfUntimelyResponse = df[df["Company response to consumer"] == "Untimely response"]
    df = df[df["Company response to consumer"] != "Untimely response"]

    twoOutputsDict = {"Closed with explanation":"Closed without relief", 
                    "Closed with non-monetary relief":"Closed with relief",
                    "Closed with monetary relief":"Closed with relief",
                    "Closed without relief":"Closed without relief", 
                    "Closed":"Closed without relief",
                    "Closed with relief":"Closed with relief"}
    df["Company response to consumer"] = df["Company response to consumer"].map(twoOutputsDict)

    return df

def dropUnusedCols(df):
    #data columns not be used for the model
    dropList = ["Consumer complaint narrative",
                "Company public response",
                "State",
                "Tags",
                "Consumer disputed?",
                "Date received", 
                "Date sent to company",
                "Company response to consumer"]
    X = df.drop(dropList, axis=1)
    Y = df["Company response to consumer"]

    return X, Y 

def createPreprocessorAndTrain(X):
    #Columns to be standard scaled/imputed
    numeric_features = ['ZIP code',
                        'Date received day',
                        'Date received month',
                        'Date received year',
                        'Date sent to company day',
                        'Date sent to company month',
                        'Date sent to company year']
    numeric_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='median')),
        ('scaler', StandardScaler())])

    #Columns to one hot encoded
    categorical_features = ['Product',
            'Sub-product',
            'Issue',
            'Sub-issue',
            'Company',
            'Consumer consent provided?',
            'Submitted via',
            'Timely response?']
    categorical_transformer = Pipeline(steps=[('onehot', OneHotEncoder(handle_unknown='ignore'))])

    #building the column transformer with both transformers
    preprocessor = ColumnTransformer(
        transformers=[
            ('num', numeric_transformer, numeric_features),
            ('cat', categorical_transformer, categorical_features)])

    #fit the preprocessor
    preprocessor.fit(X)

    return preprocessor

@ignore_warnings
def gridSearchTrainLogisticRegression(encX_train,encX_test, y_train):

    lr = LogisticRegression(n_jobs=-1, solver='saga', penalty='l1')
    lr_para = {'C':[1.0,0.1,0.01], 
            'class_weight':[None,'balanced'],
            'max_iter':[50,100,150]}

    #Apply grid search with above parameters specified
    fitmodel = GridSearchCV(lr, lr_para,cv=4, scoring='roc_auc', n_jobs=-1)
    fitmodel.fit(encX_train,y_train)

    #store the best fitting LogisiticRegression(), create prediciton from X_test data
    bestfitLR = fitmodel.best_estimator_
    y_pred = bestfitLR.predict(encX_test)

    return bestfitLR, y_pred

def scoreLogger(y_test, y_pred):
    (precision, recall, fscore, support) = precision_recall_fscore_support(y_test, y_pred, 
                                    labels = ['Closed with relief', 'Closed without relief'], 
                                    pos_label= 'Closed with relief', average= 'binary')
    
    accuracy = accuracy_score(y_test,y_pred)

    scoreString = " | ".join(str(score) for score in [precision,recall,fscore,accuracy])

    message = 'Scores in this order (precision, recall, fscore, accuracy) | ' + scoreString

    logger.info(message)

    return fscore, accuracy

def saveModel(preprocessor, bestfitLR, fscore, accuracy):
    #creates pipeline object to save
    clf = Pipeline(steps=[('preprocessor', preprocessor),
                        ('logregclassifier', bestfitLR)])

    #save model with date label to logandmodel folder
    timestring = datetime.datetime.now().strftime('%Y-%m-%d_%H-%M-%S')
    pipeline_filename = "./LogsAndModels/lrmodelpipeline" + timestring + ".save"
    joblib.dump(clf, pipeline_filename)

    #checks bestscore, writes if new model score is better or there is no model to begin with
    scoreMessage = "Fscore and Accuracy of best model |" + str(fscore) + "|" + str(accuracy)
    best_pipeline_filename = "lrmodelpipeline.save"

    if os.stat("./LogsAndModels/BestScore.log").st_size == 0:
        logbestscore.info(scoreMessage)
        joblib.dump(clf, best_pipeline_filename)
    else:
        with open("./LogsAndModels/BestScore.log") as f:
            scoreString = f.readlines()[0]
            scoreString = float(scoreString.split("|")[2])
        if (scoreString < fscore):
            logbestscore.info(scoreMessage)
            joblib.dump(clf, best_pipeline_filename)
            return 200
        else:
            return 201


def main():

    downloadCFPBDataset()

    df = readCSVToDataFrame()

    df = cleanDf(df)

    X, Y = dropUnusedCols(df)

    preprocessor = createPreprocessorAndTrain(X)

    X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.3)

    encX_train = preprocessor.transform(X_train)
    encX_test = preprocessor.transform(X_test)

    bestfitLR, y_pred = gridSearchTrainLogisticRegression(encX_train, encX_test, y_train)

    fscore, accuracy = scoreLogger(y_test, y_pred)

    status_code = saveModel(preprocessor, bestfitLR, fscore, accuracy)

    return readScore(), status_code

def run():
    try:
        model_score, status_code = main()
    except Exception:
        logger.exception('An ERROR has occured that stopped the model traing from finishing. Providing stack trace')
    else:
        return model_score, status_code

if __name__ == '__main__':
    run()