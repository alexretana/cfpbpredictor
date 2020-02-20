import sys
import getopt
import requests
import pandas as pd
import numpy as np
import joblib

#retrieves arguments and assigns it to complaintID
def getRespUsinArgv(argv):
    complaintID = None
    try:
        opts, args = getopt.getopt(argv, "hi:", ['complaint='])
    except getopt.GetoptError:
        print ('LoadModelPredict.py -i <complaintID>')
        sys.exit(2)
    for opt, arg in opts:
        if opt == 'h':
            print('LoadModelPredict.py -i <complaintID>')
        elif opt in ("-i", "--complaint"):
            complaintID = int(arg)
    if complaintID:
        print('Complaint #', complaintID, "'s outcome will be predicted:")
    else: 
        print('Complaint ID unable to be read.')
        sys.exit()
    return complaintID

#does API call and returns response
def checkAndGetComplaintData(complaintID):
    apiurl = 'https://www.consumerfinance.gov/data-research/consumer-complaints/search/api/v1/'
    queriedUrl = apiurl + str(complaintID)
    resp = requests.get(queriedUrl)
    if resp.status_code != 200:
        # This means something went wrong.
        print("ApiError: response status code", resp.status_code)
        sys.exit()
    if resp.json()['hits']['total'] == 0:
        print("Complaint ID yielded 0 result. Check to make sure you inputed it correctly.")
        sys.exit()
    return resp

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

#Transforms Narrative column to boolean response
def entryOrNull(strVal):
    return 1.0 if strVal is not np.nan else 0.0

#splits date column into three columns per dataframe column meant to be split
def dtToCols(df, dtcolumn):
        df["{} day".format(dtcolumn)] = df[dtcolumn].dt.day
        df["{} month".format(dtcolumn)] = df[dtcolumn].dt.month
        df["{} year".format(dtcolumn)] = df[dtcolumn].dt.year

#creates data frame and transforms it to agree with original model's trained dataframe
def createDF(complaintID, resp):
    #Creates DataFrame from REST API response
    df1 = pd.DataFrame(resp.json()['hits']['hits'][0]['_source'], index=[complaintID])

    #generate drop list for before preprocessing
    droplist1 = ['date_indexed_formatted',
                'complaint_id',
                'date_received_formatted',
                ':updated_at',
                'date_indexed',
                'date_sent_to_company_formatted',
                'has_narrative']
    #drop columns
    df_drop1 = df1.drop(droplist1,axis=1)

    #list that corrects names to agree with preprocessor's accepted name
    corrected_cols_dict = {'tags':'Tags',
                        'zip_code':'ZIP code',
                        'issue':'Issue',
                        'date_received':'Date received',
                        'state':'State',
                        'consumer_disputed':'Consumer disputed?',
                        'product':'Product',
                        'company_response':'Company response to consumer',
                        'submitted_via':'Submitted via',
                        'company':'Company',
                        'date_sent_to_company':'Date sent to company',
                        'company_public_response':'Company public response',
                        'sub_product':'Sub-product',
                        'timely':'Timely response?',
                        'complaint_what_happened':'Consumer complaint narrative',
                        'sub_issue':'Sub-issue',
                        'consumer_consent_provided':'Consumer consent provided?'}

    df_drop1 = df_drop1.rename(corrected_cols_dict, axis=1)

    #match order of columns. Generated with list(df.columns.values) from other notebooks
    reordered_cols= ['Date received',
                    'Product',
                    'Sub-product',
                    'Issue',
                    'Sub-issue',
                    'Consumer complaint narrative',
                    'Company public response',
                    'Company',
                    'State',
                    'ZIP code',
                    'Tags',
                    'Consumer consent provided?',
                    'Submitted via',
                    'Date sent to company',
                    'Company response to consumer',
                    'Timely response?',
                    'Consumer disputed?']
    #actually reorderes columns
    df_reordered = df_drop1[reordered_cols]

    #set index name to match
    df_reordered.index.name='Complaint ID'

    #define dictionary to define new dtypes
    dtype_dict = {'Product':"category",
                'Consumer consent provided?': "category",
                'Submitted via': "category",
                'Consumer disputed?': "category",
                'Date received':'<M8[ns]',
                'Date sent to company':'<M8[ns]'}

    #change dtypes
    df = df_reordered.astype(dtype_dict)

    #use old code to transfrom data

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

#data columns not be used for the model
def dropUnusedCols(df):
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

def loadModelPredict(X):

    pipeline_filename = "lrmodelpipeline.save"

    loaded_clf = joblib.load(pipeline_filename)
    prediction = loaded_clf.predict(X)[0]
    pred_proba_perc = loaded_clf.predict_proba(X)[0][0] * 100

    print("Prediction of Outcome: ", prediction)
    print("With a ", round(pred_proba_perc, 2) , "% chance of being Closed with relief")

    return prediction, pred_proba_perc

def main(complaintID):
    resp = checkAndGetComplaintData(complaintID)

    df = createDF(complaintID,resp)

    X, Y = dropUnusedCols(df)

    pred, pred_proba = loadModelPredict(X)

    return  [complaintID, pred, pred_proba]
    
if __name__ == '__main__':
    complaintID = getRespUsinArgv(sys.argv[1:])
    
    main(complaintID)