import unittest
import TrainSaveModel as tsm
import pandas as pd
import numpy as np
import io
import os
import random
from datetime import datetime, timedelta
from sklearn.utils.testing import ignore_warnings
from sklearn.exceptions import ConvergenceWarning
import shutil


def gen_datetime(min_year=1900, max_year=datetime.now().year):
    # generate a datetime in format yyyy-mm-dd hh:mm:ss.000000
    start = datetime(min_year, 1, 1, 00, 00, 00)
    years = max_year - min_year + 1
    end = start + timedelta(days=365 * years)
    return start + (end - start) * random.random()

def tempFile(source):
    if source[-5:] == '_temp':
        finDir = source[:-5]
        finDir = shutil.copyfile(source, finDir)
        os.remove(source)
        return finDir
    else:
        tempDir = source + '_temp'
        tempDir = shutil.copyfile(source, tempDir)
        return tempDir

class TestModel(unittest.TestCase):

    tsm.downloadCFPBDataset()

    head = pd.read_csv('complaints.csv', nrows=1000)
    head.to_csv('complaints.csv', index=False)

    def test_downloadCFPBDataset(self):
        #Ensure file was downloaded and is openable
        with open('complaints.csv') as f:
            self.assertIsInstance(f, io.TextIOWrapper)
        #ensures that complaints.csv.zip is deleted after extraction
        with self.assertRaises(FileNotFoundError):
            open('complaints.csv.zip')


    def test_convertToOther(self):
        keepList = ['One','Two','Three']
        testList = ['Three', 'Four', 'Five', 'Two', '']
        testAns = ['Three','Other','Other','Two', 'Other']
        #asserts empty string converts to 'Other'
        self.assertEqual(tsm.convertToOther('',[]), 'Other')
        #aasserts words not in keepList are turned to 'Other'
        self.assertEqual(tsm.convertToOther('Not', keepList), 'Other')
        #asserts that words in the keepList remain the same
        self.assertEqual(tsm.convertToOther('One', keepList), 'One')
        #performs test over any interable and asserts testAns is returned
        testedList = [tsm.convertToOther(val, keepList) for val in testList]
        self.assertEqual(testAns, testedList)

    def test_cleanReduceConvert(self):
        valList = ['val'+ str(val) for val in range(30)]
        df = pd.DataFrame({'vals':valList})
        #asserts columns with more than 23 unique values are grouped into 23 + 1('Other')
        self.assertEqual(24, len(tsm.cleanReduceConvert(df, 'vals').value_counts()))
        #asserst function what returns is a pandas.Series instance
        self.assertIsInstance(tsm.cleanReduceConvert(df,'vals'), pd.Series)
        #assert that after the clean up, there are no null values
        self.assertTrue(tsm.cleanReduceConvert(df, 'vals').notna().any())

    def test_entryOrNull(self):
        val1 = np.nan
        val2 = ''
        #asserts the values are transfromed correctly for both outcomes
        self.assertEqual(0.0, tsm.entryOrNull(val1))
        self.assertEqual(1.0, tsm.entryOrNull(val2))
        valList = [2, '12', np.nan, 'test', np.nan]
        df = pd.DataFrame({'vals':valList})
        testSeries = pd.Series(index=[1.0, 0.0])
        #asserts outcome of list are changed correctly
        self.assertEqual(df['vals'].apply(tsm.entryOrNull).value_counts().index[0], testSeries.index[0])
        self.assertEqual(df['vals'].apply(tsm.entryOrNull).value_counts().index[1], testSeries.index[1])

    def test_dtToCols(self):
        vals = [gen_datetime(), gen_datetime(), gen_datetime(), gen_datetime(), gen_datetime()]
        dtdf = pd.DataFrame({'test':vals})
        dtdf['test'] = pd.to_datetime(dtdf.test)
        tsm.dtToCols(dtdf, 'test')
        #asserst the final columns after creation
        self.assertEqual(dtdf.columns.tolist(), ['test','test day', 'test month','test year'])
        #asserts day stay within the maximum range of 31
        self.assertTrue(dtdf['test day'].le(31).all())
        #asserts month stay within the maximum range of 12
        self.assertTrue(dtdf['test month'].le(12).all())

    def test_readCSVToDataFrame(self):
        confirmedCols = ['Date received',
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
        #asserst index col assigned to complaint id
        df = tsm.readCSVToDataFrame()
        self.assertEqual(df.index.name, 'Complaint ID')
        #asserts the columns are as expected
        self.assertEqual(df.columns.to_list(),confirmedCols)
        #assert date columns were parsed
        self.assertEqual(df["Date received"].dtype, np.dtype('<M8[ns]'))
        self.assertEqual(df["Date sent to company"].dtype, np.dtype('<M8[ns]'))


    def test_cleanDf(self):
        df = tsm.readCSVToDataFrame()
        df = tsm.cleanDf(df)
        final_cols = ['Date received',
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
                    'Consumer disputed?',
                    'Consumer complaint narrative submitted?',
                    'Date received day',
                    'Date received month',
                    'Date received year',
                    'Date sent to company day',
                    'Date sent to company month',
                    'Date sent to company year']
        #asserts resulting dataframe has the columns in correct order
        self.assertEqual(df.columns.to_list(), final_cols)

    def test_dropUnusedCols(self):
        df = tsm.readCSVToDataFrame()
        df = tsm.cleanDf(df)
        X, Y = tsm.dropUnusedCols(df)
        #ensure data has no null values
        self.assertFalse(X.isna().any().any())
        final_cols = ['Product',
                    'Sub-product',
                    'Issue',
                    'Sub-issue',
                    'Company',
                    'ZIP code',
                    'Consumer consent provided?',
                    'Submitted via',
                    'Timely response?',
                    'Consumer complaint narrative submitted?',
                    'Date received day',
                    'Date received month',
                    'Date received year',
                    'Date sent to company day',
                    'Date sent to company month',
                    'Date sent to company year']
        #make sure after transformations, cols match up
        self.assertEqual(X.columns.to_list(), final_cols)

    def test_createPreprocessorAndTrain(self):
        df = tsm.readCSVToDataFrame()
        df = tsm.cleanDf(df)
        X, Y = tsm.dropUnusedCols(df)
        preprocessor = tsm.createPreprocessorAndTrain(X)
        #preprocessor is returned (Not nothing is returned)
        self.assertIsNotNone(preprocessor)

    def test_gridSearchTrainLogisticRegression(self):
        df = tsm.readCSVToDataFrame()
        df = tsm.cleanDf(df)
        X, Y = tsm.dropUnusedCols(df)
        preprocessor = tsm.createPreprocessorAndTrain(X)

        X_train, X_test, y_train, y_test = tsm.train_test_split(X, Y, test_size=0.3)

        encX_train = preprocessor.transform(X_train)
        encX_test = preprocessor.transform(X_test)

        bestfitLR, y_pred = tsm.gridSearchTrainLogisticRegression(encX_train, encX_test, y_train)

        #confrims the output of the model training
        self.assertIsInstance(y_pred, np.ndarray)

    def test_scoreLogger(self):
        y_pred = np.array(['Closed with relief','Closed with relief','Closed with relief','Closed without relief','Closed without relief'])
        y_test = np.array(['Closed with relief','Closed with relief','Closed with relief','Closed with relief','Closed without relief'])
        tsm.scoreLogger(y_test, y_pred)

        #asserts log files are created
        self.assertTrue(os.path.exists('./LogsAndModels/BestScore.log'))
        self.assertTrue(os.path.exists('./LogsAndModels/ModelScore.log'))
        self.assertTrue(os.path.exists('./LogsAndModels/ModelTrainingFailure.log'))
    
    def test_saveModel(self):
        df = tsm.readCSVToDataFrame()
        df = tsm.cleanDf(df)
        X, Y = tsm.dropUnusedCols(df)
        preprocessor = tsm.createPreprocessorAndTrain(X)

        X_train, X_test, y_train, y_test = tsm.train_test_split(X, Y, test_size=0.3)

        encX_train = preprocessor.transform(X_train)
        encX_test = preprocessor.transform(X_test)

        bestfitLR, y_pred = tsm.gridSearchTrainLogisticRegression(encX_train, encX_test, y_train)

        fscore, accuracy = tsm.scoreLogger(y_test, y_pred)

        self.assertTrue(os.path.exists('lrmodelpipeline.save'))

        tempLog = tempFile('./LogsAndModels/BestScore.log')
        tempModel = tempFile('lrmodelpipeline.save')

        with open('./LogsAndModels/BestScore.log', mode='w') as f:
            f.write("2020-02-06 11:02:30,200 | Fscore and Accuracy of best model |1.0| 0.1")
            f.close()
        
        tsm.saveModel(preprocessor, bestfitLR, fscore, accuracy)

        with open('./LogsAndModels/BestScore.log') as f:
            self.assertEqual(f.readlines()[0], "2020-02-06 11:02:30,200 | Fscore and Accuracy of best model |1.0| 0.1")
        
        with open('./LogsAndModels/BestScore.log', mode='w') as f:
            f.write("2020-02-06 11:02:30,200 | Fscore and Accuracy of best model |0.0| 0.1")
        
        tsm.saveModel(preprocessor, bestfitLR, fscore, accuracy)

        with open('./LogsAndModels/BestScore.log') as f:
            self.assertNotEqual(f.readlines()[0], "2020-02-06 11:02:30,200 | Fscore and Accuracy of best model |1.0| 0.1")

        tempFile(tempLog)
        tempFile(tempModel)

    def test_readScore(self):
        confirmDictKeys = ['date', 'precision', 'recall', 'fscore', 'accuracy']

        test_result = tsm.readScore()
        self.assertListEqual(list(test_result.keys()), confirmDictKeys)
        self.assertEqual(test_result, tsm.readScore(-1))

if __name__ == '__main__':
    unittest.main(exit=False)