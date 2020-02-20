import unittest
import LoadModelPredict as lmp
import numpy as np
import pandas as pd
import random
from datetime import datetime, timedelta
import pdb

def gen_datetime(min_year=1900, max_year=datetime.now().year):
    # generate a datetime in format yyyy-mm-dd hh:mm:ss.000000
    start = datetime(min_year, 1, 1, 00, 00, 00)
    years = max_year - min_year + 1
    end = start + timedelta(days=365 * years)
    return start + (end - start) * random.random()

class TestModel(unittest.TestCase):
    random.seed(42)
    complaintID = 3398126
    resp = lmp.checkAndGetComplaintData(complaintID)

    def test_getRespUsinArgv(self):
        with self.assertRaises(SystemExit) as cm:
            lmp.getRespUsinArgv(None)
        self.assertIsNone(cm.exception.code)

    
    def test_checkAndGetComplaintData(self):
        complaintIDreturn = int(self.__class__.resp.json()['hits']['hits'][0]['_source']['complaint_id'])
        #asserts complaint from api call equals the one inquired
        self.assertEqual(complaintIDreturn,self.__class__.complaintID)

        #asserts the status code for the response is indeed 200
        self.assertEqual(self.__class__.resp.status_code, 200)

        #Checks that 0 hits is returned correctly

        with self.assertRaises(SystemExit):
            badResp = lmp.checkAndGetComplaintData(00000000)
            self.assertEqual(badResp.json()['hits']['total'], 0)
    
    def test_convertToOther(self):
        keepList = ['One','Two','Three']
        testList = ['Three', 'Four', 'Five', 'Two', '']
        testAns = ['Three','Other','Other','Two', 'Other']
        #asserts empty string converts to 'Other'
        self.assertEqual(lmp.convertToOther('',[]), 'Other')
        #aasserts words not in keepList are turned to 'Other'
        self.assertEqual(lmp.convertToOther('Not', keepList), 'Other')
        #asserts that words in the keepList remain the same
        self.assertEqual(lmp.convertToOther('One', keepList), 'One')
        #performs test over any interable and asserts testAns is returned
        testedList = [lmp.convertToOther(val, keepList) for val in testList]
        self.assertEqual(testAns, testedList)

    def test_cleanReduceConvert(self):
        valList = ['val'+ str(val) for val in range(30)]
        df = pd.DataFrame({'vals':valList})
        #asserts columns with more than 23 unique values are grouped into 23 + 1('Other')
        self.assertEqual(24, len(lmp.cleanReduceConvert(df, 'vals').value_counts()))
        #asserst function what returns is a pandas.Series instance
        self.assertIsInstance(lmp.cleanReduceConvert(df,'vals'), pd.Series)
        #assert that after the clean up, there are no null values
        self.assertTrue(lmp.cleanReduceConvert(df, 'vals').notna().any())

    def test_entryOrNull(self):
        val1 = np.nan
        val2 = ''
        #asserts the values are transfromed correctly for both outcomes
        self.assertEqual(0.0, lmp.entryOrNull(val1))
        self.assertEqual(1.0, lmp.entryOrNull(val2))
        valList = [2, '12', np.nan, 'test', np.nan]
        df = pd.DataFrame({'vals':valList})
        testSeries = pd.Series(index=[1.0,0.0])
        #asserts outcome of list are changed correctly
        self.assertEqual(df['vals'].apply(lmp.entryOrNull).value_counts().index[0], testSeries.index[0])
        self.assertEqual(df['vals'].apply(lmp.entryOrNull).value_counts().index[1], testSeries.index[1])

    def test_dtToCols(self):
        vals = [gen_datetime(), gen_datetime(), gen_datetime(), gen_datetime(), gen_datetime()]
        dtdf = pd.DataFrame({'test':vals})
        dtdf['test'] = pd.to_datetime(dtdf.test)
        lmp.dtToCols(dtdf, 'test')
        #asserst the final columns after creation
        self.assertEqual(dtdf.columns.to_list(), ['test','test day', 'test month','test year'])
        #asserts day stay within the maximum range of 31
        self.assertTrue(dtdf['test day'].le(31).all())
        #asserts month stay within the maximum range of 12
        self.assertTrue(dtdf['test month'].le(12).all())

    def test_createDF(self):
        df = lmp.createDF(self.__class__.complaintID, self.__class__.resp)
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
        df = lmp.createDF(self.__class__.complaintID, self.__class__.resp)
        X, Y = lmp.dropUnusedCols(df)
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

    def test_loadModelPredict(self):
        df = lmp.createDF(self.__class__.complaintID, self.__class__.resp)
        X, Y = lmp.dropUnusedCols(df)
        pred, perc = lmp.loadModelPredict(X)
        #outcomes are only one of two outcomes
        self.assertTrue((pred in ['Closed with relief', 'Closed without relief']))
        #percentage is between 0 and 100
        self.assertGreaterEqual(perc, 0)
        self.assertLessEqual(perc,100)

if __name__ == '__main__':
    unittest.main(exit=False)