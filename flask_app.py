from flask import Flask, request, Blueprint, abort
from flask_restplus import Resource, Api, fields, Model
import TrainSaveModel as tsm
import LoadModelPredict as lmp

app = Flask(__name__)
api = Api(app, version='1.0.0', title= 'CFPB Complaint Predictor', description= 'An API to predict the Resolution Outcome of Complaints submitted to the Consumer Financial Protection Bureau (CFPB)')
dev_ns = api.namespace('devs', description= 'Tools for developers')
admin_ns = api.namespace('admin', description= 'Tools for admin to interact with models')

adminkey = api.model('adminKey', {'adminkey': fields.String(default=None, required=True, description='Passcode to access admin tools')})

prediction = api.model('prediction', 
                        {'complaintID': fields.Integer(default= None, required=True, example = 3398126, description= 'Compalint ID Number'),
                        'Outcome_Prediction': fields.String(default=None, example= 'Closed without relief', description='Outcome of prediction'),
                        'Proba': fields.Float(default=None, example= 20.31, description='Percent chance of resolution with relief')})

model_scores = api.model('model_scores',
                        {'date': fields.String(default=None,description='String describing time and date of training for a model', example='2020-02-11 14:50:40', required=True),
                        'precision': fields.Float(default=None,description='Precision score of model', example=0.324879),
                        'recall': fields.Float(default=None,description='Recall score of model', example=0.666638),
                        'fscore': fields.Float(default=None,description='F1-score of model', example=0.436859),
                        'accuracy': fields.Float(default=None,description='Accuracy score of model', example=0.6749668)})

score_log = api.model('score_log', 
                    {'score_log': fields.List(fields.Nested(model_scores), description= "List of model_scores for all scores currently store in server logfile")})


@dev_ns.route('/prediction/<int:complaintID>', endpoint='prediciton')
class Prediction(Resource):
    @dev_ns.doc(params={ 'complaintID': 'Number that indentifies consumer\'s complaint'})
    @dev_ns.response(200, 'OK', prediction)
    @dev_ns.response(400, 'Error Occured')
    @dev_ns.response(401, 'Complaint could not be found')
    @dev_ns.marshal_with(prediction)
    def get(self, complaintID):
        """
        Returns the prediction of a complaint submitted. Requires valid complaintID
        """
        apiModelPrediction = ['complaintID', 'Outcome_Prediction', 'Proba']
        try:
            pred = lmp.main(complaintID)
        except:
            abort(401)
        else:
            predictionObj = dict(zip(apiModelPrediction, pred))
            return predictionObj

@admin_ns.route('/train_model/<string:adminkey>')
@admin_ns.param('adminkey', "Passcode to access admin tools")
class Train_model(Resource):
    @admin_ns.response(200, 'OK', model_scores)
    @admin_ns.response(201, 'Model Trained, but did not out preform the previous model', model_scores)
    @admin_ns.response(400, 'Model Training Failed')
    @admin_ns.response(403, "Request is Unauthorized")
    @admin_ns.marshal_with(model_scores)
    def put(self, adminkey):
        """
        Trains the model in the server. If the model is best to date, it becomes the primary model
        """
        if adminkey != 'glasszhuanimals':
            abort(403)
    
        model_score, status_code = tsm.run()
        return model_score, status_code

    @admin_ns.response(200, 'OK', score_log)
    @admin_ns.response(400, 'Unable to fetch log')
    @admin_ns.response(403, "Request is Unauthorized")
    @admin_ns.marshal_list_with(model_scores)
    def get(self, adminkey):
        """
        Returns json of all model scores. All the model scores in the ModelScore.log file are returned
        """
        if adminkey != 'glasszhuanimals':
            abort(403)

        entry_count = len(open('./LogsAndModels/ModelScore.log').readlines())
        score_log = []

        for i in range(entry_count):
            log_entry = tsm.readScore(i)
            score_log.append(log_entry)
        
        return score_log

if __name__ == '__main__':
    app.run(host='0.0.0.0')
