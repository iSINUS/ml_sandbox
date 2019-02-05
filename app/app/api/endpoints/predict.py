from . import preprocess
from . import utils
import random
import pickle
import feather
import time
import datetime
import sys
import os
import numpy as np
import pandas as pd
import json
from ...main import api
from flask import request
from flask_restplus import Resource, fields

import warnings
warnings.simplefilter("ignore")


predict_fields = api.model('Prediction Data', {
})

predict_accounts = api.model('Prediction Data By Employee', {
    
})

prediction = api.model('Prediction', {'attritionproba': fields.Float(
    example=0.345), 'attritiondate': fields.String(example='2020-10-06T00:00:00.000Z')})

predictionByEmployee = api.model('Prediction By Employee', {})

model = api.model(
    'Predictions', {'predictions': fields.List(fields.Nested(prediction))})

modelByEmployee = api.model(
    'Predictions By Employee', {'predictions': fields.List(fields.Nested(predictionByEmployee))})

parser = api.parser()
parser.add_argument('predictdate', location='args', default=datetime.date.today().strftime("%Y-%m-%d"), help='Predict date', required=True)


@api.route("/predict")
@api.expect(parser)
class Predict(Resource):
    @api.expect(predict_fields)
    @api.marshal_with(model)
    def post(self):
        args = parser.parse_args()
        return getPrediction(request.get_json(), args['predictdate'])


@api.route("/predict/<string:companyid>/<string:accountid>")
@api.expect(parser)
class PredictEmployeeByCompany(Resource):
    @api.marshal_with(modelByEmployee)
    def get(self, companyid, accountid):
        args = parser.parse_args()
        return getPredictionByEmployee(companyid, [int(accountid)], args['predictdate'])


@api.route("/predict/<string:companyid>")
@api.expect(parser)
class PredictByCompany(Resource):
    @api.marshal_with(modelByEmployee)
    def get(self, companyid):
        args = parser.parse_args()
        return getPredictionByEmployee(companyid, None, args['predictdate'])

    @api.expect(predict_accounts)
    @api.marshal_with(modelByEmployee)
    def post(self, companyid):
        args = parser.parse_args()
        return getPredictionByEmployee(companyid, request.get_json()['accountids'], args['predictdate'])


package_directory = os.path.dirname(os.path.abspath(__file__))


def predict_class(local_model, df):
    if os.path.isfile(local_model):
        model = pickle.load(open(local_model, 'rb'))
        result = pd.Series(model.predict_proba(df)[:, 1])
    else:
        result = pd.Series(random.sample(
            range(1000), df.shape[0])).divide(10000)

    return result


def predict_reg(local_model, df):
    if os.path.isfile(local_model):
        model = pickle.load(open(local_model, 'rb'))
        result = pd.Series(model.predict(df)).apply(int).clip(lower=0)
    else:
        result = pd.Series(random.sample(range(100, 1000), df.shape[0]))
    return result


def getPrediction(data, predictdate=np.datetime64('today')):

    request_json = data

    if request_json and 'instances' in request_json and 'companyid' in request_json and 'columns' in request_json:
        sys.stdout = open(utils.log_dir + time.strftime("%Y%m%d-%H%M%S") + '_predict.txt', 'w')
        # copy model
        companyid = str(request_json['companyid'])
        print(datetime.datetime.now(), 'Predict for company', companyid)
        local_class_model = utils.model_dir + companyid + '/classification/model.pkl'
        local_reg_model = utils.model_dir + companyid + '/regression/model.pkl'
        columns = request_json['columns']
        df = pd.DataFrame(request_json['instances'], columns=columns)
        df_1 = preprocess.preprocessDF(df, utils.model_dir + companyid + '/', predictdate)
        df_1 = df_1.drop(['CompId', 'AccountId', 'AttritionReasonId', 'AttritionDays', 'IsAttrition', 'ReasonId'], axis=1, errors='ignore')
        data = {}
        result_class = predict_class(local_class_model, df_1)

        result_reg = predict_reg(local_reg_model, df_1)

        df['HiredOrReHired'] = df['HiredOrReHired'].astype('datetime64[D]')
        result_date = df['HiredOrReHired'] + pd.to_timedelta(result_reg, 'D')

        data['predictions'] = json.loads(pd.DataFrame({'attritionproba': result_class, 'attritiondate': result_date}).to_json(orient='records', date_format='iso'))
        sys.stdout.close()
        return data
    else:
        return {'attritionproba': 0, 'attritiondate': ''}


def getPredictionByEmployee(companyid, accountid=None, predictdate=np.datetime64('today')):
    sys.stdout = open(
        utils.log_dir + time.strftime("%Y%m%d-%H%M%S") + '_predict.txt', 'w')
    # copy model

    print(datetime.datetime.now(), 'Predict for company', companyid)
    local_class_model = utils.model_dir + companyid + '/classification/model.pkl'
    local_reg_model = utils.model_dir + companyid + '/regression/model.pkl'

    if np.datetime64(predictdate) >= np.datetime64('today'):
        strtodate = ''
    else:
        strtodate = np.datetime64(predictdate).astype(datetime.datetime).strftime('%Y%m')
   
    if os.path.isfile(utils.data_dir + companyid + '/preparedData_test' + strtodate + '.feather'):
        df = feather.read_dataframe(utils.data_dir + companyid + '/preparedData_test' + strtodate + '.feather')
    else:
        df = pd.read_csv(utils.data_dir + companyid + '/preparedData_test' + strtodate + '.csv', low_memory=False)
        feather.write_dataframe(df, utils.data_dir + companyid + '/preparedData_test' + strtodate + '.feather')
    
    if os.path.isfile(utils.model_dir + companyid + '/preprocessedData_test' + strtodate + '.feather'):
        df_1 = feather.read_dataframe(utils.model_dir + companyid + '/preprocessedData_test' + strtodate + '.feather')
    else:
        df_1 = pd.read_csv(utils.model_dir + companyid + '/preprocessedData_test' + strtodate + '.csv', low_memory=False)
        feather.write_dataframe(df_1, utils.model_dir + companyid + '/preprocessedData_test' + strtodate + '.feather')

    if accountid:
        df = df.loc[(df['CompId'] == int(companyid)) & (df['AccountId'].isin(accountid))].reset_index(drop=True)
        df_1 = df_1.loc[(df_1['CompId'] == int(companyid)) & (df_1['AccountId'].isin(accountid))].reset_index(drop=True)
    else:
        df = df.loc[(df['CompId'] == int(companyid))]
        df_1 = df_1.loc[(df['CompId'] == int(companyid))]

    #df_1 = preprocess.preprocessDF(df, utils.model_dir + companyid + '/', np.datetime64(predictdate))

    df_1 = df_1.drop(['CompId', 'AccountId', 'AttritionReasonId', 'AttritionDays', 'IsAttrition', 'ReasonId'], axis=1, errors='ignore')
    print(datetime.datetime.now(), 'Predict for data', df_1.shape)

    data = {}
    result_class = predict_class(local_class_model, df_1)

    result_reg = predict_reg(local_reg_model, df_1)

    df['HiredOrReHired'] = df['HiredOrReHired'].astype('datetime64[D]')
    result_date = df['HiredOrReHired'] + pd.to_timedelta(result_reg, 'D')

    data['predictions'] = json.loads(pd.DataFrame(
        {'accountid': df['AccountId'], 'attritionproba': result_class, 'attritiondate': result_date}).to_json(orient='records', date_format='iso'))
    sys.stdout.close()
    return data
