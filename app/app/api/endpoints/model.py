from . import utils
import time
import datetime
import sys
import os
import pandas as pd
import json
from ...main import api
from flask import request
from flask_restplus import Resource, fields

import warnings
warnings.simplefilter("ignore")


feature = api.model('Feature', {'feature': fields.String(example='EmployeeType'), 'importance': fields.Float(example=0.0026227492)})


model = api.model('Model Data', { 'algorithm': fields.String(example='classification'), 'trained': fields.String(example='2018-10-06T00:00:00.000Z'), 'score': fields.Float(example=0.6742780779733891), 'trainset': fields.String(example='(18237, 445)'), 'testset': fields.String(example='(17211, 445)'), 'featureimportance': fields.List(fields.Nested(feature))})


modelByCompany = api.model('Model By Company', {
    'companyid': fields.Integer(required=True, description='Comapny ID', example=506433),
    'models': fields.List(fields.Nested(model))
})

allmodels = api.model('Companies Models', {
    'companies': fields.List(fields.Nested(modelByCompany))
})

@api.route("/model")
class Model(Resource):
    @api.marshal_with(allmodels)
    def get(self):
        return getModels()


@api.route("/model/<string:companyid>")
class ModelByCompany(Resource):
    @api.marshal_with(modelByCompany)
    def get(self, companyid):
        return getModel(companyid)


package_directory = os.path.dirname(os.path.abspath(__file__))

def getModelInfo(local_model):
    model = {}
    if os.path.isdir(local_model):  
        model['algorithm']  = local_model.split('/')[-1]
        model['trained'] = str(time.ctime(os.path.getctime(local_model + '/model.pkl')))
        model['score'] = open(local_model + '/model.score').read().splitlines()[0]
        model['trainset'] = open(local_model + '/train.set').read().splitlines()[0]
        model['testset'] = open(local_model + '/test.set').read().splitlines()[0]
        model['featureimportance'] = json.loads(pd.read_csv(local_model + '/featuresImportance.csv').to_json(orient='records', date_format='iso'))
    return model

def getModelByCompany(companyid): 
    local_class_model = utils.model_dir + companyid + '/classification'
    local_reg_model = utils.model_dir + companyid + '/regression'
    
    data = {}
    data['companyid'] = companyid
    data['models'] = [getModelInfo(local_class_model), getModelInfo(local_reg_model)]

    return data

def getModel(companyid):
    sys.stdout = open(utils.log_dir + time.strftime("%Y%m%d-%H%M%S") + '_model.txt', 'w')
    
    print(datetime.datetime.now(), 'Get model for company', companyid)
    
    data = getModelByCompany(companyid)
    
    sys.stdout.close()
    return data

def getModels():
    sys.stdout = open(utils.log_dir + time.strftime("%Y%m%d-%H%M%S") + '_model.txt', 'w')
    
    result = {}
    result['companies'] = []

    for companyid in [name for name in os.listdir(utils.model_dir) if os.path.isdir(os.path.join(utils.model_dir, name)) &  (name != '.git')]:
        print(datetime.datetime.now(), 'Get model for company', companyid)
        data = getModelByCompany(companyid)
        result['companies'].extend([data])

    sys.stdout.close()
    return result