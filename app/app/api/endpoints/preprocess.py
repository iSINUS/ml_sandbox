from ...main import celery, api
from flask_restplus import Resource, fields

import pandas as pd
import numpy as np
import os
import sys
import datetime
import time
import pickle
from sklearn import preprocessing

from . import utils
parser = api.parser()
parser.add_argument('traindate', location='args',
                    default=datetime.date.today().strftime("%Y-%m-%d"),
                    help='Train date for dataset', required=True)

model = api.model('Task', {'task_id': fields.String(
    example='ac435180-23d4-48aa-82e5-30af102847cc')})


@api.route("/preprocess")
@api.expect(parser)
class Preprocess(Resource):
    @api.marshal_with(model, code=202)
    def get(self):
        args = parser.parse_args()
        task = preprocessData.delay(args['traindate'])
        return {'task_id': task.id}


def preprocessDF(df, path, traindate):
    # ...

    return df


@celery.task(bind=True)
def preprocessData(self, traindate):
    def preprocess(companyid, traindate):
        df_train = pd.read_csv(utils.data_dir + companyid + '/preparedData_train.csv', low_memory=False).set_index(['CompId', 'AccountId'])
        df_test = pd.read_csv(utils.data_dir + companyid + '/preparedData_test.csv', low_memory=False).set_index(['CompId', 'AccountId'])

        company_dir = utils.model_dir + companyid
        if not os.path.exists(company_dir):
            os.makedirs(company_dir)

        df_train = preprocessDF(df_train, company_dir + '/', traindate)
        self.update_state(state='PROGRESS', meta={'Train Data Shape': str(df_train.shape)})
        print('Train Data Shape', df_train.shape)
        df_train.to_csv(company_dir + '/preprocessedData_train.csv')

        df_test = preprocessDF(df_test, company_dir + '/', np.datetime64('today'))
        self.update_state(state='PROGRESS', meta={'Test Data Shape': str(df_test.shape)})
        print('Test Data Shape', df_test.shape)
        df_test.to_csv(company_dir + '/preprocessedData_test.csv')

    traindate = np.datetime64(traindate)
    sys.stdout = open(utils.log_dir + time.strftime("%Y%m%d-%H%M%S") + '_preprocess.txt', 'w')
    self.update_state(state='PROGRESS', meta={'Preprocess for train date': str(traindate)})
    print(datetime.datetime.now(), 'Preprocess for train date', str(traindate))

    for _, dirs, _ in os.walk(utils.data_dir):
        for d in [d for d in dirs if d not in ['raw', 'upload', '.git']]:
            print("Preprocess CompanyID:", d)
            preprocess(d, traindate)

    utils.gitCommitData('Preprocess data')

    sys.stdout.close()

    return "OK"
