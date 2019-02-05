from ...main import celery, api
from flask_restplus import Resource, fields

import pandas as pd
import numpy as np
import sys
import os
import datetime
import time

from . import utils

parser = api.parser()
parser.add_argument('splitdate', location='args', default='2018-01-01',
                    help='Split date for dataset', required=True)

model = api.model('Task', {'task_id': fields.String(
    example='ac435180-23d4-48aa-82e5-30af102847cc')})


@api.route("/prepare")
@api.expect(parser)
class Prepare(Resource):
    @api.marshal_with(model, code=202)
    def get(self):
        args = parser.parse_args()
        task = prepareData.delay(args['splitdate'])
        return {'task_id': task.id}


def trendline(data):
    coeffs = np.polyfit(range(data.shape[0]), list(data.values), 1)
    slope = coeffs[-2]
    return float(slope)


def trendDF(df):
    df_1 = pd.DataFrame()
    for idx, df_select in df.groupby(level=[0, 1]):
        if df_select.shape[0] > 1:
            tmp = pd.DataFrame(df_select.aggregate(trendline, axis=0)).T
            tmp['CompId'] = idx[0]
            tmp['AccountId'] = idx[1]
            df_1 = df_1.append(tmp.set_index(['CompId', 'AccountId']))

    return df_1


def aggregateMeanTrend(df_1, df_train, df_test, split_date):
    df_2 = df_1.reset_index()
    df_2['EntryDate'] = pd.to_datetime(df_2['Year'].astype('str') + '-' + df_2['Month'].astype('str'))
    df_2 = df_2.set_index(['CompId', 'AccountId', 'Year', 'Month'])
    df_1_train = filtrateToDate(df_2, 'EntryDate', split_date, True) 
    del df_2

    if df_1_train.shape[0] > 0:
        df_1_test = df_1.copy()

        df_1_train.columns = [str(col[0]) + '_' + str(col[1]) if isinstance(col, tuple) else col for col in df_1_train.columns.values.tolist()]
        df_1_test.columns = [str(col[0]) + '_' + str(col[1]) if isinstance(col, tuple) else col for col in df_1_test.columns.values.tolist()]

        print(datetime.datetime.now(), 'Starting mean calculation!')
        df_train = df_train.join(df_1_train.mean(level=1), rsuffix='_Mean')
        df_test = df_test.join(df_1_test.mean(level=1), rsuffix='_Mean')

        print(datetime.datetime.now(), 'Starting trends calculation!')
        df_2_train = trendDF(df_1_train)
        df_2_test = trendDF(df_1_test)

        df_train = df_train.join(df_2_train.mean(level=1), rsuffix='_Trend')
        df_test = df_test.join(df_2_test.mean(level=1), rsuffix='_Trend')

        df_train = df_train.fillna(0)
        df_test = df_test.fillna(0)
        del df_1_train
        del df_2_train
        del df_1_test
        del df_2_test
    return df_train, df_test


def divideByCompany(df, filename):
    for company in df.index.levels[0]:
        if not os.path.exists(utils.data_dir + str(company)):
            os.makedirs(utils.data_dir + str(company))
        df.loc[[company]].to_csv(utils.data_dir + str(company) + '/' + filename, index_label=['CompId', 'AccountId'])


def filtrateToDate(df, columnname, to_date, isdrop=False):
    df[columnname] = pd.to_datetime(df[columnname])
    df = df[(df[columnname] < to_date)]
    if isdrop:
        df = df.drop([columnname], axis=1, errors='ignore')
    return df


def filtrateFromDate(df, columnname, from_date, isdrop=False):
    df[columnname] = pd.to_datetime(df[columnname])
    df = df[(df[columnname] >= from_date)]
    if isdrop:
        df = df.drop([columnname], axis=1, errors='ignore')
    return df


@celery.task(bind=True)
def prepareData(self, split_date, scorenames=['Performance', 'Reliability', 'Risk'], from_date=np.datetime64('2017-01-01'), to_date=np.datetime64('2100-01-01'), withTLM=True, num_largest=100):
    def loaddata(filename, getcolumns, dtype=None, skiprows=None):
        df = pd.read_csv(utils.data_dir + 'raw/' + filename, names=getcolumns,
                         low_memory=False, dtype=dtype, index_col=False, skiprows=skiprows)
        return df

    split_date = np.datetime64(split_date)
    sys.stdout = open(utils.log_dir + time.strftime("%Y%m%d-%H%M%S") + '_prepare.txt', 'w')
    print(datetime.datetime.now(), 'Prepare for split date', str(split_date))
    print(datetime.datetime.now(), 'Prepare data till', to_date)
    
    # ...
    
    print(datetime.datetime.now(), 'Saving!')
    df_train.to_csv(utils.data_dir + 'preparedData_train' + strtodate + '.csv',  index_label=['CompId', 'AccountId'])
    df_test.to_csv(utils.data_dir + 'preparedData_test' + strtodate + '.csv', index_label=['CompId', 'AccountId'])

    divideByCompany(df_train, 'preparedData_train' + strtodate + '.csv')
    divideByCompany(df_test, 'preparedData_test' + strtodate + '.csv')
    data = {}
    data['companyid'] = company_list.tolist()
    df_test = df_test.reset_index()
    data['columns'] = list(df_test.columns)

    del df_test
    del df_train
    utils.gitCommitData('Prepare data')

    sys.stdout.close()

    return data
