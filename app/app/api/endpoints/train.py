from ...main import celery, api
from flask_restplus import Resource, fields

import pandas as pd
import os
import sys
import datetime
import time

from sklearn.ensemble import GradientBoostingRegressor
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import KFold
from sklearn.model_selection import cross_val_score, cross_val_predict
from sklearn.metrics import accuracy_score
from sklearn.metrics import mean_squared_error
from sklearn.metrics import explained_variance_score

import pickle
from . import utils

parser = api.parser()
parser.add_argument('train_type', location='path',
                    help='Classification Type', choices=('classification', 'regression'))
parser.add_argument('withtlm', location='args',
                    default='0',
                    help='Use TLM data in training', required=True)
parser.add_argument('tune', location='args',
                    default='0',
                    help='Use Hyperparameters tuning during training', required=True)

model = api.model('Task', {'task_id': fields.String(
    example='ac435180-23d4-48aa-82e5-30af102847cc')})


@api.route("/train/<train_type>")
@api.expect(parser)
class Train(Resource):
    @api.marshal_with(model, code=202)
    def get(self, train_type):
        def switch_train(x):
            args = parser.parse_args()
            return {
                'classification': train_classification.delay,
                'regression': train_regression.delay,
            }[x](args['withtlm'], args['tune'])

        task = switch_train(train_type)
        return {'task_id': task.id}


result_folders = ['classification', 'regression',
                  'classification_notlm', 'regression_notlm', '.git']


@celery.task(bind=True)
def train_classification(self, withTLM, tune):
    def trainModel(folder, tune, X_train, y_train, X_test, y_test, df_index):
        print('Train data:', X_train.shape)
        if not os.path.exists(folder):
            os.makedirs(folder)
        f = open(folder + '/train.set', 'w')
        f.write(str(X_train.shape))
        f.close()
        if not os.path.exists(folder):
            os.makedirs(folder)
        f = open(folder + '/test.set', 'w')
        f.write(str(X_test.shape))
        f.close()
        clf = RandomForestClassifier(max_features='sqrt', oob_score=True,
                                     criterion='entropy', random_state=0, class_weight='balanced')
        # Use a grid over parameters of interest
        if tune:
            param_grid = {
                "n_estimators": [45, 63, 113, 145],
                "min_samples_split": [2, 10, 20],
                "min_samples_leaf": [2, 6, 10]}

            CV_rfc = GridSearchCV(
                estimator=clf, param_grid=param_grid, cv=5, scoring='f1')
            CV_rfc.fit(X_train, y_train)
            print('Best score: %0.2f' % (100 * CV_rfc.best_score_))
            print('Best params: ', CV_rfc.best_params_)

            # Optimized RF classifier
            clf = CV_rfc.best_estimator_
            kfold = KFold(n_splits=5, random_state=0)

            # fit the model with training set
            scores = cross_val_score(
                clf, X_train, y_train, cv=kfold, scoring='accuracy')
            print("Train accuracy %0.2f (+/- %0.2f)" %
                  (scores.mean() * 100, scores.std() * 100))

            # predict on testing set
            preds = cross_val_predict(clf, X_test, y_test, cv=kfold)
            ac_score = accuracy_score(y_test, preds)
            print("Test accuracy %0.2f" % (100 * ac_score))

        clf.fit(X_train, y_train)
        ac_score = clf.score(X_test, y_test)
        print('Model F1 Score', ac_score)

        y_pred = clf.predict(X_train)
        print('Classication report for TRAIN set')
        print(classification_report(y_train, y_pred))

        # Compute confusion matrix
        cnf_matrix = confusion_matrix(y_train, y_pred)
        print('Confusion matrix for TRAIN set')
        print(cnf_matrix)
        if not os.path.exists(folder):
            os.makedirs(folder)

        f = open(folder + '/model.score', 'w')
        f.write(str(ac_score))
        f.close()
        y_pred = clf.predict(X_test)
        print('Classication report for TEST set')
        print(classification_report(y_test, y_pred))

        # Compute confusion matrix
        cnf_matrix = confusion_matrix(y_test, y_pred)
        print('Confusion matrix for TEST set')
        print(cnf_matrix)

        pickle.dump(clf, open(folder + '/model.pkl', 'wb'))

        feat_imp = pd.DataFrame({'importance': clf.feature_importances_})
        feat_imp['feature'] = X_train.columns
        feat_imp = feat_imp.set_index(['feature'])
        feat_imp.to_csv(folder + '/featuresImportance.csv')

    withTLM = (withTLM == '1')
    tune = (tune == '1')
    sys.stdout = open(
        utils.log_dir + time.strftime("%Y%m%d-%H%M%S") + '_train_classification.txt', 'w')
    print(datetime.datetime.now(),
          'Train (with TLM, tune)', str(withTLM), str(tune))

    for _, dirs, _ in os.walk(utils.model_dir):
        for d in [d for d in dirs if d not in result_folders]:
            company_dir = utils.model_dir + d

            if os.path.isfile(company_dir + '/VoluntaryReasons.csv'):
                voluntary_reasons = list(pd.read_csv(
                    company_dir + '/VoluntaryReasons.csv', names=['ReasonId'])['ReasonId'])
            else:
                voluntary_reasons = []

            df_train = pd.read_csv(company_dir + '/preprocessedData_train.csv')
            df_test = pd.read_csv(company_dir + '/preprocessedData_test.csv')
           

            X_train = df_train.drop(['CompId', 'AccountId', 'AttritionReasonId', 'AttritionDays', 'IsAttrition'],
                                    axis=1, errors='ignore')
            y_train = df_train['IsAttrition']
            X_test = df_test.drop(['CompId', 'AccountId', 'AttritionReasonId', 'AttritionDays', 'IsAttrition'],
                                  axis=1, errors='ignore')
            y_test = df_test['IsAttrition']
            df_index = df_test[['CompId', 'AccountId']]

            if withTLM:
                print("Train Classification CompanyID:", d)
            else:
                tlm_cols = [c for c in X_train.columns if any(
                    c.startswith(prefix) for prefix in tlm_prefixes)]
                X_train = X_train.drop(tlm_cols, axis=1, errors='ignore')
                X_test = X_test.drop(tlm_cols, axis=1, errors='ignore')

                print("Train Classification (no TLM) CompanyID:", d)

            trainModel(company_dir + '/classification', tune,
                       X_train, y_train, X_test, y_test, df_index)

    utils.gitCommitModel('Train classification')

    sys.stdout.close()

    return 'OK'


@celery.task(bind=True)
def train_regression(self, withTLM, tune):
    def trainModel(folder, tune, X_train, y_train, X_test, y_test, df_index):
        print('Train data:', X_train.shape)
        if not os.path.exists(folder):
            os.makedirs(folder)
        f = open(folder + '/train.set', 'w')
        f.write(str(X_train.shape))
        f.close()
        if not os.path.exists(folder):
            os.makedirs(folder)
        f = open(folder + '/test.set', 'w')
        f.write(str(X_test.shape))
        f.close()
        if y_test.shape[0] == 0:
            return
        reg = GradientBoostingRegressor(random_state=0)

        if tune:
            # Use a grid over parameters of interest
            param_grid = {
                "n_estimators": [50, 100, 500],
                "max_depth": [3, 5, 20],
                "min_samples_split": [4, 8, 15],
                "min_samples_leaf": [2, 5, 10]}

            CV_rfc = GridSearchCV(
                estimator=reg, param_grid=param_grid, cv=5, scoring='neg_mean_squared_error')
            CV_rfc.fit(X_train, y_train)
            print('Best score: %.4f' % (CV_rfc.best_score_))
            print('Best params: ', CV_rfc.best_params_)

            # Optimized RF classifier
            reg = CV_rfc.best_estimator_

        reg.fit(X_train, y_train)
        score = reg.score(X_test, y_test)
        print('Model score', score)
        if not os.path.exists(folder):
            os.makedirs(folder)
        f = open(folder + '/model.score', 'w')
        f.write(str(score))
        f.close()
        y_pred = reg.predict(X_test)
        mse = mean_squared_error(y_test, y_pred)
        print("Test MSE: %.4f" % mse)
        var = explained_variance_score(y_test, y_pred)
        print("Test Variance: %.6f" % var)

        pickle.dump(reg, open(folder + '/model.pkl', 'wb'))

        feat_imp = pd.DataFrame({'importance': reg.feature_importances_})
        feat_imp['feature'] = X_train.columns
        feat_imp = feat_imp.set_index(['feature'])
        feat_imp.to_csv(folder + '/featuresImportance.csv')

    withTLM = (withTLM == '1')
    tune = (tune == '1')
    sys.stdout = open(
        utils.log_dir + time.strftime("%Y%m%d-%H%M%S") + '_train_regression.txt', 'w')
    print(datetime.datetime.now(),
          'Train (with TLM, tune)', str(withTLM), str(tune))

    for _, dirs, _ in os.walk(utils.model_dir):
        for d in [d for d in dirs if d not in result_folders]:
            company_dir = utils.model_dir + d

            if os.path.isfile(company_dir + '/VoluntaryReasons.csv'):
                voluntary_reasons = list(pd.read_csv(
                    company_dir + '/VoluntaryReasons.csv', names=['ReasonId'])['ReasonId'])
            else:
                voluntary_reasons = []

            df_train = pd.read_csv(company_dir + '/preprocessedData_train.csv')
            df_test = pd.read_csv(company_dir + '/preprocessedData_test.csv')
            

            X_train = df_train[df_train.IsAttrition].drop(['CompId', 'AccountId', 'AttritionReasonId', 'AttritionDays', 'IsAttrition'],
                                                          axis=1, errors='ignore')
            y_train = df_train[df_train.IsAttrition]['AttritionDays']
            X_test = df_test[df_test.IsAttrition].drop(['CompId', 'AccountId', 'AttritionReasonId', 'AttritionDays', 'IsAttrition'],
                                                       axis=1, errors='ignore')
            y_test = df_test[df_test.IsAttrition]['AttritionDays']
            df_index = df_test[df_test.IsAttrition][['CompId', 'AccountId']]

            if withTLM:
                print("Train Regression CompanyID:", d)
            else:
                tlm_cols = [c for c in X_train.columns if any(
                    c.startswith(prefix) for prefix in tlm_prefixes)]
                X_train = X_train.drop(tlm_cols, axis=1, errors='ignore')
                X_test = X_test.drop(tlm_cols, axis=1, errors='ignore')

                print("Train Regression (no TLM) CompanyID:", d)

            trainModel(company_dir + '/regression', tune, X_train,
                       y_train, X_test, y_test, df_index)

    utils.gitCommitModel('Train regression')

    sys.stdout.close()

    return 'OK'
