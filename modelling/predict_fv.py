import os
import pickle
import pandas as pd
import numpy as np

from sklearn.preprocessing import LabelEncoder, OneHotEncoder
# for Regression
from sklearn.metrics import mean_squared_error, explained_variance_score, median_absolute_error
# for multi-class classification
from sklearn.metrics import accuracy_score, f1_score, recall_score, precision_score
from sklearn.metrics import balanced_accuracy_score, confusion_matrix, hinge_loss, matthews_corrcoef, roc_auc_score
from sklearn.metrics import classification_report

from xgboost import XGBRegressor, XGBClassifier
import shap

from matplotlib import pyplot as plt

_BASE_DIR = '/home/costam/Documents'
_DATA_DIR = os.path.join(_BASE_DIR, 'fantacalcio/data')


def main(model_type='classifier'):

    # getting the data
    X_train, y_train, X_test, y_test = get_train_test_data(model_type)
    if model_type == 'classifier':
        label_encoder = LabelEncoder()
        y_tot = pd.concat([y_train, y_test])
        label_encoder.fit(y_tot['fwd_fv_scaled'])
        y_train = label_encoder.transform(y_train['fwd_fv_scaled'])
        y_test = label_encoder.transform(y_test['fwd_fv_scaled'])

    # # getting the model trained with regression objective
    # model_type='regressor'
    # regress_pkl_filename = os.path.join(_DATA_DIR, 'models', 'xgboost_nogrid_20201003.pkl')
    # # Open the file to save as pkl file
    # with open(regress_pkl_filename, 'rb') as f:
    #     regres_model = pickle.load(f)

    # getting the model trained with multiclass objective
    softmax_pkl_filename = os.path.join(_DATA_DIR, 'models', 'xgboost_softmax_gridsearch_20201012.pkl')
    # Open the file to save as pkl files
    with open(softmax_pkl_filename, 'rb') as f:
        classi_model =  pickle.load(f)

    # np.unique(y_tot, return_counts=True)
    # np.unique(y_train, return_counts=True)

    # in sample predictions
    ins_preds = classi_model.predict(X_train)
    # np.unique(ins_preds, return_counts=True)
    ins_prob_preds = classi_model.predict_proba(X_train)

    # out of sample predictions
    oos_preds = classi_model.predict(X_test)
    oos_prob_preds = classi_model.predict_proba(X_test)

    return None


def get_train_test_data(model_type):

    X_train = pd.read_csv(os.path.join(_DATA_DIR, 'target_features', 'x_train_dt.csv'), header=0, index_col=0)
    y_train = pd.read_csv(os.path.join(_DATA_DIR, 'target_features', 'y_train_dt.csv'), header=0, index_col=0)

    X_test = pd.read_csv(os.path.join(_DATA_DIR, 'target_features', 'x_test_dt.csv'), header=0, index_col=0)
    y_test = pd.read_csv(os.path.join(_DATA_DIR, 'target_features', 'y_test_dt.csv'), header=0, index_col=0)

    return X_train, y_train, X_test, y_test
