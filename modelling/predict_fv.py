import os
import pickle
import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split, KFold, GridSearchCV
from sklearn.model_selection import cross_val_score
# for Regression
from sklearn.metrics import mean_squared_error, explained_variance_score, median_absolute_error
# for classification
from sklearn.metrics import coverage_error, label_ranking_loss, label_ranking_average_precision_score

from xgboost import XGBRegressor, XGBClassifiser

from matplotlib import pyplot as plt

_BASE_DIR = '/home/costam/Documents'
_DATA_DIR = os.path.join(_BASE_DIR, 'fantacalcio/data')


def main():

    # getting the data
    X_train, y_train, X_test, y_test = get_train_test_data()

    # getting the model trained with regression objective
    model_type='regressor'
    regress_pkl_filename = os.path.join(_DATA_DIR, 'models', 'xgboost_nogrid_20201003.pkl')
    # Open the file to save as pkl file
    with open(regress_pkl_filename, 'rb') as f:
        regres_model = pickle.load(f)

    analyse_ins_prediction(regres_model, X_train, y_train)

    # getting the model trained with multiclass objective
    model_type='classifier'
    softmax_pkl_filename = os.path.join(_DATA_DIR, 'models', 'xgboost_softmax_nogrid_20201003.pkl')
    # Open the file to save as pkl files
    with open(softmax_pkl_filename, 'rb') as f:
        classi_model =  pickle.load(f)


    # full_grid = gsearch.grid_scores_
    best_params = gsearch.best_params_
    best_score = gsearch.best_score_

    return None


def get_train_test_data():

    X_train = pd.read_csv(os.path.join(_DATA_DIR, 'target_features', 'x_train_dt.csv'), header=0, index_col=0)
    y_train = pd.read_csv(os.path.join(_DATA_DIR, 'target_features', 'y_train_dt.csv'), header=0, index_col=0)

    X_test = pd.read_csv(os.path.join(_DATA_DIR, 'target_features', 'x_test_dt.csv'), header=0, index_col=0)
    y_test = pd.read_csv(os.path.join(_DATA_DIR, 'target_features', 'y_test_dt.csv'), header=0, index_col=0)

    return X_train, y_train, X_test, y_test


def analyse_ins_prediction(model, X_train, y_train, model_type):

    # model = regres_model
    # model = classi_model
    feat_importance = pd.Series(model.best_estimator_.feature_importances_)
    feat_importance.index = X_train.columns.values
    feat_importance.sort_values(ascending=True, inplace=True)

    # Setting the figure size
    fig = plt.figure(figsize=(10,8))
    # plt.barh(feat_importance.tail(40).index, feat_importance.tail(40), alpha=0.3)
    plt.barh(feat_importance.index, feat_importance, alpha=0.3)
    plt.xticks(rotation=90)
    plt.title('XGBoost - Feature Importance')
    #Saving the plot as an image
    fig.savefig(os.path.join(_DATA_DIR, 'figs', 'xgb_'+model_type+'_top_features.jpg'), bbox_inches='tight', dpi=150)

    fig = plt.figure(figsize=(10,8))
    plt.barh(feat_importance.head(40).index, feat_importance.head(40), alpha=0.3)
    plt.xticks(rotation=90)
    plt.title('XGBoost - Feature Importance')
    #Saving the plot as an image
    fig.savefig(os.path.join(_DATA_DIR, 'figs', 'xgb_'+model_type+'_bot_features.jpg'), bbox_inches='tight', dpi=150)

    ins_preds = model.predict(X_train)

    if model_type == 'regressor':
        ins_metrics = dict({
            'mse' : mean_squared_error(y_train, ins_preds),
            'exvar' : explained_variance_score(y_train, ins_preds),
            'mae' : median_absolute_error(y_train, ins_preds)
        })
    elif model_type == 'regressor':
        ins_metrics = dict({
            'mse' : mean_squared_error(y_train, ins_preds),
            'exvar' : explained_variance_score(y_train, ins_preds),
            'mae' : median_absolute_error(y_train, ins_preds)
        })

    compare_dt = pd.DataFrame({'true' : y_train['fwd_fv_scaled'], 'pred' : ins_preds})
    kwargs = dict(histtype='stepfilled', alpha=0.3, bins=200)
    fig = plt.figure(figsize=(10,8))
    plt.hist(data=compare_dt[['true']], x='true', **kwargs)
    plt.hist(data=compare_dt[['pred']], x='pred', **kwargs)
    fig.savefig(os.path.join(_DATA_DIR, 'figs', 'xgb_'+model_type+'_ins_pred.jpg'), bbox_inches='tight', dpi=150)

    X_train.shape

    return None


def analyse_oos_prediction(model, X_test, y_test):

    oos_preds = model.predict(X_test)
    oos_mse = mean_squared_error(y_test, oos_preds)
    oos_exvar = explained_variance_score(y_test, oos_preds)
    oos_mae = median_absolute_error(y_test, oos_preds)

    compare_dt = pd.DataFrame({'true' : y_test['fwd_fv_scaled'], 'pred' : oos_preds})
    kwargs = dict(histtype='stepfilled', alpha=0.3, bins=200)
    fig = plt.figure(figsize=(10,8))
    plt.hist(data=compare_dt[['true']], x='true', **kwargs)
    plt.hist(data=compare_dt[['pred']], x='pred', **kwargs)
    fig.savefig(os.path.join(_DATA_DIR, 'figs', 'xgb_reg_oos_pred.jpg'), bbox_inches='tight', dpi=150)

    return None
