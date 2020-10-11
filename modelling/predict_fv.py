import os
import pickle
import pandas as pd
import numpy as np

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
    X_train, y_train, X_test, y_test = get_train_test_data()

    # # getting the model trained with regression objective
    # model_type='regressor'
    # regress_pkl_filename = os.path.join(_DATA_DIR, 'models', 'xgboost_nogrid_20201003.pkl')
    # # Open the file to save as pkl file
    # with open(regress_pkl_filename, 'rb') as f:
    #     regres_model = pickle.load(f)

    # getting the model trained with multiclass objective
    softmax_pkl_filename = os.path.join(_DATA_DIR, 'models', 'xgboost_softmax_gridsearch_20201003.pkl')
    # Open the file to save as pkl files
    with open(softmax_pkl_filename, 'rb') as f:
        classi_model =  pickle.load(f)
    analyse_ins_prediction(classi_model, X_train, y_train, model_type)
    analyse_oos_prediction(classi_model, X_test, y_test, model_type)
    return None


def get_train_test_data(model_type):

    X_train = pd.read_csv(os.path.join(_DATA_DIR, 'target_features', 'x_train_dt.csv'), header=0, index_col=0)
    y_train = pd.read_csv(os.path.join(_DATA_DIR, 'target_features', 'y_train_dt.csv'), header=0, index_col=0)

    X_test = pd.read_csv(os.path.join(_DATA_DIR, 'target_features', 'x_test_dt.csv'), header=0, index_col=0)
    y_test = pd.read_csv(os.path.join(_DATA_DIR, 'target_features', 'y_test_dt.csv'), header=0, index_col=0)

    if model_type == 'classifier':
        y_train['fwd_fv_scaled'] = y_train['fwd_fv_scaled'].astype('category')
        y_test['fwd_fv_scaled'] = y_test['fwd_fv_scaled'].astype('category')

    return X_train, y_train, X_test, y_test


def analyse_ins_prediction(model, X_train, y_train, model_type):

    # model = regres_model
    # model = classi_model
    feat_importance = pd.Series(model.best_estimator_.feature_importances_)
    feat_importance.index = X_train.columns.values
    feat_importance.sort_values(ascending=True, inplace=True)

    # Setting the figure size
    fig = plt.figure(figsize=(10,8))
    plt.barh(feat_importance.tail(40).index, feat_importance.tail(40), alpha=0.3)
    # plt.barh(feat_importance.index, feat_importance, alpha=0.3)
    plt.xticks(rotation=90)
    plt.title('XGBoost Classifier - Feature Importance')
    #Saving the plot as an image
    fig.savefig(os.path.join(_DATA_DIR, 'figs', 'xgb_'+model_type+'_top_features.jpg'), bbox_inches='tight', dpi=150)

    fig = plt.figure(figsize=(10,8))
    plt.barh(feat_importance.head(40).index, feat_importance.head(40), alpha=0.3)
    plt.xticks(rotation=90)
    plt.title('XGBoost Classifier - bottom feature Importance')
    #Saving the plot as an image
    fig.savefig(os.path.join(_DATA_DIR, 'figs', 'xgb_'+model_type+'_bot_features.jpg'), bbox_inches='tight', dpi=150)

    ins_preds = model.predict(X_train)
    ins_prob_preds = model.predict_proba(X_train)

    if model_type == 'regressor':
        ins_metrics = dict({
            'mse' : mean_squared_error(y_train.iloc[:,0], ins_preds),
            'exvar' : explained_variance_score(y_train.iloc[:,0], ins_preds),
            'mae' : median_absolute_error(y_train.iloc[:,0], ins_preds)
        })
    elif model_type == 'classifier':
        labels=y_train.iloc[:,0].cat.categories.values
        ins_metrics = dict({
            'accuracy' : accuracy_score(y_train.iloc[:,0], ins_preds),
            'f1' : f1_score(y_train.iloc[:,0], ins_preds, average=None),
            'recall' : recall_score(y_train.iloc[:,0], ins_preds, average=None),
            'precision' : precision_score(y_train.iloc[:,0], ins_preds, average=None),
            balanced_accuracy_score(y_train.iloc[:,0], ins_preds),
            pd.DataFrame(confusion_matrix(y_train.iloc[:,0], ins_preds, labels=labels)),
            hinge_loss(y_train.iloc[:,0], ins_prob_preds, labels=labels),
            matthews_corrcoef(y_train.iloc[:,0], ins_preds, average=None),
            roc_auc_score(y_train.iloc[:,0], ins_preds, average=None)
        })
    print(classification_report(y_train.iloc[:,0], ins_preds))

    compare_dt = pd.DataFrame({'true' : y_train, 'pred' : ins_preds})
    kwargs = dict(histtype='stepfilled', alpha=0.3, bins=200)
    fig = plt.figure(figsize=(10,8))
    plt.hist(data=compare_dt[['true']], x='true', **kwargs)
    plt.hist(data=compare_dt[['pred']], x='pred', **kwargs)
    fig.savefig(os.path.join(_DATA_DIR, 'figs', 'xgb_'+model_type+'_ins_pred.jpg'), bbox_inches='tight', dpi=150)

    X_train.shape

    return None


def analyse_oos_prediction(model, X_test, y_test, model_type):

    oos_preds = model.predict(X_test)
    print(classification_report(y_test.iloc[:,0], oos_preds))



    compare_dt = pd.DataFrame({'true' : y_test['fwd_fv_scaled'], 'pred' : oos_preds})
    kwargs = dict(histtype='stepfilled', alpha=0.3, bins=200)
    fig = plt.figure(figsize=(10,8))
    plt.hist(data=compare_dt[['true']], x='true', **kwargs)
    plt.hist(data=compare_dt[['pred']], x='pred', **kwargs)
    fig.savefig(os.path.join(_DATA_DIR, 'figs', 'xgb_reg_oos_pred.jpg'), bbox_inches='tight', dpi=150)

    return None


def analyse_shap(model, X_test, y_test):

    explainer = shap.TreeExplainer(mcl)
    shap_values = explainer.shap_values(X_train)
    shap.initjs()
    for which_class in range(0,3):
        display(shap.force_plot(explainer.expected_value[which_class], shap_values[which_class], X_rand))

    #Display all features and SHAP values
    df1=pd.DataFrame(data=shap_values[0], columns=X.columns, index=[0])
    df2=pd.DataFrame(data=shap_values[1], columns=X.columns, index=[1])
    df3=pd.DataFrame(data=shap_values[2], columns=X.columns, index=[2])
    df=pd.concat([df1,df2,df3])
    display(df.transpose())
