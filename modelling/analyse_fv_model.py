# five different feature attribution methods on our simple tree models:
# 1) Tree SHAP. A new individualized method we are proposing.
# 2) Saabas. An individualized heuristic feature attribution method.
# 3) mean(|Tree SHAP|). A global attribution method based on the average magnitude of the individualized Tree SHAP attributions.
# 4) Gain. The same method used above in XGBoost, and also equivalent to the Gini importance measure used in scikit-learn tree models.
# 5) Split count. Represents both the closely related “weight” and “cover” methods in XGBoost, but is computed using the “weight” method.
# 6) Permutation. The resulting drop in accuracy of the model when a single feature is randomly permuted in the test data set.

import os
import pickle
import pandas as pd
import numpy as np

from sklearn.preprocessing import LabelEncoder
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

    # getting the model trained with multiclass objective
    softmax_validated_filename = os.path.join(_DATA_DIR, 'models', 'xgboost_softmax_validated_20201012.pkl')
    # Open the file to save as pkl files
    with open(softmax_validated_filename, 'rb') as f:
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

    ins_metrics, ins_report = analyse_ins_prediction(classi_model, X_train,
                                                     y_train, ins_preds,
                                                     ins_prob_preds, model_type)
    oos_metrics, oos_report = analyse_oos_prediction(classi_model,
                                                     y_test, oos_preds,
                                                     oos_prob_preds, model_type)
    return None


def get_train_test_data(model_type):

    X_train = pd.read_csv(os.path.join(_DATA_DIR, 'target_features', 'x_train_dt.csv'), header=0, index_col=0)
    y_train = pd.read_csv(os.path.join(_DATA_DIR, 'target_features', 'y_train_dt.csv'), header=0, index_col=0)

    X_test = pd.read_csv(os.path.join(_DATA_DIR, 'target_features', 'x_test_dt.csv'), header=0, index_col=0)
    y_test = pd.read_csv(os.path.join(_DATA_DIR, 'target_features', 'y_test_dt.csv'), header=0, index_col=0)

    return X_train, y_train, X_test, y_test


def analyse_ins_prediction(model, X_train, y_train, ins_preds, ins_prob_preds, model_type):

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

    if model_type == 'regressor':
        ins_metrics = dict({
            'mse' : mean_squared_error(y_train, ins_preds),
            'exvar' : explained_variance_score(y_train, ins_preds),
            'mae' : median_absolute_error(y_train, ins_preds)
        })
        ins_report = None
    elif model_type == 'classifier':
        # oh_encoder = OneHotEncoder()
        # ohe_y_train = oh_encoder.fit_transform(pd.DataFrame({"label" : y_train}))
        ins_metrics = dict({
            'accuracy' : accuracy_score(y_train, ins_preds),
            'f1' : f1_score(y_train, ins_preds, average=None),
            'recall' : recall_score(y_train, ins_preds, average=None),
            'precision' : precision_score(y_train, ins_preds, average=None),
            'balanced_accuracy' : balanced_accuracy_score(y_train, ins_preds),
            'confusion' : pd.DataFrame(confusion_matrix(y_train, ins_preds)),
            # 'hinge_loss' : hinge_loss(y_train, ins_preds),
            'matt_corr' : matthews_corrcoef(y_train, ins_preds)
            # ,'roc_auc' : roc_auc_score(y_train, ins_preds, multi_class='ovr', average="macro")
        })
        ins_report = classification_report(y_train, ins_preds)
        print(ins_report)

    compare_dt = pd.DataFrame({'true' : y_train, 'pred' : ins_preds})
    kwargs = dict(histtype='stepfilled', alpha=0.3, bins=200)
    fig = plt.figure(figsize=(10,8))
    plt.hist(data=compare_dt[['true']], x='true', **kwargs)
    plt.hist(data=compare_dt[['pred']], x='pred', **kwargs)
    fig.savefig(os.path.join(_DATA_DIR, 'figs', 'xgb_'+model_type+'_ins_pred.jpg'), bbox_inches='tight', dpi=150)

    return ins_metrics, ins_report


def analyse_oos_prediction(model, y_test, oos_preds, oos_prob_preds, model_type):

    if model_type == 'regressor':
        oos_metrics = dict({
            'mse' : mean_squared_error(y_test, oos_preds),
            'exvar' : explained_variance_score(y_test, oos_preds),
            'mae' : median_absolute_error(y_test, oos_preds)
        })
        oos_report = None
    elif model_type == 'classifier':
        oos_metrics = dict({
            'accuracy' : accuracy_score(y_test, oos_preds),
            'f1' : f1_score(y_test, oos_preds, average=None),
            'recall' : recall_score(y_test, oos_preds, average=None),
            'precision' : precision_score(y_test, oos_preds, average=None),
            'balanced_accuracy' : balanced_accuracy_score(y_test, oos_preds),
            'confusion' : pd.DataFrame(confusion_matrix(y_test, oos_preds)),
            # 'hinge_loss' : hinge_loss(y_train, ins_preds),
            'matt_corr' : matthews_corrcoef(y_test, oos_preds)
            # ,'roc_auc' : roc_auc_score(y_train, ins_preds, multi_class='ovr', average="macro")
        })
        oos_report = classification_report(y_test, oos_preds)
        print(oos_report)

    compare_dt = pd.DataFrame({'true' : y_test, 'pred' : oos_preds})
    kwargs = dict(histtype='stepfilled', alpha=0.3, bins=200)
    fig = plt.figure(figsize=(10,8))
    plt.hist(data=compare_dt[['true']], x='true', **kwargs)
    plt.hist(data=compare_dt[['pred']], x='pred', **kwargs)
    fig.savefig(os.path.join(_DATA_DIR, 'figs', 'xgb_'+model_type+'_oos_pred.jpg'), bbox_inches='tight', dpi=150)

    return oos_metrics, oos_report


def analyse_shap(model, X_train, y_train, X_test, y_test):

    rnd_idx = np.rnd()

    # model = classi_model['estimator'][1]
    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(X_train)
    shap.initjs()

    fig = plt.figure(figsize=(10,8))
    shap.summary_plot(shap_values, X_train)
    fig.savefig(os.path.join(_DATA_DIR, 'figs', 'xgb_'+model_type+'_shap_features_by_label.jpg'), bbox_inches='tight', dpi=150)


    shap.force_plot(explainer.expected_value[0], shap_values[0], X_train)
