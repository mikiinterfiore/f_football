import os
import pickle
import json
import pandas as pd
import numpy as np

from sklearn.preprocessing import LabelEncoder, OneHotEncoder
from xgboost import XGBRegressor, XGBClassifier
from utils.utils_features_target_model import get_ffdata_combined

_BASE_DIR = '/home/costam/Documents'
_DATA_DIR = os.path.join(_BASE_DIR, 'fantacalcio/data')


def get_ffm_predictions(final_round):

    # final_round=6

    # loading the specific players for Fantasy Football
    ff_players_filename = os.path.join(_DATA_DIR, 'master', 'fantacalcio_players.csv')
    ff_players = pd.read_csv(ff_players_filename, header=0, index_col=None)

    # getting the data
    select_seasons = [2019, 2020]
    full_dt = get_ffdata_combined(select_seasons)

    # players skeleton
    players_dt = full_dt.loc[:, ['name','surname', 'season','match']]
    players_dt = players_dt.merge(ff_players, on = ['name', 'surname'], how='left')
    season_idx = (full_dt['season'] == max(select_seasons)) & (full_dt['match'] <= final_round)
    ff_players_idx = players_dt.loc[(~players_dt['team'].isna()) & season_idx].index

    # creating features data
    exclude_col = ['name', 'surname', 'season', 'match', 'fwd_fantavoto', 'fwd_fv_scaled', 'role']
    feat_cols = full_dt.columns.values[~full_dt.columns.isin(exclude_col)].tolist()
    X = full_dt.loc[ff_players_idx, feat_cols]

    # getting the model trained with multiclass objective
    # softmax_validated_filename = 'xgboost_softmax_validated_20201018.pkl'
    softmax_validated_filename = 'xgboost_softmax_validated_20210101.pkl'
    softmax_validated_filename = os.path.join(_DATA_DIR, 'models', softmax_validated_filename)
    # Open the file to save as pkl files
    with open(softmax_validated_filename, 'rb') as f:
        classi_model =  pickle.load(f)

    preds = classi_model.predict(X)
    preds = pd.DataFrame({'expected_fv_scaled_binned' : preds})
    preds.index = X.index

    pred_fv_dt = players_dt.loc[(~players_dt['team'].isna()) & season_idx]
    pred_fv_dt = pred_fv_dt.merge(preds, left_index=True, right_index=True)

    # get the original encoding map
    encoded_map_filename = 'xgboost_softmax_label_encoder.pkl'
    encoded_map_filename = os.path.join(_DATA_DIR, 'models', encoded_map_filename)
    with open(encoded_map_filename, 'r') as f:
        encoded_map = json.load(f)
    encoded_map_rev = {v: k for k, v in encoded_map.items()}
    pred_fv_dt['expected_fv_scaled'] = pred_fv_dt['expected_fv_scaled_binned'].map(encoded_map_rev)
    pred_fv_dt['expected_fv'] = pred_fv_dt['expected_fv_scaled'].astype('float')+6
    pred_fv_dt = pred_fv_dt.sort_values(['season', 'surname', 'name', 'match'])

    ff_players_predictions_filename = os.path.join(_DATA_DIR, 'fantacalcio_players_predictions.csv')
    pred_fv_dt.to_csv(ff_players_predictions_filename, header=True, index=False)

    return None


def _get_train_test_data(model_type):

    X_train = pd.read_csv(os.path.join(_DATA_DIR, 'target_features', 'x_train_dt.csv'), header=0, index_col=0)
    y_train = pd.read_csv(os.path.join(_DATA_DIR, 'target_features', 'y_train_dt.csv'), header=0, index_col=0)

    X_test = pd.read_csv(os.path.join(_DATA_DIR, 'target_features', 'x_test_dt.csv'), header=0, index_col=0)
    y_test = pd.read_csv(os.path.join(_DATA_DIR, 'target_features', 'y_test_dt.csv'), header=0, index_col=0)

    return X_train, y_train, X_test, y_test
