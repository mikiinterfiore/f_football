import os
import pickle
import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split, KFold, GridSearchCV
from sklearn.model_selection import cross_val_score
from sklearn.metrics import mean_squared_error

from xgboost import XGBRegressor

_BASE_DIR = '/home/costam/Documents'
_CODE_DIR = os.path.join(_BASE_DIR, 'fantacalcio/fanta_code')
_DATA_DIR = os.path.join(_BASE_DIR, 'fantacalcio/data')


def main():

    # getting the data
    team_map = get_masters()
    tm_feat, pl_feat = get_features()
    pl_fv = get_targets()
    # combining the data
    select_seasons = [2019]
    full_dt = combine_data(team_map, tm_feat, pl_feat, pl_fv, select_seasons)
    # setting the random seed before the split
    rng = np.random.RandomState(1298)
    # preparing the features
    X_train, X_test, y_train, y_test = prepare_model_data(full_dt, rng)
    # running the model
    gsearch = run_model(X_train, y_train, rng)

    model_pkl_filename = os.path.join(_DATA_DIR, 'models', 'xgboost_nogrid_20200928.pkl')
    # Open the file to save as pkl file
    with open(model_pkl_filename, 'wb') as f:
        pickle.dump(gsearch, f)

    os.getcwd()

    return None


def get_masters():

    tm_master_file = os.path.join(_DATA_DIR, 'master', 'rapid_teamnames_map.csv')
    team_map = pd.read_csv(tm_master_file, header=0, index_col=False)

    return team_map


def get_features():

    pl_feat_file = os.path.join(_DATA_DIR, 'target_features', 'players_features.csv')
    pl_feat = pd.read_csv(pl_feat_file, header=0, index_col=False)
    pl_feat = pl_feat.sort_values(['name', 'surname', 'season', 'match'])

    tm_feat_file = os.path.join(_DATA_DIR, 'target_features', 'team_features.csv')
    tm_feat = pd.read_csv(tm_feat_file, header=0, index_col=False)
    tm_feat = tm_feat.sort_values(['team_id', 'event_date'])
    # todo(Michele) : add the correct match to each fixture as from API
    # creating the match column based on the time of the game for each team
    tm_feat['match'] = 1
    tm_feat['match'] = tm_feat.groupby('team_id')['match'].cumsum()
    # todo(Michele) : mapping of each Rapid API season to the year

    return tm_feat, pl_feat


def get_targets():

    pl_fv_file = os.path.join(_DATA_DIR, 'target_features', 'players_target.csv')
    pl_fv = pd.read_csv(pl_fv_file, header=0, index_col=False)
    pl_fv = pl_fv.sort_values(['name', 'surname', 'season', 'match'])
    # filling the fv with 6 if they have not played and centering fv on 6
    pl_fv['fwd_fv_scaled'] = pl_fv['fwd_fantavoto'].fillna(6.0) - 6

    return pl_fv


def combine_data(team_map, tm_feat, pl_feat, pl_fv, select_seasons):

    pl_feat_ssn_idx = pl_feat['season'].isin(select_seasons)
    pl_fv_ssn_idx = pl_fv['season'].isin(select_seasons)

    # getting the features split between team and player
    tm_cols = tm_feat.columns.values[~tm_feat.columns.isin(['team_id', 'team_name', 'event_date', 'fixture_id', 'match'])].tolist()
    pl_cols = pl_feat.columns.values[~pl_feat.columns.isin(['team', 'name', 'surname', 'role', 'season', 'match'])].tolist()

    # correct name of team coming from Gazzetta
    tm_feat = tm_feat.merge(team_map, left_on = 'team_name', right_on = 'team_name')

    # merging players target and players features
    full_dt = pl_fv.loc[pl_fv_ssn_idx,:].merge(pl_feat.loc[pl_feat_ssn_idx],
                                               on=['name', 'surname', 'season', 'match'],
                                               how ='left')
    full_dt['team'] = full_dt['team_x']
    full_dt.drop(['team_x', 'team_y'], axis=1, inplace=True)
    # filling NAs for player features
    pl_pivot = ['name', 'surname', 'season', 'match']
    full_dt = full_dt.sort_values(pl_pivot)
    # filling the players features
    full_dt.loc[:, pl_cols] = full_dt.groupby(pl_pivot[0:3])[pl_cols].fillna(method='ffill')
    # fixing role
    full_dt['role'] = full_dt.groupby(pl_pivot[0:3])['role'].fillna(method='ffill')
    full_dt['role'] = full_dt.groupby(pl_pivot[0:3])['role'].fillna(method='bfill')

    full_dt = full_dt.merge(tm_feat,
                            left_on=['team','match'],
                            right_on=['team_gaz', 'match'],
                            how='left')
    # na_idx = combo_feat.isnull().sum(axis=1) >1
    # combo_feat.loc[na_idx].to_csv(os.path.join(_DATA_DIR, 'check_nas_features.csv') ,header=True, index=False)

    # filling the team features
    tm_pivot = ['team', 'season', 'match']
    full_dt = full_dt.sort_values(tm_pivot)
    full_dt.loc[:, tm_cols] = full_dt.groupby(tm_pivot[0:2])[tm_cols].fillna(method='ffill')

    # cleaning up the features data
    drop_cols = ['team_id', 'team', 'team_name', 'team_gaz', 'event_date', 'fixture_id', 'league_id']
    full_dt.drop(drop_cols, axis=1, inplace=True)
    main_cols = ['name', 'surname', 'season', 'match', 'role']
    all_cols = main_cols + full_dt.columns.values[~full_dt.columns.isin(main_cols)].tolist()
    full_dt = full_dt.loc[:, all_cols]

    # dropping the features that have not enough data
    feat_na_check = full_dt.isna().sum()/full_dt.shape[0]
    feat_na_check = feat_na_check[feat_na_check>0.25]
    feat_na_check = feat_na_check.index.values[feat_na_check.index.values != 'fwd_fantavoto']
    if len(feat_na_check)>0:
        full_dt.drop(feat_na_check, axis=1, inplace=True)

    # dropping the rows that have not enough data
    obs_na_check = full_dt.isna().sum(axis=1)/full_dt.shape[1]
    obs_na_check = obs_na_check[obs_na_check>0.3]
    if len(obs_na_check)>0:
        full_dt.drop(obs_na_check.index.values, axis=0, inplace=True)
        full_dt.reset_index(inplace=True, drop=True)

    full_dt['role'] = full_dt['role'].astype('category')

    # print(full_dt.columns.values)
    # len(full_dt.columns.values)

    return full_dt


def prepare_model_data(full_dt, rng, target_var='fwd_fv_scaled'):

    # creating features and target data
    exclude_col = ['name', 'surname', 'season', 'match', 'role', 'fwd_fantavoto',
                   'fwd_fv_scaled']
    feat_cols = full_dt.columns.values[~full_dt.columns.isin(exclude_col)].tolist()

    X = full_dt.loc[:, feat_cols]
    y = full_dt.loc[:, target_var]
    X_train, X_test, y_train, y_test = train_test_split(X, y,
                                                        test_size=0.2,
                                                        random_state=rng,
                                                        stratify=full_dt['role'])

    return X_train, X_test, y_train, y_test


def run_model(X_train, y_train, rng):

    # initialise the model
    # https://xgboost.readthedocs.io/en/latest/python/python_api.html#module-xgboost.sklearn
    model = XGBRegressor(objective='reg:squarederror',
                             booster='gbtree',
                             tree_method='hist', # vs 'gpu_hist'
                             n_jobs=3,
                             random_state=rng,
                             # scoring = 'mse',
                             eval_metric='rmse')

    # parameter grid
    # gamma_params = [i/1000.0 for i in range(10,50,7)] + [0.08, 0.1, 0.2]
    # param_search = {'n_estimators':[1000, 1500, 2000, 2500], # Number of gradient boosted trees.
    #                 'max_depth':[4,5,6,7], # Maximum tree depth for base learners.
    #                 'learning_rate': [0.003, 0.004, 0.005], # equivalent to eta
    #                 # Minimum loss reduction required to make a further partition on a leaf node
    #                 'gamma': gamma_params,
    #
    #                  # Minimum sum of instance weight(hessian) needed in a child
    #                 # 'min_child_weight':[j for j in range(1,6,2)],
    #
    #                  # Subsample ratio of the training instance.
    #                 'subsample':[i/10.0 for i in range(7,10)],
    #                 # Subsample ratio of columns when constructing each tree.
    #                 'colsample_bytree':[i/10.0 for i in range(7,10)],
    #
    #                 # L1 regularization term on weights
    #                 # 'reg_alpha':[0, 1e-5, 1e-2, 0.1, 1, 10],
    #
    #                 # L2 regularization term on weights
    #                 'reg_lambda':[0, 1e-5, 1e-2, 0.1, 1, 10]}

    param_search = {'n_estimators':[1000], # Number of gradient boosted trees.
                    'max_depth':[7], # Maximum tree depth for base learners.
                    'learning_rate': [0.005], # equivalent to eta
                    # Minimum loss reduction required to make a further partition on a leaf node
                    'gamma': [0.1],

                     # Minimum sum of instance weight(hessian) needed in a child
                    # 'min_child_weight':[j for j in range(1,6,2)],

                     # Subsample ratio of the training instance.
                    'subsample':[0.8],
                    # Subsample ratio of columns when constructing each tree.
                    'colsample_bytree':[0.8],

                    # L1 regularization term on weights
                    # 'reg_alpha':[0, 1e-5, 1e-2, 0.1, 1, 10],

                    # L2 regularization term on weights
                    'reg_lambda':[0.1]}

    # cross validate: we stratify the train, test split by role, so use simple kfold here
    fold5_cv = KFold(n_splits=5, random_state=rng)
    # for an overview of the cv scoring :
    # https://scikit-learn.org/stable/modules/model_evaluation.html#scoring-parameter
    gsearch = GridSearchCV(estimator=model, cv=fold5_cv, param_grid=param_search,
                           pre_dispatch=3, scoring='neg_mean_squared_error', verbose=True)

    gsearch.fit(X_train, y_train)

    # full_grid = gsearch.grid_scores_
    best_params = gsearch.best_params_
    best_score = gsearch.best_score_

    # for train_index, test_index in kf.split(X):
    #         xgb_model = xgb.XGBClassifier().fit(X[train_index], y[train_index])
    #         predictions = xgb_model.predict(X[test_index])
    #         actuals = y[test_index]
    #         print(confusion_matrix(actuals, predictions))

    # # predict the target on the train dataset
    predict_train = gsearch.predict(X_train)
    print('\nTarget on train data',predict_train)

    return gsearch
