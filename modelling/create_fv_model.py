import os
import pickle
import pandas as pd
import numpy as np

from scipy import stats
from sklearn.preprocessing import LabelEncoder
from sklearn.utils.multiclass import type_of_target
# hyper parameter search
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV
# overfitting prevention
from sklearn.model_selection import train_test_split, KFold, StratifiedKFold
# unknown
from sklearn.model_selection import cross_validate

from xgboost import XGBRegressor, XGBClassifier

_BASE_DIR = '/home/costam/Documents'
_CODE_DIR = os.path.join(_BASE_DIR, 'fantacalcio/fanta_code')
_DATA_DIR = os.path.join(_BASE_DIR, 'fantacalcio/data')


def main(model_type='classifier'):

    # getting the data
    team_map, fix_master = get_masters()
    tm_feat, pl_feat = get_features()
    pl_fv = get_targets()
    # combining the data
    select_seasons = [2018, 2019]
    full_dt = combine_data(team_map, fix_master, tm_feat, pl_feat, pl_fv, select_seasons)
    # setting the random seed before the split
    rng = np.random.RandomState(1298)
    # preparing the features
    X_train, X_test, y_train, y_test = prepare_model_data(full_dt, rng, model_type)

    # # running the model for regression
    # gsearch = run_model(X_train, y_train, rng, model_type='regressor', full_grid=False)
    # regress_pkl_filename = os.path.join(_DATA_DIR, 'models', 'xgboost_nogrid_20201003.pkl')
    # # Open the file to save as pkl file
    # with open(regress_pkl_filename, 'wb') as f:
    #     pickle.dump(gsearch, f)

    # running the model for multi-label classification
    full_grid = True
    grid_iter = 80
    softmax_gsearch = run_grid(X_train, y_train, rng, model_type, full_grid, grid_iter)
    softmax_pkl_filename = 'xgboost_softmax_gridsearch_20201012.pkl'
    softmax_pkl_filename = os.path.join(_DATA_DIR, 'models', softmax_pkl_filename)
    # Open the file to save as pkl files
    with open(softmax_pkl_filename, 'wb') as f:
        pickle.dump(softmax_gsearch, f)

    # softmax_gsearch = gsearch

    grid_best_params = softmax_gsearch.best_params_
    run_model(grid_best_params, X_train, y_train, rng, model_type)

    softmax_gsearch.estimator

    return None


def get_masters():

    tm_master_file = os.path.join(_DATA_DIR, 'master', 'rapid_teamnames_map.csv')
    team_map = pd.read_csv(tm_master_file, header=0, index_col=False)

    fixture_master_file = os.path.join(_DATA_DIR, 'rapid_football/masters', 'fixture_master.csv')
    fix_master = pd.read_csv(fixture_master_file, header=0, index_col=False)

    return team_map, fix_master


def get_features():

    pl_feat_file = os.path.join(_DATA_DIR, 'target_features', 'players_features.csv')
    pl_feat = pd.read_csv(pl_feat_file, header=0, index_col=False)
    pl_feat = pl_feat.sort_values(['name', 'surname', 'season', 'match'])

    tm_feat_file = os.path.join(_DATA_DIR, 'target_features', 'team_features.csv')
    tm_feat = pd.read_csv(tm_feat_file, header=0, index_col=False)
    tm_feat = tm_feat.sort_values(['team_id', 'event_date'])
    # # todo(Michele) : add the correct match to each fixture as from API
    # # creating the match column based on the time of the game for each team
    # tm_feat['match'] = 1
    # tm_feat['match'] = tm_feat.groupby('team_id')['match'].cumsum()
    # todo(Michele) : mapping of each Rapid API season to the year
    tm_feat.loc[tm_feat['league_id']==94, 'season'] = int(2018)
    tm_feat.loc[tm_feat['league_id']==891, 'season'] = int(2019)

    return tm_feat, pl_feat


def get_targets():

    pl_fv_file = os.path.join(_DATA_DIR, 'target_features', 'players_target.csv')
    pl_fv = pd.read_csv(pl_fv_file, header=0, index_col=False)
    pl_fv = pl_fv.sort_values(['name', 'surname', 'season', 'match'])
    # filling the fv with 6 if they have not played and centering fv on 6
    pl_fv['fwd_fv_scaled'] = pl_fv['fwd_fantavoto'].fillna(6.0) - 6

    return pl_fv


def combine_data(team_map, fix_master, tm_feat, pl_feat, pl_fv, select_seasons):

    pl_feat_ssn_idx = pl_feat['season'].isin(select_seasons)
    pl_fv_ssn_idx = pl_fv['season'].isin(select_seasons)

    # getting the features split between team and player
    tm_remove = ['league_id', 'team_id', 'team_name', 'event_date', 'fixture_id', 'season','match']
    tm_cols = tm_feat.columns.values[~tm_feat.columns.isin(tm_remove)].tolist()
    pl_remove = ['team', 'name', 'surname', 'role', 'season', 'match']
    pl_cols = pl_feat.columns.values[~pl_feat.columns.isin(pl_remove)].tolist()

    # getting the home and away info for fixtures
    fix_cols = ['league_id','fixture_id','home_team_name','away_team_name', 'home_team_id', 'away_team_id']
    tm_feat = tm_feat.merge(fix_master.loc[:, fix_cols],
                            on = ['league_id', 'fixture_id'])
    # core team is home team
    core_home_idx = tm_feat['team_id'] == tm_feat['home_team_id']
    tm_feat.loc[core_home_idx, 'opponent_name'] = tm_feat.loc[core_home_idx,'away_team_name']
    tm_feat.loc[core_home_idx, 'opponent_id'] = tm_feat.loc[core_home_idx,'away_team_id']
    # core team is away team
    core_away_idx = tm_feat['team_id'] == tm_feat['away_team_id']
    tm_feat.loc[core_away_idx, 'opponent_name'] = tm_feat.loc[core_away_idx,'home_team_name']
    tm_feat.loc[core_away_idx, 'opponent_id'] = tm_feat.loc[core_away_idx,'home_team_id']
    tm_feat.drop(['away_team_name','home_team_name', 'home_team_id', 'away_team_id'], axis=1, inplace=True)

    # merging the opponents data onto the main df
    tm_oppo_feat = tm_feat.copy()
    tm_oppo_feat.drop(['team_name', 'event_date', 'match', 'season', 'opponent_name', 'opponent_id'],
                      axis=1, inplace=True)
    new_col_names = 'op_'+tm_oppo_feat.columns[3:].values
    new_col_dict = dict(zip(tm_oppo_feat.columns[3:].values, new_col_names))
    tm_oppo_feat.rename(columns=new_col_dict, inplace=True)
    tm_feat = tm_feat.merge(tm_oppo_feat, how='left',
                            left_on = ['league_id', 'fixture_id', 'opponent_id'],
                            right_on = ['league_id', 'fixture_id', 'team_id'])
    tm_feat['team_id'] = tm_feat['team_id_x']
    tm_feat.drop(['team_id_x','team_id_y'], axis=1, inplace=True)

    # correct name of team coming from Gazzetta
    tm_feat = tm_feat.merge(team_map, left_on = 'team_name', right_on = 'team_name')
    tm_feat['team_name'] = tm_feat['team_gaz']
    tm_feat.drop(['team_gaz'], axis=1, inplace=True)

    # team_id	event_date	fixture_id	league_id	match	season	opponent_name	team_name

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

    # adding the features for the player's team
    full_dt = full_dt.merge(tm_feat,
                            left_on=['team','season','match'],
                            right_on=['team_name','season', 'match'],
                            how='left')
    # filling the team features
    tm_pivot = ['team', 'season', 'match']
    full_dt = full_dt.sort_values(tm_pivot)
    full_dt.loc[:, tm_cols] = full_dt.groupby(tm_pivot[0:2])[tm_cols].fillna(method='ffill')
    oppo_tm_cols = ['op_' + x for x in tm_cols]
    full_dt.loc[:, oppo_tm_cols] = full_dt.groupby(tm_pivot[0:2])[oppo_tm_cols].fillna(method='ffill')

    # cleaning up the features data
    full_dt.columns.values
    drop_cols = ['team_id', 'opponent_id', 'team', 'team_name', 'opponent_name',
                 'event_date', 'fixture_id', 'league_id']
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


def bin_target_values(v):

    neg_fv_groups = [-20] + np.arange(-2.5, 0, 2).tolist()
    pos_fv_groups = [0] + np.arange(0.5, 5.5, 2).tolist() + [20]
    fv_groups = neg_fv_groups + pos_fv_groups
    bins = []
    labels = []
    for i in range(1,len(fv_groups)):
        bins.append((fv_groups[i-1], fv_groups[i]))
        labels.append(fv_groups[i-1])
    bins = pd.IntervalIndex.from_tuples(bins, closed='left') # in this way 0 is counted as positive

    labels_dict = dict(zip(bins, labels))
    v_bin = pd.cut(v, bins, include_lowest=True)
    v_bin = v_bin.map(labels_dict)

    return v_bin


def prepare_model_data(full_dt, rng, model_type, target_var='fwd_fv_scaled'):

    # creating features and target data
    exclude_col = ['name', 'surname', 'season', 'match', 'fwd_fantavoto', 'fwd_fv_scaled', 'role']
    feat_cols = full_dt.columns.values[~full_dt.columns.isin(exclude_col)].tolist()

    X = full_dt.loc[:, feat_cols]
    # causing issue, should try one hot encoding
    # X['role'] = X['role'].astype('category')
    y = full_dt.loc[:, target_var]

    # set the labels for the target if we are using the multiclass
    if model_type == 'classifier':
        y = bin_target_values(v=y)

    stratify_df = pd.DataFrame({'fv' : y, 'role' : full_dt['role']})
    # stratify_df.groupby(['fv', 'role'])['fv'].agg('count')

    X_train, X_test, y_train, y_test = train_test_split(X, y,
                                                        test_size=0.2,
                                                        random_state=rng,
                                                        stratify=stratify_df)
    # saving the data to file to simplify the analysis after
    save_all_data(X_train, X_test, y_train, y_test)

    return X_train, X_test, y_train, y_test


def save_all_data(X_train, X_test, y_train, y_test ):

    X_train.to_csv(os.path.join(_DATA_DIR, 'target_features', 'x_train_dt.csv'), header=True, index=True)
    X_test.to_csv(os.path.join(_DATA_DIR, 'target_features', 'x_test_dt.csv'), header=True, index=True)
    y_train.to_csv(os.path.join(_DATA_DIR, 'target_features', 'y_train_dt.csv'), header=True, index=True)
    y_test .to_csv(os.path.join(_DATA_DIR, 'target_features', 'y_test_dt.csv'), header=True, index=True)


def run_grid(X_train, y_train, rng, model_type, full_grid, grid_iter):

    # initialise the model
    # https://xgboost.readthedocs.io/en/latest/python/python_api.html#module-xgboost.sklearn
    if model_type == 'regressor':
        model = XGBRegressor(objective='reg:squarederror',
                                 booster='gbtree',
                                 tree_method='hist', # vs auto, exact, approx, hist, gpu_hist,
                                 n_jobs=-1,
                                 random_state=rng,
                                 # scoring = 'mse',
                                 eval_metric='rmse')
    elif model_type == 'classifier':
        # possible to run also multiple metrics as eval : eval_metric = ["merror", "map", "auc"]
        model = XGBClassifier(objective='multi:softmax',
                              booster='gbtree',
                              tree_method='hist', # vs auto, exact, approx, hist, gpu_hist,
                              n_jobs=-1,
                              random_state=rng,
                              # scoring = 'mse',
                              num_class = len(y_train.cat.categories),
                              eval_metric='merror') # vs error, mlogloss, auc

    # cross validate: we stratify the train, test split by role, and we further
    # stratify based on y labels if classifier is requested
    if model_type == 'regressor':
        fold5_cv = KFold(n_splits=5, shuffle=False)
    elif model_type == 'classifier':
        label_encoder = LabelEncoder()
        y_train = label_encoder.fit_transform(y_train)
        fold5_cv = StratifiedKFold(n_splits=5, shuffle=False)

    # parameter grid
    # for an overview of the cv scoring :
    # https://scikit-learn.org/stable/modules/model_evaluation.html#scoring-parameter
    # grid example : https://gist.github.com/fpleoni/c44c0c3654c12abe686a80937d9da200

    if full_grid:
        # determining distribution
        estim_p = stats.randint(300, 1000) #[300, 500, 700, 1000]
        depth_p = stats.randint(4, 13) #[5,7,10,15]
        learn_p = [0.01, 0.1, 0.5]
        # gamma_p = [0.02]+ [i/100.0 for i in range(10,100,30)] + [1, 1.5]
        gamma_p = stats.invgamma(1.8)
        lambd_p = stats.lognorm(0.75)

        param_search = {
            'n_estimators':estim_p, # Number of gradient boosted trees.
            'max_depth':depth_p, # Maximum tree depth for base learners.
            'learning_rate':learn_p, # equivalent to eta
            # Minimum loss reduction required to make a further partition on a leaf node
            'gamma': gamma_p,
            # Minimum sum of instance weight(hessian) needed in a child
            # 'min_child_weight':[j for j in range(1,6,2)],
            # Subsample ratio of the training instance.
            'subsample': [0.85], #[i/10.0 for i in range(7,10)],
            # Subsample ratio of columns when constructing each tree.
            'colsample_bytree': [0.85], #[i/10.0 for i in range(7,10)],
            # L1 regularization term on weights
            # 'reg_alpha':[0, 1e-5, 1e-2, 0.1, 1, 10],
            # L2 regularization term on weights
            'reg_lambda':lambd_p
            }
        gsearch = RandomizedSearchCV(estimator=model, cv=fold5_cv, n_iter=grid_iter,
                                     param_distributions=param_search,n_jobs=-1,
                                     scoring='neg_mean_squared_error',
                                     verbose=True)
    else:
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
        gsearch = GridSearchCV(estimator=model, cv=fold5_cv,
                               param_grid=param_search, pre_dispatch=3,
                               scoring='neg_mean_squared_error', verbose=True)


    # fitting
    gsearch.fit(X_train, y_train)

    return gsearch


def run_model(grid_best_params, X_train, y_train, rng, model_type):

    if model_type == 'regressor':
        model = XGBRegressor(objective='reg:squarederror',
                             booster='gbtree',
                             tree_method='hist',
                             n_jobs=-1,
                             random_state=rng,
                             eval_metric='rmse',
                             # start of parameters taken from the grid
                             n_estimators = grid_best_params['n_estimators'],
                             max_depth = grid_best_params['max_depth'],
                             learning_rate = grid_best_params['learning_rate'],
                             gamma = grid_best_params['gamma'],
                             subsample = grid_best_params['subsample'],
                             colsample_bytree = grid_best_params['colsample_bytree'],
                             reg_lambda = grid_best_params['reg_lambda'])

    elif model_type == 'classifier':
        # possible to run also multiple metrics as eval : eval_metric = ["merror", "map", "auc"]
        model = XGBClassifier(objective='multi:softmax',
                              booster='gbtree',
                              tree_method='hist',
                              n_jobs=-1,
                              random_state=rng,
                              num_class = len(y_train.cat.categories),
                              eval_metric='merror',
                              # start of parameters taken from the grid
                              n_estimators = grid_best_params['n_estimators'],
                              max_depth = grid_best_params['max_depth'],
                              learning_rate = grid_best_params['learning_rate'],
                              subsample = grid_best_params['subsample'],
                              colsample_bytree = grid_best_params['colsample_bytree'],
                              reg_lambda = grid_best_params['reg_lambda'])

    # #Set our final hyperparameters to the tuned values
    if model_type == 'regressor':
        fold5_cv = KFold(n_splits=5, shuffle=False)
    elif model_type == 'classifier':
        fold5_cv = StratifiedKFold(n_splits=5, shuffle=False)

    # Evaluate metric(s) by cross-validation and also record fit/score times
    cv_scoring = ['accuracy', 'roc_auc', 'precision', 'recall', 'f1']
    cv_scores = cross_validate(estimator=model,
                               X=X_train,
                               y=y_train,
                               cv=fold5_cv,
                               n_jobs=-1,
                               scoring=cv_scoring)

    print('Training 5-fold Cross Validation Results:\n')
    print('AUC: ', cv_scores['test_roc_auc'].mean())
    print('Accuracy: ', cv_scores['test_accuracy'].mean())
    print('Precision: ', cv_scores['test_precision'].mean())
    print('Recall: ', cv_scores['test_recall'].mean())
    print('F1: ', cv_scores['test_f1'].mean(), '\n')

    model.fit(X_train, y_train)

    return model
