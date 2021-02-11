import os
import pickle
import json
import pandas as pd
import numpy as np

from scipy import stats
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
# hyper parameter search
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV, cross_validate
# overfitting prevention
from sklearn.model_selection import train_test_split, KFold, StratifiedKFold
from sklearn.metrics import accuracy_score

from xgboost import XGBRegressor, XGBClassifier

from utils.utils_features_target_model import get_ffdata_combined
from utils.utils_setup_model import bin_target_values, assign_label_relative_importance

_BASE_DIR = '/home/costam/Documents'
_DATA_DIR = os.path.join(_BASE_DIR, 'fantacalcio/data')


def main(model_type='classifier', use_class_scale=True, full_grid=True, grid_iter=80):

    model_type='classifier'
    use_class_scale=True
    full_grid = False
    grid_iter = 80

    # getting the data
    select_seasons = [2018, 2019, 2020]
    full_dt = get_ffdata_combined(select_seasons)
    print("Dropping the observations with NA fantavoto!")
    full_dt = full_dt.loc[~full_dt["fwd_fantavoto"].isna()]
    # setting the random seed before the split
    rng = np.random.RandomState(1298)
    # preparing the features
    X_train, X_test, y_train, y_test, label_encoder = prepare_model_data(full_dt, rng, model_type)

    scale_weight = np.repeat(1, len(y_test) + len(y_train))
    if model_type=='classifier' and use_class_scale:
        scale_weight, labels_weight_map = assign_label_relative_importance(y_train,
                                                                           y_test,
                                                                           label_encoder)

    # running the model for multi-label classification
    softmax_gsearch = run_grid(X_train, y_train, rng, model_type, full_grid,
                               grid_iter, scale_weight)
    # softmax_pkl_filename = 'xgboost_softmax_gridsearch_20201018.pkl'
    softmax_pkl_filename = 'xgboost_softmax_gridsearch_20210101.pkl'
    softmax_pkl_filename = os.path.join(_DATA_DIR, 'models', softmax_pkl_filename)
    # Open the file to save as pkl files
    with open(softmax_pkl_filename, 'wb') as f:
        pickle.dump(softmax_gsearch, f)
    # with open(softmax_pkl_filename, 'rb') as f:
    #     softmax_gsearch = pickle.load(f)

    grid_best_params = softmax_gsearch.best_params_
    validation_out, validated_model = run_model(grid_best_params, X_train, y_train,
                                                rng, model_type, scale_weight)
    # softmax_validated_filename = 'xgboost_softmax_validated_20201018.pkl'
    softmax_validated_filename = 'xgboost_softmax_validated_20210101.pkl'
    softmax_validated_filename = os.path.join(_DATA_DIR, 'models', softmax_validated_filename)
    # Open the file to save as pkl files
    with open(softmax_validated_filename, 'wb') as f:
        pickle.dump(validated_model, f)

    return None


def prepare_model_data(full_dt, rng, model_type, target_var='fwd_fv_scaled', test_size=0.2):

    # creating features and target data
    exclude_col = ['name', 'surname', 'season', 'match', 'fwd_fantavoto', 'fwd_fv_scaled', 'role']
    feat_cols = full_dt.columns.values[~full_dt.columns.isin(exclude_col)].tolist()

    X = full_dt.loc[:, feat_cols]
    # causing issue, should try one hot encoding
    # X['role'] = X['role'].astype('category')
    y = full_dt.loc[:, target_var]

    # set the labels for the target if we are using the multiclass
    label_encode = None
    if model_type == 'classifier':
        y = bin_target_values(v=y)
        label_encoder = LabelEncoder()
        y = label_encoder.fit_transform(y)

    stratify_df = pd.DataFrame({'fv' : y, 'role' : full_dt['role']})
    # stratify_df.groupby(['fv', 'role'])['fv'].agg('count')

    X_train, X_test, y_train, y_test = train_test_split(X, y,
                                                        test_size=test_size,
                                                        random_state=rng,
                                                        stratify=stratify_df)
    # saving the data to file to simplify the analysis after
    save_all_data(X_train, X_test, y_train, y_test, target_var)

    return X_train, X_test, y_train, y_test, label_encoder


def save_all_data(X_train, X_test, y_train, y_test, target_var):

    X_train.to_csv(os.path.join(_DATA_DIR, 'target_features', 'x_train_dt.csv'), header=True, index=True)
    X_test.to_csv(os.path.join(_DATA_DIR, 'target_features', 'x_test_dt.csv'), header=True, index=True)
    pd.DataFrame({target_var:y_train}).to_csv(os.path.join(_DATA_DIR, 'target_features', 'y_train_dt.csv'), header=True, index=True)
    pd.DataFrame({target_var:y_test}).to_csv(os.path.join(_DATA_DIR, 'target_features', 'y_test_dt.csv'), header=True, index=True)


def run_grid(X_train, y_train, rng, model_type, full_grid, grid_iter, scale_weight):

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
                              num_class = len(np.unique(y_train)),
                              eval_metric='mlogloss') # vs error, mlogloss, auc

    # cross validate: we stratify the train, test split by role, and we further
    # stratify based on y labels if classifier is requested
    if model_type == 'regressor':
        fold5_cv = KFold(n_splits=5, shuffle=False)
    elif model_type == 'classifier':
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
    ins_scale_weight = scale_weight[0:len(y_train)]
    gsearch.fit(X=X_train, y=y_train,sample_weight=ins_scale_weight)

    return gsearch


def run_model(grid_best_params, X_train, y_train, rng, model_type, scale_weight):

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
                              num_class = len(np.unique(y_train)),
                              eval_metric='mlogloss',
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

    validation_out = cross_validate(model,
                                     X=X_train.iloc[:1000,:],
                                     y=y_train[:1000,],
                                     scoring=['accuracy'],
                                     cv=fold5_cv,
                                     n_jobs=-1,
                                     verbose=1,
                                     return_estimator=True)

    # Fit the final model
    ins_scale_weight = scale_weight[0:len(y_train)]
    validated_model = model.fit(X_train, y_train, sample_weight=ins_scale_weight)

    return validation_out, validated_model
