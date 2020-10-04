
# five different feature attribution methods on our simple tree models:
# 1) Tree SHAP. A new individualized method we are proposing.
# 2) Saabas. An individualized heuristic feature attribution method.
# 3) mean(|Tree SHAP|). A global attribution method based on the average magnitude of the individualized Tree SHAP attributions.
# 4) Gain. The same method used above in XGBoost, and also equivalent to the Gini importance measure used in scikit-learn tree models.
# 5) Split count. Represents both the closely related “weight” and “cover” methods in XGBoost, but is computed using the “weight” method.
# 6) Permutation. The resulting drop in accuracy of the model when a single feature is randomly permuted in the test data set.


def analyse_ins_prediction(model, X_train, y_train):

    best_model_feat = model.best_estimator_
    feat_importance = pd.Series(best_model.feature_importances_)
    feat_importance.index = X_train.columns.values
    feat_importance.sort_values(ascending=True, inplace=True)

    # Setting the figure size
    fig = plt.figure(figsize=(10,8))
    plt.barh(feat_importance.tail(40).index, feat_importance.tail(40), alpha=0.3)
    plt.xticks(rotation=90)
    plt.title('XGBoost Regression - Feature Importance')
    #Saving the plot as an image
    fig.savefig(os.path.join(_DATA_DIR, 'figs', 'xgb_reg_top_features.jpg'), bbox_inches='tight', dpi=150)

    fig = plt.figure(figsize=(10,8))
    plt.barh(feat_importance.head(40).index, feat_importance.head(40), alpha=0.3)
    plt.xticks(rotation=90)
    plt.title('XGBoost Regression - Feature Importance')
    #Saving the plot as an image
    fig.savefig(os.path.join(_DATA_DIR, 'figs', 'xgb_reg_bot_features.jpg'), bbox_inches='tight', dpi=150)


    ins_preds = model.predict(X_train)
    ins_mse = mean_squared_error(y_train, ins_preds)
    ins_exvar = explained_variance_score(y_train, ins_preds)
    ins_mae = median_absolute_error(y_train, ins_preds)

    compare_dt = pd.DataFrame({'true' : y_train['fwd_fv_scaled'], 'pred' : ins_preds})
    kwargs = dict(histtype='stepfilled', alpha=0.3, bins=30)
    plt.hist(data=compare_dt[['true']], x='true', **kwargs)
    plt.hist(data=compare_dt[['pred']], x='pred', **kwargs)
    plt.show()

    return None
