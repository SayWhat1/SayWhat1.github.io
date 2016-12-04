---
title: This is just a test
date: 2016-12-03 21:42:36
tags:
---

This is an example of a helper function I used when building my XGBoost model for the Kaggle competition, Santander Customer Satisfaction.

```python
def modelFit(alg, dtrain, predictors, target, useTrainCV = True, early_stopping_rounds = 50, cv_folds = 5):
    '''.'''

    if useTrainCV:
		xgb_param = alg.get_xgb_params()
		xgtrain = xgb.DMatrix(dtrain[predictors].values, label = dtrain[target].values)
		cvresult = xgb.cv(xgb_param, xgtrain, num_boost_round = alg.get_params()['n_estimators'], nfold = cv_folds,\
                          metrics = 'auc', early_stopping_rounds = early_stopping_rounds, verbose_eval = 100)
        alg.set_params(n_estimators = cvresult.shape[0])

        alg.fit(dtrain[predictors], dtrain[target], eval_metric = 'auc')

        dtrain_predictions = alg.predict(dtrain[predictors])
        dtrain_predprob = alg.predict_proba(dtrain[predictors])[:,1]

        print('\\nModel Report')
        print('Accuracy : {:.4g}'.format(metrics.accuracy_score(dtrain[target].values, dtrain_predictions)))
        print('AUC Score (Train): {:f}'.format(metrics.roc_auc_score(dtrain[target], dtrain_predprob)))

        feat_imp = pd.Series(alg.booster().get_fscore()).sort_values(ascending = False)
        feat_imp.plot(kind = 'bar', title = 'Feature Importances')
        plt.ylabel('Feature Importance Score')"
```

The End?
