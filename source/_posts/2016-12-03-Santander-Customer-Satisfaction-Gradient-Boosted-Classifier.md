---
title: Santander Customer Satisfaction - Gradient Boosted Classifier
date: 2016-12-03 21:43:01
tags: Python; Data Analysis; Kaggle
---


I started this blog to show off some of the things I'm doing to learn more about data analysis. So, let's get started. Recently, I've been looking at the [Santander Customer Satisfaction](https://www.kaggle.com/c/santander-customer-satisfaction) competition on Kaggle. I started off with gradient boosting classifier. I took a lot of inspiration from [Analytics Vidhya](http://www.analyticsvidhya.com/). He has some really good guides on parameter tuning in both Python and R.

First off, I made all the required imports and then adapted a function from the previous website to help check out the performance of the model.

```python
def modelFit(alg, dtrain, predictors, target, performCV = True, printFeatReport = True, cv_folds = 5):

#alg is the model to fit, dtrain is the training dataframe, predictors is a string or list of
#strings of the column names to use as predictors, target a string of the column with the target
#performCV will crossvalidate the model, printFeatImport will print a graph showing the important
#features, and cv_folds is the number of folds to use in cross validation.'''

alg.fit(dtrain[predictors], dtrain[target])

dtrain_predictions = alg.predict(dtrain[predictors])
dtrain_predprob = alg.predict_proba(dtrain[predictors])[:,1]

if performCV:
    cv_score = cross_validation.cross_val_score(alg, dtrain[predictors], dtrain[target], \
                                               cv = cv_folds, scoring = 'roc_auc' )

print('\nModel Report')
print('Accuracy : {:.4g}'.format(metrics.accuracy_score(dtrain[target].values, dtrain_predictions)))
print('AUC Score (Train): {:f}'.format(metrics.roc_auc_score(dtrain[target], dtrain_predprob)))

if performCV:
    print('CV Score: Mean - {:.7g} | Std - {:.7g} | Min - {:.7g} | Max - {:.7g}'.format(np.mean(cv_score),\
                                                                                        np.std(cv_score),\
                                                                                        np.min(cv_score),\
                                                                                        np.max(cv_score)))

if printFeatReport:
    feat_imp = pd.Series(alg.feature_importances_, predictors).sort_values(ascending = False)
    feat_imp.plot(kind = 'bar', title = 'Feature Importances')
    plt.ylabel('Feature Importance Score')
```

I haven't been working with it long, but there are already a few tweeks that I'd like to make. Mainly, to list out the top performing features so I can start thinking about cutting down on the features I'm training with.

I import the data, and do some very minor cleaning. I first drop the columns that are uniform, i.e. have zero variance, then I drop columns that are identical. Then, there is only one column of the data that has farily obvious outliers and several invalid entries. Discounting the outliers and NaN's, the column has so little variance that instead of working with it futher, I didn't use it to train the model. All told, 63 columns were removed from the training and testing data.

```python
train = pd.read_csv('Data/Santander/train.csv')
test = pd.read_csv('Data/Santander/test.csv')

dropCols = []
for i in train.columns.values:
    if len(train[i].unique()) == 1:
        dropCols.append(i)
print('Dropping {} columns due to non-unique entries'.format(len(dropCols)))
train.drop(dropCols, axis = 1,  inplace = True)
test.drop(dropCols, axis = 1, inplace = True)

dropCols = []
c = train.columns
for i in range(len(c)-1):
    v = train[c[i]].values
    for j in range(i+1,len(c)):
        if np.array_equal(v,train[c[j]].values):
            dropCols.append(c[j])
print('Dropping {} columns due to duplicate columns'.format(len(dropCols)))
train.drop(dropCols, axis = 1,  inplace = True)
test.drop(dropCols, axis = 1, inplace = True)

train.loc[train.var3 < -10000, 'var3'] = np.nan
test.loc[test.var3 < -10000, 'var3'] = np.nan

colsWithNAN = []
for i in train.columns.values:
    if train[i].isnull().sum() > 0:
        colsWithNAN.append(i)
train.drop(colsWithNAN, axis = 1,  inplace = True)
test.drop(colsWithNAN, axis = 1, inplace = True)
```

Next, I trained the model with all of the default values to get a baseline for the models performance.

```python
pred = [i for i in train.columns if i not in ['ID', 'TARGET']]
gbm0 = GradientBoostingClassifier(random_state = 10)
modelFit(gbm0, train, pred, 'TARGET', printFeatReport = False)
```

The results it gave me were decent for a first run. The AUC score was around .85 and it cross validated at .8352\. Next, I ran a few different grid searches and refined the model.

```python
paramTest1 = {'n_estimators' : range(80,141,10)}
gSearch1 = GridSearchCV(estimator = GradientBoostingClassifier(learning_rate = 0.1,\
                                                               min_samples_split = 500,\
                                                               min_samples_leaf = 50,\
                                                               max_depth = 8,\
                                                               max_features = 'sqrt',\
                                                               subsample = 0.8,\
                                                               random_state = 10),
                       param_grid = paramTest1, scoring = 'roc_auc', n_jobs = 8, iid = False, cv = 5)
gSearch1.fit(train[pred], train['TARGET'])
gSearch1.grid_scores_, gSearch1.best_params_, gSearch1.best_score_

([mean: 0.83770, std: 0.00833, params: {'n_estimators': 80},
  mean: 0.83784, std: 0.00819, params: {'n_estimators': 90},
  mean: 0.83793, std: 0.00867, params: {'n_estimators': 100},
  mean: 0.83757, std: 0.00844, params: {'n_estimators': 110},
  mean: 0.83734, std: 0.00855, params: {'n_estimators': 120},
  mean: 0.83726, std: 0.00826, params: {'n_estimators': 130},
  mean: 0.83698, std: 0.00809, params: {'n_estimators': 140}],
 {'n_estimators': 100},
 0.83792968856907224)
```

First I checked the number of estimators, and as you can see from above, it found that 100 was the best fit.

```python
paramTest2 = {'max_depth' : range(5,16,2), 'min_samples_split' : range(200,1001, 200)}
gSearch2 = GridSearchCV(estimator = GradientBoostingClassifier(learning_rate = 0.1,\
                                                               n_estimators = 100,\
                                                               min_samples_leaf = 50,\
                                                               max_features = 'sqrt',\
                                                               subsample = 0.8,\
                                                               random_state = 10),
                       param_grid = paramTest2, scoring = 'roc_auc', n_jobs = 8, iid = False, cv = 5)
gSearch2.fit(train[pred], train['TARGET'])
gSearch2.grid_scores_, gSearch2.best_params_, gSearch2.best_score_

([mean: 0.83576, std: 0.00915, params: {'min_samples_split': 200, 'max_depth': 5},
  mean: 0.83705, std: 0.00813, params: {'min_samples_split': 400, 'max_depth': 5},
  mean: 0.83613, std: 0.00887, params: {'min_samples_split': 600, 'max_depth': 5},
  mean: 0.83639, std: 0.00912, params: {'min_samples_split': 800, 'max_depth': 5},
  mean: 0.83573, std: 0.00873, params: {'min_samples_split': 1000, 'max_depth': 5},
  mean: 0.83491, std: 0.00716, params: {'min_samples_split': 200, 'max_depth': 7},
  mean: 0.83665, std: 0.00764, params: {'min_samples_split': 400, 'max_depth': 7},
  mean: 0.83653, std: 0.00902, params: {'min_samples_split': 600, 'max_depth': 7},
  mean: 0.83680, std: 0.00873, params: {'min_samples_split': 800, 'max_depth': 7},
  mean: 0.83667, std: 0.00902, params: {'min_samples_split': 1000, 'max_depth': 7},
  mean: 0.83550, std: 0.00747, params: {'min_samples_split': 200, 'max_depth': 9},
  mean: 0.83760, std: 0.00702, params: {'min_samples_split': 400, 'max_depth': 9},
  mean: 0.83716, std: 0.00751, params: {'min_samples_split': 600, 'max_depth': 9},
  mean: 0.83675, std: 0.00851, params: {'min_samples_split': 800, 'max_depth': 9},
  mean: 0.83663, std: 0.00870, params: {'min_samples_split': 1000, 'max_depth': 9},
  mean: 0.83276, std: 0.00819, params: {'min_samples_split': 200, 'max_depth': 11},
  mean: 0.83558, std: 0.00950, params: {'min_samples_split': 400, 'max_depth': 11},
  mean: 0.83589, std: 0.00749, params: {'min_samples_split': 600, 'max_depth': 11},
  mean: 0.83713, std: 0.00861, params: {'min_samples_split': 800, 'max_depth': 11},
  mean: 0.83527, std: 0.00898, params: {'min_samples_split': 1000, 'max_depth': 11},
  mean: 0.83208, std: 0.00747, params: {'min_samples_split': 200, 'max_depth': 13},
  mean: 0.83346, std: 0.00818, params: {'min_samples_split': 400, 'max_depth': 13},
  mean: 0.83536, std: 0.00780, params: {'min_samples_split': 600, 'max_depth': 13},
  mean: 0.83700, std: 0.00883, params: {'min_samples_split': 800, 'max_depth': 13},
  mean: 0.83624, std: 0.00855, params: {'min_samples_split': 1000, 'max_depth': 13},
  mean: 0.83061, std: 0.00964, params: {'min_samples_split': 200, 'max_depth': 15},
  mean: 0.83242, std: 0.00796, params: {'min_samples_split': 400, 'max_depth': 15},
  mean: 0.83445, std: 0.00908, params: {'min_samples_split': 600, 'max_depth': 15},
  mean: 0.83434, std: 0.00896, params: {'min_samples_split': 800, 'max_depth': 15},
  mean: 0.83583, std: 0.00880, params: {'min_samples_split': 1000, 'max_depth': 15}],
 {'max_depth': 9, 'min_samples_split': 400},
 0.83759736130706453)
```

Next, I searched for the `max_depth` and `min_samples_split` parameters. As you can see, the best values were 9 and 400 respectively.

```python
paramTest3 = {'min_samples_leaf' : range(30,71, 10)}
gSearch3 = GridSearchCV(estimator = GradientBoostingClassifier(learning_rate = 0.1,\
                                                               n_estimators = 100,\
                                                               max_depth = 9,\
                                                               min_samples_split = 400,\
                                                               max_features = 'sqrt',\
                                                               subsample = 0.8,\
                                                               random_state = 10),
                        param_grid = paramTest3, scoring = 'roc_auc', n_jobs = 8, iid = False, cv = 5)
gSearch3.fit(train[pred], train['TARGET'])
gSearch3.grid_scores_, gSearch3.best_params_, gSearch3.best_score_

([mean: 0.83655, std: 0.00843, params: {'min_samples_leaf': 30},
  mean: 0.83702, std: 0.00998, params: {'min_samples_leaf': 40},
  mean: 0.83760, std: 0.00702, params: {'min_samples_leaf': 50},
  mean: 0.83666, std: 0.00899, params: {'min_samples_leaf': 60},
  mean: 0.83603, std: 0.00967, params: {'min_samples_leaf': 70}],
 {'min_samples_leaf': 50},
 0.83759736130706453)

 paramTest4 = {'max_features' : range(17,30, 2)}
gSearch4 = GridSearchCV(estimator = GradientBoostingClassifier(learning_rate = 0.1,\
                                                               n_estimators = 100,\
                                                               max_depth = 9,\
                                                               min_samples_split = 400,\
                                                               min_samples_leaf = 50,\
                                                               subsample = 0.8,\
                                                               random_state = 10),
                       param_grid = paramTest4, scoring = 'roc_auc', n_jobs = 8, iid = False, cv = 5)
gSearch4.fit(train[pred], train['TARGET'])
gSearch4.grid_scores_, gSearch4.best_params_, gSearch4.best_score_

([mean: 0.83760, std: 0.00702, params: {'max_features': 17},
  mean: 0.83686, std: 0.00983, params: {'max_features': 19},
  mean: 0.83567, std: 0.00880, params: {'max_features': 21},
  mean: 0.83631, std: 0.00850, params: {'max_features': 23},
  mean: 0.83735, std: 0.00862, params: {'max_features': 25},
  mean: 0.83588, std: 0.00836, params: {'max_features': 27},
  mean: 0.83722, std: 0.00796, params: {'max_features': 29}],
 {'max_features': 17},
 0.83759736130706453)

 paramTest5 = {'subsample' : [0.6, 0.7, 0.75, 0.8, 0.85, 0.9]}
gSearch5 = GridSearchCV(estimator = GradientBoostingClassifier(learning_rate = 0.1,\
                                                                n_estimators = 100,\
                                                                max_depth = 9,\
                                                                min_samples_split = 400,\
                                                                min_samples_leaf = 50,\
                                                                max_features = 'sqrt',\
                                                                random_state = 10),\
                                                                param_grid = paramTest5, scoring = 'roc_auc', n_jobs = 8, iid = False, cv = 5)
gSearch5.fit(train[pred], train['TARGET'])
gSearch5.grid_scores_, gSearch5.best_params_, gSearch5.best_score_

([mean: 0.83661, std: 0.00957, params: {'subsample': 0.6},
  mean: 0.83638, std: 0.00789, params: {'subsample': 0.7},
  mean: 0.83606, std: 0.00882, params: {'subsample': 0.75},
  mean: 0.83760, std: 0.00702, params: {'subsample': 0.8},
  mean: 0.83678, std: 0.00788, params: {'subsample': 0.85},
  mean: 0.83663, std: 0.00895, params: {'subsample': 0.9}],
 {'subsample': 0.8},
 0.83759736130706453)
```

The final three grid searches I ran were for `min_samples_leaf`, `max_features`, and `subsample`. After tuning these 6 parameters, I got the cross validation score to .8379 or an increase of .002\. The following is the final feature importance chart.<br>
![GBC Final Feature Importance Chart](/images/GBC-Final-Feat-Importance.png)

The next thing I plan to do with this model is reduce the number of features using the above chart and adjust the learning rate. I'm hoping to be able to squeeze a few more hundreths of a point out of this model.

This will be an ongoing series, as I have already trained a random forest and used XGBoost to model the same data and will be posting about them in the future. Stay tuned, because eventually I'll start to ensemble these models and see how high I can finish in the leaderboards. As of writing this, I'm sitting just above the 50th percentile. You can find the [notebook file](https://github.com/SayWhat1/Santander-Customer-Satisfaction-Kaggle/blob/master/Santander%20Customer%20Satisfaction%20GBC.ipynb) on my Github, but be warned that it is constantly changing as I find time to mess with it.
