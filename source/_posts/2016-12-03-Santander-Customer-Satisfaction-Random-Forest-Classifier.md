---
title: Santander Customer Satisfaction - Random Forest Classifier
date: 2016-12-03 21:43:30
tags:
  - Python
  - Data Analysis
  - Kaggle
categories:
  - Data Analysis
---

This is the second in a series of post detailing the analysis I've performed on the [Santander Customer Satisfaction](https://www.kaggle.com/c/santander-customer-satisfaction) competition on Kaggle.

This is the first algorithm that I tried to tune the parameters on my own.  I couldn't find a good guide, so I did my best based on the guide I used for the gradient boosted algorithm.  I used the same helper function, modified slightly to fit the algorithm.

{% codeblock lang:python %}
def modelFit(alg, dtrain, predictors, target, performCV = True, printFeatReport = True, cv_folds = 5):

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
{% endcodeblock %}

Along with what I mentioned last time, I think I want to modify this to automatically remove features that don't contribute to the model.  I could do it iteratively and stop when the cross validation score starts to drop.  These models already take a few minutes each to run, and when I start to cross validate, it could take 10 or 15 minutes to train.  In the grand scheme of things, that's not that long considering what it could take with more advanced models, but if I start trying to ensemble I'd rather not have to wait overnight to check my results.  

I use the exact same code to import and clean the data to get it ready for training the model, but I'll include it below for the sake of completeness.

{% codeblock lang:python %}
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
{% endcodeblock %}
I ran the model with the default parameter values and set the random state so that I could train multiple times and get consistent results.

{% codeblock lang:python %}
pred = [i for i in train.columns if i not in ['ID', 'TARGET']]
rfc0 = RandomForestClassifier(n_estimators = 500, \
                              random_state = 42)
modelFit(rfc0, train, pred, 'TARGET')
{% endcodeblock %}
The default parameters resulted in a CV score of 0.7653.  Not bad for a first run, but there is a lot of room for improvement.

Next, instead of running several different grid searches on each parameter, I ran a randomized search on all of the relevant parameters.  The function to report the findings of the random search was taken from the [scikit-learn](http://scikit-learn.org/stable/auto_examples/model_selection/randomized_search.html) example.  It was modified very slightly to fit my needs though.

{% codeblock lang:python %}
def report(grid_scores, n_top=3):
    top_scores = sorted(grid_scores, key=itemgetter(1), reverse=True)[:n_top]
    for i, score in enumerate(top_scores):
        print("\nModel with rank: {0}".format(i + 1))
        print("Mean validation score: {0:.3f} (std: {1:.3f})".format(score.mean_validation_score,np.std(score.cv_validation_scores)))
        print("Parameters: {0}".format(score.parameters))


param_dist = {"max_depth": sp_randint(3,10),
              "max_features": sp_randint(15, 29),
              "min_samples_split": sp_randint(400, 600),
              "min_samples_leaf": sp_randint(40, 60),
              "bootstrap": [True, False],
              "criterion": ["gini", "entropy"]}

random_search = RandomizedSearchCV(rfc0, param_distributions = param_dist, n_iter = 20)
random_search.fit(train[pred], train['TARGET'])
{% endcodeblock %}
After running it, it produced the following output:

{% codeblock lang:python %}
Model with rank: 1
Mean validation score: 0.960 (std: 0.000)
Parameters: {'bootstrap': True, 'min_samples_leaf': 45, 'min_samples_split': 404, 'criterion': 'gini', 'max_features': 24, 'max_depth': 8}

Model with rank: 2
Mean validation score: 0.960 (std: 0.000)
Parameters: {'bootstrap': False, 'min_samples_leaf': 45, 'min_samples_split': 554, 'criterion': 'gini', 'max_features': 24, 'max_depth': 6}

Model with rank: 3
Mean validation score: 0.960 (std: 0.000)
Parameters: {'bootstrap': True, 'min_samples_leaf': 52, 'min_samples_split': 597, 'criterion': 'entropy', 'max_features': 28, 'max_depth': 4}
report(random_search.grid_scores_)
{% endcodeblock %}
Then, I created a model with the best parameters and trained it to check out the cross validation score.

{% codeblock lang:python %}
rfc1 = RandomForestClassifier(n_estimators = 500, \
                              max_depth = 8, \
                              max_features = 24, \
                              min_samples_split = 404, \
                              min_samples_leaf = 45, \
                              bootstrap = True, \
                              random_state = 42)
modelFit(rfc1, train, pred, 'TARGET')
{% endcodeblock %}
This lead to a CV score of 0.8143.  That's an increase of almost 0.05.  That's a substantial improvement over the default parameters.  This is as far as I took this model, but in the future, I plan on following up with a few grid searches to further refine the parameters.  This is because of the way the RandomizedSearchCV searches for the best parameters.  It only selects a sample from the given range, so while what it found is an improvement, it might not have provided the optimal parameters.

That's it for this edition.  The next post will detail how I used the XGBoost model to fit the same data.  That's all I already have prepared, so following that, it's hard to say what will be coming.  I might go back and do the things I've already suggested with the models I already have, or I'll start to ensemble.  I also need to put up something on the initial data exploration and do some visualizations.  So, stay tuned.  You can find the [notebook](https://github.com/SayWhat1/Santander-Customer-Satisfaction-Kaggle/blob/master/Santander%20Customer%20Satisfaction%20RandomForest.ipynb) on my GitHub, but as I work on this model and the others, it's subject to change.
