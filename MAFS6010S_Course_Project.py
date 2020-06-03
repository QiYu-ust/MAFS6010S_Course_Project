#!/usr/bin/env python
# coding: utf-8

# In[1]:


# Load Python libraries cross_validation
from sklearn import metrics, model_selection, ensemble
import xgboost as xgb
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib as mpl
import warnings
from xgboost.sklearn import XGBClassifier
from sklearn.model_selection import GridSearchCV  #Perforing grid search
warnings.filterwarnings('ignore')
plt.style.use('ggplot')

import matplotlib.pylab as plt
get_ipython().run_line_magic('matplotlib', 'inline')
from matplotlib.pylab import rcParams
rcParams['figure.figsize'] = 12, 4


# In[2]:


# Load data
df = pd.read_csv('train.csv')

# 1% sample of items
df = df.sample(frac=0.01)


# In[3]:


# Load and join songs data
songs = pd.read_csv('songs.csv')
df = pd.merge(df, songs, on='song_id', how='left')
del songs

# Load and join songs data
members = pd.read_csv('members.csv')
df = pd.merge(df, members, on='msno', how='left')
del members

df.info()


# In[4]:


# Count Na in %
df.isnull().sum()/df.isnull().count()*100

for i in df.select_dtypes(include=['object']).columns:
    df[i][df[i].isnull()] = 'unknown'
df = df.fillna(value=0)


# In[5]:


# Create Dates

# registration_init_time
df.registration_init_time = pd.to_datetime(df.registration_init_time, format='%Y%m%d', errors='ignore')
df['registration_init_time_year'] = df['registration_init_time'].dt.year
df['registration_init_time_month'] = df['registration_init_time'].dt.month
df['registration_init_time_day'] = df['registration_init_time'].dt.day

# expiration_date
df.expiration_date = pd.to_datetime(df.expiration_date,  format='%Y%m%d', errors='ignore')
df['expiration_date_year'] = df['expiration_date'].dt.year
df['expiration_date_month'] = df['expiration_date'].dt.month
df['expiration_date_day'] = df['expiration_date'].dt.day


# In[6]:


# Dates to categoty
df['registration_init_time'] = df['registration_init_time'].astype('category')
df['expiration_date'] = df['expiration_date'].astype('category')


# In[7]:


# Object data to category
for col in df.select_dtypes(include=['object']).columns:
    df[col] = df[col].astype('category')
    
# Encoding categorical features
for col in df.select_dtypes(include=['category']).columns:
    df[col] = df[col].cat.codes


# In[8]:


# Сorrelation matrix
plt.figure(figsize=[7,5])
sns.heatmap(df.corr())
plt.show()


# # Random Forest

# In[9]:


# Drop columns
df = df.drop(['expiration_date', 'lyricist'], 1)


# In[10]:


# Model with the best estimator
model = ensemble.RandomForestClassifier(n_estimators=250, max_depth=25)
model.fit(df[df.columns[df.columns != 'target']], df.target)


# In[11]:


df_plot = pd.DataFrame({'features': df.columns[df.columns != 'target'],
                        'importances': model.feature_importances_})
df_plot = df_plot.sort_values('importances', ascending=False)


# In[12]:


plt.figure(figsize=[11,5])
sns.barplot(x = df_plot.importances, y = df_plot.features)
plt.title('Importances of Features Plot')
plt.show()


# In[13]:


# Drop columns with importances < 0.02
df = df.drop(df_plot.features[df_plot.importances < 0.02].tolist(), 1)


# In[14]:


# Selected columns
df.columns


# # XGBoost

# In[15]:


# Train & Test split
target = df.pop('target')
train_data, test_data, train_labels, test_labels = model_selection.train_test_split(df, target, test_size = 0.3)

# Delete df
del df


# In[16]:


def modelfit(alg, dtrain, dtest, predictors, useTrainCV=True, cv_folds=5, early_stopping_rounds=50):
    if useTrainCV:
        xgb_param = alg.get_xgb_params()
        xgtrain = xgb.DMatrix(dtrain[predictors].values, label=dtrain[target].values)
        cvresult = xgb.cv(xgb_param, xgtrain, num_boost_round=alg.get_params()['n_estimators'], nfold=cv_folds,
                          metrics='auc', early_stopping_rounds=early_stopping_rounds, verbose_eval=False)
        alg.set_params(n_estimators=cvresult.shape[0])

#Fit the algorithm on the data
    alg.fit(dtrain[predictors].values, dtrain['target'],eval_metric='auc')

#Predict training set:
    dtrain_predictions = alg.predict(dtrain[predictors].values)
    dtrain_predprob = alg.predict_proba(dtrain[predictors].values)[:,1]

#Predict test set
    dtest_predictions = alg.predict(dtest[predictors].values)
    dtest_predprob = alg.predict_proba(dtest[predictors].values)[:,1]

#Print model report:
    print ("\nModel Report")
    print ("Accuracy(Train) : %.4g" % metrics.accuracy_score(dtrain['target'].values, dtrain_predictions))
    print ("Accuracy(Test) : %.4g" % metrics.accuracy_score(dtest['target'].values, dtest_predictions))
    print ("AUC Score(Train) : %f" % metrics.roc_auc_score(dtrain['target'], dtrain_predprob))
    print ("AUC Score(Test) : %f" % metrics.roc_auc_score(dtest['target'], dtest_predprob))

    feat_imp = pd.Series(alg.get_booster().get_fscore()).sort_values(ascending=False)
    feat_imp.plot(kind='bar', title='Feature Importances')
    plt.ylabel('Feature Importance Score')
    
    print(metrics.classification_report(dtest['target'], dtest_predictions))


# In[17]:


#Choose all predictors except target
train = pd.concat([train_data, train_labels], axis=1)
test = pd.concat([test_data, test_labels], axis=1)
target = 'target'


# In[18]:


predictors = [x for x in train.columns if x not in [target]]
#predictors = train.drop(['target'], axis = 1)
xgb1 = XGBClassifier(
    base_score=0.5, 
    booster='gbtree', 
    colsample_bylevel=1,
    colsample_bynode=1, 
    colsample_bytree=1, 
    gamma=0, 
    gpu_id=-1,
    importance_type='gain', 
    interaction_constraints='',
    learning_rate=0.1, 
    max_delta_step=0, 
    max_depth=10,
    min_child_weight=5,  
    monotone_constraints='()',
    n_estimators=250, 
    n_jobs=0, 
    num_parallel_tree=1,
    objective='binary:logistic', 
    random_state=0, 
    reg_alpha=0,
    reg_lambda=1, 
    scale_pos_weight=1, 
    subsample=1,
    tree_method='exact', 
    validate_parameters=1, 
    verbosity=None)

modelfit(xgb1, train, test, predictors)


# In[19]:


param_test1 = {
 'max_depth':range(3,10,2),
 'min_child_weight':range(1,6,2)
}

gsearch1 = GridSearchCV(estimator = XGBClassifier(base_score=0.5, booster='gbtree', colsample_bylevel=1,colsample_bynode=1, 
                                                  colsample_bytree=1, gamma=0, gpu_id=-1,importance_type='gain', 
                                                  interaction_constraints='',learning_rate=0.1, max_delta_step=0, 
                                                  max_depth=10, min_child_weight=5,  monotone_constraints='()', n_estimators=250, 
                                                  n_jobs=0, num_parallel_tree=1, objective='binary:logistic', random_state=0, 
                                                  reg_alpha=0, reg_lambda=1, scale_pos_weight=1, subsample=1, tree_method='exact', 
                                                  validate_parameters=1, verbosity=None), 
                        param_grid = param_test1, scoring='roc_auc',n_jobs=4,iid=False, cv=5)

gsearch1.fit(train[predictors],train[target])
gsearch1.cv_results_, gsearch1.best_params_, gsearch1.best_score_


# In[20]:


param_test3 = {
 'gamma':[i/10.0 for i in range(0,5)]
}
gsearch3 = GridSearchCV(estimator =XGBClassifier(base_score=0.5, booster='gbtree', colsample_bylevel=1,colsample_bynode=1, 
                                                  colsample_bytree=1, gamma=0, gpu_id=-1,importance_type='gain', 
                                                  interaction_constraints='',learning_rate=0.1, max_delta_step=0, 
                                                  max_depth=5, min_child_weight=3,  monotone_constraints='()', n_estimators=250, 
                                                  n_jobs=0, num_parallel_tree=1, objective='binary:logistic', random_state=0, 
                                                  reg_alpha=0, reg_lambda=1, scale_pos_weight=1, subsample=1, tree_method='exact', 
                                                  validate_parameters=1, verbosity=None),
                        param_grid = param_test3, scoring='roc_auc',n_jobs=4,iid=False, cv=5)

gsearch3.fit(train[predictors],train[target])
gsearch3.cv_results_, gsearch3.best_params_, gsearch3.best_score_


# In[22]:


param_test4 = {
 'subsample':[i/10.0 for i in range(6,10)],
 'colsample_bytree':[i/10.0 for i in range(6,10)]
}

gsearch4 = GridSearchCV(estimator =XGBClassifier(base_score=0.5, booster='gbtree', colsample_bylevel=1,colsample_bynode=1, 
                                                  colsample_bytree=1, gamma=0, gpu_id=-1,importance_type='gain', 
                                                  interaction_constraints='',learning_rate=0.1, max_delta_step=0, 
                                                  max_depth=5, min_child_weight=3,  monotone_constraints='()', n_estimators=250, 
                                                  n_jobs=0, num_parallel_tree=1, objective='binary:logistic', random_state=0, 
                                                  reg_alpha=0, reg_lambda=1, scale_pos_weight=1, subsample=1, tree_method='exact', 
                                                  validate_parameters=1, verbosity=None), 
                        param_grid = param_test4, scoring='roc_auc',n_jobs=4,iid=False, cv=5)

gsearch4.fit(train[predictors],train[target])
gsearch4.cv_results_, gsearch4.best_params_, gsearch4.best_score_


# In[23]:


param_test6 = {
 'reg_alpha':[1e-5, 1e-2, 0.1, 1, 100]
}
gsearch6 = GridSearchCV(estimator =XGBClassifier(base_score=0.5, booster='gbtree', colsample_bylevel=1,colsample_bynode=1, 
                                                  colsample_bytree=0.7, gamma=0, gpu_id=-1,importance_type='gain', 
                                                  interaction_constraints='',learning_rate=0.1, max_delta_step=0, 
                                                  max_depth=5, min_child_weight=3,  monotone_constraints='()', n_estimators=250, 
                                                  n_jobs=0, num_parallel_tree=1, objective='binary:logistic', random_state=0, 
                                                  reg_alpha=0, reg_lambda=1, scale_pos_weight=1, subsample=0.9, tree_method='exact', 
                                                  validate_parameters=1, verbosity=None), 
                        param_grid = param_test6, scoring='roc_auc',n_jobs=4,iid=False, cv=5)

gsearch6.fit(train[predictors],train[target])
gsearch6.cv_results_, gsearch6.best_params_, gsearch6.best_score_


# In[24]:


xgb4 = XGBClassifier(base_score=0.5, booster='gbtree', colsample_bylevel=1,colsample_bynode=1, 
                    colsample_bytree=0.7, gamma=0, gpu_id=-1,importance_type='gain', 
                    interaction_constraints='',learning_rate=0.1, max_delta_step=0, 
                    max_depth=5, min_child_weight=3,  monotone_constraints='()', n_estimators=250, 
                    n_jobs=0, num_parallel_tree=1, objective='binary:logistic', random_state=0, 
                    reg_alpha=1, reg_lambda=1, scale_pos_weight=1, subsample=0.9, tree_method='exact', 
                    validate_parameters=1, verbosity=None)
modelfit(xgb4, train, test, predictors)


# In[ ]:




