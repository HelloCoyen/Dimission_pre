# -*- coding: utf-8 -*-
"""
Created on Mon Dec 30 21:04:55 2019

@author: wuhaoyu
"""
from sklearn.model_selection import GridSearchCV
from xgboost import XGBClassifier
import xgboost as xgb
from xgboost import plot_importance
#import matplotlib.pyplot  as plt
from sklearn.model_selection import train_test_split
#from sklearn.metrics import accuracy_score  # 准确率
import pandas as pd

# 最优迭代次数
def model_cv(model, X, y, cv_folds=5, early_stopping_rounds=50, seed=0):
    xgb_param = model.get_xgb_params()
    xgtrain = xgb.DMatrix(X, label=y)
    cvresult = xgb.cv(xgb_param, xgtrain, num_boost_round=model.get_params()['n_estimators'], nfold=cv_folds,
                    metrics='auc', seed=seed, callbacks=[
            xgb.callback.print_evaluation(show_stdv=False),
            xgb.callback.early_stop(early_stopping_rounds)
       ])
    num_round_best = cvresult.shape[0] - 1
    print('Best round num: ', num_round_best)
    return num_round_best
# 参数选择
def gridsearch_cv(model, test_param, X, y, cv=5):
    gsearch = GridSearchCV(estimator=model, param_grid=test_param, scoring='roc_auc', n_jobs=4, iid=False, cv=cv)
    gsearch.fit(X, y)
    print('CV Results: ', gsearch.cv_results_)
    print('Best Params: ', gsearch.best_params_)
    print('Best Score: ', gsearch.best_score_)
    return gsearch.best_params_

# 记载样本数据集
data = pd.read_csv("pfm_train.csv", index_col = "EmployeeNumber")
data = data.drop(['Over18', 'StandardHours'], axis=1)
# char_var
char_var = ['BusinessTravel', 'Department', 'EducationField', 'Gender', 
            'JobRole', 'MaritalStatus', 'OverTime','PerformanceRating']
# one-hot
data = pd.get_dummies(data, prefix_sep="_", columns=char_var)

X,y = data[data.columns.difference(["Attrition"])],data["Attrition"]
# 数据集分割
X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=0.0,random_state=123)
X_train,y_train = X,y
# 算法参数
params = {
    'booster':'gbtree',
    'objective':'binary:logistic',
    'gamma':0.7,
    'max_depth':2,
    'lambda':0.1,
    'subsample':0.8,
    'colsample_bytree':0.2,
    'min_child_weight':15,
    'slient':1,
    'eta':0.01,
    'seed':1000,
    'scale_pos_weight':1,
    'alpha':0.01
}

plst = params.items()
# 生成数据集格式
dtrain = xgb.DMatrix(X_train,y_train)
num_rounds = 900
# xgboost模型训练
model = xgb.train(plst,dtrain,num_rounds)
# 对测试集进行预测
test = pd.read_csv("pfm_test.csv", index_col = "EmployeeNumber")
test = test.drop(['Over18', 'StandardHours'], axis=1)
# one-hot
test = pd.get_dummies(test, prefix_sep="_", columns=char_var)
test = test[X_train.columns]
dtest = xgb.DMatrix(test)
y_pred = model.predict(dtest)
y_pred[y_pred>=0.5] = 1
y_pred[y_pred<0.5] = 0
(350-sum(y_pred))/350
result = pd.DataFrame(y_pred)
result.columns = ['result']
result['result'] = result['result'].map(int)
result.to_csv('result_python_xgboost.csv', index=0)
## 计算准确率
#accuracy = accuracy_score(y_test,y_pred)
#print('accuarcy:%.2f%%'%(accuracy*100))
## 显示重要特征
#plot_importance(model)
#plt.show()

# 参数优化
num_round = 900
seed = 0
max_depth = 2
min_child_weight = 15
gamma = 0.7
subsample = 0.8
colsample_bytree = 0.2
scale_pos_weight = 1
reg_alpha = 0.01
reg_lambda = 0.1
learning_rate = 0.01
model = XGBClassifier(learning_rate=learning_rate, n_estimators=num_round, max_depth=max_depth,
                      min_child_weight=min_child_weight, gamma=gamma, subsample=subsample, reg_alpha=reg_alpha,
                      reg_lambda=reg_lambda, colsample_bytree=colsample_bytree, objective='binary:logistic',
                      scale_pos_weight=scale_pos_weight, seed=seed)

num_round = model_cv(model, X_train, y_train)
# best rond = 891

# tune max_depth & min_child_weight
param_test1 = {
    'max_depth': range(1, 10, 1),
    'min_child_weight': range(1, 20, 1)
}
gridsearch_cv(model, param_test1, X_train, y_train)
# 2 15
# tune gamma
param_test2 = {
    'gamma': [i / 10.0 for i in range(0, 10)]
}
gridsearch_cv(model, param_test2, X_train, y_train)
# 0.7
# tune subsample & colsample_bytree
param_test3 = {
    'subsample': [i / 10.0 for i in range(1, 10)],
    'colsample_bytree': [i / 10.0 for i in range(1, 10)]
}
gridsearch_cv(model, param_test3, X_train, y_train)
#0.8 0.2
# tune scale_pos_weight
param_test4 = {
    'scale_pos_weight': [i /10.0 for i in range(1, 100, 1)]
}
gridsearch_cv(model, param_test4, X_train, y_train)
#1
# tune reg_alpha
param_test5 = {
    'reg_alpha': [1e-5, 1e-2, 0.1, 1, 100, 1000]
}
gridsearch_cv(model, param_test5, X_train, y_train)
#0.01
# tune reg_lambda
param_test6 = {
    'reg_lambda': [1e-5, 1e-2, 0.1, 1, 100, 1000]
}
gridsearch_cv(model, param_test6,X_train, y_train)
#0.1