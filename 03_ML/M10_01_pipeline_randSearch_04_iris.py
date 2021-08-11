# Pipeline 

import numpy as np
import warnings 
warnings.filterwarnings(action='ignore')
from sklearn.pipeline import Pipeline, make_pipeline

# 1. data
x_data = np.load('./_save/_NPY/k55_x_data_iris.npy')
y_data = np.load('./_save/_NPY/k55_y_data_iris.npy')

# 2. model 구성

from sklearn.svm import LinearSVC, SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier

from sklearn.model_selection import RandomizedSearchCV, GridSearchCV, KFold, cross_val_score, train_test_split
from sklearn.preprocessing import MinMaxScaler, StandardScaler, RobustScaler, MaxAbsScaler, QuantileTransformer, PowerTransformer

parameters = [{
    "randomforestclassifier__max_depth": [6, 8, 10, 12],
    "randomforestclassifier__min_samples_leaf": [3, 5, 7],
    "randomforestclassifier__min_samples_split": [2, 3, 5, 10],
}] # model and param connection : model__param

n_split = 5
kfold = KFold(n_splits=n_split, shuffle=True, 
                random_state=66)

pipe = make_pipeline(MinMaxScaler(), RandomForestClassifier())

# model = RandomizedSearchCV(pipe, parameters, cv=kfold, verbose=1)
model = GridSearchCV(pipe, parameters, cv=kfold, verbose=1)

# 3. 컴파일 훈련
import time
st = time.time()
model.fit(x_data, y_data)
et = time.time() - st

# 4. 평가 예측

print('totla time : ', et)
print('Best estimator : ', model.best_estimator_)
print('Best score  :', model.best_score_)

'''
RandomizedSearchCV
totla time :  4.536161184310913
Best estimator :  Pipeline(steps=[('minmaxscaler', MinMaxScaler()),
                ('randomforestclassifier',
                 RandomForestClassifier(max_depth=8, min_samples_leaf=3))])
Best score  : 0.96

GridSearchCV
totla time :  20.9615478515625
Best estimator :  Pipeline(steps=[('minmaxscaler', MinMaxScaler()),
                ('randomforestclassifier',
                 RandomForestClassifier(max_depth=10, min_samples_leaf=3,
                                        min_samples_split=5))])
Best score  : 0.9666666666666666
'''

