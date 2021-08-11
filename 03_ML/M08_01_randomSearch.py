# RandomizedSearchCV

import numpy as np
import warnings 
warnings.filterwarnings(action='ignore')

# 1. data
from sklearn.model_selection import RandomizedSearchCV, GridSearchCV, KFold, cross_val_score, train_test_split

x_data = np.load('./_save/_NPY/k55_x_data_iris.npy')
y_data = np.load('./_save/_NPY/k55_y_data_iris.npy')

x_train, x_test, y_train, y_test = train_test_split(x_data, y_data,
      test_size=0.25, shuffle=True, random_state=1)

n_split = 5
kfold = KFold(n_splits=n_split, shuffle=True, 
                random_state=66)

# 2. model 구성

from sklearn.svm import LinearSVC, SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier

parameters = [
    {"C":[1,10,100,1000], "kernel":["linear"]},
    {"C":[1,10,100], "kernel":["rbf"], "gamma":[0.001, 0.0001]},
    {"C":[1,10,100,1000], "kernel":["sigmoid"], "gamma":[0.001, 0.0001]} 
    ]

# model = GridSearchCV(SVC(), parameters, cv=kfold, verbose=1 )
# Fitting 5 folds for each of 18 candidates, totalling 90 fits
model = RandomizedSearchCV(SVC(), parameters, cv=kfold, verbose=1 )
# Fitting 5 folds for each of 10 candidates, totalling 50 fits

# 3. 컴파일 훈련
model.fit(x_train, y_train)

# 4. 평가 예측
from sklearn.metrics import accuracy_score

print('Best estimator : ', model.best_estimator_)
print('Best score  :', model.best_score_)


y_predict = model.predict(x_test)
acc = accuracy_score(y_test, y_predict)
print('acc_score   :', acc)
print('model score :', model.score(x_test, y_test))


