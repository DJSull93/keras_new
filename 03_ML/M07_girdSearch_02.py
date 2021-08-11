# Practice
# make model ith 07_01's best parameter 

# GridSearchCV - iris example
# Dictionary (key, value) -> all parameter setting
# {model: node_5, 10, 15}...

import numpy as np
import warnings 
warnings.filterwarnings(action='ignore')

# 1. data
from sklearn.model_selection import GridSearchCV, KFold, cross_val_score, train_test_split

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

# parameters = [
#     {"C":[1,10,100,1000], "kernel":["linear"]},
#     # [] in 4 * kfold 5 = 20
#     {"C":[1,10,100], "kernel":["rbf"], "gamma":[0.001, 0.0001]},
#     # [] in 3 * kernel 1 * gamma 2 * kfold 5= 30
#     {"C":[1,10,100,1000], "kernel":["sigmoid"], "gamma":[0.001, 0.0001]} 
#     # [] in 4 * kernel 1 * gamma 2 * kfold 5= 40
#     # 20 + 30 + 40 = 90times run model
#     ]

# model = GridSearchCV(SVC(), parameters, cv=kfold)

model = SVC(C=10, kernel='linear')

# 3. 컴파일 훈련
model.fit(x_train, y_train)

# 4. 평가 예측
from sklearn.metrics import accuracy_score

y_predict = model.predict(x_test)
acc = accuracy_score(y_test, y_predict)
print('acc_score   :', acc)
print('model score :', model.score(x_test, y_test))

# 01. - best score
# acc_score   : 0.9736842105263158 
# model score : 0.9736842105263158 

# 02. - set param in best score
# acc_score   : 0.9736842105263158
# model score : 0.9736842105263158

# 01. 02. same 