# PipeLine

import numpy as np
from sklearn.model_selection import train_test_split

# 1. data
x_data = np.load('./_save/_NPY/k55_x_data_wine.npy')
y_data = np.load('./_save/_NPY/k55_y_data_wine.npy')

x_train, x_test, y_train, y_test = train_test_split(x_data, y_data,
      test_size=0.1, shuffle=True, random_state=66)

from sklearn.preprocessing import MinMaxScaler, StandardScaler, RobustScaler, MaxAbsScaler, QuantileTransformer, PowerTransformer

# 2. model 구성
from sklearn.svm import LinearSVC, SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier

from sklearn.pipeline import Pipeline, make_pipeline

model = make_pipeline(MinMaxScaler(), RandomForestClassifier(), )
# model = LogisticRegression()

# 3. 컴파일 훈련
import time
st = time.time()
model.fit(x_train, y_train)
et = time.time() - st

# 4. 평가 예측
from sklearn.metrics import r2_score, accuracy_score

y_predict = model.predict(x_test)
acc = accuracy_score(y_test, y_predict)

print('acc_score :', acc)
print('model_score :', model.score(x_test, y_test))

'''
acc_score : 0.7
model_score : 0.7
'''