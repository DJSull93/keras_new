# PipeLine

import numpy as np
from sklearn.model_selection import train_test_split

# 1. data

x_data = np.load('./_save/_NPY/k55_x_data_diabetes.npy')
y_data = np.load('./_save/_NPY/k55_y_data_diabetes.npy')

x_train, x_test, y_train, y_test = train_test_split(x_data, y_data,
      test_size=0.1, shuffle=True, random_state=66)

from sklearn.preprocessing import MinMaxScaler, StandardScaler, RobustScaler, MaxAbsScaler, QuantileTransformer, PowerTransformer

# 2. model 구성
from sklearn.svm import LinearSVC, SVC, SVR
from sklearn.neighbors import KNeighborsRegressor
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor

from sklearn.pipeline import Pipeline, make_pipeline

model = make_pipeline(MinMaxScaler(), RandomForestRegressor(), )
# model = LogisticRegression()

# 3. 컴파일 훈련
import time
st = time.time()
model.fit(x_train, y_train)
et = time.time() - st

# 4. 평가 예측
from sklearn.metrics import r2_score, accuracy_score

y_predict = model.predict(x_test)
r2 = r2_score(y_test, y_predict)

print('r2_score :', r2)
print('model_score :', model.score(x_test, y_test))

'''
r2_score : 0.326515389580727
model_score : 0.326515389580727
'''