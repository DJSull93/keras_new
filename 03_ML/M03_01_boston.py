# Classifier -> Regression 
# 1. check Classifier 2. check Regression
'''
# 1. -> ValueError: Unknown label type: 'continuous'
'''

import numpy as np

x_data = np.load('./_save/_NPY/k55_x_data_boston.npy')
y_data = np.load('./_save/_NPY/k55_y_data_boston.npy')

from sklearn.metrics import r2_score
from sklearn.model_selection import train_test_split

# 1. data 구성

x_train, x_test, y_train, y_test = train_test_split(x_data, y_data,
      test_size=0.2, shuffle=True, random_state=66)

# print(x_train.shape, x_test.shape) # (404, 13) (102, 13)

from sklearn.preprocessing import MinMaxScaler, StandardScaler, RobustScaler, MaxAbsScaler, QuantileTransformer, PowerTransformer
scaler = RobustScaler()
x_train = scaler.fit_transform(x_train)
x_test = scaler.transform(x_test)

# 2. model 구성
from sklearn.svm import LinearSVC, SVC
from sklearn.neighbors import KNeighborsRegressor
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor

model = RandomForestRegressor()

# 3. 컴파일 훈련

import time 
start_time = time.time()
hist = model.fit(x_train, y_train)
end_time = time.time() - start_time

# 4. 평가 예측
y_predict2 = model.predict(x_test)

result = model.score(x_test, y_test)
print('model_score :', result)

from sklearn.metrics import r2_score, accuracy_score

y_predict = model.predict(x_test)
# acc = accuracy_score(y_test, y_predict)

# print('acc_score :', acc)

r2 = r2_score(y_test, y_predict)
print('R^2 score : ', r2)


'''
KNeighborsRegressor
model_score : 0.825402539162744
R^2 score :  0.825402539162744

LinearRegression
model_score : 0.8111288663608665
R^2 score :  0.8111288663608665

DecisionTreeRegressor
model_score : 0.7870343555050473
R^2 score :  0.7870343555050473

RandomForestRegressor
model_score : 0.9251964923494207
R^2 score :  0.9251964923494207
'''