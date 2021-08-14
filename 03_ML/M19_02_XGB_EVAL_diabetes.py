from xgboost import XGBClassifier, XGBRegressor
from sklearn.model_selection import train_test_split
from sklearn.datasets import load_diabetes
import numpy as np
from sklearn.metrics import r2_score
from sklearn.preprocessing import MinMaxScaler, StandardScaler, RobustScaler

# 1-1. data
datasets = load_diabetes()

# 1. data
x_data = datasets["data"] # (506, 13) 
y_data = datasets["target"] # (506,)

x_train, x_test, y_train, y_test = train_test_split(x_data, y_data,
      test_size=0.2, shuffle=True, random_state=66)

# print(x_train.shape, x_test.shape) # (404, 13) (102, 13)

scaler = RobustScaler()
x_train = scaler.fit_transform(x_train)
x_test = scaler.transform(x_test)

model = XGBRegressor(n_setimators=100, learning_rate=0.1,
                     n_jobs=-1)

model.fit(x_train, y_train, verbose=1, 
            # eval_metric=['rmse', 'mae', 'logloss'],
            eval_set=[(x_train, y_train), (x_test, y_test)]
)

result = model.score(x_test, y_test)
print('result :', result)

y_pred = model.predict(x_test)
r2 = r2_score(y_test, y_pred)
print('r2 score :', r2)

'''
default : rmse
result : 0.20522865765443532
r2 score : 0.20522865765443532

DNN
epo, batch, run, random = 200, 32, 2, 9
$ MaxAbsScaler
loss :  2247.06396484375
R^2 score :  0.6105307266558146
'''