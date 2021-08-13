from xgboost import XGBClassifier, XGBRegressor
from sklearn.model_selection import train_test_split
from sklearn.datasets import load_boston
import numpy as np
from sklearn.metrics import r2_score
from sklearn.preprocessing import MinMaxScaler, StandardScaler, RobustScaler

# 1-1. data
datasets = load_boston()

# 1. data
x_data = datasets["data"] # (506, 13) 
y_data = datasets["target"] # (506,)

x_train, x_test, y_train, y_test = train_test_split(x_data, y_data,
      test_size=0.2, shuffle=True, random_state=66)

# print(x_train.shape, x_test.shape) # (404, 13) (102, 13)

scaler = RobustScaler()
x_train = scaler.fit_transform(x_train)
x_test = scaler.transform(x_test)

model = XGBRegressor(n_setimators=100, learning_rate=0.052,
                     n_jobs=1)

model.fit(x_train, y_train, verbose=1, 
            # eval_metric=['rmse', 'mae', 'logloss'],
            eval_set=[(x_train, y_train), (x_test, y_test)]
            # train set 명시해야 validation 지정 가능
)

result = model.score(x_test, y_test)
print('result :', result)

y_pred = model.predict(x_test)
r2 = r2_score(y_test, y_pred)
print('r2 score :', r2)

'''
default : rmse
[99]    
validation_0-rmse:0.90645      
validation_0-mae:0.68880      
validation_0-logloss:-791.72473 
validation_1-rmse:2.27729   
validation_1-mae:1.73750    
validation_1-logloss:-799.52997

result : 0.922318947095854
r2 score : 0.922318947095854

DNN
epo, batch, run, random = 1000, 32, 2, 66
$ MinMaxScaler
loss :  4.442183971405029
R^2 score :  0.9547103416059415
'''