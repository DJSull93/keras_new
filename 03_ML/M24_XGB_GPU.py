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

model = XGBRegressor(n_estimators=10000, learning_rate=0.001,
                    # n_jobs=2
                    tree_method='gpu_hist',
                    gpu_id=0, # choose gpu      
                    predictor='gpu_predictor'    # 'cpu_predictor'
)

import time
st = time.time()
model.fit(x_train, y_train, verbose=1, 
            # early_stopping_rounds=10,
            eval_metric=['rmse', 'mae', 'logloss'],
            eval_set=[(x_train, y_train), (x_test, y_test)]
)
et = time.time() - st

print('time :', et)
# result = model.score(x_test, y_test)
# print('result :', result)

# y_pred = model.predict(x_test)
# r2 = r2_score(y_test, y_pred)
# print('r2 score :', r2)

'''
epo = 10000

n_jobs=1 
time : 16.14333415031433

n_jobs=2
time : 11.757595777511597

n_jobs=4
time : 10.193732500076294

n_jobs=8
time : 10.75623083114624

tree_method='gpu_hist'                     
time : 53.32287001609802
'''