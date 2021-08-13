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
      test_size=0.2, shuffle=True, random_state=661)

# print(x_train.shape, x_test.shape) # (404, 13) (102, 13)

scaler = MinMaxScaler()
x_train = scaler.fit_transform(x_train)
x_test = scaler.transform(x_test)

model = XGBRegressor(n_estimators=100, learning_rate=0.098,
                     n_jobs=-1)

model.fit(x_train, y_train, verbose=1, 
            # eval_metric=['rmse', 'mae', 'logloss'],
            early_stopping_rounds=20,
            eval_set=[(x_train, y_train), (x_test, y_test)]
)

result = model.score(x_test, y_test)
print('result :', result)

y_pred = model.predict(x_test)
r2 = r2_score(y_test, y_pred)
print('r2 score :', r2)

# print("=========================================")
# import matplotlib.pyplot as plt

# hist = model.evals_result()
# print(hist)

# epochs = len(hist['validation_0']['logloss'])
# x_axis = range(0,epochs)

# fig, ax = plt.subplots()
# ax.plot(x_axis, hist['validation_0']['logloss'], label='Train')
# ax.plot(x_axis, hist['validation_1']['logloss'], label='Test')
# ax.legend()
# plt.ylabel('Log loss')
# plt.title('XGBoost Log Loss')

# fig, ax = plt.subplots()
# ax.plot(x_axis, hist['validation_0']['rmse'], label='Train')
# ax.plot(x_axis, hist['validation_1']['rmse'], label='Test')
# ax.legend()
# plt.ylabel('Rmse')
# plt.title('XGBoost RMSE')
# plt.show()

'''
default : rmse
result : 0.4433414286258561
r2 score : 0.4433414286258561

DNN
epo, batch, run, random = 200, 32, 2, 9
$ MaxAbsScaler
loss :  2247.06396484375
R^2 score :  0.6105307266558146
'''