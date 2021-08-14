from xgboost import XGBClassifier, XGBRegressor
from sklearn.model_selection import train_test_split
from sklearn.datasets import load_wine
import numpy as np
from sklearn.metrics import r2_score, accuracy_score
from sklearn.preprocessing import MinMaxScaler, StandardScaler, RobustScaler

# 1-1. data
datasets = load_wine()

# 1. data
x_data = datasets["data"] # (506, 13) 
y_data = datasets["target"] # (506,)

x_train, x_test, y_train, y_test = train_test_split(x_data, y_data,
      test_size=0.2, shuffle=True, random_state=66)

# print(x_train.shape, x_test.shape) # (404, 13) (102, 13)

scaler = RobustScaler()
x_train = scaler.fit_transform(x_train)
x_test = scaler.transform(x_test)

model = XGBClassifier(n_setimators=100, learning_rate=0.1,
                     n_jobs=-1)

model.fit(x_train, y_train, verbose=1, 
            # eval_metric=['auc', 'merror'],
            eval_set=[(x_train, y_train), (x_test, y_test)]
)

result = model.score(x_test, y_test)
print('result :', result)

y_pred = model.predict(x_test)
acc = accuracy_score(y_test, y_pred)
print('acc score :', acc)

'''
default : logloss
result : 1.0
acc score : 1.0

QuantileTransformer
loss[category] :  0.0029777565505355597
loss[accuracy] :  1.0
'''