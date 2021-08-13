from xgboost import XGBClassifier, XGBRegressor
from sklearn.model_selection import train_test_split
from sklearn.datasets import load_iris
import numpy as np
from sklearn.metrics import r2_score, accuracy_score
from sklearn.preprocessing import MinMaxScaler, StandardScaler, RobustScaler

# 1-1. data
datasets = load_iris()

# 1. data
x_data = datasets["data"] # (506, 13) 
y_data = datasets["target"] # (506,)

x_train, x_test, y_train, y_test = train_test_split(x_data, y_data,
      test_size=0.2, shuffle=True, random_state=66)

# print(x_train.shape, x_test.shape) # (404, 13) (102, 13)

scaler = RobustScaler()
x_train = scaler.fit_transform(x_train)
x_test = scaler.transform(x_test)

model = XGBClassifier(n_estimators=100, learning_rate=0.1,
                     n_jobs=-1)

model.fit(x_train, y_train, verbose=1, 
            eval_metric=['auc', 'merror'],
            eval_set=[(x_train, y_train), (x_test, y_test)]
)

result = model.score(x_test, y_test)
print('result :', result)

y_pred = model.predict(x_test)
acc = accuracy_score(y_test, y_pred)
print('acc score :', acc)

print("=========================================")
import matplotlib.pyplot as plt

hist = model.evals_result()
print(hist)

epochs = len(hist['validation_0']['auc'])
x_axis = range(0,epochs)

fig, ax = plt.subplots()
ax.plot(x_axis, hist['validation_0']['auc'], label='Train')
ax.plot(x_axis, hist['validation_1']['auc'], label='Test')
ax.legend()
plt.ylabel('auc')
plt.title('XGBoost auc')

fig, ax = plt.subplots()
ax.plot(x_axis, hist['validation_0']['merror'], label='Train')
ax.plot(x_axis, hist['validation_1']['merror'], label='Test')
ax.legend()
plt.ylabel('merror')
plt.title('XGBoost merror')
plt.show()

'''
default : logloss
result : 0.9
acc score : 0.9

DNN
QuantileTransformer
loss[binary] :  0.00010815416317200288
loss[accuracy] :  1.0
'''