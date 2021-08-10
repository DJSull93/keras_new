from sklearn.utils import all_estimators
import numpy as np
from sklearn.datasets import load_iris
from sklearn.metrics import r2_score
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, accuracy_score

# 1. data
datasets = load_iris()

x = datasets.data 
y = datasets.target 

x_train, x_test, y_train, y_test = train_test_split(x, y,
      test_size=0.1, shuffle=True, random_state=1)

from sklearn.preprocessing import MinMaxScaler, StandardScaler, RobustScaler, MaxAbsScaler, QuantileTransformer, PowerTransformer
scaler = MaxAbsScaler()
x_train = scaler.fit_transform(x_train) 
x_test = scaler.transform(x_test) 

# 2. model 구성

allAlgorithms = all_estimators(type_filter='classifier') # or regressor
# print(allAlgorithms)

for (name, algorithm) in allAlgorithms:
    model = algorithm()

    model.fit(x_train, y_train)

    y_pred = model.predict(x_test)
    acc = model.accuracy_score(y_test, y_pred)

    print(name, ' accuracy : ', acc)

