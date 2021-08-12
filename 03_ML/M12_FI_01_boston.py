# 실습
# 피쳐임포턴스 진행 후 필요 없는 칼럼 (하위20~25%) 제거 후 결과 관측
# 각 모델 결과 도출
# 기존 모델과 비교

from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor, GradientBoostingClassifier
from xgboost import XGBRegressor, XGBClassifier
from sklearn.model_selection import train_test_split
import numpy as np
from sklearn.datasets import load_boston

# 1-1. data
datasets = load_boston()

# 1. data
x_data = datasets.data
y_data = datasets.target

print(datasets.feature_names)
# ['CRIM' 'ZN' 'INDUS' 'CHAS' 'NOX' 'RM' 'AGE' 'DIS' 'RAD' 'TAX' 'PTRATIO'
#  'B' 'LSTAT']

import pandas as pd

datadf = pd.DataFrame(x_data, columns=datasets.feature_names)

# print(datadf)

# x_data = datadf[['LSTAT', 'DIS', 'RM']]

x_train, x_test, y_train, y_test = train_test_split(x_data, y_data,
      test_size=0.25, shuffle=True, random_state=1234)

# 2. model
model = XGBRegressor()

# 3. fit
model.fit(x_train, y_train)

# 4. pred
acc = model.score(x_test, y_test)
print('acc : ', acc) 

print(model.feature_importances_)

# import matplotlib.pyplot as plt

# def plot_feature_importance_dataset(model):
#       n_features = datasets.data.shape[1]
#       plt.barh(np.arange(n_features), model.feature_importances_,
#             align='center')
#       plt.yticks(np.arange(n_features), datasets.feature_names)
#       plt.xlabel("Feature Importances")
#       plt.ylabel("Features")
#       plt.ylim(-1, n_features)

# plot_feature_importance_dataset(model)
# plt.show()

'''
DecisionTreeRegressor
banilar
acc :  0.8640020461997446
data cut
acc :  0.795514909422874

RandomForestRegressor
banilar
acc :  0.893412012870183
data cut
acc :  0.8333690192891234

GradientBoostingRegressor
banilar
acc :  0.9134457277386153
data cut
acc :  0.8317864931862278

XGBRegressor
banilar
acc :  0.8941188975894797
data cut
acc :  0.8204399093158405
'''