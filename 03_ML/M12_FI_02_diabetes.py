# 실습
# 피쳐임포턴스 진행 후 필요 없는 칼럼 (하위20~25%) 제거 후 결과 관측
# 각 모델 결과 도출
# 기존 모델과 비교

from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor, GradientBoostingClassifier
from xgboost import XGBRegressor, XGBClassifier
from sklearn.model_selection import train_test_split
import numpy as np
from sklearn.datasets import load_diabetes

# 1-1. data
datasets = load_diabetes()

# 1. data
x_data = datasets.data
y_data = datasets.target

print(datasets.feature_names)
# ['age', 'sex', 'bmi', 'bp', 's1', 's2', 's3', 's4', 's5', 's6']

import pandas as pd

datadf = pd.DataFrame(x_data, columns=datasets.feature_names)

# print(datadf)

# x_data = datadf[['s6', 's5', 's3', 'bp', 'bmi', 'age', 'sex']]

x_train, x_test, y_train, y_test = train_test_split(x_data, y_data,
      test_size=0.25, shuffle=True, random_state=1234)

# 2. model
model = DecisionTreeRegressor()

# 3. fit
model.fit(x_train, y_train)

# 4. pred
acc = model.score(x_test, y_test)
print('acc : ', acc) 

print(model.feature_importances_)

import matplotlib.pyplot as plt

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
acc :  0.030067595639747702
data cut
acc :  -0.047730583142881366

RandomForestRegressor
banilar
acc :  0.4384052684736741
data cut
acc :  0.4539785524980814

GradientBoostingRegressor
banilar
acc :  0.4139240852077556
data cut
acc :  0.49482470692250646

XGBRegressor
banilar
acc :  0.3462785763449687
data cut
acc :  0.4111940531311685
'''