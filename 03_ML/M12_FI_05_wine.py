# 실습
# 피쳐임포턴스 진행 후 필요 없는 칼럼 (하위20~25%) 제거 후 결과 관측
# 각 모델 결과 도출
# 기존 모델과 비교

from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingRegressor, GradientBoostingClassifier
from xgboost import XGBRegressor, XGBClassifier
from sklearn.model_selection import train_test_split
import numpy as np
from sklearn.datasets import load_wine

# 1-1. data
datasets = load_wine()

# 1. data
x_data = datasets.data
y_data = datasets.target

print(datasets.feature_names)
# ['alcohol', 'malic_acid', 'ash', 'alcalinity_of_ash', 'magnesium', 'total_phenols', 'flavanoids', 'nonflavanoid_phenols', 'proanthocyanins', 'color_intensity', 'hue', 'od280/od315_of_diluted_wines', 'proline']

import pandas as pd

datadf = pd.DataFrame(x_data, columns=datasets.feature_names)

# print(datadf)

# x_data = datadf[['proline', 'od280/od315_of_diluted_wines',
#                 'color_intensity',
#                 'flavanoids']]

x_train, x_test, y_train, y_test = train_test_split(x_data, y_data,
      test_size=0.25, shuffle=True, random_state=1234)

# 2. model
model = XGBClassifier()

# 3. fit
model.fit(x_train, y_train)

# 4. pred
acc = model.score(x_test, y_test)
print('acc : ', acc) 

print(model.feature_importances_)

import matplotlib.pyplot as plt

# def plot_feature_importance_dataset(model):
#     #   n_features = datasets.data.shape[1]
#       n_features = datadf.shape[1]
#       plt.barh(np.arange(n_features), model.feature_importances_,
#             align='center')
#     #   plt.yticks(np.arange(n_features), datasets.feature_names)
#       plt.yticks(np.arange(n_features), datadf.feature_names)
#       plt.xlabel("Feature Importances")
#       plt.ylabel("Features")
#       plt.ylim(-1, n_features)

# plot_feature_importance_dataset(model)
# plt.show()

'''
DecisionTreeClassifier
banilar
acc :  0.8666666666666667
data cut
acc :  0.8666666666666667

RandomForestClassifier
banilar
acc :  0.9555555555555556
data cut
acc :  0.9333333333333333

GradientBoostingClassifier
banilar
acc :  0.8666666666666667
data cut
acc :  0.9111111111111111

XGBClassifier
banilar
acc :  0.9555555555555556
data cut
acc :  0.9333333333333333
'''