# 실습
# 피쳐임포턴스 진행 후 필요 없는 칼럼 (하위20~25%) 제거 후 결과 관측
# 각 모델 결과 도출
# 기존 모델과 비교

from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingRegressor, GradientBoostingClassifier
from xgboost import XGBRegressor, XGBClassifier
from sklearn.model_selection import train_test_split
import numpy as np
from sklearn.datasets import load_breast_cancer

# 1-1. data
datasets = load_breast_cancer()

# 1. data
x_data = datasets.data
y_data = datasets.target

print(datasets.feature_names)
# ['mean radius' 'mean texture' 'mean perimeter' 'mean area'
#  'mean smoothness' 'mean compactness' 'mean concavity'
#  'mean concave points' 'mean symmetry' 'mean fractal dimension'
#  'radius error' 'texture error' 'perimeter error' 'area error'
#  'smoothness error' 'compactness error' 'concavity error'
#  'concave points error' 'symmetry error' 'fractal dimension error'
#  'worst radius' 'worst texture' 'worst perimeter' 'worst area'
#  'worst smoothness' 'worst compactness' 'worst concavity'
#  'worst concave points' 'worst symmetry' 'worst fractal dimension']
import pandas as pd

datadf = pd.DataFrame(x_data, columns=datasets.feature_names)

# print(datadf)

# x_data = datadf[['worst perimeter', 
#                 'mean concave points',
#                 'worst concave points']]

x_train, x_test, y_train, y_test = train_test_split(x_data, y_data,
      test_size=0.25, shuffle=True, random_state=1234)

# 2. model
model = DecisionTreeClassifier()

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
DecisionTreeClassifier
banilar
acc :  0.9300699300699301
data cut
acc :  0.8741258741258742

RandomForestClassifier
banilar
acc :  0.9300699300699301
data cut
acc :  0.9020979020979021

GradientBoostingClassifier
banilar
acc :  0.9370629370629371
data cut
acc :  0.9020979020979021

XGBClassifier
banilar
acc :  0.9300699300699301
data cut
acc :  0.9020979020979021
'''