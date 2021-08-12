from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingRegressor, GradientBoostingClassifier
from xgboost import XGBRegressor, XGBClassifier
from sklearn.model_selection import train_test_split
import numpy as np
from sklearn.datasets import load_breast_cancer

# 1-1. data
datasets = load_breast_cancer()

# 1. data
x_data = np.load('./_save/_NPY/k55_x_data_cancer.npy')
y_data = np.load('./_save/_NPY/k55_y_data_cancer.npy')

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

'''
acc :  0.9230769230769231
[0.04607065 0.01157968 0.04411715 0.03047664 0.00580009 0.00472953
 0.06753042 0.16007771 0.00293302 0.00303689 0.00847924 0.00459108
 0.01441352 0.02071447 0.00136204 0.00575602 0.0032967  0.00180149
 0.00440394 0.00420838 0.10309655 0.01492929 0.10693843 0.11320165
 0.01210023 0.02702782 0.04052561 0.12297291 0.00612343 0.00770545]
'''

import matplotlib.pyplot as plt

def plot_feature_importance_dataset(model):
      n_features = datasets.data.shape[1]
      plt.barh(np.arange(n_features), model.feature_importances_,
            align='center')
      plt.yticks(np.arange(n_features), datasets.feature_names)
      plt.xlabel("Feature Importances")
      plt.ylabel("Features")
      plt.ylim(-1, n_features)

plot_feature_importance_dataset(model)
plt.show()