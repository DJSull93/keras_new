from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
import numpy as np
from sklearn.datasets import load_wine

# 1-1. data
datasets = load_wine()

# 1. data
x_data = np.load('./_save/_NPY/k55_x_data_wine.npy')
y_data = np.load('./_save/_NPY/k55_y_data_wine.npy')

x_train, x_test, y_train, y_test = train_test_split(x_data, y_data,
      test_size=0.25, shuffle=True, random_state=1234)

# 2. model
model = RandomForestClassifier()

# 3. fit
model.fit(x_train, y_train)

# 4. pred
acc = model.score(x_test, y_test)
print('acc : ', acc) 

print(model.feature_importances_)

'''
acc :  0.6710204081632654
[0.07364588 0.10232124 0.08157261 0.08736527 0.0834641  0.09156399
 0.09347665 0.1030764  0.08596974 0.08171103 0.11583309]
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