'''
PCA -> use when downgrading dimension -> delete usless columns and merge
'''

from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor, GradientBoostingClassifier
from xgboost import XGBRegressor, XGBClassifier
from sklearn.model_selection import train_test_split
import numpy as np
from sklearn.datasets import load_boston

from sklearn.decomposition import PCA

# 1-1. data
datasets = load_boston()

# 1. data
x = datasets.data # (442, 10)
y = datasets.target # (442,) 

pca = PCA(n_components=7)
x = pca.fit_transform(x)

# print(x)
# print(x.shape) # (442, 7)

x_train, x_test, y_train, y_test = train_test_split(x, y,
      test_size=0.3, shuffle=True, random_state=66)

# 2. model
model = RandomForestRegressor()

# 3. fit
model.fit(x_train, y_train)

# 4. pred
acc = model.score(x_test, y_test)
print('acc : ', acc) 

# pca
# acc :  0.8827466393455596
# not pca
# acc :  0.8905866309557223