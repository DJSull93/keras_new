'''
PCA -> use when downgrading dimension -> delete usless columns and merge
: check with EVR, usually use 0.9~0.95
'''

from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor, GradientBoostingClassifier
from xgboost import XGBRegressor, XGBClassifier
from sklearn.model_selection import train_test_split
import numpy as np
from sklearn.datasets import load_diabetes

from sklearn.decomposition import PCA

# 1-1. data
datasets = load_diabetes()

# 1. data
x = datasets.data # (442, 10)
y = datasets.target # (442,) 

pca = PCA(n_components=10)
x = pca.fit_transform(x)

# print(x)
# print(x.shape) # (442, 7)

pca_EVR = pca.explained_variance_ratio_
print(pca_EVR)
# [0.40242142 0.14923182 0.12059623 0.09554764 0.06621856 0.06027192
#  0.05365605]
# -> feature importance in pca

print(sum(pca_EVR)) 
# n_components : 10 -> 1.0
# n_components : 7 -> 0.9479436357350414
# n_components : 9 -> 0.9991439470098977
# n_components : 4 -> 0.7677971110209652
# -> n_com downgade -> min colum value deleted

cumsum = np.cumsum(pca_EVR)
print(cumsum)
# [0.40242142 0.55165324 0.67224947 0.76779711 0.83401567 0.89428759
#  0.94794364 0.99131196 0.99914395 1.        ]
# -> 초기값부터 누적 합으로 표기

print(np.argmax(cumsum >= 0.94)+1) # 7

import matplotlib.pyplot as plt

plt.plot(cumsum)
plt.grid()
plt.show()

'''
x_train, x_test, y_train, y_test = train_test_split(x, y,
      test_size=0.3, shuffle=True, random_state=66)

# 2. model
model = RandomForestRegressor()

# 3. fit
model.fit(x_train, y_train)

# 4. pred
acc = model.score(x_test, y_test)
print('acc : ', acc) 

# pca n = 7
# acc :  0.4203796646254544
# not pca
# acc :  0.41152654438447955
'''