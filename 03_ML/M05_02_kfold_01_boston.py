import numpy as np
from sklearn.model_selection import train_test_split, KFold, cross_val_score
import warnings 
warnings.filterwarnings(action='ignore')

# 1. data
x_data = np.load('./_save/_NPY/k55_x_data_boston.npy')
y_data = np.load('./_save/_NPY/k55_y_data_boston.npy')

n_split = 5
kfold = KFold(n_splits=n_split, shuffle=True, 
                random_state=12)

# 2. model 구성
from sklearn.svm import LinearSVC, SVC
from sklearn.neighbors import KNeighborsRegressor
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor

# model = LinearSVC()

# model = SVC()

# model = KNeighborsRegressor()
# acc score :  [0.54471958 0.38861387 0.54167384 0.50340188 0.63774585]
# avg score : 0.5232
# model = LinearRegression()
# acc score :  [0.74840317 0.67767952 0.72592914 0.73157826 0.69457569]
# avg score : 0.7156
# model = DecisionTreeRegressor()
# acc score :  [0.83199372 0.53684701 0.6642313  0.82668098 0.8489956 ]
# avg score : 0.7417
model = RandomForestRegressor()
# acc score :  [0.88129518 0.81780431 0.87318708 0.92540567 0.89552029]
# avg score : 0.8786

# 3. 컴파일 훈련
# 4. 평가 예측

score = cross_val_score(model, x_data, y_data, cv=kfold)
print('acc score : ', score)
# =. fit + model.score / training (n_splits) times
print('avg score :', np.round(np.mean(score), 4)) # get avg acc
# categorical -> score : accuracy