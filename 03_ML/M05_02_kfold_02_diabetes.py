import numpy as np
from sklearn.model_selection import train_test_split, KFold, cross_val_score
import warnings 
warnings.filterwarnings(action='ignore')

# 1. data
x_data = np.load('./_save/_NPY/k55_x_data_diabetes.npy')
y_data = np.load('./_save/_NPY/k55_y_data_diabetes.npy')

n_split = 5
kfold = KFold(n_splits=n_split, shuffle=True, 
                random_state=66)

# 2. model 구성
from sklearn.svm import LinearSVC, SVC
from sklearn.neighbors import KNeighborsRegressor
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor

# model = LinearSVC()

# model = SVC()

model = KNeighborsRegressor()
# acc score :  [0.39683913 0.32569788 0.43311217 0.32635899 0.35466969]
# avg score : 0.3673
# model = LinearRegression()
# acc score :  [0.50638911 0.48684632 0.55366898 0.3794262  0.51190679]
# avg score : 0.4876
# model = DecisionTreeRegressor()
# acc score :  [-0.30707277 -0.259948   -0.22940019 -0.03014968  0.0845184 ]
# avg score : -0.1484
# model = RandomForestRegressor()
# acc score :  [0.34812678 0.4830282  0.49330458 0.40076964 0.44179215]
# avg score : 0.4334

# 3. 컴파일 훈련
# 4. 평가 예측

score = cross_val_score(model, x_data, y_data, cv=kfold)
print('acc score : ', score)
# =. fit + model.score / training (n_splits) times
print('avg score :', np.round(np.mean(score), 4)) # get avg acc
# categorical -> score : accuracy