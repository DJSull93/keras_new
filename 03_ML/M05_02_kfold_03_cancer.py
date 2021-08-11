import numpy as np
from sklearn.model_selection import train_test_split, KFold, cross_val_score
import warnings 
warnings.filterwarnings(action='ignore')

# 1. data
x_data = np.load('./_save/_NPY/k55_x_data_cancer.npy')
y_data = np.load('./_save/_NPY/k55_y_data_cancer.npy')

n_split = 5
kfold = KFold(n_splits=n_split, shuffle=True, 
                random_state=66)

# 2. model 구성
from sklearn.svm import LinearSVC, SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier

# model = LinearSVC()
# acc score :  [0.89473684 0.81578947 0.85087719 0.90350877 0.97345133]
# avg score : 0.8877
# model = SVC()
# acc score :  [0.89473684 0.92982456 0.89473684 0.92105263 0.96460177]
# avg score : 0.921
# model = KNeighborsClassifier()
# acc score :  [0.92105263 0.92105263 0.92105263 0.92105263 0.95575221]
# avg score : 0.928
# model = LogisticRegression()
# acc score :  [0.93859649 0.95614035 0.88596491 0.94736842 0.96460177]
# avg score : 0.9385
# model = DecisionTreeClassifier()
# acc score :  [0.92982456 0.9122807  0.93859649 0.92105263 0.94690265]
# avg score : 0.9297
model = RandomForestClassifier()
# acc score :  [0.96491228 0.96491228 0.96491228 0.94736842 0.98230088]
# avg score : 0.9649

# 3. 컴파일 훈련
# 4. 평가 예측

score = cross_val_score(model, x_data, y_data, cv=kfold)
print('acc score : ', score)
# =. fit + model.score / training (n_splits) times
print('avg score :', np.round(np.mean(score), 4)) # get avg acc
# categorical -> score : accuracy