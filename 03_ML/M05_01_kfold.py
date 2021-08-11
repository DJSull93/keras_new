'''
train, test split -> test(label) unused in training
solution 
train -> test ;
test -> train -> /2 => use all label data for training
'''
'''
국화꽃 종 다중 분류 예제
'''
import numpy as np
import warnings 
warnings.filterwarnings(action='ignore')

# 1. data
from sklearn.model_selection import KFold, cross_val_score, train_test_split

x_data = np.load('./_save/_NPY/k55_x_data_iris.npy')
y_data = np.load('./_save/_NPY/k55_y_data_iris.npy')

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
# acc score :  [0.96666667 0.96666667 1.         0.9        1.        ]
# avg score : 0.9667
# model = SVC()
# acc score :  [0.96666667 0.96666667 1.         0.93333333 0.96666667]
# avg score : 0.9667
# model = KNeighborsClassifier()
# acc score :  [0.96666667 0.96666667 1.         0.9        0.96666667]
# avg score : 0.96
# model = LogisticRegression()
# acc score :  [1.         0.96666667 1.         0.9        0.96666667]
# avg score : 0.9667
# model = DecisionTreeClassifier()
# acc score :  [0.93333333 0.96666667 1.         0.9        0.93333333]
# avg score : 0.9467
model = RandomForestClassifier()
# acc score :  [0.9        0.96666667 1.         0.9        0.96666667]
# avg score : 0.9533

# 3. 컴파일 훈련
# 4. 평가 예측

score = cross_val_score(model, x_data, y_data, cv=kfold)
print('acc score : ', score)
# =. fit + model.score / training (n_splits) times
print('avg score :', np.round(np.mean(score), 4)) # get avg acc
# categorical -> score : accuracy
