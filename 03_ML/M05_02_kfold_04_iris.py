import numpy as np
import warnings 
warnings.filterwarnings(action='ignore')

# 1. data
from sklearn.model_selection import KFold, cross_val_score, train_test_split

x_data = np.load('./_save/_NPY/k55_x_data_iris.npy')
y_data = np.load('./_save/_NPY/k55_y_data_iris.npy')

x_train, x_test, y_train, y_test = train_test_split(x_data, y_data,
      test_size=0.25, shuffle=True, random_state=1)

n_split = 5
kfold = KFold(n_splits=n_split, shuffle=True, 
                random_state=66)

# 2. model 구성
from sklearn.svm import LinearSVC, SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier

model = LinearSVC()
# acc score :  [0.86956522 0.95652174 0.90909091 1.         0.90909091]
# avg score : 0.9289
# model = SVC()
# acc score :  [0.86956522 1.         0.95454545 1.         0.90909091]
# avg score : 0.9466
# model = KNeighborsClassifier()
# acc score :  [0.86956522 1.         0.95454545 1.         0.90909091]
# avg score : 0.9466
# model = LogisticRegression()
# acc score :  [0.91304348 1.         0.95454545 1.         0.90909091]
# avg score : 0.9553
# model = DecisionTreeClassifier()
# acc score :  [0.91304348 0.82608696 0.95454545 1.         0.90909091]
# avg score : 0.9206
# model = RandomForestClassifier()
# acc score :  [0.86956522 1.         0.95454545 1.         0.90909091]
# avg score : 0.9466

# 3. 컴파일 훈련
# 4. 평가 예측

score = cross_val_score(model, x_train, y_train, cv=kfold)
print('acc score : ', score)
# =. fit + model.score / training (n_splits) times
print('avg score :', np.round(np.mean(score), 4)) # get avg acc
# categorical -> score : accuracy
