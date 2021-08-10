'''
국화꽃 종 다중 분류 예제
ML Models practice

LogisticRegression - Classification or Regression? 
=> Classification!!!!!!!!
'''

import numpy as np
from sklearn.datasets import load_iris
from sklearn.metrics import r2_score
from sklearn.model_selection import train_test_split

# 1. data
datasets = load_iris()

x = datasets.data # (150, 4)
y = datasets.target # (150, ) - incoding-> (150,3)

x_train, x_test, y_train, y_test = train_test_split(x, y,
      test_size=0.1, shuffle=True, random_state=1)

from sklearn.preprocessing import MinMaxScaler, StandardScaler, RobustScaler, MaxAbsScaler, QuantileTransformer, PowerTransformer
scaler = MaxAbsScaler()
scaler.fit(x_train) 
x_train = scaler.transform(x_train) 
x_test = scaler.transform(x_test) 

# 2. model 구성
from sklearn.svm import LinearSVC, SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier

# model = LinearSVC()
# acc_score : 0.9333333333333333
model = SVC()
# acc_score : 1.0
# model = KNeighborsClassifier()
# acc_score : 1.0
# model = LogisticRegression()
# acc_score : 0.9333333333333333
# model = DecisionTreeClassifier()
# acc_score : 1.0
# model = RandomForestClassifier()
# acc_score : 1.0

# 3. 컴파일 훈련
model.fit(x_train, y_train)

# 4. 평가 예측
y_predict2 = model.predict(x_test)
# print(y_test[:5])
# print(y_predict2[:5])

result = model.score(x_test, y_test)
print('model_score :', result)


from sklearn.metrics import r2_score, accuracy_score

y_predict = model.predict(x_test)
acc = accuracy_score(y_test, y_predict)

print('acc_score :', acc)
'''
model_score : 0.9333333333333333
acc_score : 0.9333333333333333
'''
