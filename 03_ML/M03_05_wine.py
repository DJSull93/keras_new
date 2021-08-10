import numpy as np
from sklearn.model_selection import train_test_split

# 1. data
x_data = np.load('./_save/_NPY/k55_x_data_wine.npy')
y_data = np.load('./_save/_NPY/k55_y_data_wine.npy')

x_train, x_test, y_train, y_test = train_test_split(x_data, y_data,
      test_size=0.25, shuffle=True, random_state=66)

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

model = SVC()

# 3. 컴파일 훈련
model.fit(x_train, y_train)

# 4. 평가 예측
y_predict2 = model.predict(x_test)

result = model.score(x_test, y_test)
print('model_score :', result)


from sklearn.metrics import accuracy_score

y_predict = model.predict(x_test)
acc = accuracy_score(y_test, y_predict)

print('acc_score :', acc)

'''
SVC
model_score : 0.5224489795918368
acc_score : 0.5224489795918368

LinearSVC
model_score : 0.5314285714285715
acc_score : 0.5314285714285715

KNeighborsClassifier
model_score : 0.5820408163265306
acc_score : 0.5820408163265306

LogisticRegression
model_score : 0.5395918367346939
acc_score : 0.5395918367346939

DecisionTreeClassifier
model_score : 0.6081632653061224
acc_score : 0.6081632653061224

RandomForestClassifier
model_score : 0.7020408163265306
acc_score : 0.7020408163265306
'''
