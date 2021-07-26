# 실습 diabetes 
# 1. loss, R2 평가
# MinMax, Standard 결과들 명시

from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import Dense, Input
import numpy as np
from sklearn import datasets
from sklearn.datasets import load_diabetes
from sklearn.metrics import r2_score
from sklearn.model_selection import train_test_split

# 1. data 구성
datasets = load_diabetes()
x = datasets.data # (442, 10)
y = datasets.target # (442,)

x_train, x_test, y_train, y_test = train_test_split(x, y,
      test_size=0.25, shuffle=True, random_state=9)

from sklearn.preprocessing import MinMaxScaler, StandardScaler
scaler = MinMaxScaler()
scaler.fit(x_train) 
x_train = scaler.transform(x_train) 
x_test = scaler.transform(x_test) 

# 2. model 구성
input1 = Input(shape=(10,))
dense1 = Dense(32, activation= 'relu')(input1)
dense2 = Dense(16, activation= 'relu')(dense1)
dense3 = Dense(8, activation= 'relu')(dense2)
dense4 = Dense(4, activation= 'relu')(dense3)
output1 = Dense(1)(dense4)

model = Model(inputs=input1, outputs=output1)

# 3. 컴파일 훈련
model.compile(loss='mse', optimizer='adam')
model.fit(x_train, y_train, epochs=200, batch_size=32, verbose=2,
    validation_split=0.2 )

# 4. 평가 예측
y_predict = model.predict([x_test])
# print('x의 예측값 : ', y_predict)

loss = model.evaluate(x_test, y_test)
print('loss : ', loss)

r2 = r2_score(y_test, y_predict)
print('R^2 score : ', r2)

'''
epo, batch, run, random = 200, 32, 2, 9
$ MinMaxScaler
loss :  2289.67333984375
R^2 score :  0.6031454895718897

$ StandardScaler
loss :  2448.52978515625
R^2 score :  0.5756118781244629
'''