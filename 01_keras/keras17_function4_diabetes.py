from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import Dense, Input
import numpy as np
from sklearn import datasets
# import pandas as pd
from sklearn.datasets import load_diabetes
from sklearn.metrics import r2_score
from sklearn.model_selection import train_test_split

# R2 0.62 이상
# 1. data 구성
datasets = load_diabetes()
x = datasets.data # (442, 10)
y = datasets.target # (442,)

x_train, x_test, y_train, y_test = train_test_split(x, y,
      test_size=0.9, shuffle=True, random_state=10)

# 2. model 구성
input1 = Input(shape=(10,))
dense1 = Dense(270, activation= 'relu')(input1)
dense2 = Dense(240, activation= 'relu')(dense1)
dense3 = Dense(200, activation= 'relu')(dense2)
dense4 = Dense(124, activation= 'relu')(dense3)
dense5 = Dense(110, activation= 'relu')(dense4)
output1 = Dense(1)(dense5)

model = Model(inputs=input1, outputs=output1)
'''
model = Sequential()
model.add(Dense(70,input_dim=10))
model.add(Dense(60))
model.add(Dense(50))
model.add(Dense(30))
model.add(Dense(20))
model.add(Dense(10))
model.add(Dense(1))
'''

# 3. 컴파일 훈련
model.compile(loss='mse', optimizer='adam', metrics=['mae'])
model.fit(x_train, y_train, epochs=96, batch_size=32,
 verbose=2, validation_split=0.01)


# 4. 평가 예측
y_predict = model.predict([x_test])
# print('x의 예측값 : ', y_predict)

loss = model.evaluate(x_test, y_test)
print('loss : ', loss)

r2 = r2_score(y_test, y_predict)
print('R^2 score : ', r2)

'''
seq
epo = 95 
loss :  [2051.656982421875, 32.97710418701172]
R^2 score :  0.6602601796180484
mod
epo = 96
loss :  3519.072021484375
R^2 score :  0.4159600447539217
'''