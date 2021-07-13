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
      test_size=0.5, shuffle=True, random_state=66)

# 2. model 구성
input1 = Input(shape=(10,))
dense1 = Dense(70)(input1)
dense2 = Dense(60)(dense1)
dense3 = Dense(50)(dense2)
dense4 = Dense(30)(dense3)
dense5 = Dense(20)(dense4)
dense6 = Dense(10)(dense5)
output1 = Dense(1)(dense6)

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
model.compile(loss='mse', optimizer='adam')
model.fit(x_train, y_train, epochs=100, batch_size=8,
 verbose=2, validation_split=0.2, shuffle=True)


# 4. 평가 예측
y_predict = model.predict([x_test])
# print('x의 예측값 : ', y_predict)

loss = model.evaluate(x_test, y_test)
print('loss : ', loss)

r2 = r2_score(y_test, y_predict)
print('R^2 score : ', r2)

'''
seq
epo = 100
loss :  2876.104736328125
R^2 score :  0.5056882002869799
mod
epo = 100
loss :  3053.63525390625
R^2 score :  0.47517629853902
'''