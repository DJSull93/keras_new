# 과제 3 0709
# boston housing 
# loss, R2 print
# train 70%
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import Dense, Input
import numpy as np
import matplotlib.pyplot as plt
from sklearn import datasets
from sklearn.datasets import load_boston
from sklearn.metrics import r2_score
from sklearn.model_selection import train_test_split

datasets = load_boston()

x = datasets.data # (506, 13) input_dim = 13
y = datasets.target # (506,) output_dim = 1

x_train, x_test, y_train, y_test = train_test_split(x, y,
      test_size=0.2, shuffle=True, random_state=12)

print(x.shape)
print(y.shape)

print(datasets.feature_names) # ['CRIM' 'ZN' 'INDUS' 'CHAS' 'NOX' 'RM' 'AGE' 'DIS' 'RAD' 'TAX' 'PTRATIO' 'B' 'LSTAT'] B = 흑인 비율
print(datasets.DESCR)


input1 = Input(shape=(13,))
dense1 = Dense(15)(input1)
dense2 = Dense(22)(dense1)
dense3 = Dense(22)(dense2)
dense4 = Dense(11)(dense3)
dense5 = Dense(33)(dense4)
dense6 = Dense(33)(dense5)
dense7 = Dense(22)(dense6)
dense8 = Dense(21)(dense7)
output1 = Dense(1)(dense8)

model = Model(inputs=input1, outputs=output1)

'''
model = Sequential()
model.add(Dense(15, input_dim=13))
model.add(Dense(22))
model.add(Dense(22))
model.add(Dense(11))
model.add(Dense(33))
model.add(Dense(33))
model.add(Dense(22))
model.add(Dense(21))
model.add(Dense(1))
'''

model.compile(loss='mse', optimizer='adam')
model.fit(x_train, y_train, epochs=100, batch_size=32, verbose=0)

y_predict = model.predict([x_test])
print('x의 예측값 : ', y_predict)

loss = model.evaluate(x_test, y_test)
print('loss : ', loss)

r2 = r2_score(y_test, y_predict)
print('R^2 score : ', r2)
'''
seq
epo, batch = 100, 1
loss :  21.215435028076172
R^2 score :  0.7432077221153058
model 
epo, batch = 100, 32
loss :  52.22905349731445
R^2 score :  0.3608020825496324
'''