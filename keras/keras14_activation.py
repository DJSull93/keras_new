from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, Activation
import numpy as np
from sklearn import datasets
# import pandas as pd
from sklearn.datasets import load_diabetes
from sklearn.metrics import r2_score
from sklearn.model_selection import train_test_split

# R2 0.62 이상
# epo, batch, node, layer, activation=relu or not -> tuning
# 1. data 구성
datasets = load_diabetes()
x = datasets.data # (442, 10)
y = datasets.target # (442,)

x_train, x_test, y_train, y_test = train_test_split(x, y,
      test_size=0.3, shuffle=True, random_state=66)

# 2. model 구성
# activation - relu => 활성화 함수 
model = Sequential()
model.add(Dense(65, activation="relu" ,input_dim=10))
model.add(Dense(86, activation="relu"))
model.add(Dense(105, activation="relu"))
model.add(Dense(75, activation="relu"))
model.add(Dense(45, activation="relu"))
model.add(Dense(1))


# 3. 컴파일 훈련
model.compile(loss='mse', optimizer='adam', metrics=['mae'])
model.fit(x_train, y_train, epochs=36, batch_size=16,
 verbose=2, validation_split=0.4)


# 4. 평가 예측
y_predict = model.predict([x_test])
# print('x의 예측값 : ', y_predict)

loss = model.evaluate(x_test, y_test)
print('loss : ', loss)

r2 = r2_score(y_test, y_predict)
print('R^2 score : ', r2)

'''
epo = 100
loss :  2876.104736328125
R^2 score :  0.5056882002869799
'''