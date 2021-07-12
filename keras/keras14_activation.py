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
      train_size=0.9, shuffle=True, random_state=10)

# 2. model 구성
# activation - relu => 활성화 함수 
model = Sequential()
model.add(Dense(270, input_dim=(10), activation="relu"))
model.add(Dense(240, activation="relu"))
model.add(Dense(200, activation="relu"))
model.add(Dense(124, activation="relu"))
model.add(Dense(110, activation="relu"))
model.add(Dense(1))




# 3. 컴파일 훈련
model.compile(loss='mse', optimizer='adam', metrics=['mae'])
model.fit(x_train, y_train, epochs=97, batch_size=32,
 verbose=2, validation_split=0.01)


# 4. 평가 예측
y_predict = model.predict([x_test])
# print('x의 예측값 : ', y_predict)

loss = model.evaluate(x_test, y_test)
print('loss : ', loss)

r2 = r2_score(y_test, y_predict)
print('R^2 score : ', r2)

'''
epo = 97
loss :  [2088.807373046875, 33.21739959716797]
R^2 score :  0.6541083487898958
'''