from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import Dense, Input
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score

# 완성 한 뒤, 출력 결과 스샷

x = np.array([1,2,3,4,5])
y = np.array([1,2,4,3,5])

input1 = Input(shape=(1,))
dense1 = Dense(10)(input1)
dense2 = Dense(10)(dense1)
dense3 = Dense(10)(dense2)
dense4 = Dense(10)(dense3)
dense5 = Dense(10)(dense4)
dense6 = Dense(10)(dense5)
dense7 = Dense(10)(dense6)
dense8 = Dense(10)(dense7)
dense9 = Dense(10)(dense8)
output1 = Dense(1)(dense9)

model = Model(inputs=input1, outputs=output1)
'''
model = Sequential()
model.add(Dense(10, input_dim=1))
model.add(Dense(10))
model.add(Dense(10))
model.add(Dense(10))
model.add(Dense(10))
model.add(Dense(10))
model.add(Dense(10))
model.add(Dense(10))
model.add(Dense(10))
model.add(Dense(1))
'''

x_train, x_test, y_train, y_test = train_test_split(x, y,
      train_size=0.6, shuffle=True)

model.compile(loss='mse', optimizer='adam')
model.fit(x_train, y_train, epochs=500, batch_size=1)

loss = model.evaluate(x_test, y_test)
print('loss : ', loss)

y_predict = model.predict([x_test])
print('x_test의 예측값 : ', y_predict)

r2 = r2_score(y_test, y_predict)
print('R^2 score : ', r2)

'''
sqe
epo = 5000
loss :  0.3800278306007385
R^2 score :  0.809986093514469
mod
epo = 500
loss :  0.9698692560195923
R^2 score :  0.5689470073806938
'''

# 06_R2_2 copy
# 함수형 리폼
# 서머리 확인
