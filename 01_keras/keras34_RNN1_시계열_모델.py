'''
데이터가 순서에 따라 상관관계가 존재 : 시계열 데이터
이전 데이터가 다음 데이터에 영향을 끼치므로 
전의 가중치가 다음 연산의 가중치에 매개함수로 포함되어야함

RNN(시계열) -> LSTM, GRU, simpleRNN 세가지 실습 예정

$ RNN(시계열)은 통과 후 차원이 자동으로 축소됨
3차원 -> 2차원
'''

import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, SimpleRNN

# 1. data

x = np.array([[1,2,3], [2,3,4], [3,4,5], [4,5,6]]) # (4, 3) 
y = np.array([4,5,6,7]) # (4, )

# print(x.shape, y.shape)

x = x.reshape(4, 3, 1) # (batch_size, timesteps, feature)

# 2. model

model = Sequential()
model.add(SimpleRNN(units=10, activation='relu', input_shape=(3, 1)))
model.add(Dense(10, activation='relu'))
model.add(Dense(1))

# 3. compile train

model.compile(loss='mse', optimizer='adam')

from tensorflow.keras.callbacks import EarlyStopping
es = EarlyStopping(monitor='loss', patience=50, mode='min', verbose=1)

model.fit(x, y, epochs=10000, batch_size=1, callbacks=[es])

# 4. pred eval
x_input = np.array([5, 6, 7]).reshape(1, 3, 1)
y_pred = model.predict(x_input)
print(y_pred) # [[8.]] epochs = 851

# model.summary()
'''
input_shape (3, 1)
_________________________________________________________________
Layer (type)                 Output Shape              Param #
=================================================================
simple_rnn (SimpleRNN)       (None, 11)                143
_________________________________________________________________
dense (Dense)                (None, 10)                120
_________________________________________________________________
dense_1 (Dense)              (None, 1)                 11
=================================================================
Total params: 274
Trainable params: 274
Non-trainable params: 0
_________________________________________________________________
'''
# params = (Input + bias + output ) * output
# 143 = (11+1+1)*11