'''
RNN레이어는 디폴트로 차원을 축소시켜 중첩이 불가능해 보임
-> return_sequences=True 로 차원 유지 가능함 : 튜닝 대상인 파라미터
'''
# pred 80  

import numpy as np

# 1. data

x = np.array([[1,2,3], [2,3,4], [3,4,5], [4,5,6],
                [5,6,7], [6,7,8], [7,8,9], [8,9,10],
                [9,10,11], [10,11,12], [20,30,40],
                [30,40,50], [40,50,60]]) # (13, 3) 
y = np.array([4,5,6,7,8,9,10,11,12,13,50,60,70]) # (13, )
x_predict = np.array([50,60,70])
x_predict = x_predict.reshape(1, x_predict.shape[0], 1)
# print(x.shape, y.shape)

x = x.reshape(13, 3, 1) # (batch_size, timesteps, feature)
x = x.reshape(x.shape[0], x.shape[1], 1)

# 2. model
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, SimpleRNN, LSTM, Dropout

model = Sequential()
model.add(LSTM(units=5, activation='relu', return_sequences=True, input_shape=(3, 1)))
model.add(LSTM(units=5, activation='relu'))
model.add(Dense(16, activation='relu'))
model.add(Dense(1))

# model.summary()

# 3. compile train

model.compile(loss='mse', optimizer='adam')

from tensorflow.keras.callbacks import EarlyStopping
es = EarlyStopping(monitor='loss', patience=30, mode='min', verbose=1)

model.fit(x, y, epochs=1000, batch_size=32, callbacks=[es])

# 4. pred eval
y_pred = model.predict(x_predict)
print('y_pred : ', y_pred) 

'''
_________________________________________________________________
Layer (type)                 Output Shape              Param #
=================================================================
lstm (LSTM)                  (None, 3, 10)             480
_________________________________________________________________
lstm_1 (LSTM)                (None, 8)                 608
_________________________________________________________________
dense (Dense)                (None, 16)                144
_________________________________________________________________
dense_1 (Dense)              (None, 1)                 17
=================================================================

units : 20
y_pred :  [[79.897575]]
y_pred :  [[80.00862]]
'''
