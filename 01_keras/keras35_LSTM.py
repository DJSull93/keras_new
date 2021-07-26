# LSTM 시계열 끝판왕 

import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, SimpleRNN, LSTM

# 1. data

x = np.array([[1,2,3], [2,3,4], [3,4,5], [4,5,6]]) # (4, 3) 
y = np.array([4,5,6,7]) # (4, )

# print(x.shape, y.shape)

x = x.reshape(4, 3, 1) # (batch_size, timesteps, feature)

# 2. model

model = Sequential()
# model.add(SimpleRNN(units=10, activation='relu', input_shape=(3, 1)))
model.add(LSTM(units=10, activation='relu', input_shape=(3, 1)))
model.add(Dense(5, activation='relu'))
model.add(Dense(1))

# 3. compile train

model.compile(loss='mse', optimizer='adam')

from tensorflow.keras.callbacks import EarlyStopping
es = EarlyStopping(monitor='loss', patience=20, mode='min', verbose=1)

model.fit(x, y, epochs=10000, batch_size=1, callbacks=[es])

# 4. pred eval
x_input = np.array([5, 6, 7]).reshape(1, 3, 1)
y_pred = model.predict(x_input)
print(y_pred) # [[8.006084]] epochs = 1745

# model.summary()
'''
_________________________________________________________________
Layer (type)                 Output Shape              Param #
=================================================================
lstm (LSTM)                  (None, 10)                480
_________________________________________________________________
dense (Dense)                (None, 10)                110
_________________________________________________________________
dense_1 (Dense)              (None, 1)                 11
=================================================================
Total params: 601
Trainable params: 601
Non-trainable params: 0
_________________________________________________________________
'''
# params = 4 * (Input + bias + output ) * output
# 480 = 4 * (10 + 1 + 1) * 10