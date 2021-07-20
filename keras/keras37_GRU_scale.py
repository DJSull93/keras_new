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
from tensorflow.keras.layers import Dense, GRU, Dropout

model = Sequential()
model.add(GRU(units=20, activation='relu', input_shape=(3, 1)))
model.add(Dense(256, activation='relu'))
model.add(Dense(128, activation='relu'))
model.add(Dense(64, activation='relu'))
model.add(Dense(32, activation='relu'))
# model.add(Dense(16, activation='relu'))
model.add(Dense(1))

# 3. compile train

model.compile(loss='mse', optimizer='adam')

from tensorflow.keras.callbacks import EarlyStopping
es = EarlyStopping(monitor='loss', patience=30, mode='min', verbose=1)

model.fit(x, y, epochs=1000, batch_size=32)

# 4. pred eval
y_pred = model.predict(x_predict)
print('y_pred : ', y_pred) 

'''
units : 20
y_pred :  [[80.289474]]
y_pred :  [[80.21963]]

epo = 1000
y_pred :  [[80.023636]]
'''
