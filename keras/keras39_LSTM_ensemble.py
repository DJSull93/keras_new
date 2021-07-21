# ensemble model

import numpy as np
from numpy import array

# 1. data

x1 = array([[1,2,3], [2,3,4], [3,4,5], [4,5,6],
                [5,6,7], [6,7,8], [7,8,9], [8,9,10],
                [9,10,11], [10,11,12], [20,30,40],
                [30,40,50], [40,50,60]]) # (13, 3) 
x2 = array([[10,20,30], [20,30,40], [30,40,50], [40,50,60],
                [30,60,70], [60,70,80], [70,80,90], [80,90,100],
                [90,100,110], [100,110,120], 
                [2,3,4], [3,4,5],[4,5,6]]) # (13, 3) 
y = array([4,5,6,7,8,9,10,11,12,13,50,60,70]) # (13, )
x1_predict = array([55,65,75])
x2_predict = array([65,75,85])
x1_predict = x1_predict.reshape(1, x1_predict.shape[0], 1)
x2_predict = x2_predict.reshape(1, x2_predict.shape[0], 1)
# print(x1.shape, x2.shape, y.shape)

x1 = x1.reshape(x1.shape[0], x1.shape[1], 1)
x2 = x2.reshape(x2.shape[0], x2.shape[1], 1)

# 2. model
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import Dense, LSTM, Dropout, Input

# 2-1. model1
input1 = Input(shape=(3,1))
xx = LSTM(units=20, activation='relu', input_shape=(3, 1))(input1)
xx = Dense(256, activation='relu')(xx)
xx = Dense(128, activation='relu')(xx)
xx = Dense(64, activation='relu')(xx)
xx = Dense(32, activation='relu')(xx)
output1 = Dense(10)(xx)

# 2-2. model2
input2 = Input(shape=(3,1))
xx = LSTM(units=20, activation='relu', input_shape=(3, 1))(input1)
xx = Dense(256, activation='relu')(xx)
xx = Dense(128, activation='relu')(xx)
xx = Dense(64, activation='relu')(xx)
xx = Dense(32, activation='relu')(xx)
output2 = Dense(10)(xx)

# 2-3. model 1, 2 merge
from tensorflow.keras.layers import concatenate

merge1 = concatenate([output1, output2]) # merge 도 layer 임
merge2 = Dense(10)(merge1)
merge3 = Dense(5, activation='relu')(merge2)
last_output = Dense(1)(merge3)

model = Model(inputs=[input1, input2], outputs=last_output)

# 3. compile train

model.compile(loss='mse', optimizer='adam')

from tensorflow.keras.callbacks import EarlyStopping
es = EarlyStopping(monitor='loss', patience=20, mode='min', verbose=1)

model.fit([x1, x2], y, epochs=1000, batch_size=32, callbacks=[es])

# 4. pred eval
y_pred = model.predict([x1_predict, x2_predict])
print('y_pred : ', y_pred) 

'''
y_pred :  [[89.98565]]
'''