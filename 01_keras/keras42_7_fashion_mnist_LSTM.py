# example fasion_mnist  
# LSTM

import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.datasets import fashion_mnist

# 1. data
(x_train, y_train), (x_test, y_test) = fashion_mnist.load_data() # (60000, 28, 28) (60000,) (10000, 28, 28) (10000,)

x_train = x_train.reshape(60000, 28*28) # (60000, 28, 28, 1)
x_test = x_test.reshape(10000, 28*28) # (10000, 28, 28, 1)

from sklearn.preprocessing import MinMaxScaler, StandardScaler, RobustScaler, MaxAbsScaler, QuantileTransformer, PowerTransformer
scaler = QuantileTransformer()
# scaler.fit(x_train) 
# x_train = scaler.transform(x_train) 
x_train = scaler.fit_transform(x_train)
x_test = scaler.transform(x_test) 

x_train = x_train.reshape(60000, 28*28, 1)
x_test = x_test.reshape(10000, 28*28, 1)

from sklearn.preprocessing import OneHotEncoder
one = OneHotEncoder()
y_train = y_train.reshape(-1,1)
y_test = y_test.reshape(-1,1)
one.fit(y_train)
y_train = one.transform(y_train).toarray() # (60000, 10)
y_test = one.transform(y_test).toarray() # (10000, 10)

# 2. model
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import Dense, LSTM, Dropout, Input

input1 = Input(shape=(28*28, 1))
xx = LSTM(units=10, activation='relu')(input1)
xx = Dense(128, activation='relu')(xx)
xx = Dense(64, activation='relu')(xx)
xx = Dense(32, activation='relu')(xx)
xx = Dense(16, activation='relu')(xx)
output1 = Dense(10, activation='softmax')(xx)

model = Model(inputs=input1, outputs=output1)

# 3. comple fit // metrics 'acc'
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics='acc')

from tensorflow.keras.callbacks import EarlyStopping
es = EarlyStopping(monitor='val_loss', patience=15, mode='min', verbose=1)

import time 

start_time = time.time()
model.fit(x_train, y_train, epochs=30, batch_size=2000, verbose=2,
    validation_split=0.05)
end_time = time.time() - start_time

# 4. predict eval -> no need to

loss = model.evaluate(x_test, y_test)
print("time : ", end_time)
print('loss : ', loss[0])
print('acc : ', loss[1])

'''
CNN - Conv1D
loss :  0.4134199619293213
acc :  0.923799991607666

DNN
time =  19.189976692199707
loss :  0.3708308935165405
acc :  0.8871999979019165

LSTM
time :  362.24685978889465
loss :  2.3025922775268555
acc :  0.10000000149011612
'''