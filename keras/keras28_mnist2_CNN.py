# mnist example 
# acc upper than 0.98 

import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.datasets import mnist
from icecream import ic

# 1. data
# y = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]
(x_train, y_train), (x_test, y_test) = mnist.load_data() # (60000, 28, 28) (60000,) (10000, 28, 28) (10000,)

x_train = x_train.reshape(60000, 28*28*1)
x_test = x_test.reshape(10000, 28*28*1)

from sklearn.preprocessing import MinMaxScaler, StandardScaler, RobustScaler, MaxAbsScaler, QuantileTransformer, PowerTransformer
scaler = MinMaxScaler()
# scaler.fit(x_train) 
# x_train = scaler.transform(x_train) 
x_train = scaler.fit_transform(x_train)
x_test = scaler.transform(x_test) 

x_train = x_train.reshape(60000, 28, 28, 1)
x_test = x_test.reshape(10000, 28, 28, 1)

from sklearn.preprocessing import OneHotEncoder
one = OneHotEncoder()
y_train = y_train.reshape(-1,1)
y_test = y_test.reshape(-1,1)
one.fit(y_train)
y_train = one.transform(y_train).toarray() # (60000, 10)
y_test = one.transform(y_test).toarray() # (10000, 10)

# 2. model
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Conv2D, Flatten, MaxPool2D
from tensorflow.python.keras.layers.core import Flatten

model = Sequential()
model.add(Conv2D(filters=32, kernel_size=(5, 5),                          
                        padding='same', activation='relu' ,input_shape=(28, 28, 1))) 
model.add(Conv2D(32, (5, 5), padding='same', activation='relu'))                   
model.add(MaxPool2D())                                         
model.add(Conv2D(64, (3, 3), padding='same', activation='relu'))                   
model.add(Conv2D(64, (3, 3), padding='same', activation='relu'))    
model.add(Flatten())                                              
model.add(Dense(256, activation='relu'))
model.add(Dense(124, activation='relu'))
model.add(Dense(84, activation='relu'))
model.add(Dense(10, activation='softmax'))

# 3. comple fit // metrics 'acc'
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics='accuracy')

from tensorflow.keras.callbacks import EarlyStopping
es = EarlyStopping(monitor='val_loss', patience=15, mode='min', verbose=1)

import time

start_time = time.time()
model.fit(x_train, y_train, epochs=10000, batch_size=576, verbose=2,
    validation_split=0.0005, callbacks=[es])
end_time = time.time() - start_time

# 4. predict eval -> no need to

loss = model.evaluate(x_test, y_test)
print("time = ", end_time)
print('loss[category] : ', loss[0])
print('loss[accuracy] : ', loss[1])

'''
loss[category] :  0.06172657012939453
loss[accuracy] :  0.9883999824523926

loss[category] :  0.06967199593782425
loss[accuracy] :  0.9894000291824341

loss[category] :  0.0706218034029007
loss[accuracy] :  0.9886000156402588

loss[category] :  0.06398986279964447
loss[accuracy] :  0.989300012588501

loss[category] :  0.056856293231248856
loss[accuracy] :  0.9918000102043152

loss[category] :  0.04294644296169281
loss[accuracy] :  0.9919999837875366

loss[category] :  0.057616908103227615
loss[accuracy] :  0.9922000169754028

loss[category] :  0.04006421938538551
loss[accuracy] :  0.994700014591217
'''