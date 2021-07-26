# Dense -> possible with N dimention

import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.datasets import mnist

# 1. data
(x_train, y_train), (x_test, y_test) = mnist.load_data() # (60000, 28, 28) (60000,) (10000, 28, 28) (10000,)

x_train = x_train.reshape(60000, 28*28*1)
x_test = x_test.reshape(10000, 28*28*1)

from sklearn.preprocessing import MinMaxScaler, StandardScaler, RobustScaler, MaxAbsScaler, QuantileTransformer, PowerTransformer
scaler = MinMaxScaler()
x_train = scaler.fit_transform(x_train)
x_test = scaler.transform(x_test) 

x_train = x_train.reshape(60000, 28, 28)
x_test = x_test.reshape(10000, 28, 28)

from sklearn.preprocessing import OneHotEncoder
one = OneHotEncoder()
y_train = y_train.reshape(-1,1)
y_test = y_test.reshape(-1,1)
one.fit(y_train)
y_train = one.transform(y_train).toarray() # (60000, 10)
y_test = one.transform(y_test).toarray() # (10000, 10)

# 2. model
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Conv2D, Flatten, MaxPool2D, GlobalAveragePooling2D, Dropout

model = Sequential()
model.add(Dense(units=270, activation='relu', input_shape=(28, 28)))
model.add(Flatten())
model.add(Dense(64, activation='relu'))
model.add(Dense(10, activation='softmax'))

# 3. compile fit // metrics 'acc'
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics='accuracy')

from tensorflow.keras.callbacks import EarlyStopping
es = EarlyStopping(monitor='val_loss', patience=15, mode='min', verbose=1)

import time

start_time = time.time()
model.fit(x_train, y_train, epochs=1000, batch_size=576, verbose=2,
    validation_split=0.0005, callbacks=[es])
end_time = time.time() - start_time

# 4. predict eval -> no need to

loss = model.evaluate(x_test, y_test)
print("time : ", end_time)
print('loss : ', loss[0])
print('acc : ', loss[1])

'''
3D Dense
time :  95.94592499732971
loss :  0.25713950395584106
acc :  0.978600025177002
'''
'''
_________________________________________________________________
Layer (type)                 Output Shape              Param #
=================================================================
dense (Dense)                (None, 11, 22, 33, 270)   12150
_________________________________________________________________
dense_1 (Dense)              (None, 11, 22, 33, 64)    17344
_________________________________________________________________
dense_2 (Dense)              (None, 11, 22, 33, 10)    650
=================================================================
Total params: 30,144
Trainable params: 30,144
Non-trainable params: 0
_________________________________________________________________
'''