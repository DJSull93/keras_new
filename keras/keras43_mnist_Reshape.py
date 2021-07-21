# Reshape

import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.datasets import mnist
from tensorflow.python.keras.layers.convolutional import Conv2D
from tensorflow.python.keras.layers.pooling import MaxPooling2D

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
from tensorflow.keras.layers import Dense, Flatten, Dropout, Reshape

model = Sequential()
model.add(Dense(units=10, activation='relu', input_shape=(28, 28)))
model.add(Flatten()) # (N, 280)
model.add(Dense(784)) # (N, 784)
model.add(Reshape((28, 28, 1), input_shape=(784, ))) # (N, 28, 28, 1)
model.add(Conv2D(64, (2,2)))
model.add(Conv2D(64, (2,2)))
model.add(MaxPooling2D())
model.add(Flatten())
# model.add(Dense(64, activation='relu'))
model.add(Dense(10, activation='softmax'))

# 3. comple fit // metrics 'acc'
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics='accuracy')

from tensorflow.keras.callbacks import EarlyStopping
es = EarlyStopping(monitor='val_loss', patience=15, mode='min', verbose=1)

import time

start_time = time.time()
model.fit(x_train, y_train, epochs=1000, batch_size=576, verbose=2,
    validation_split=0.05, callbacks=[es])
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

Reshape
time :  34.75011372566223
loss :  0.11863075196743011
acc :  0.9787999987602234
'''
'''
_________________________________________________________________
Layer (type)                 Output Shape              Param #
=================================================================
dense (Dense)                (None, 28, 10)            290
_________________________________________________________________
flatten (Flatten)            (None, 280)               0
_________________________________________________________________
dense_1 (Dense)              (None, 784)               220304
_________________________________________________________________
reshape (Reshape)            (None, 28, 28, 1)         0
_________________________________________________________________
conv2d (Conv2D)              (None, 27, 27, 64)        320
_________________________________________________________________
conv2d_1 (Conv2D)            (None, 26, 26, 64)        16448
_________________________________________________________________
max_pooling2d (MaxPooling2D) (None, 13, 13, 64)        0
_________________________________________________________________
flatten_1 (Flatten)          (None, 10816)             0
_________________________________________________________________
dense_2 (Dense)              (None, 10)                108170
=================================================================
Total params: 345,532
Trainable params: 345,532
Non-trainable params: 0
_________________________________________________________________
'''