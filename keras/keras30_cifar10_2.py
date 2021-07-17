# example cifar10
# make perfect model

# image color ; (32, 32, 3)
from tensorflow.keras.datasets import cifar10

import numpy as np
import matplotlib.pyplot as plt
from icecream import ic

# 1. data
(x_train, y_train), (x_test, y_test) = cifar10.load_data() 

# ic(x_train.shape, x_test.shape) # (50000, 32, 32, 3) (10000, 32, 32, 3)
# ic(y_train.shape, y_test.shape) # (50000, 1) (10000, 1)

x_train = x_train.reshape(50000, 32, 32, 3)/255. # (50000, 32, 32, 3)
x_test = x_test.reshape(10000, 32, 32, 3)/255. # (10000, 32, 32, 3)

from sklearn.preprocessing import OneHotEncoder
one = OneHotEncoder()
y_train = y_train.reshape(-1,1)
y_test = y_test.reshape(-1,1)
one.fit(y_train)
y_train = one.transform(y_train).toarray() # (50000, 10)
y_test = one.transform(y_test).toarray() # (10000, 10)

# 2. model
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Conv2D, Flatten, MaxPool2D
from tensorflow.python.keras.layers.core import Flatten


model = Sequential()
model.add(Conv2D(filters=32, kernel_size=(3, 3),                          
                        padding='same', activation='relu', 
                        input_shape=(32, 32, 3))) 
model.add(Conv2D(32, (3, 3), padding='same', activation='relu'))                   
model.add(MaxPool2D())                                         
model.add(Conv2D(64, (3, 3), padding='same', activation='relu'))                   
model.add(Conv2D(64, (3, 3), padding='same', activation='relu'))    
model.add(MaxPool2D())                                         
model.add(Conv2D(128, (3, 3), padding='same', activation='relu'))                   
model.add(Conv2D(128, (3, 3), padding='same', activation='relu'))
model.add(Flatten())                                              
model.add(Dense(128, activation='relu'))
# model.add(Dense(124, activation='relu'))
model.add(Dense(84, activation='relu'))
model.add(Dense(10, activation='softmax'))

# 3. comple fit // metrics 'acc'
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['mse', 'accuracy'])

from tensorflow.keras.callbacks import EarlyStopping
es = EarlyStopping(monitor='val_loss', patience=20, mode='min', verbose=1)

model.fit(x_train, y_train, epochs=10000, batch_size=576, verbose=2,
    validation_split=0.0012, callbacks=[es])

# 4. predict eval -> no need to

loss = model.evaluate(x_test, y_test)
print('loss[category] : ', loss[0])
print('loss[accuracy] : ', loss[2])

'''
loss[category] :  2.1953704357147217
loss[accuracy] :  0.6863999962806702

loss[category] :  1.6478464603424072
loss[accuracy] :  0.7605000138282776
'''