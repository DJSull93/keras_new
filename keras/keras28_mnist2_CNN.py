# mnist example 
# acc upper than 0.98 

import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.datasets import mnist
from icecream import ic

# 1. data
# y = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]
(x_train, y_train), (x_test, y_test) = mnist.load_data() # (60000, 28, 28) (60000,) (10000, 28, 28) (10000,)

x_train = x_train.reshape(60000, 28, 28, 1)/255. # (60000, 28, 28, 1)
x_test = x_test.reshape(10000, 28, 28, 1)/255. # (10000, 28, 28, 1)

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
model.add(Conv2D(filters=30, kernel_size=(2, 2),                          
                        padding='same' ,input_shape=(28, 28, 1))) 
model.add(Conv2D(20, (2,2), activation='relu'))                   
model.add(Conv2D(10, (2,2), activation='relu'))  
model.add(MaxPool2D())                                            
model.add(Conv2D(15, (2,2)))    
model.add(MaxPool2D())                                            
model.add(Flatten())                                              
model.add(Dense(64, activation='relu'))
model.add(Dense(32, activation='relu'))
model.add(Dense(16, activation='relu'))
model.add(Dense(10, activation='softmax'))

# 3. comple fit // metrics 'acc'
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['mse', 'accuracy'])

from tensorflow.keras.callbacks import EarlyStopping
es = EarlyStopping(monitor='val_loss', patience=20, mode='min', verbose=1)

model.fit(x_train, y_train, epochs=10000, batch_size=1152, verbose=2,
    validation_split=0.0015, callbacks=[es])

# 4. predict eval -> no need to

loss = model.evaluate(x_test, y_test)
print('loss[category] : ', loss[0])
print('loss[accuracy] : ', loss[2])

'''
loss[category] :  0.06172657012939453
loss[accuracy] :  0.9883999824523926

loss[category] :  0.06967199593782425
loss[accuracy] :  0.9894000291824341
'''