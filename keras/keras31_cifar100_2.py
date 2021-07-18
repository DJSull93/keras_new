# example cifar10
# make perfect model
# top 3 coffee

from tensorflow.keras.datasets import cifar100

import numpy as np
import matplotlib.pyplot as plt
from icecream import ic

# 1. data
(x_train, y_train), (x_test, y_test) = cifar100.load_data() 

# ic(x_train.shape, x_test.shape) # (50000, 32, 32, 3) (10000, 32, 32, 3)
# ic(y_train.shape, y_test.shape) # (50000, 1) (10000, 1)

x_train = x_train.reshape(50000, 32, 32, 3)/255. # (50000, 32, 32, 3)
x_test = x_test.reshape(10000, 32, 32, 3)/255. # (10000, 32, 32, 3)

from sklearn.preprocessing import OneHotEncoder
one = OneHotEncoder()
y_train = y_train.reshape(-1,1)
y_test = y_test.reshape(-1,1)
one.fit(y_train)
y_train = one.transform(y_train).toarray() # (50000, 100)
y_test = one.transform(y_test).toarray() # (10000, 100)

# 2. model
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Conv2D, Flatten, MaxPool2D

model = Sequential()
model.add(Conv2D(filters=32, kernel_size=(2, 2),                          
                        padding='same', activation='relu', 
                        input_shape=(32, 32, 3))) 
model.add(Conv2D(32, (2, 2), padding='same', activation='relu'))                   
model.add(MaxPool2D())                                         
model.add(Conv2D(64, (3, 3), padding='same', activation='relu'))                   
model.add(Conv2D(64, (3, 3), padding='same', activation='relu'))    
model.add(MaxPool2D())                                         
model.add(Conv2D(128, (4, 4), activation='relu'))                   
model.add(Conv2D(128, (4, 4), activation='relu'))
model.add(Flatten())                                              
model.add(Dense(1024, activation='relu'))
model.add(Dense(512, activation='relu'))
model.add(Dense(256, activation='relu'))
model.add(Dense(100, activation='softmax'))

# 3. comple fit // metrics 'acc'
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['mse', 'accuracy'])

from tensorflow.keras.callbacks import EarlyStopping
es = EarlyStopping(monitor='val_loss', patience=20, mode='min', verbose=1)

model.fit(x_train, y_train, epochs=10000, batch_size=512, verbose=2,
    validation_split=0.0045, callbacks=[es])

# 4. predict eval -> no need to

loss = model.evaluate(x_test, y_test)
print('loss[category] : ', loss[0])
print('loss[accuracy] : ', loss[2])

'''
loss[category] :  6.727427959442139
loss[accuracy] :  0.3682999908924103

loss[category] :  7.6312689781188965
loss[accuracy] :  0.37220001220703125

loss[category] :  6.800853729248047
loss[accuracy] :  0.3847000002861023

loss[category] :  7.04472017288208
loss[accuracy] :  0.3855000138282776

loss[category] :  6.220393180847168
loss[accuracy] :  0.4025000035762787

loss[category] :  4.9466118812561035
loss[accuracy] :  0.4169999957084656
'''