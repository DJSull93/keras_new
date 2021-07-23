# example cifar100 #######################
# make perfect model in DNN

from tensorflow.keras.datasets import cifar100

import numpy as np
import matplotlib.pyplot as plt

# 1. data
(x_train, y_train), (x_test, y_test) = cifar100.load_data() 

x_train = x_train.reshape(50000, 32*32*3) # (50000, 32, 32, 3)
x_test = x_test.reshape(10000, 32*32*3) # (10000, 32, 32, 3)

from sklearn.preprocessing import MinMaxScaler, StandardScaler, RobustScaler, MaxAbsScaler, QuantileTransformer, PowerTransformer
scaler = StandardScaler()
# scaler.fit(x_train) 
# x_train = scaler.transform(x_train) 
x_train = scaler.fit_transform(x_train)
x_test = scaler.transform(x_test) 

x_train = x_train.reshape(50000, 32*32, 3) # (50000, 32, 32, 3)
x_test = x_test.reshape(10000, 32*32, 3) # (10000, 32, 32, 3)

from sklearn.preprocessing import OneHotEncoder
one = OneHotEncoder()
y_train = y_train.reshape(-1,1)
y_test = y_test.reshape(-1,1)
one.fit(y_train)
y_train = one.transform(y_train).toarray() # (50000, 100)
y_test = one.transform(y_test).toarray() # (10000, 100)

# 2. model
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import Dense, Conv1D, Flatten, MaxPool1D, GlobalAveragePooling1D, Dropout

# model = Sequential()
# model.add(Conv1D(filters=32, kernel_size=2, padding='same',                        
#                         activation='relu' ,input_shape=(32*32, 3))) 
# model.add(Conv1D(32, 2, padding='same', activation='relu'))                   
# model.add(MaxPool1D())                                         
# model.add(Conv1D(64, 2, padding='same', activation='relu'))                   
# model.add(Conv1D(64, 2, padding='same', activation='relu'))    
# model.add(Flatten())                                              
# model.add(Dense(256, activation='relu'))
# model.add(Dense(124, activation='relu'))
# # model.add(Dense(84, activation='relu'))
# model.add(Dense(100, activation='softmax'))

# # 3. comple fit // metrics 'acc'
# model.compile(loss='categorical_crossentropy', optimizer='adam', metrics='acc')

# from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
# es = EarlyStopping(monitor='val_loss', patience=20, mode='min', verbose=1)
# cp = ModelCheckpoint(monitor='val_loss',save_best_only=True, mode='auto',
#                      filepath='./_save/ModelCheckPoint/keras48_cifar100_MCP.hdf5')

# import time 
# start_time = time.time()
# model.fit(x_train, y_train, epochs=100, batch_size=1024, verbose=2,
#     validation_split=0.05, callbacks=[es, cp])
# end_time = time.time() - start_time

# model.save('./_save/ModelCheckPoint/keras48_cifar100_model_save.h5')
model = load_model('./_save/ModelCheckPoint/keras48_cifar100_model_save.h5')
# model = load_model('./_save/ModelCheckPoint/keras48_cifar100_MCP.hdf5')

# 4. predict eval -> no need to

loss = model.evaluate(x_test, y_test)
# print("time : ", end_time)
print('loss : ', loss[0])
print('acc : ', loss[1])

'''
CNN Conv2D
loss :  4.9466118812561035
acc :  0.4169999957084656

CNN Conv1D
time :  76.25080919265747
loss :  8.077665328979492
acc :  0.27320000529289246

DNN
time =  35.59278154373169
loss :  3.2117340564727783
acc :  0.257999986410141

LSTM
time :  1009.927225112915
loss :  4.605213642120361
acc :  0.009999999776482582

MCP CNN Conv1D
loss :  2.9477133750915527
acc :  0.2962000072002411

model save Conv1D
loss :  8.101597785949707
acc :  0.2736000120639801
'''