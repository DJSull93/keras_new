# example cifar10
# make perfect model in DNN

from tensorflow.keras.datasets import cifar10

import numpy as np
import matplotlib.pyplot as plt

# 1. data
(x_train, y_train), (x_test, y_test) = cifar10.load_data() 

# ic(x_train.shape, x_test.shape) # (50000, 32, 32, 3) (10000, 32, 32, 3)
# ic(y_train.shape, y_test.shape) # (50000, 1) (10000, 1)

x_train = x_train.reshape(50000, 32*32*3)
x_test = x_test.reshape(10000, 32*32*3)

from sklearn.preprocessing import MinMaxScaler, StandardScaler, RobustScaler, MaxAbsScaler, QuantileTransformer, PowerTransformer
scaler = QuantileTransformer()
# scaler.fit(x_train) 
# x_train = scaler.transform(x_train) 
x_train = scaler.fit_transform(x_train)
x_test = scaler.transform(x_test) 

x_train = x_train.reshape(50000, 32*32, 3)
x_test = x_test.reshape(10000, 32*32, 3)

from sklearn.preprocessing import OneHotEncoder
one = OneHotEncoder()
y_train = y_train.reshape(-1,1)
y_test = y_test.reshape(-1,1)
one.fit(y_train)
y_train = one.transform(y_train).toarray() # (50000, 10)
y_test = one.transform(y_test).toarray() # (10000, 10)

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
# model.add(Dense(84, activation='relu'))
# model.add(Dense(10, activation='softmax'))

# # 3. comple fit // metrics 'acc'
# model.compile(loss='categorical_crossentropy', optimizer='adam', metrics='acc')

# from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
# es = EarlyStopping(monitor='val_loss', patience=20, mode='min', verbose=1)
# cp = ModelCheckpoint(monitor='val_loss',save_best_only=True, mode='auto',
#                      filepath='./_save/ModelCheckPoint/keras48_cifar10_MCP.hdf5')
# import time 

# start_time = time.time()
# model.fit(x_train, y_train, epochs=100, batch_size=1024, verbose=2,
#     validation_split=0.05, callbacks=[es, cp])
# end_time = time.time() - start_time

# model.save('./_save/ModelCheckPoint/keras48_cifar10_model_save.h5')
model = load_model('./_save/ModelCheckPoint/keras48_cifar10_model_save.h5')
# model = load_model('./_save/ModelCheckPoint/keras48_cifar10_MCP.hdf5')

# 4. predict eval -> no need to

loss = model.evaluate(x_test, y_test)
# print("time : ", end_time)
print('loss : ', loss[0])
print('acc : ', loss[1])

'''
CNN Conv2D
loss :  1.6478464603424072
acc :  0.7605000138282776

CNN Conv1D
time :  86.63126993179321
loss :  3.5127007961273193
acc :  0.6090999841690063

DNN
time =  36.721885681152344
loss :  1.3962748050689697
acc :  0.5027999877929688

LSTM
time :  981.9731032848358
loss :  2.0776593685150146
acc :  0.24459999799728394

MCP CNN Conv1D
loss :  1.0730032920837402
acc :  0.6187999844551086

model save Conv1D
loss :  2.865736246109009
acc :  0.6172000169754028
'''