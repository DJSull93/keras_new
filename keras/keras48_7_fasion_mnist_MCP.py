# example fasion_mnist  
# make perfect model in DNN

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

x_train = x_train.reshape(60000, 28, 28) # (60000, 28, 28, 1)
x_test = x_test.reshape(10000, 28, 28) # (10000, 28, 28, 1)

from sklearn.preprocessing import OneHotEncoder
one = OneHotEncoder()
y_train = y_train.reshape(-1,1)
y_test = y_test.reshape(-1,1)
one.fit(y_train)
y_train = one.transform(y_train).toarray() # (60000, 10)
y_test = one.transform(y_test).toarray() # (10000, 10)

# 2. model
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import Dense, Conv1D, Flatten, MaxPool1D, GlobalAveragePooling1D, Dropout

# model = Sequential()
# model.add(Conv1D(filters=32, kernel_size=2, padding='same',                        
#                         activation='relu' ,input_shape=(28, 28))) 
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
# es = EarlyStopping(monitor='val_loss', patience=15, mode='min', verbose=1)
# cp = ModelCheckpoint(monitor='val_loss',save_best_only=True, mode='auto',
#                      filepath='./_save/ModelCheckPoint/keras48_fasion_mnist_MCP.hdf5')
# import time 

# start_time = time.time()
# model.fit(x_train, y_train, epochs=10000, batch_size=576, verbose=2,
#     validation_split=0.005, callbacks=[es, cp])
# end_time = time.time() - start_time

# model.save('./_save/ModelCheckPoint/keras48_fasion_mnist_model_save.h5')
model = load_model('./_save/ModelCheckPoint/keras48_fasion_mnist_model_save.h5')
# model = load_model('./_save/ModelCheckPoint/keras48_fasion_mnist_MCP.hdf5')

# 4. predict eval -> no need to

loss = model.evaluate(x_test, y_test)
# print("time : ", end_time)
print('loss : ', loss[0])
print('acc: ', loss[1])

'''
CNN Conv1D
time :  18.8833224773407
loss :  0.34899044036865234
acc:  0.8956999778747559

DNN
time =  19.189976692199707
loss :  0.3708308935165405
acc :  0.8871999979019165

LSTM
time :  362.24685978889465
loss :  2.3025922775268555
acc :  0.10000000149011612

MCP CNN Conv1D
loss :  0.30312579870224
acc:  0.8948000073432922

model save Conv1D
loss :  0.40952491760253906
acc:  0.8906999826431274
'''