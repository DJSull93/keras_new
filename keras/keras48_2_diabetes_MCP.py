# diabetes 
# make model to CNN

import numpy as np
from sklearn import datasets
from sklearn.datasets import load_diabetes
from sklearn.metrics import r2_score
from sklearn.model_selection import train_test_split

# 1. data 구성
datasets = load_diabetes()

x = datasets.data # (442, 10)
y = datasets.target # (442,)

x_train, x_test, y_train, y_test = train_test_split(x, y,
      test_size=0.25, shuffle=True, random_state=9)

from sklearn.preprocessing import MinMaxScaler, StandardScaler, RobustScaler, MaxAbsScaler, QuantileTransformer, PowerTransformer
scaler = MinMaxScaler()
scaler.fit(x_train) 
x_train = scaler.transform(x_train) 
x_test = scaler.transform(x_test) 

x_train = x_train.reshape(x_train.shape[0], x_train.shape[1], 1)
x_test = x_test.reshape(x_test.shape[0], x_test.shape[1], 1)

# 2. model 구성
from tensorflow.keras.models import Sequential, Model, load_model
from tensorflow.keras.layers import Dense, Input, Conv1D, Flatten, Dropout, GlobalAveragePooling1D, MaxPool1D

# model = Sequential()
# model.add(Conv1D(filters=32, kernel_size=2, padding='same',                          
#                         activation='relu', input_shape=(10, 1))) 
# model.add(Dropout(0.2))
# model.add(Conv1D(32, 2, padding='same', activation='relu'))
# model.add(MaxPool1D())
# model.add(Conv1D(64, 2, padding='same', activation='relu'))
# model.add(Dropout(0.2))
# model.add(Conv1D(64, 2, padding='same', activation='relu'))
# model.add(MaxPool1D())
# model.add(Conv1D(128, 2, padding='same', activation='relu'))
# model.add(Dropout(0.2))
# model.add(Conv1D(128, 2, padding='same', activation='relu'))
# model.add(MaxPool1D())
# model.add(GlobalAveragePooling1D())
# model.add(Dense(1))

# # 3. 컴파일 훈련
# model.compile(loss='mse', optimizer='adam')

# from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
# es = EarlyStopping(monitor='val_loss', patience=50, mode='min', verbose=1)
# cp = ModelCheckpoint(monitor='val_loss',save_best_only=True, mode='auto',
#                      filepath='./_save/ModelCheckPoint/keras48_diabetes_MCP.hdf5')

# import time 
# start_time = time.time()
# model.fit(x_train, y_train, epochs=1000, batch_size=32, verbose=2,
#     validation_split=0.2, callbacks=[es, cp])
# end_time = time.time() - start_time

# model.save('./_save/ModelCheckPoint/keras48_diabetes_model_save.h5')
model = load_model('./_save/ModelCheckPoint/keras48_diabetes_model_save.h5')
# model = load_model('./_save/ModelCheckPoint/keras48_diabetes_MCP.hdf5')

# 4. 평가 예측
y_predict = model.predict([x_test])
# print('x의 예측값 : ', y_predict)

loss = model.evaluate(x_test, y_test)
# print("time : ", end_time)
print('loss : ', loss)
r2 = r2_score(y_test, y_predict)
print('R^2 score : ', r2)

'''
DNN
epo, batch, run, random = 200, 32, 2, 9
$ MaxAbsScaler
loss :  2247.06396484375
R^2 score :  0.6105307266558146

CNN
time =  10.436100006103516
loss :  2641.959228515625
R^2 score :  0.542086011867093

LSTM
time :  91.9576826095581
loss :  3112.569091796875
R^2 score :  0.4605181975203496

MCP CNN Conv1D
loss :  3098.157958984375
R^2 score :  0.4630160404118471

model save Conv1D
loss :  2643.99267578125
R^2 score :  0.5417335315949131
'''