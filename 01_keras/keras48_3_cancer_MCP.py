# cancer 
# make model to CNN

import numpy as np
from sklearn.datasets import load_breast_cancer
from sklearn.metrics import r2_score
from sklearn.model_selection import train_test_split

# 1. data
datasets = load_breast_cancer()

# print(datasets.DESCR)
# print(datasets.feature_names)

x = datasets.data # (569, 30) 
y = datasets.target # (569,)

x_train, x_test, y_train, y_test = train_test_split(x, y,
      test_size=0.25, shuffle=True, random_state=66)

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
#                         activation='relu', input_shape=(30, 1))) 
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
# model.add(Dense(1, activation="sigmoid"))

# # 3. 컴파일 훈련
# # data 형태가 다르므로 mse 대신 binary_crossentropy 사용
# model.compile(loss='binary_crossentropy', optimizer='adam', metrics='acc')

# from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
# es = EarlyStopping(monitor='loss', patience=20, mode='min', verbose=1)
# cp = ModelCheckpoint(monitor='val_loss',save_best_only=True, mode='auto',
#                      filepath='./_save/ModelCheckPoint/keras48_cancer_MCP.hdf5')

# import time 
# start_time = time.time()
# model.fit(x_train, y_train, epochs=10000, batch_size=32, verbose=2,
#     validation_split=0.2, callbacks=[es, cp])
# end_time = time.time() - start_time

# model.save('./_save/ModelCheckPoint/keras48_cancer_model_save.h5')
# model = load_model('./_save/ModelCheckPoint/keras48_cancer_model_save.h5')
model = load_model('./_save/ModelCheckPoint/keras48_cancer_MCP.hdf5')

# 4. 평가 예측

loss = model.evaluate(x_test, y_test)
# print("time : ", end_time)
print('loss : ', loss[0])
print('acc : ', loss[1])

'''
DNN
QuantileTransformer val 0.2
loss :  0.3074130117893219
acc :  0.9720279574394226

CNN
time =  15.102147340774536
loss :  0.11470630764961243
acc :  0.9720279574394226

LSTM
time :  86.30018758773804
loss :  0.06054418906569481
acc :  0.9860140085220337

MCP CNN Conv1D
loss :  0.23556265234947205
acc :  0.9300699234008789

model save Conv1D
loss :  0.23556265234947205
acc :  0.9300699234008789
'''