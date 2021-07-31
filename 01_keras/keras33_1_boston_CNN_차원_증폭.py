'''
2차원의 데이터도 reshape로 차원 증폭하여 다른 레이어에 사용 가능
'''

# boston 
# make model to CNN

import numpy as np
from sklearn import datasets
from sklearn.datasets import load_boston
from sklearn.metrics import r2_score
from sklearn.model_selection import train_test_split

# 1. data 구성
datasets = load_boston()

x = datasets.data # (506, 13) input_dim = 13
y = datasets.target # (506,) output_dim = 1

x_train, x_test, y_train, y_test = train_test_split(x, y,
      test_size=0.2, shuffle=True, random_state=66)

# print(x_train.shape, x_test.shape) # (404, 13) (102, 13)

from sklearn.preprocessing import MinMaxScaler, StandardScaler, RobustScaler, MaxAbsScaler, QuantileTransformer, PowerTransformer
scaler = RobustScaler()
# scaler.fit(x_train) 
# x_train = scaler.transform(x_train) 
x_train = scaler.fit_transform(x_train)
x_test = scaler.transform(x_test)

x_train = x_train.reshape(x_train.shape[0], x_train.shape[1], 1, 1)
x_test = x_test.reshape(x_test.shape[0], x_test.shape[1], 1, 1)

# 2. model 구성
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import Dense, Input, Conv2D, Flatten, Dropout, GlobalAveragePooling2D, MaxPool2D

model = Sequential()
model.add(Conv2D(filters=32, kernel_size=2,                          
                        padding='same', activation='relu', input_shape=(13, 1, 1))) 
model.add(Dropout(0.2))
model.add(Conv2D(32, 2, padding='same', activation='relu'))
# model.add(MaxPool2D())
model.add(Conv2D(64, 2, padding='same', activation='relu'))
model.add(Dropout(0.2))
model.add(Conv2D(64, 2, padding='same', activation='relu'))
# model.add(MaxPool2D())
model.add(Conv2D(128, 2, padding='same', activation='relu'))
model.add(Dropout(0.2))
model.add(Conv2D(128, 2, padding='same', activation='relu'))
# model.add(MaxPool2D())
model.add(GlobalAveragePooling2D())
model.add(Dense(1))

# 3. 컴파일 훈련
model.compile(loss='mse', optimizer='adam')

from tensorflow.keras.callbacks import EarlyStopping
es = EarlyStopping(monitor='val_loss', patience=50, mode='min', verbose=1)

import time 
start_time = time.time()
model.fit(x_train, y_train, epochs=1000, batch_size=32, verbose=2,
    validation_split=0.15, callbacks=[es])
end_time = time.time() - start_time

# 4. 평가 예측
y_predict = model.predict([x_test])
# print('x의 예측값 : ', y_predict)

loss = model.evaluate(x_test, y_test)
print("time : ", end_time)
print('loss : ', loss)
r2 = r2_score(y_test, y_predict)
print('R^2 score : ', r2)

'''
DNN
epo, batch, run, random = 1000, 32, 2, 66
$ MinMaxScaler
loss :  4.442183971405029
R^2 score :  0.9547103416059415

CNN - Conv1D
time =  8.77216625213623
loss :  9.604047775268555
R^2 score :  0.885095590560277

CNN--Conv2D
time =  24.576046228408813
loss :  12.115398406982422
R^2 score :  0.8550493745054127
'''