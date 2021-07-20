# wine2 
# make model to LSTM

from sklearn.metrics import r2_score
from sklearn.model_selection import train_test_split
import numpy as np
import pandas as pd

# 1. data
datasets = pd.read_csv('../_data/winequality-white.csv', sep=';',
                       index_col=None, header=0 ) # (4898, 12)

x = datasets.iloc[:,0:11] # (4898, 11)
y = datasets.iloc[:,[11]] # (4898, 10)

from sklearn.preprocessing import OneHotEncoder
one = OneHotEncoder()
one.fit(y)
y = one.transform(y).toarray()

x_train, x_test, y_train, y_test = train_test_split(x, y,
      test_size=0.15, shuffle=True, random_state=24)

from sklearn.preprocessing import MinMaxScaler, StandardScaler, RobustScaler, MaxAbsScaler, QuantileTransformer, PowerTransformer
scaler = StandardScaler()
scaler.fit(x_train) 
x_train = scaler.transform(x_train) 
x_test = scaler.transform(x_test) 

x_train = x_train.reshape(x_train.shape[0], x_train.shape[1], 1)
x_test = x_test.reshape(x_test.shape[0], x_test.shape[1], 1)

# 2. model 구성
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import Dense, LSTM, Dropout, Input

input1 = Input(shape=(11, 1))
xx = LSTM(units=20, activation='relu')(input1)
xx = Dense(128, activation='relu')(xx)
xx = Dense(64, activation='relu')(xx)
xx = Dense(32, activation='relu')(xx)
xx = Dense(16, activation='relu')(xx)
output1 = Dense(7, activation='softmax')(xx)

model = Model(inputs=input1, outputs=output1)

# 3. 컴파일 훈련
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics='acc')

from tensorflow.keras.callbacks import EarlyStopping
es = EarlyStopping(monitor='val_loss', patience=20, mode='min', verbose=1)

import time 
start_time = time.time()
hist = model.fit(x_train, y_train, epochs=1000, batch_size=256, verbose=2,
    validation_split=0.1, callbacks=[es])
end_time = time.time() - start_time

# 4. 평가 예측

loss = model.evaluate(x_test, y_test)
print("time : ", end_time)
print('loss : ', loss[0])
print('acc : ', loss[1])

# 5. plt visualize
# import matplotlib.pyplot as plt
# plt.figure(figsize=(9,5))

# plot 1 
# plt.subplot(2,1,1)
# plt.plot(hist.history["loss"], marker='.', c='red', label='loss')
# plt.plot(hist.history["val_loss"], marker='.', c='blue', label='val_loss')
# plt.grid()
# plt.title("loss")
# plt.ylabel("loss")
# plt.xlabel("epochs")
# plt.legend(loc='upper right')

# plot 2
# plt.subplot(2,1,2)
# plt.plot(hist.history["acc"])
# plt.plot(hist.history["val_acc"])
# plt.grid()
# plt.title("acc")
# plt.ylabel("acc")
# plt.xlabel("epochs")
# plt.legend(['acc', 'val_acc'])

# plt.show()
'''
DNN
patience, val = 20, 0.0024
loss :  0.6754863858222961
acc:  0.8399999737739563

CNN
time =  13.718469142913818
loss :  1.0421748161315918
acc :  0.6217687129974365

LSTM
time :  36.65963530540466
loss :  1.0574597120285034
acc :  0.5129251480102539
'''
