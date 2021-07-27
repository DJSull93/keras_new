import numpy as np

x_data = np.load('./_save/_NPY/k55_x_data_wine.npy')
y_data = np.load('./_save/_NPY/k55_y_data_wine.npy')

# print(type(x_data), type(y_data)) 
# <class 'numpy.ndarray'> <class 'numpy.ndarray'>

from sklearn.metrics import r2_score
from sklearn.model_selection import train_test_split

# 1. data


from sklearn.preprocessing import OneHotEncoder
one = OneHotEncoder()
one.fit(y_data)
y_data = one.transform(y_data).toarray()

x_train, x_test, y_train, y_test = train_test_split(x_data, y_data,
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
from tensorflow.keras.layers import Dense, Input, Conv1D, Flatten, Dropout, GlobalAveragePooling1D, MaxPool1D

model = Sequential()
model.add(Conv1D(filters=32, kernel_size=2, padding='same',                          
                        activation='relu', input_shape=(11, 1))) 
model.add(Dropout(0.1))
model.add(Conv1D(32, 2, padding='same', activation='relu'))
model.add(MaxPool1D())
model.add(Conv1D(64, 2, padding='same', activation='relu'))
model.add(Dropout(0.1))
model.add(Conv1D(64, 2, padding='same', activation='relu'))
model.add(MaxPool1D())
model.add(Conv1D(128, 2, padding='same', activation='relu'))
model.add(Dropout(0.2))
model.add(Conv1D(128, 2, padding='same', activation='relu'))
model.add(MaxPool1D())
model.add(Conv1D(256, 2, padding='same', activation='relu'))
model.add(Dropout(0.2))
model.add(Conv1D(256, 2, padding='same', activation='relu'))
model.add(GlobalAveragePooling1D())
model.add(Dense(7, activation="softmax"))

# 3. 컴파일 훈련
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics='acc')

from tensorflow.keras.callbacks import EarlyStopping
es = EarlyStopping(monitor='val_loss', patience=20, mode='min', verbose=1)

import time 
start_time = time.time()
hist = model.fit(x_train, y_train, epochs=100, batch_size=256, verbose=2,
    validation_split=0.1, callbacks=[es])
end_time = time.time() - start_time

# 4. 평가 예측

loss = model.evaluate(x_test, y_test)
print("time : ", end_time)
print('loss : ', loss[0])
print('acc : ', loss[1])

'''
time :  9.43043565750122
loss :  0.9773891568183899
acc :  0.5795918107032776
'''