import numpy as np

x_data = np.load('./_save/_NPY/k55_x_data_boston.npy')
y_data = np.load('./_save/_NPY/k55_y_data_boston.npy')

# print(type(x_data), type(y_data)) 
# <class 'numpy.ndarray'> <class 'numpy.ndarray'>

from sklearn.metrics import r2_score
from sklearn.model_selection import train_test_split

# 1. data 구성


x_train, x_test, y_train, y_test = train_test_split(x_data, y_data,
      test_size=0.2, shuffle=True, random_state=66)

# print(x_train.shape, x_test.shape) # (404, 13) (102, 13)

from sklearn.preprocessing import MinMaxScaler, StandardScaler, RobustScaler, MaxAbsScaler, QuantileTransformer, PowerTransformer
scaler = RobustScaler()
x_train = scaler.fit_transform(x_train)
x_test = scaler.transform(x_test)

x_train = x_train.reshape(x_train.shape[0], x_train.shape[1], 1)
x_test = x_test.reshape(x_test.shape[0], x_test.shape[1], 1)

# 2. model 구성
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import Dense, Input, Conv1D, Flatten, Dropout, GlobalAveragePooling1D, MaxPool1D

model = Sequential()
model.add(Conv1D(filters=64, kernel_size=2, padding='same',                          
                        activation='relu', input_shape=(13, 1))) 
model.add(Dropout(0.2))
model.add(Conv1D(64, 2, padding='same', activation='relu'))
# model.add(MaxPool2D())
model.add(Conv1D(128, 2, padding='same', activation='relu'))
model.add(Dropout(0.2))
model.add(Conv1D(128, 2, padding='same', activation='relu'))
# model.add(MaxPool2D())
model.add(Conv1D(256, 2, padding='same', activation='relu'))
model.add(Dropout(0.3))
model.add(Conv1D(256, 2, padding='same', activation='relu'))
# model.add(MaxPool2D())
model.add(GlobalAveragePooling1D())
model.add(Dense(1))

# 3. 컴파일 훈련
from tensorflow.keras.optimizers import Adam

op = Adam(lr = 0.001)

# model.compile(loss='categorical_crossentropy', 
#                 optimizer='adam', metrics='acc')
model.compile(loss='mse', 
                optimizer=op, metrics='acc')

from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
es = EarlyStopping(monitor='val_loss', patience=20, 
                mode='min', verbose=1)
lr = ReduceLROnPlateau(monitor='val_loss', patience=5, 
                mode='auto', verbose=1, factor=0.8)

import time 
start_time = time.time()
hist = model.fit(x_train, y_train, epochs=1000, batch_size=32, verbose=2,
    validation_split=0.15, callbacks=[es, lr])
end_time = time.time() - start_time

# 4. 평가 예측
y_predict = model.predict([x_test])
loss = model.evaluate(x_test, y_test, batch_size=256)
print("======================================")

acc = hist.history['acc']
val_acc = hist.history['val_acc']
loss = hist.history['loss']
val_loss = hist.history['val_loss']

print("total time : ", end_time)
# print('acc : ',acc[-20])
# print('val_acc : ',val_acc[-20])
print('loss : ',loss[-20])
print('val_loss : ',val_loss[-20])

r2 = r2_score(y_test, y_predict)
print('R^2 score : ', r2)

'''
time :  10.910688638687134
loss :  15.667049407958984
R^2 score :  0.8125568409123047

with ir
total time :  12.336548805236816
loss :  18.767131805419922
val_loss :  19.854581832885742
R^2 score :  0.8374119752548896
'''