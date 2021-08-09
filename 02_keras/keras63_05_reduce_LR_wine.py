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
      test_size=0.005, shuffle=True, random_state=24)

from sklearn.preprocessing import MinMaxScaler, StandardScaler, RobustScaler, MaxAbsScaler, QuantileTransformer, PowerTransformer
scaler = RobustScaler()
scaler.fit(x_train) 
x_train = scaler.transform(x_train) 
x_test = scaler.transform(x_test) 

x_train = x_train.reshape(x_train.shape[0], x_train.shape[1])
x_test = x_test.reshape(x_test.shape[0], x_test.shape[1])

# 2. model 구성
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import Dense, Input, Conv1D, Flatten, Dropout, GlobalAveragePooling1D, MaxPool1D

model = Sequential()
model.add(Dense(240, input_dim=(11), activation="relu"))
model.add(Dense(240, activation="relu"))
model.add(Dense(200, activation="relu"))
model.add(Dense(124, activation="relu"))
model.add(Dense(60, activation="relu"))
model.add(Dense(7, activation="softmax"))
# 3. 컴파일 훈련
from tensorflow.keras.optimizers import Adam

op = Adam(lr = 0.001)

# model.compile(loss='categorical_crossentropy', 
#                 optimizer='adam', metrics='acc')
model.compile(loss='categorical_crossentropy', 
                optimizer=op, metrics='acc')

from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
es = EarlyStopping(monitor='val_loss', patience=20, 
                mode='min', verbose=1)
lr = ReduceLROnPlateau(monitor='val_loss', patience=5, 
                mode='auto', verbose=1, factor=0.9)

import time 
start_time = time.time()
hist = model.fit(x_train, y_train, epochs=100, batch_size=64, verbose=2,
    validation_split=0.0024, callbacks=[es, lr])
end_time = time.time() - start_time

# 4. 평가 예측
loss = model.evaluate(x_test, y_test, batch_size=256)
print("======================================")

acc = hist.history['acc']
val_acc = hist.history['val_acc']
loss = hist.history['loss']
val_loss = hist.history['val_loss']

print("total time : ", end_time)
print('acc : ',acc[-20])
print('val_acc : ',val_acc[-20])
print('loss : ',loss[-20])
print('val_loss : ',val_loss[-20])

'''
time :  9.43043565750122
loss :  0.9773891568183899
acc :  0.5795918107032776

with lr
total time :  23.467710971832275
acc :  0.6597597599029541
val_acc :  0.5738295316696167
loss :  0.8209688067436218
val_loss :  1.041482925415039
'''