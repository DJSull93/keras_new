# Timeseries data make train, test example function
# preprocess, traintest split, early stop

import numpy as np

x_data = np.array(range(1, 101))
x_pred = np.array(range(96, 106))

size1 = 6
size2 = 5

def split_x(dataset, size):
    aaa = []
    for i in range(len(dataset) - size + 1):
        subset = dataset[i : (i + size)]
        aaa.append(subset)
    return np.array(aaa)

dataset = split_x(x_data, size1)

x_pred = split_x(x_pred, size2) # (6, 5)

x = dataset[:, :-1] # (95, 5)  
y = dataset[:, -1] # (95,)

# print(x.shape, y.shape, x_pred.shape)
from sklearn.model_selection import train_test_split

x_train, x_test, y_train, y_test = train_test_split(x, y,
      test_size=0.2, shuffle=True, random_state=66)

from sklearn.preprocessing import MinMaxScaler, StandardScaler, RobustScaler, MaxAbsScaler, QuantileTransformer, PowerTransformer
scaler = MinMaxScaler()
x_train = scaler.fit_transform(x_train)
x_test = scaler.transform(x_test)

# 2. model
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Conv2D, Flatten, MaxPool2D, GlobalAveragePooling2D, Dropout

model = Sequential()
model.add(Dense(270, activation='relu', input_shape=(5,)))
model.add(Dense(240, activation="relu"))
model.add(Dense(200, activation="relu"))
model.add(Dense(124, activation='relu'))
model.add(Dense(64, activation='relu'))
model.add(Dense(1))

# 3. compile train

model.compile(loss='mse', optimizer='adam')

from tensorflow.keras.callbacks import EarlyStopping
es = EarlyStopping(monitor='loss', patience=20, mode='min', verbose=1)
import time 

start_time = time.time()
model.fit(x_train, y_train, epochs=100, batch_size=64,
        validation_split=0.1, callbacks=[es])
end_time = time.time() - start_time

# 4. pred eval
from sklearn.metrics import r2_score, mean_squared_error
y_pred = model.predict(x_test)
print("time : ", end_time)
# print('y_pred : \n', y_pred) 


def RMSE(y_test, y_pred):
    return np.sqrt(mean_squared_error(y_test, y_pred))
rmse = RMSE(y_test, y_pred)
print('rmse score : ', rmse)

r2 = r2_score(y_test, y_pred)
print('R^2 score : ', r2)

'''
time :  3.1982839107513428
rmse score :  0.20088487453566456
R^2 score :  0.9999349146291963
'''