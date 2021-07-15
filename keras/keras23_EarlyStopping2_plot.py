# Early Stopping 
# loss plot

# boston housing minmax

from math import e
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
import numpy as np
from sklearn import datasets
from sklearn.datasets import load_boston
from sklearn.metrics import r2_score
from sklearn.model_selection import train_test_split

# 1. data
datasets = load_boston()

x = datasets.data # (506, 13) input_dim = 13
y = datasets.target # (506,) output_dim = 1

# print(x[:])
# print(np.min(x), np.max(x))
# print(np.min(x), np.max(x)) # 0.0 711.0

x_train, x_test, y_train, y_test = train_test_split(x, y,
      test_size=0.1, shuffle=True, random_state=12)

from sklearn.preprocessing import MinMaxScaler, StandardScaler
scaler = MinMaxScaler()
scaler.fit(x_train) 
x_train = scaler.transform(x_train) 
x_test = scaler.transform(x_test) 
'''
x_val = scaler.transform(x_val) 
x_pred = scaler.transform(x_pred) 
'''

# print(x.shape)
# print(y.shape)

print(datasets.feature_names) # ['CRIM' 'ZN' 'INDUS' 'CHAS' 'NOX' 'RM' 'AGE' 'DIS' 'RAD' 'TAX' 'PTRATIO' 'B' 'LSTAT'] B = 흑인 비율
print(datasets.DESCR)

# 2. model
model = Sequential()
model.add(Dense(270, input_dim=(13), activation="relu"))
model.add(Dense(240, activation="relu"))
model.add(Dense(200, activation="relu"))
model.add(Dense(124, activation="relu"))
model.add(Dense(100, activation="relu"))
model.add(Dense(1))

# 3. complile train
model.compile(loss='mse', optimizer='adam')
# Earlysttoping 도 verbose 존재 
from tensorflow.keras.callbacks import EarlyStopping
es = EarlyStopping(monitor='loss', patience=20, mode='min', verbose=1)

hist = model.fit(x_train, y_train, epochs=10000, batch_size=32, verbose=2,
    validation_split=0.115, callbacks=[es] )

print('=================loss=================')
print(hist.history['loss'])

print('===============val_loss===============')
print(hist.history['val_loss'])

print('===============compile, train===============')
# 4. eval, pred
y_predict = model.predict([x_test])
# print('x의 예측값 : ', y_predict)

# eval 도 batch_size 있으나. 이미 w 구했으므로 큰 의미 x
loss = model.evaluate(x_test, y_test, batch_size=32)
print('loss : ', loss)

r2 = r2_score(y_test, y_predict)
print('R^2 score : ', r2)

# plot 
import matplotlib.pyplot as plt 
from matplotlib import font_manager, rc

plt.plot(hist.history['loss'], color='red') # x = epoch, y = hist['loss']
plt.plot(hist.history['val_loss'], color='blue')
plt.title('로스, 발로스')
plt.xlabel('epochs')
plt.ylabel('loss, val_loss')
plt.legend(['train loss', 'val loss']) # 범례 추가
plt.show()

'''
2/2 [==============================] - 0s 997us/step - loss: 8.3174
loss :  8.317400932312012
R^2 score :  0.9152011176086197
'''