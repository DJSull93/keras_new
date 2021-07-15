# Early Stopping 
# parameter setting -> loss 최소치 감지 시 정지 

# boston housing minmax

from math import e
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
import numpy as np
import matplotlib.pyplot as plt
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
es = EarlyStopping(monitor='val_loss', patience=3, mode='min', verbose=1)

hist = model.fit(x_train, y_train, epochs=10000, batch_size=32, verbose=2,
    validation_split=0.015, callbacks=[es] )


# loss, val_loss 는 다음 연산을 위해 어딘가에 저장
# print(hist)
# <tensorflow.python.keras.callbacks.History object at 0x000001BFC7DC2820>

# print(hist.history.keys()) # dict_keys(['loss', 'val_loss'])
print('=================loss=================')
print(hist.history['loss'])
print('===============val_loss===============')
print(hist.history['val_loss'])
'''
=================loss=================
[420.1285705566406, 127.08382415771484, 69.29115295410156, 50.45737838745117, 41.95061111450195, 33.09235763549805, 26.69041633605957, 22.768295288085938, 19.7364501953125, 19.65814208984375, 18.23590850830078, 17.72934341430664, 16.495927810668945, 16.34779167175293, 14.733175277709961, 14.760472297668457]
===============val_loss===============
[119.02879333496094, 100.83172607421875, 43.409427642822266, 48.97944259643555, 30.57598876953125, 25.496646881103516, 20.825347900390625, 18.704187393188477, 16.027315139770508, 14.97414493560791, 14.575732231140137, 15.470197677612305, 13.857688903808594, 14.150918960571289, 13.943397521972656, 13.8618745803833]
'''
'''
# 4. eval, pred
y_predict = model.predict([x_test])
# print('x의 예측값 : ', y_predict)

# eval 도 batch_size 있으나. 이미 w 구했으므로 큰 의미 x
loss = model.evaluate(x_test, y_test, batch_size=32)
print('loss : ', loss)

r2 = r2_score(y_test, y_predict)
print('R^2 score : ', r2)

'''
'''
verbose - none
14/14 - 0s - loss: 2.1382 - val_loss: 5.9347
2/2 [==============================] - 0s 2ms/step - loss: 6.1444
loss :  6.144405364990234
R^2 score :  0.937355587389544

verbose = 1
Epoch 302/10000
14/14 - 0s - loss: 1.3875 - val_loss: 6.6796
Epoch 00302: early stopping
2/2 [==============================] - 0s 997us/step - loss: 5.5425
loss :  5.542469501495361
R^2 score :  0.9434925355084461
'''