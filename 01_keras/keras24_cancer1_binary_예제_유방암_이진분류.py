'''
새로운 예제로, 여태까지와 다르게 반환값이 0, 1 두가지로 
True or False분류이며 이진분류에 따른 하단에 소개할 내용들의
적용이 필요하다
'''

# cancer 예제 / y -> 0, 1 이진분류 

# 이진분류 데이터에 대해서는 다음이 강제됨
# output activation = sigmoid
# loss = binary_crossentropy

from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import Dense, Input
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

# print(x.shape, y.shape)
print(y[:20])
print(np.unique(y)) # [0 1] -> 구성 요소 탐색

x_train, x_test, y_train, y_test = train_test_split(x, y,
      test_size=0.25, shuffle=True, random_state=66)

from sklearn.preprocessing import MinMaxScaler, StandardScaler, RobustScaler, MaxAbsScaler, QuantileTransformer, PowerTransformer
scaler = QuantileTransformer()
scaler.fit(x_train) 
x_train = scaler.transform(x_train) 
x_test = scaler.transform(x_test) 

# 2. model 구성
# 결과가 0, 이므로 새로운 activation(활성화) func. -> sigmoid 사용
model = Sequential()
model.add(Dense(270, input_dim=(30), activation="relu"))
model.add(Dense(240, activation="relu"))
model.add(Dense(200, activation="relu"))
model.add(Dense(124, activation="relu"))
model.add(Dense(100, activation="relu"))
model.add(Dense(1, activation="sigmoid"))

# 3. 컴파일 훈련
# data 형태가 다르므로 mse 대신 binary_crossentropy 사용
model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['mse', 'accuracy'])

from tensorflow.keras.callbacks import EarlyStopping
es = EarlyStopping(monitor='loss', patience=20, mode='min', verbose=1)

model.fit(x_train, y_train, epochs=10000, batch_size=32, verbose=2,
    validation_split=0.2, callbacks=[es])


# 4. 평가 예측
# y_predict = model.predict([x_test])
# print('x의 예측값 : ', y_predict)
# r2 = r2_score(y_test, y_predict)
# print('R^2 score : ', r2)

loss = model.evaluate(x_test, y_test)
print('loss[binary] : ', loss[0])
print('loss[accuracy] : ', loss[2])

# original
print(y_test[-5:-1])
# after softmax
y_predict = model.predict(x_test[-5:-1])
print(y_predict)

'''
using linear
power
loss :  0.020754193887114525
R^2 score :  0.9095476332529246
using sigmoid
quantile
loss :  0.5398237705230713
Sequential
loss :  0.5177478790283203
loss :  0.2344427853822708

using accuracy in metrics
Sequential
loss[binary] :  0.4892100393772125
loss[accuracy] :  0.9590643048286438

loss[binary] :  0.44642624258995056
loss[accuracy] :  0.9650349617004395

QuantileTransformer val 0.2
loss[binary] :  0.3074130117893219
loss[accuracy] :  0.9720279574394226
'''
''' 
origin
[1 1 0 1]
adfter signoid 
[[9.8342741e-01]
 [1.0000000e+00]
 [2.0116324e-14]
 [1.0000000e+00]]
 '''