# wine 예제
# acc 0.8 이상 만들 것

from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import Dense, Input
import numpy as np
from sklearn.datasets import load_wine
from sklearn.metrics import r2_score
from sklearn.model_selection import train_test_split


# 1. data
datasets = load_wine()

print(datasets.DESCR)
print(datasets.feature_names) # ['sepal length (cm)', 'sepal width (cm)', 'petal length (cm)', 'petal width (cm)']     

x = datasets.data # (178, 13)
y = datasets.target # (178, ) - incoding-> (178,3)

# print(x.shape, y.shape)

# incoding -> One_Hot_Incoding 
from tensorflow.keras.utils import to_categorical
y = to_categorical(y)

# print(y[:5])
# print(y.shape)

x_train, x_test, y_train, y_test = train_test_split(x, y,
      test_size=0.2, shuffle=True, random_state=1)

from sklearn.preprocessing import MinMaxScaler, StandardScaler, RobustScaler, MaxAbsScaler, QuantileTransformer, PowerTransformer
scaler = QuantileTransformer()
scaler.fit(x_train) 
x_train = scaler.transform(x_train) 
x_test = scaler.transform(x_test) 

# 2. model 구성
model = Sequential()
model.add(Dense(270, input_dim=(13), activation="relu"))
model.add(Dense(240, activation="relu"))
model.add(Dense(200, activation="relu"))
model.add(Dense(124, activation="relu"))
model.add(Dense(100, activation="relu"))
model.add(Dense(3, activation="softmax"))

# 3. 컴파일 훈련
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['mse', 'accuracy'])

from tensorflow.keras.callbacks import EarlyStopping
es = EarlyStopping(monitor='loss', patience=20, mode='min', verbose=1)

model.fit(x_train, y_train, epochs=10000, batch_size=32, verbose=2,
    validation_split=0.5, callbacks=[es])

# 4. 평가 예측
# y_predict = model.predict([x_test])
# print('x의 예측값 : ', y_predict)
# r2 = r2_score(y_test, y_predict)
# print('R^2 score : ', r2)

loss = model.evaluate(x_test, y_test)
print('loss[category] : ', loss[0])
print('loss[accuracy] : ', loss[2])

# original
print(y_test[:5])
# after softmax
y_predict = model.predict(x_test[:5])
print(y_predict)


'''
test size = 0.1
quanile
Epoch 01673: early stopping
loss[category] :  2.0449801013455726e-05
loss[accuracy] :  1.0
[[0. 0. 1.]
 [0. 1. 0.]
 [1. 0. 0.]
 [0. 1. 0.]
 [1. 0. 0.]]
[[7.0223860e-10 2.0880429e-11 1.0000000e+00]
 [5.9794682e-12 1.0000000e+00 6.5260731e-14]
 [1.0000000e+00 1.0158904e-09 6.3176109e-09]
 [8.5711171e-10 1.0000000e+00 2.5768632e-12]
 [1.0000000e+00 2.8179817e-10 1.8027558e-09]]

MinMaxScaler
Epoch 01392: early stopping
loss[category] :  7.768218893033918e-06
loss[accuracy] :  1.0
[[0. 0. 1.]
 [0. 1. 0.]
 [1. 0. 0.]
 [0. 1. 0.]
 [1. 0. 0.]]
[[5.0608018e-11 4.0945514e-10 1.0000000e+00]
 [3.9799188e-12 1.0000000e+00 6.5529434e-13]
 [1.0000000e+00 7.3859324e-10 7.1088273e-09]
 [1.0897119e-10 1.0000000e+00 1.1662272e-12]
 [1.0000000e+00 1.5750894e-10 1.5820854e-08]]
'''
'''
test size = 0.2
QuantileTransformer
Epoch 00880: early stopping
loss[category] :  0.0029777565505355597
loss[accuracy] :  1.0
[[0. 0. 1.]
 [0. 1. 0.]
 [1. 0. 0.]
 [0. 1. 0.]
 [1. 0. 0.]]
[[3.45059137e-09 1.53771648e-10 1.00000000e+00]
 [1.11740464e-13 1.00000000e+00 1.63463874e-13]
 [1.00000000e+00 1.83690641e-13 2.81967921e-11]
 [7.03233516e-09 1.00000000e+00 2.05429992e-10]
 [1.00000000e+00 5.20571948e-14 2.39953439e-11]]

MinMaxScaler
Epoch 01231: early stopping
loss[category] :  0.03374214842915535
loss[accuracy] :  0.9722222089767456
[[0. 0. 1.]
 [0. 1. 0.]
 [1. 0. 0.]
 [0. 1. 0.]
 [1. 0. 0.]]
[[7.4443946e-08 1.6727853e-06 9.9999821e-01]
 [8.0170752e-11 1.0000000e+00 2.7280180e-13]
 [1.0000000e+00 1.3937996e-18 2.0273587e-11]
 [3.5023309e-09 1.0000000e+00 1.1602218e-11]
 [1.0000000e+00 1.0965372e-19 2.6145601e-11]]
'''

# 0714 과제 내용 처럼 Qiantile Transformer 사용이 적합함