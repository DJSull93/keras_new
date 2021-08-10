'''
국화꽃 종 다중 분류 예제
svm - support vetor machine
'''

import numpy as np
from sklearn.datasets import load_iris
from sklearn.metrics import r2_score
from sklearn.model_selection import train_test_split

# 1. data
datasets = load_iris()

x = datasets.data # (150, 4)
y = datasets.target # (150, ) - incoding-> (150,3)

'''
Machine Learning doesn't need Onehotencode
'''
# incoding -> One_Hot_Incoding 
# from tensorflow.keras.utils import to_categorical
# y = to_categorical(y)

x_train, x_test, y_train, y_test = train_test_split(x, y,
      test_size=0.1, shuffle=True, random_state=1)

from sklearn.preprocessing import MinMaxScaler, StandardScaler, RobustScaler, MaxAbsScaler, QuantileTransformer, PowerTransformer
scaler = MaxAbsScaler()
scaler.fit(x_train) 
x_train = scaler.transform(x_train) 
x_test = scaler.transform(x_test) 

# 2. model 구성
# from tensorflow.keras.models import Sequential, Model
# from tensorflow.keras.layers import Dense, Input
from sklearn.svm import LinearSVC

model = LinearSVC()
# model = Sequential()
# model.add(Dense(270, input_dim=(4), activation="relu"))
# model.add(Dense(240, activation="relu"))
# model.add(Dense(200, activation="relu"))
# model.add(Dense(124, activation="relu"))
# model.add(Dense(100, activation="relu"))
# model.add(Dense(3, activation="softmax"))

# 3. 컴파일 훈련
model.fit(x_train, y_train)
# model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

# from tensorflow.keras.callbacks import EarlyStopping
# es = EarlyStopping(monitor='loss', patience=10, mode='min', verbose=1)

# hist = model.fit(x_train, y_train, epochs=10000, batch_size=32, verbose=2,
#     validation_split=0.25, callbacks=[es])

# 4. 평가 예측
y_predict2 = model.predict(x_test)
print(y_test[:5])
print(y_predict2[:5])

result = model.score(x_test, y_test)
print('model_score :', result)

# loss = model.evaluate(x_test, y_test)
# print('loss : ', loss[0])
# print('acc : ', loss[1])

from sklearn.metrics import r2_score, accuracy_score

y_predict = model.predict(x_test)
acc = accuracy_score(y_test, y_predict)

print('acc_score :', acc)
'''
y_test : [0 1 1 0 2]
y_pred : [0 1 1 0 2]
model_score : 0.9333333333333333
acc_score : 0.9333333333333333
'''

'''
preprocessing -> o
onehot encoding -> x
y -> 1D array
'''