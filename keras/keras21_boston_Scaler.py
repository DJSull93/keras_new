# 실습 boston_Scaler 
# 1. loss, R2 평가
# RobustScaler, MaxAbsScaler, 
# QuantileTransformer, PowerTransformer 결과들 명시

from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import Dense, Input
import numpy as np
from sklearn import datasets
from sklearn.datasets import load_boston
from sklearn.metrics import r2_score
from sklearn.model_selection import train_test_split

# 1. data 구성
datasets = load_boston()

x = datasets.data # (506, 13) input_dim = 13
y = datasets.target # (506,) output_dim = 1

x_train, x_test, y_train, y_test = train_test_split(x, y,
      test_size=0.25, shuffle=True, random_state=66)

from sklearn.preprocessing import MinMaxScaler, StandardScaler, RobustScaler, MaxAbsScaler, QuantileTransformer, PowerTransformer
scaler = RobustScaler()
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
model.add(Dense(1))

# 3. 컴파일 훈련
model.compile(loss='mse', optimizer='adam')
model.fit(x_train, y_train, epochs=1000, batch_size=32, verbose=2,
    validation_split=0.015 )

# 4. 평가 예측
y_predict = model.predict([x_test])
# print('x의 예측값 : ', y_predict)

loss = model.evaluate(x_test, y_test)
print('loss : ', loss)

r2 = r2_score(y_test, y_predict)
print('R^2 score : ', r2)

'''
epo, batch, run = 1000, 32, 2
$ MinMaxScaler
loss :  4.442183971405029
R^2 score :  0.9547103416059415

$ StandardScaler
loss :  6.393800258636475
R^2 score :  0.9348129216145945

$ RobustScaler
loss :  9.409444808959961
R^2 score :  0.8868064747537897

$ MaxAbsScaler
loss :  7.375861167907715
R^2 score :  0.9112700225293082

$ QuantileTransformer
loss :  10.57460880279541
R^2 score :  0.8727898147822734

$ PowerTransformer
loss :  10.439437866210938
R^2 score :  0.8744158791547259
'''