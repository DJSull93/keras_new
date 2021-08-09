'''
첫 예제인 보스턴 집값 예측문제. 앞으로 이러한 예저들은 
반복적으로 다양한 툴로 재활용하므로 대략적인 이해 필요
'''


# 과제 3 0709
# boston housing 
# loss, R2 print
# train 70%
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
import numpy as np
import matplotlib.pyplot as plt
from sklearn import datasets
from sklearn.metrics import r2_score
from sklearn.model_selection import train_test_split
from sklearn.datasets import load_boston
# 데이터셋은 sklearn에서 예제를 제공하므로, load_boston로 불러옴

datasets = load_boston()
# 임포트한 데이터를 datasets으로 지정

x = datasets.data # (506, 13) input_dim = 13
y = datasets.target # (506,) output_dim = 1
'''
각각의 데이터는 이미 지정이 되어있음, 사용법은 위처럼 datasets의 data과,
target으로 분류하여 x, y에 지정

이후 모델 구성 및 컴파일, 훈련, 예측, 평가는 동일함
'''
x_train, x_test, y_train, y_test = train_test_split(x, y,
      test_size=0.1, shuffle=True, random_state=12)

print(x.shape)
print(y.shape)

print(datasets.feature_names) # ['CRIM' 'ZN' 'INDUS' 'CHAS' 'NOX' 'RM' 'AGE' 'DIS' 'RAD' 'TAX' 'PTRATIO' 'B' 'LSTAT'] B = 흑인 비율
print(datasets.DESCR)


# 완성 한 뒤, 출력 결과 스샷

model = Sequential()
model.add(Dense(270, input_dim=(13), activation="relu"))
model.add(Dense(240, activation="relu"))
model.add(Dense(200, activation="relu"))
model.add(Dense(124, activation="relu"))
model.add(Dense(110, activation="relu"))
model.add(Dense(1))

model.compile(loss='mse', optimizer='adam')
model.fit(x_train, y_train, epochs=350, batch_size=32, verbose=2,
    validation_split=0.02 )

y_predict = model.predict([x_test])
# print('x의 예측값 : ', y_predict)

loss = model.evaluate(x_test, y_test)
print('loss : ', loss)

r2 = r2_score(y_test, y_predict)
print('R^2 score : ', r2)
'''
epo = 300
loss :  9.119855880737305
R^2 score :  0.9070198081147383
loss :  8.989311218261719
R^2 score :  0.9083507513852449
epo = 350 
loss :  7.438326835632324
R^2 score :  0.9241635942624012
'''