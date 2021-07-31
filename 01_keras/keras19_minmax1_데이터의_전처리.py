# boston housing minmax
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
import numpy as np
import matplotlib.pyplot as plt
from sklearn import datasets
from sklearn.datasets import load_boston
from sklearn.metrics import r2_score
from sklearn.model_selection import train_test_split

datasets = load_boston()

x = datasets.data # (506, 13) input_dim = 13
y = datasets.target # (506,) output_dim = 1
'''
$$$ 중요 포인트

모델의 정확도를 높이는 작업은 

1. 파라미터 튜닝
2. 전처리

두개로 크게 나뉨



-> MinMaxScaler
데이터가 양의 정수 이므로 연산 많아질수록 데이터 값 폭증
0~1 사이의 수로 데이터의 변환 필요하나, 데이터 조작임
때문에 시작 시 데이터 변환, 종료 후 환원 필요
하지만 x 변환 시 매칭되는 y는 변함 없으므로 환원 불필요
x data 0~1 변환 : data preprocessing 전처리
데이터 전처리는 필수적으로 수행해야 함

=> MinMaxScaler / normalization 정규화 中 하나
(x-np.min(x))/(np.max(x)-np.min(x))

'''

x = x/np.max(x)
'''
단순히 행렬의 가장 큰 값으로 나누어 데이터를 0~1 사이값으로 변환

하지만 이렇게 단순히 작업하면 학습, 테스트 세트에 최댓값, 최솟값이 포함이 
안될 수 도 있고, 이러면 데이터 셋들이 0~1이라는 1의 길이가 아닌
다른 길이를 가질 수 있음. -> 개선 필요
'''
# print(np.min(x), np.max(x)) # 0.0 711.0

x_train, x_test, y_train, y_test = train_test_split(x, y,
      test_size=0.1, shuffle=True, random_state=12)

# print(x.shape)
# print(y.shape)

print(datasets.feature_names) # ['CRIM' 'ZN' 'INDUS' 'CHAS' 'NOX' 'RM' 'AGE' 'DIS' 'RAD' 'TAX' 'PTRATIO' 'B' 'LSTAT'] B = 흑인 비율
print(datasets.DESCR)

model = Sequential()
model.add(Dense(270, input_dim=(13), activation="relu"))
model.add(Dense(240, activation="relu"))
model.add(Dense(200, activation="relu"))
model.add(Dense(124, activation="relu"))
model.add(Dense(80, activation="relu"))
model.add(Dense(1))

model.compile(loss='mse', optimizer='adam')
model.fit(x_train, y_train, epochs=350, batch_size=32, verbose=2,
    validation_split=0.015 )

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
