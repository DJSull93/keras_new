# boston housing minmax
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

'''
데이터 전처리를 경험해봤고, 역시나 불편하고 쉽지 않은건 이미 함수로 
다 나와있다.

이제 다양한 전처리 함수에 대해 알아보자.
'''

# x_train에 대해 fit 한 scaler를 x_test에 적용
# 컬럼별 minmax -> sklearn.preproccessing
from sklearn.preprocessing import MinMaxScaler
scaler = MinMaxScaler()
scaler.fit(x_train) 
x_train = scaler.transform(x_train) 
x_test = scaler.transform(x_test) 
'''
사용법은 위와 같다.
먼저 호출해주고, 스케일러를 지정 후 
##### 같은 길이를 가지는게 중점 ##### 이므로 
학습 데이터에 스케일러를 훈련시키고 (35라인)
그 결과를 바탕으로 학습, 테스트 세트를 적용한다 (36,37라인)
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
model.fit(x_train, y_train, epochs=1000, batch_size=32, verbose=2,
    validation_split=0.015 )

# 4. eval, pred
y_predict = model.predict([x_test])
# print('x의 예측값 : ', y_predict)

loss = model.evaluate(x_test, y_test)
print('loss : ', loss)

r2 = r2_score(y_test, y_predict)
print('R^2 score : ', r2)

'''
MinMax Scaler
epo = 1000 
loss :  4.89172887802124
R^2 score :  0.9501270749506096
'''