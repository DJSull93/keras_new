from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
import numpy as np
import matplotlib.pyplot as plt

# 1. 데이터
x = np.array([1,2,3,4,5,6,7,8,9,10,11,12,13])
y = np.array([1,2,3,4,5,6,7,8,9,10,11,12,13])

# 잘라서 만들어
'''
넘파이 행렬로 바꾼 데이터를 조작하여 원하는 
훈련, 테스트, 검증 세트로 변환하는 능력이 필요,

행렬에 대한 이해 필요
'''
x_train = x[:7]
y_train = y[:7]
x_test = x[7:10]
y_test = y[7:10]
x_val = x[10:13]
y_val = y[10:13]

'''
예제는 (13, 1)로 일반적인 벡터 형태의 리스트며, 이는 간단하게
인덱스를 지정해서 몇번째 자료까지 편리하게 자를 수 있음.

넘파이는 0부터 시작하므로, 항상 프린트를 이용해서 확인 필요
'''

# x_train = np.array([1,2,3,4,5,6,7]) # 훈련
# y_train = np.array([1,2,3,4,5,6,7])
# x_test = np.array([8,9,10])         # 평가
# y_test = np.array([8,9,10])
# x_val = np.array([11, 12, 13])      # 검증
# y_val = np.array([11, 12, 13])


# # 2. 모델 구성
# model = Sequential()
# model.add(Dense(5, input_dim=1))
# model.add(Dense(5))
# model.add(Dense(5))
# model.add(Dense(5))
# model.add(Dense(5))
# model.add(Dense(5))
# model.add(Dense(5))
# model.add(Dense(5))
# model.add(Dense(5))
# model.add(Dense(5))
# model.add(Dense(1))

# # 3. 컴파일, 훈련
# model.compile(loss='mse', optimizer='adam', metrics=['mse'])
# model.fit(x_train, y_train, epochs=80, verbose=0, batch_size=1, validation_data=(x_val, y_val))

# # 4. 평가 예측
# loss = model.evaluate(x_test, y_test)
# print('loss : ', loss)
# result = model.predict([11])
# print('11의 예측값 : ', result)


# 5. 시각화
# y_predict = model.predict(x)
# plt.scatter(x,y)
# plt.plot(x,y_predict, color='red')
# plt.show()

"""

"""