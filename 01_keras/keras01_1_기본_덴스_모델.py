from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
import numpy as np
import matplotlib.pyplot as plt

# 1. 데이터
x = np.array([1,2,3])
y = np.array([1,2,3])
# 기초적인 형태의 1,2,3 1,2,3 배열로 데이터 셋 준비

# 2. 모델
model = Sequential()
model.add(Dense(1, input_dim=1))
# 모델 구성 -> 모델은 시퀀셜, 함수형 두가지 존재, 그중 시퀀셜 모델 사용

# 컴파일, 훈련
model.compile(loss='mse', optimizer='adam')
# x, y 관계 학습(y = 'w'x + 'b') 중 mse 방식(하단 참조)를 사용한 모델 구조의 일부

model.fit(x, y, epochs=10000, batch_size=1)
# x 와 y 의 관계를 찾기 위해 데이터셋을 훈련. (x, y, 반복 횟수, 한번에 훈련할 최소단위)
# 위의 예시에서 배치사이즈 1일때 "[1]. [2], [3] 각각 훈련 = 1회 훈련"
# 배치사이즈가 3이라면 "[1,2,3] 한번에 훈련 = 1회 훈련"

# 결과, 예측
loss = model.evaluate(x, y)
# 결과로 나온 y = 'w'x + 'b' 와 데이터 셋이 벌어진 정도를 측정 = loss

print('loss : ', loss)

result = model.predict([4])
# 학습을 진행한 y = 'w'x + 'b' 의 x 자리에 4라는 별개 값을 대입했을때, result는 그에 대응하는 y값

print('4의 예측값 : ', result)


# 정리 

# "AI  학습 : 최소의 loss인 최적의 weight 탐색 과정"
# 정제된 data 확보가 목표 - 형태를 외울 것

# MSE
# Lose Fuction(손실 함수) 중의 한 종류로, 
# 평균 값과 예측 값의 차이의 제곱의 평균합으로 나타냅니다.