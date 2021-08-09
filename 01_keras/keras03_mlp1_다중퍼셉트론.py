from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
import numpy as np
import matplotlib.pyplot as plt
'''
다중 퍼셉트론 mlp는 이전까지의 실습에서 x, y의 차원이 1차원으로 동일하였으나,
앞으로 학습할 인공지능에서는 다수의 x를 대입하여 y를 구하게 됨

따라서 다양한 형태의 행렬 계산이 필요 하며, 과제의 수기로 쓴 행렬 구분은
이 학습을 수월히 진행하기 위함이었음

하기 데이터셋은 x 는 (10, 2), y 는 (10, 1) 형태
즉 3개의 x의 행을 바탕으로 하나의 y 행을 출력하는 모델
'''
# 1. 데이터
x = np.array([[1,2,3,4,5,6,7,8,9,10], [1, 1.1, 1.2, 1.3, 1.4, 1.5, 1.6, 1.5, 1.4, 1.3]])
x1 = np.transpose(x)
# 일반적으로 transpose를 사용하지 않지만, 해당 실습에서 입력하기 편하게 열을 행으로
# 기입하여, transpose를 통해 행과 열 사이 변환을 함

# 행렬데이터에서 최전방의 데이터((ㄱ, ㄴ, ㄷ, ...)에서 ㄱ)는
# 전체 데이터를 대표하는 수의 의미로 사용되며 
# 이는 인풋 데이터에 입력이 생략됨

# 또한 명칭이 행 무시일 뿐, 실제 벡터에서 최전단의 명칭은 
# 차원마다 다름

print(x1.shape) #( 10,2)
y = np.array([[11,12,13,14,15,16,17,18,19,20]])
y1 = np.transpose(y)
print(y.shape) # (10, ) <-> (10, 1)

# 2. 모델 구성
model = Sequential()
model.add(Dense(5, input_dim=2))
model.add(Dense(5))
model.add(Dense(5))
model.add(Dense(5))
model.add(Dense(5))
model.add(Dense(5))
model.add(Dense(5))
model.add(Dense(1))

# 3. 컴파일, 훈련
model.compile(loss='mse', optimizer='adam')
model.fit(x1, y1, epochs=10, batch_size=1)

# 4. 평가 예측
x_pred = np.array([[10, 1.3]])
loss = model.evaluate(x1, y1)
print('loss : ', loss)
result = model.predict(x_pred)
print('[10, 1.3]의 예측값 : ', result)

y_predict = model.predict(x1)
plt.scatter(x1[:,:1],y1)
plt.plot(x1,y_predict, color='red')
plt.show()

"""
epo = 1000
loss :  0.0017816139152273536
[10, 1.3]의 예측값 :  [[20.019432]]
epo = 1200
loss :  0.00035092225880362093
[10, 1.3]의 예측값 :  [[20.014517]]
"""