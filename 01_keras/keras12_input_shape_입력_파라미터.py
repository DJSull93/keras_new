from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
import numpy as np
import matplotlib.pyplot as plt

# 1. 데이터
x = np.array([[1,2,3,4,5,6,7,8,9,10],
 [1, 1.1, 1.2, 1.3, 1.4, 1.5, 1.6, 1.5, 1.4, 1.3],
 [10, 9, 8, 7, 6, 5, 4, 3, 2, 1]])
x = np.transpose(x)
print(x.shape) #(10,3)
y = np.array([[11,12,13,14,15,16,17,18,19,20]])
y = np.transpose(y)
print(y.shape) # (10, ) <-> (10, 1)

'''
모델에 대입할 때 인풋 디멘션이 맞지 않으면 에러

input_shape를 통해 벡터 형태로 인풋 가능
'''

# 2. 모델 구성
model = Sequential()
# model.add(Dense(5, input_dim=3))
model.add(Dense(5, input_shape=(3, )))
model.add(Dense(5))
model.add(Dense(5))
model.add(Dense(5))
model.add(Dense(5))
model.add(Dense(5))
model.add(Dense(5))
model.add(Dense(1))

# 3. 컴파일, 훈련
model.compile(loss='mse', optimizer='adam')
model.fit(x, y, epochs=100, batch_size=1, verbose=0)

# 4. 평가 예측
x_pred = np.array([[10, 1.3, 1]])
loss = model.evaluate(x, y)
print('loss : ', loss)
result = model.predict(x_pred)
print('[10, 1.3, 1]의 예측값 : ', result)

# # 5. 시각화
# y_predict = model.predict(x)
# plt.scatter(x[:,0],y)
# plt.scatter(x[:,1],y)
# plt.scatter(x[:,2],y)
# plt.plot(x,y_predict, color='red')
# plt.show()

"""

"""