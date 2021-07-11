from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
import numpy as np
import matplotlib.pyplot as plt

# 1. 데이터
x = np.array([range(10), range(21,31), range(201,211)])
x1 = np.transpose(x)
print(x1.shape) #( 10,3)
y = np.array([[1,2,3,4,5,6,7,8,9,10], [1, 1.1, 1.2, 1.3, 1.4, 1.5, 1.6, 1.5, 1.4, 1.3], [10, 9, 8, 7, 6, 5, 4, 3, 2, 1]])
y1 = np.transpose(y)
print(y.shape) # (10, ) <-> (10, 1)

# 2. 모델 구성
model = Sequential()
model.add(Dense(5, input_dim=3))
model.add(Dense(5))
model.add(Dense(5))
model.add(Dense(5))
model.add(Dense(5))
model.add(Dense(5))
model.add(Dense(5))
model.add(Dense(5))
model.add(Dense(5))
model.add(Dense(5))
model.add(Dense(3))

# 3. 컴파일, 훈련
model.compile(loss='mse', optimizer='adam')
model.fit(x1, y1, epochs=1200, batch_size=1)

# 4. 평가 예측
x_pred = np.array([[0, 21, 201]])
loss = model.evaluate(x1, y1)
print('loss : ', loss)
result = model.predict(x_pred)
print('[0, 21, 201]의 예측값 : ', result)

y_predict = model.predict(x1)
plt.scatter(x1[:,:1],y1[:,:1])
plt.plot(x1,y_predict, color='red')
plt.show()
"""
epo = 900
loss :  0.005745985079556704
[0, 21, 201]의 예측값 :  [[ 1.0106049  1.094575  10.024266 ]]
"""