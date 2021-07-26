from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
import numpy as np
import matplotlib.pyplot as plt

# 1. 데이터
x = np.array([range(10)])
x1 = np.transpose(x)
print(x1.shape) #( 10,1)
y = np.array([[1,2,3,4,5,6,7,8,9,10], [1, 1.1, 1.2, 1.3, 1.4, 1.5, 1.6, 1.5, 1.4, 1.3], [10, 9, 8, 7, 6, 5, 4, 3, 2, 1]])
y1 = np.transpose(y)
print(y.shape) # (10, ) <-> (10, 3)

# 2. 모델 구성
model = Sequential()
model.add(Dense(5, input_dim=1))
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
model.fit(x1, y1, epochs=900, batch_size=1)

# 4. 평가 예측
x_pred = np.array([[9]])
loss = model.evaluate(x1, y1)
print('loss : ', loss)
result = model.predict(x_pred)
print('[9]의 예측값 : ', result)


# 5. 시각화
y_predict = model.predict(x1)
plt.scatter(x1,y1[:,:1])
plt.plot(x1,y_predict, color='red')
plt.show()
"""
epo = 900
loss :  0.011730324476957321
[0]의 예측값 :  [[ 0.9785621  1.1771258 10.025688 ]]
loss :  0.006119788624346256
[9]의 예측값 :  [[9.9694    1.521634  0.9390503]]
"""