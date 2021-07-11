from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
import numpy as np
import matplotlib.pyplot as plt

# 1. 데이터
# train, test로 일부 데이터는 훈련용으로 사용함
# model 성능 향상을 위함
x = np.array([1,2,3,4,5,6,7,8,9,10])
y = np.array([1,2,3,4,5,6,7,8,9,10])

x_train = x[0:7]
y_train = y[0:7]
x_test = x[7:10]
y_test = y[7:10]

print(x_train.shape, y_train.shape)  # (70,) (70,)
print(x_test.shape, y_test.shape)  # (30,) (30,)

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
model.add(Dense(1))

# 3. 컴파일, 훈련
model.compile(loss='mse', optimizer='adam')
model.fit(x_train, y_train, epochs=1000, batch_size=1)

# 4. 평가 예측
loss = model.evaluate(x_test, y_test)
print('loss : ', loss)
result = model.predict([11])
print('11의 예측값 : ', result)


# 5. 시각화
y_predict = model.predict(x)
plt.scatter(x,y)
plt.plot(x,y_predict, color='red')
plt.show()

"""

"""