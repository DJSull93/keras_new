from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split

# 1. 데이터
x = np.array([1,2,3,4,5,6,7,8,9,10,11,12,13])
y = np.array([1,2,3,4,5,6,7,8,9,10,11,12,13])

# train_test_split만 사용

x_train, x_test, y_train, y_test = train_test_split(x, y, random_state=66, test_size=0.4)
x_test, x_val, y_test, y_val = train_test_split(x_test, y_test, random_state=66, test_size=0.5)

print(x_train, x_val, x_test)

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