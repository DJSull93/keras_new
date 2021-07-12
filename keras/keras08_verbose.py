from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
import numpy as np
import matplotlib.pyplot as plt
import time 

# 1. 데이터
x = np.array([[1,2,3,4,5,6,7,8,9,10], [1, 1.1, 1.2, 1.3, 1.4, 1.5, 1.6, 1.5, 1.4, 1.3], [10, 9, 8, 7, 6, 5, 4, 3, 2, 1]])
x1 = np.transpose(x)
# print(x1.shape) #( 10,3)
y = np.array([[11,12,13,14,15,16,17,18,19,20]])
y1 = np.transpose(y)
# print(y.shape) # (10, ) <-> (10, 1)

# 2. 모델 구성
model = Sequential()
model.add(Dense(5, input_dim=3))
model.add(Dense(5))
model.add(Dense(5))
model.add(Dense(5))
model.add(Dense(5))
model.add(Dense(5))
model.add(Dense(5))
model.add(Dense(1))

# 3. 컴파일, 훈련
# time.time() 으로 수행 시간 측정
model.compile(loss='mse', optimizer='adam')
start = time.time()
model.fit(x1, y1, epochs=1000, batch_size=1, verbose=0)
end = time.time() - start
print("걸린 시간 : ", end)


"""
batch = 1
epo = 1000
v = 0 걸린 시간 :  18.73912239074707
v = 1 걸린 시간 :  24.556628227233887
v = 2 걸린 시간 :  21.374227046966553
v = 3 걸린 시간 :  20.799005031585693
"""
"""
batch = 10
epo = 1000
v = 0 
v = 1 
v = 2 
v = 3 
"""
# verbose / 출력 양 설정
# 0 -> hide
# 1 -> show
# 2 -> hide progress bar
# 3 -> epo only

# 4. 평가 예측
# x_pred = np.array([[10, 1.3, 1]])
# loss = model.evaluate(x1, y1)
# print('loss : ', loss)
# result = model.predict(x_pred)
# print('[10, 1.3, 1]의 예측값 : ', result)

# 5. 시각화
# y_predict = model.predict(x1)
# plt.scatter(x1[:,0],y1)
# plt.scatter(x1[:,1],y1)
# plt.scatter(x1[:,2],y1)

# plt.plot(x1,y_predict, color='red')
# plt.show()
