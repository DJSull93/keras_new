from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
import numpy as np
import matplotlib.pyplot as plt
'''
앞서 02에서 진행했던 시각화를 어긋난 데이터 셋에서 실행
코드를 실행해보면 붉은 점(데이터) 가 선(예측한 w = 웨이트, b = 바이아스)와 
차이가 큰 것을 볼 수 있음
'''
# 1. 데이터
x = np.array([1,2,3,4,5,6,7,8,9,10])
y = np.array([1,2,4,3,5,7,9,3,8,12])

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
model.fit(x, y, epochs=900, batch_size=1)

# 4. 평가 예측
loss = model.evaluate(x, y)
print('loss : ', loss)
result = model.predict([11])
print('11의 예측값 : ', result)

# 5. 시각화
y_predict = model.predict(x)
plt.scatter(x,y)
plt.plot(x,y_predict, color='red')
plt.show()

"""
epo = 900
loss :  3.6977150440216064
11의 예측값 :  [[10.317613]]
"""