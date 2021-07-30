'''
앞선 코드들에서 학습과 검증 세트를 일일히 나눴지만 
우리가 생각할때 불편하여 함수로 개선하고 싶은 것들은 
대부분 모두 나와있으며, 불러와서 사용만 하면 됨

-> train_test_split 함수
'''

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split

# 1. 데이터
x = np.array(range(100))
y = np.array(range(1, 101))

x_train, x_test, y_train, y_test = train_test_split(x, y,
      test_size=0.3, shuffle=True, random_state=66)
'''
사용법은 다음과 같음. 각 x, y 학습, 테스트 데이터 에 함수를 통해 데이터 셋을 
분리 후 각각에 지정해줌

default shuffle = true
default test_size = 0.25
수업에서 default에 대한 학습은 지속적으로 이뤄짐.
default에 대한 학습은 기본적으로 파라미터 튜닝을 할 때,
default값은 평타는 가는 성능을 보여주므로 이해를 필요로함.
'''

print(x_test) 
# [47 64 55 20 62 78  6 22 97 59 70 57 80 67 14 48 29 86 32 35 21 
# 39 76  7
#  96 81 82 75 90 24]
print(y_test)
# [48 65 56 21 63 79  7 23 98 60 71 58 81 68 15 49 30 87 33 36 22 
# 40 77  8
#  97 82 83 76 91 25]

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
model.fit(x_train, y_train, epochs=50, batch_size=1)

# 4. 평가 예측
loss = model.evaluate(x_test, y_test)
print('loss : ', loss)
result = model.predict([101])
print('101의 예측값 : ', result)


# # 5. 시각화
# y_predict = model.predict(x)
# plt.scatter(x,y)
# plt.plot(x,y_predict, color='red')
# plt.show()

"""
epo = 50
loss :  4.000128284165072e-10
101의 예측값 :  [[102.000046]]
epo = 100
loss :  6.664938045908286e-11
100의 예측값 :  [[100.99999]]
"""
