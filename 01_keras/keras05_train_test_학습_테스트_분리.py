from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
import numpy as np
import matplotlib.pyplot as plt

'''
여태껏 배운 내용은 모델을 구성하고, 데이터셋을 학습한 결과로만 
임의의 x에 대해 예측을 진햄했다.

하지만 훈련만 진행을 한다면 예측한 웨이트와 바이아스는 
기존에 학습을 진행한 x의 범위 내에서만 정확하게 작동하게되며
이를 '과적합'이라고 한다. 

-> 과적합은 자세한 내용 좀 더 찾아볼 것

따라서 모델 훈련은 훈련세트와 테스트세트로 진행하여 
훈련세트로 얻은 웨이트와 바이아스로 훈련 지표(loss, accuracy등)을
측정해서 훈련에 사용된 모든 데이터를 제외한 범위에서도 
비교적 정확한 예측을 실행할 수 있어야 한다.
'''

# 1. 데이터
# train, test로 일부 데이터는 훈련용으로 사용함
# model 성능 향상을 위함
x_train = np.array([1,2,3,4,5,6,7])
y_train = np.array([1,2,3,4,5,6,7])
x_test = np.array([8,9,10])
y_test = np.array([8,9,10])
# 학습, 검증을 위해 데이터 세트를 x, y각 두개씩 준비

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
# 모델의 훈련(fit)에서는 학습데이터(_train) 사용

# 4. 평가 예측
loss = model.evaluate(x_test, y_test)
# 모델의 평가(evaluate)에는 검증데이터(_test)를 사용해서 평가 지표 획득
print('loss : ', loss)

result = model.predict([11])
print('11의 예측값 : ', result)
# 이렇게 구분하여 학습 시 일반적으로 훈련 범위 밖 데이터의 
# 비교적 정확한 예측을 진행할 수 있다.

# 5. 시각화
# y_predict = model.predict(x)
# plt.scatter(x,y)
# plt.plot(x,y_predict, color='red')
# plt.show()

"""
epo = 10
loss :  0.006999348755925894
11의 예측값 :  [[10.8634]]
epo = 1000
loss :  1.8189894035458565e-12
11의 예측값 :  [[11.000003]]
"""