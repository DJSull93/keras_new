'''
보스턴 집값 예측만 해도 나쁘지 않은 정확도가 나오지만,
앞으로 나올 대부분의 모델은 그렇지 못하다.

많은 이유가 있어 파라미터 튜닝을 기본적으로 해야하며,
그중, 모델의 레이어 간의 계산에서 제한시켜주는 활성함수에 대해 배운다.

# 모델의 계산 횟수
앞서 잠깐 보여준 것처럼 model.summary()를 통해 모델이 제대로 구성이
되었는지와, parameter가 몇개인지 확인 할 수 있고,

여기서 모델의 계산 횟수는 전체 parameter 개수와 동일하다.
이때, 파라미터의 개수는 노드의 수에 영향을 받고, 
깊어질수록, 노드가 많아질수록 기하급수적으로 연산이 늘어나며, 
이에 따라 웨이트는 너무 큰 수가 되어버린다.

너무 큰 수를 연산하는데는 많은 리소스가 필요하고, 파이썬, 넘파이는 
소수점 연산에 최적화 되어있으므로, 

### 활성함수를 통해 하나에 연산에서 이뤄진 y=wx+b 를 
### y = activation(wx+b)로 감싸주어, 연산의 결과를 소수점의 수로
### 반환한다.

따라서 연산 효율이 상승하고, 정확도의 개선을 도모할 수 있다.

활성함수 또한 다양하게 존재하며, 기본값은 linear이며 최종 아웃풋은 
기본값을 사용해야만 한다.

그외, 입력 및 중간 레이어에서는 relu라는 강력한 함수를 사용한다.

$$$ 요약 
1. 연산 많아지만 웨이트 너무 커져서 느리고 부정확해짐
2. 활성함수로 소수점 값으로 변환 후 연산하면 좀 정확해짐
3. 렐루가 짱임 

-> 다양한 활성함수가 있고, 지금까지나온 linear, relu 구글링 해볼것
-> 앞으로 나올 대부분의 활성함수, 키워드는 이해 안될 시 구글링 해볼것


'''

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
import numpy as np
from sklearn import datasets
# import pandas as pd
from sklearn.datasets import load_diabetes
from sklearn.metrics import r2_score
from sklearn.model_selection import train_test_split

# R2 0.62 이상
# epo, batch, node, layer, activation=relu or not -> tuning
# 1. data 구성
datasets = load_diabetes()
x = datasets.data # (442, 10)
y = datasets.target # (442,)

x_train, x_test, y_train, y_test = train_test_split(x, y,
      train_size=0.9, shuffle=True, random_state=10)

# 2. model 구성
# activation - relu => 활성화 함수 
model = Sequential()
model.add(Dense(270, input_dim=(10), activation="relu"))
model.add(Dense(240, activation="relu"))
model.add(Dense(200, activation="relu"))
model.add(Dense(124, activation="relu"))
model.add(Dense(110, activation="relu"))
model.add(Dense(1))

# 3. 컴파일 훈련
model.compile(loss='mse', optimizer='adam', metrics=['mae'])
model.fit(x_train, y_train, epochs=96, batch_size=32,
 verbose=2, validation_split=0.01)

# 4. 평가 예측
y_predict = model.predict([x_test])
# print('x의 예측값 : ', y_predict)

loss = model.evaluate(x_test, y_test)
print('loss : ', loss)

r2 = r2_score(y_test, y_predict)
print('R^2 score : ', r2)

'''
epo = 97
loss :  [2088.807373046875, 33.21739959716797]
R^2 score :  0.6541083487898958
epo = 95 
loss :  [2051.656982421875, 32.97710418701172]
R^2 score :  0.6602601796180484
'''