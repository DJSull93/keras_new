from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score
'''
기존 평가지표로 loss를 사용하였으나,  다양한 모델과 데이터를 
다룸에 있어 마찬가지로 다양한 지표로 모델의 정확성을 측정하게 됨.

그중 일부인 y의 테스트데이터와 예측데이터간의 관계 지표로 활용되는 
r2_score를 학습

또한, loss 에서 사용하던 mse 외에도 mae, rmse 등 다양한 지표가 존재함
'''
# 1. 데이터
# train, test로 일부 데이터는 훈련용으로 사용함
# model 성능 향상을 위함
# default shuffle = true
x = np.array(range(100))
y = np.array(range(1, 101))

x_train, x_test, y_train, y_test = train_test_split(x, y,
      test_size=0.3, shuffle=True, random_state=66)

print(x_test) 
# [47 64 55 20 62 78  6 22 97 59 70 57 80 67 14 48 29 86 32 35 21 
# 39 76  7  96 81 82 75 90 24]
print(y_test)
# [48 65 56 21 63 79  7 23 98 60 71 58 81 68 15 49 30 87 33 36 22 
# 40 77  8  97 82 83 76 91 25]

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
model.fit(x_train, y_train, epochs=100, batch_size=1)

# 4. 평가 예측
loss = model.evaluate(x_test, y_test)
print('loss : ', loss)

y_predict = model.predict([x_test])
print('100의 예측값 : ', y_predict)

r2 = r2_score(y_test, y_predict)
print('R^2 score : ', r2)
# 기존의 모델에 r2_score 지표를 추가, 결과 R^2 score :  0.9988562899295829
# r2_score는 1에 가까울 수록 모델이 정확함을 나타내는 지표이다.


# # 5. 시각화
# y_predict = model.predict(x)
# plt.scatter(x,y)
# plt.plot(x,y_predict, color='red')
# plt.show()

"""
epo = 100
loss :  6.254519124837543e-10
100의 예측값 :  [[10.       ]
 [95.00004  ]
 [ 5.9999986]
 [ 7.       ]
 [54.000027 ]
 [43.000015 ]
 [ 1.9999964]
 [75.00003  ]
 [90.00003  ]
 [70.000046 ]
 [27.000013 ]
 [20.000002 ]
 [28.000011 ]
 [31.000015 ]
 [68.00004  ]
 [52.000023 ]
 [82.00003  ]
 [47.000027 ]
 [40.000027 ]
 [60.000034 ]
 [51.000023 ]
 [87.000046 ]
 [96.000046 ]
 [89.00003  ]
 [17.000004 ]
 [ 4.9999986]
 [16.000004 ]
 [35.00002  ]
 [25.00001  ]
 [26.000013 ]]
R^2 score :  0.9988562899295829
"""
