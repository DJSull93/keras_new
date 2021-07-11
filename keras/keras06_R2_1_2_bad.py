# 과제 1 0709
# 1. R2 score를 음수가 아닌 0.5 이하로 만들어라.
# 2. 데이터 건들지 마
# 3. 레이어는 인풋 아웃풋 포함 6개 이상
# 4. batch_size = 1 고정
# 5. epoch 는 100 이상
# 6. 히든레이어의 노드는 10개 이상 1000개 이하
# 7. train 70%

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score

# 1. 데이터
# train, test로 일부 데이터는 훈련용으로 사용함
# model 성능 향상을 위함
# default shuffle = true
x = np.array(range(100))
y = np.array(range(1, 101))

x_train, x_test, y_train, y_test = train_test_split(x, y,
      test_size=0.3, shuffle=True)

print(x_test) 
# [47 64 55 20 62 78  6 22 97 59 70 57 80 67 14 48 29 86 32 35 21 
# 39 76  7  96 81 82 75 90 24]
print(y_test)
# [48 65 56 21 63 79  7 23 98 60 71 58 81 68 15 49 30 87 33 36 22 
# 40 77  8  97 82 83 76 91 25]

# 2. 모델 구성
model = Sequential()
model.add(Dense(1000, input_dim=1))
model.add(Dense(1000))
model.add(Dense(1000))
model.add(Dense(1000))
model.add(Dense(10))
model.add(Dense(1000))
model.add(Dense(1000))
model.add(Dense(1000))
model.add(Dense(1))

# 3. 컴파일, 훈련
model.compile(loss='mse', optimizer='adam')
model.fit(x_train, y_train, epochs=120, batch_size=1)

# 4. 평가 예측
loss = model.evaluate(x_test, y_test)
print('loss : ', loss)

y_predict = model.predict([x_test])
print('100의 예측값 : ', y_predict)

r2 = r2_score(y_test, y_predict)
print('R^2 score : ', r2)

# # 5. 시각화
# y_predict = model.predict(x)
# plt.scatter(x,y)
# plt.plot(x,y_predict, color='red')
# plt.show()

"""
loss :  307.5365295410156
100의 예측값 :  [[42.31614 ]
 [47.438477]
 [63.271885]
 [49.301243]
 [71.188194]
 [38.125   ]
 [78.17332 ]
 [79.57071 ]
 [79.104546]
 [74.913574]
 [62.80629 ]
 [57.683525]
 [53.95814 ]
 [50.232655]
 [57.217808]
 [66.06571 ]
 [59.080578]
 [69.79119 ]
 [78.63914 ]
 [68.86022 ]
 [82.83025 ]
 [81.89871 ]
 [82.3643  ]
 [58.149303]
 [53.026867]
 [61.40891 ]
 [69.32591 ]
 [40.919094]
 [41.384747]
 [60.47775 ]]
R^2 score :  0.6164298682983269
"""
