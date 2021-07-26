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
y_predict = model.predict([x_test])
print('100의 예측값 : ', y_predict)

loss = model.evaluate(x_test, y_test)
print('loss : ', loss)

r2 = r2_score(y_test, y_predict)
print('R^2 score : ', r2)

# # 5. 시각화
# y_predict = model.predict(x)
# plt.scatter(x,y)
# plt.plot(x,y_predict, color='red')
# plt.show()

"""
epo = 12-
loss :  865.321044921875
R^2 score :  0.0017011482266989786
100의 예측값 :  [[111.13347   ]
 [  7.0562816 ]
 [  5.350097  ]
 [107.720085  ]
 [ 83.833275  ]
 [-45.8353    ]
 [  0.23310602]
 [ 70.18435   ]
 [-20.243145  ]
 [ 77.01128   ]
 [ 39.47396   ]
 [ 15.587001  ]
 [-27.067295  ]
 [104.30964   ]
 [ -6.5923867 ]
 [ 92.36546   ]
 [ 36.06311   ]
 [-42.423042  ]
 [114.539856  ]
 [ 85.54101   ]
 [ 25.823498  ]
 [-23.655075  ]
 [  3.6448894 ]
 [ 42.890804  ]
 [ 13.882197  ]
 [-52.660053  ]
 [-15.123513  ]
 [-21.949303  ]
 [ 37.768208  ]
 [ 54.825943  ]]
"""
