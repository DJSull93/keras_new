from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
import numpy as np
import matplotlib.pyplot as plt

# 1. 데이터
# train, test로 일부 데이터는 훈련용으로 사용함
# model 성능 향상을 위함
x = np.array(range(100))
y = np.array(range(1, 101))

# shuffle 활용해서 x,y 동등하게 랜덤, 7:3 분할
# np.random.shuffle 활용, 매개 함수 s 할당해서 인덱스로 활용
# x, y 각 두 함수가 동일한 인덱스가 셔플됨
s = np.arange(x.shape[0])
np.random.shuffle(s)
x1 = x[s]
y1 = y[s]
x_train = x1[:70]
y_train = y1[:70]
x_test = x1[-30:]
y_test = y1[-30:]
print(x_train)
'''
 [45 37  7 36 82 43 75 85 20 86 95 11 19 89 76 51 93 52 66 99 38 
 8 21 42
 57 32 78 62  1 60 55 47  9 28 49 39 61 34 30  4 46 27 83 71 70 
84 48  5
 40  3 73  6 12 92 31 87 81 64 26 24 35  2 98 79 41 18 91 22 69 
58]
'''
print(y_train)
'''
[ 46  38   8  37  83  44  76  86  21  87  96  12  20  90  77  52  94  53
  67 100  39   9  22  43  58  33  79  63   2  61  56  48  10  29  50  40
  62  35  31   5  47  28  84  72  71  85  49   6  41   4  74   7  13  93
  32  88  82  65  27  25  36   3  99  80  42  19  92  23  70  59]
'''
print(x_test) 
'''
[67 13 16 44 54 72 15 90 29 59 50 33 96 94 88 74 77 14 10 68 97 
63 65 17
  0 53 25 23 56 80]
'''
print(y_test)
'''
[68 14 17 45 55 73 16 91 30 60 51 34 97 95 89 75 78 15 11 69 98 
64 66 18
  1 54 26 24 57 81]
'''


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
'''
y_predict = model.predict(x)

# 5. 시각화
plt.scatter(x,y)
plt.plot(x,y_predict, color='red')
plt.show()

"""

"""
'''