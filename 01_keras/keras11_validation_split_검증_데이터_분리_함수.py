from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split

# 1. 데이터
x = np.array([1,2,3,4,5,6,7,8,9,10,11,12,13,14,15])
y = np.array([1,2,3,4,5,6,7,8,9,10,11,12,13,14,15])

# train_test_split 후 fit에서 split 활용

x_train, x_test, y_train, y_test = train_test_split(x, y,
 random_state=66, train_size=0.8)

'''
하지만 앞서 train_test_split 두번 하는것 조차 번거롭다.

'''

# 2. 모델 구성
model = Sequential()
model.add(Dense(5, input_dim=1))
model.add(Dense(5))
model.add(Dense(5))
model.add(Dense(5))
model.add(Dense(5))
model.add(Dense(1))

# 3. 컴파일, 훈련
model.compile(loss='mse', optimizer='adam', metrics=['mse'])
# model.fit(x_train, y_train, epochs=80,
#  verbose=0, batch_size=1, validation_data=(x_val, y_val))
# 소수점 절삭함
model.fit(x_train, y_train, epochs=80,
 verbose=1, batch_size=1, validation_split=0.3, shuffle=True)

'''
이처럼 fit에서 자체적으로 검증셋을 분리하는 파라미터가 존재하며, 
사용하면 된다. 
'''

# 4. 평가 예측
loss = model.evaluate(x_test, y_test)
print('loss : ', loss)
result = model.predict([x_test])
print('x_test의 예측값 : ', result)


# 5. 시각화
# y_predict = model.predict(x)
# plt.scatter(x,y)
# plt.plot(x,y_predict, color='red')
# plt.show()

"""
loss :  [0.0003325627476442605, 0.0003325627476442605]
x_test의 예측값 :  [[ 7.0121794]
 [ 2.027588 ]
 [13.990605 ]]
"""