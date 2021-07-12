from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score

# 완성 한 뒤, 출력 결과 스샷

x = np.array([1,2,3,4,5])
y = np.array([1,2,4,3,5])

model = Sequential()
model.add(Dense(10, input_dim=1))
model.add(Dense(10))
model.add(Dense(10))
model.add(Dense(10))
model.add(Dense(10))
model.add(Dense(10))
model.add(Dense(10))
model.add(Dense(10))
model.add(Dense(10))
model.add(Dense(1))

x_train, x_test, y_train, y_test = train_test_split(x, y,
      train_size=0.6, shuffle=True)

model.compile(loss='mse', optimizer='adam')
model.fit(x_train, y_train, epochs=1000, batch_size=1)

loss = model.evaluate(x_test, y_test)
print('loss : ', loss)

y_predict = model.predict([x_test])
print('x_test의 예측값 : ', y_predict)

r2 = r2_score(y_test, y_predict)
print('R^2 score : ', r2)

'''
epo = 5000
loss :  0.3800278306007385
6의 예측값 :  [[1.2085181]
 [2.1048973]
 [3.0012765]
 [3.8976555]
 [4.7940345]]
R^2 score :  0.809986093514469
'''

# 과제 2 0709
# R2를 0.9 이상으로 올려라
# 단톡방에 0.81 이상, top 123 커피
