from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import r2_score

# 완성 한 뒤, 출력 결과 스샷

x = np.array([1,2,3,4,5])
y = np.array([1,2,4,3,5])
model = Sequential()
model.add(Dense(1, input_dim=1))
model.compile(loss='mse', optimizer='adam')
model.fit(x, y, epochs=200, batch_size=1)

loss = model.evaluate(x, y)
print('loss : ', loss)
y_predict = model.predict([6])
print('6의 예측값 : ', y_predict)

r2 = r2_score(y, y_predict)
print('R^2 score : ', r2)

# 01_1과 유사하나, 데이터셋의 구조가 어긋나 있을때 얼마나 훈련 결과가 엉망이 되는지 보여줌
# -> 정제된 데이터 셋의 준비가 중요함!!!

# Preprocessing = 전처리 