from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
import numpy as np
import matplotlib.pyplot as plt

x = np.array([1,2,3])
y = np.array([1,2,3])


model = Sequential()
model.add(Dense(1, input_dim=1))

model.compile(loss='mse', optimizer='adam')

model.fit(x, y, epochs=10000, batch_size=1)

loss = model.evaluate(x, y)

print('loss : ', loss)

result = model.predict([4])

print('4의 예측값 : ', result)