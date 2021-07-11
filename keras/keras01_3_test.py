from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
import numpy as np
from tensorflow.python.keras.engine import input_layer

x = np.array([1,2,3,4,5,6,7,8,9,10])
y = np.array([2,1,4,3,5,6,7,8,9,10])

model = Sequential()
model.add(Dense(1, input_dim=1))

model.compile(loss='mse', optimizer='adam')
model.fit(x, y, epochs=3000, batch_size=1)

loss = model.evaluate(x, y)
print(loss)
result = model.predict([11])
print(result)