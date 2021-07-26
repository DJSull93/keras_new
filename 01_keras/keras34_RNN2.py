import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, SimpleRNN

# 1. data

x = np.array([[1,2,3], [2,3,4], [3,4,5], [4,5,6]]) # (4, 3) 
y = np.array([4,5,6,7]) # (4, )

# print(x.shape, y.shape)

x = x.reshape(4, 3, 1) # (batch_size, timesteps, feature)

# 2. model

model = Sequential()
# model.add(SimpleRNN(units=10, activation='relu', input_shape=(3, 1)))
model.add(SimpleRNN(10, activation='relu', input_length=3, input_dim=1))
# input_shape = input(length=timestep, dim=feature)
model.add(Dense(10, activation='relu'))
model.add(Dense(1))

# 3. compile train

model.compile(loss='mse', optimizer='adam')

from tensorflow.keras.callbacks import EarlyStopping
es = EarlyStopping(monitor='loss', patience=50, mode='min', verbose=1)

model.fit(x, y, epochs=10000, batch_size=1, callbacks=[es])

# 4. pred eval
x_input = np.array([5, 6, 7]).reshape(1, 3, 1)
y_pred = model.predict(x_input)
print(y_pred) # [[8.]] epochs = 851

# model.summary()
