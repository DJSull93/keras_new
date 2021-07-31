
import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, SimpleRNN, LSTM, Bidirectional

# 1. data

x = np.array([[1,2,3], [2,3,4], [3,4,5], [4,5,6]]) # (4, 3) 
y = np.array([4,5,6,7]) # (4, )

# print(x.shape, y.shape)

x = x.reshape(4, 3, 1) # (batch_size, timesteps, feature)

# 2. model

model = Sequential()
# model.add(SimpleRNN(units=10, activation='relu', input_shape=(3, 1)))
model.add(LSTM(units=10, activation='relu', input_shape=(3, 1), return_sequences=True))
model.add(LSTM(units=10, activation='relu'))
model.add(Dense(5, activation='relu'))
model.add(Dense(1))
'''
# 3. compile train

model.compile(loss='mse', optimizer='adam')

from tensorflow.keras.callbacks import EarlyStopping
es = EarlyStopping(monitor='loss', patience=20, mode='min', verbose=1)

model.fit(x, y, epochs=1000, batch_size=1, callbacks=[es])

# 4. pred eval
x_input = np.array([5, 6, 7]).reshape(1, 3, 1)
y_pred = model.predict(x_input)
print(y_pred) # [[8.006084]] epochs = 1745
'''
model.summary()

# bidirec
# _________________________________________________________________
# Layer (type)                 Output Shape              Param #
# =================================================================
# lstm (LSTM)                  (None, 3, 10)             480
# _________________________________________________________________
# bidirectional (Bidirectional (None, 20)                1680
# _________________________________________________________________
# dense (Dense)                (None, 5)                 105
# _________________________________________________________________
# dense_1 (Dense)              (None, 1)                 6
# =================================================================
# Total params: 2,271
# Trainable params: 2,271
# Non-trainable params: 0
# _________________________________________________________________

# non bidirec
# _________________________________________________________________
# Layer (type)                 Output Shape              Param #
# =================================================================
# lstm (LSTM)                  (None, 3, 10)             480
# _________________________________________________________________
# lstm_1 (LSTM)                (None, 10)                840
# _________________________________________________________________
# dense (Dense)                (None, 5)                 55
# _________________________________________________________________
# dense_1 (Dense)              (None, 1)                 6
# =================================================================
# Total params: 1,381
# Trainable params: 1,381
# Non-trainable params: 0
# _________________________________________________________________