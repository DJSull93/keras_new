from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import Dense, Activation, Input
import numpy as np
from sklearn.metrics import r2_score
from sklearn.model_selection import train_test_split

# 1. data 
x = np.array ([range(100), range(301, 401), range(1, 101), 
               range(100), range(401, 501)])
x = np.transpose(x) # (100, 5)

y = np.array([range(701, 801), range(101, 201)]) 
y = np.transpose(y) # (100, 2)

# 2. model

input1 = Input(shape=(5,))
dense1 = Dense(3)(input1)
dense2 = Dense(4)(dense1)
dense3 = Dense(10)(dense2)
output1 = Dense(2)(dense3)

model = Model(inputs=input1, outputs=output1)

# model = Sequential()
# model.add(Dense(3, input_dim=(5)))
# model.add(Dense(4))
# model.add(Dense(10))
# model.add(Dense(2))
model.summary()

# 3. compile, train

# 4. evaluate, predict

