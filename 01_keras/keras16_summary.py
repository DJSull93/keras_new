from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Activation
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
model = Sequential()
model.add(Dense(3, input_dim=(5)))
model.add(Dense(4))
model.add(Dense(10))
model.add(Dense(2))

model.summary()



