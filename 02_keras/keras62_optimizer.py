import numpy as np
from tensorflow.keras import models

# 1. data

x = np.array([1,2,3,4,5,6,7,8,9,10])
y = np.array([1,3,5,4,5,6,7,11,9,10])


# 2. model

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Conv2D, Flatten, MaxPooling2D

model = Sequential()
model.add(Dense(1000, input_dim=1))
model.add(Dense(1000))
model.add(Dense(1000))
model.add(Dense(1000))
model.add(Dense(100))
model.add(Dense(1))

# 3. compile train
from tensorflow.keras.optimizers import Adam, Adagrad, Adamax, Adadelta
from tensorflow.keras.optimizers import RMSprop, SGD, Nadam

opti = Adam(lr=0.001) # default : 0.001
# loss : 1.404402494430542 y_pred : [[10.392479]]
# opti = Adagrad(lr=0.001) # default : 0.01
# loss : 1.1597635746002197 y_pred : [[11.124821]]
# opti = Adamax(lr=0.001) # default : 0.002
# loss : 1.0801925659179688 y_pred : [[11.963009]]
# opti = Adadelta(lr=0.001) # default : 1.0
# loss : 1.1143584251403809 y_pred : [[11.816234]]
# opti = RMSprop(lr=0.001) # default : 0.001
# loss : 1.6003001928329468 y_pred : [[10.419261]]
# opti = SGD(lr=0.001) # default : 0.01
# loss : 1.0462414026260376 y_pred : [[11.451179]]
# opti = Nadam(lr=0.001) # default : 0.002
# loss : 1.3385355472564697 y_pred : [[10.475482]]

model.compile(loss='mse', optimizer=opti, metrics=['mae'])
model.fit(x, y, epochs=100, batch_size=1)

# 4. predict eval
loss, mse = model.evaluate(x, y)
y_pred = model.predict([11])

print('loss :',loss  ,'y_pred :', y_pred)


'''
adam default
loss : 1.9365031675988575e-06
y_pred : [[11.002652]]

adam 0.01
loss : 1.1817896366119385
y_pred : [[10.877981]]

adam 0.001
loss : 1.0525476932525635
y_pred : [[11.283892]]
'''