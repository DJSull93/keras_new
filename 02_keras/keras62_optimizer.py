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
# loss : 1.998802900314331 y_pred : [[13.241291]] time :  3.5485124588012695
# opti = Adagrad(lr=0.001) # default : 0.01
# loss : 1.159216046333313 y_pred : [[11.140946]] time :  3.1875057220458984
# opti = Adamax(lr=0.001) # default : 0.002
# loss : 1.823979377746582 y_pred : [[10.051725]] time :  3.3704347610473633
# opti = Adadelta(lr=0.001) # default : 1.0
# loss : 1.1149882078170776 y_pred : [[11.806165]] time :  3.447190761566162
# opti = RMSprop(lr=0.001) # default : 0.001
# loss : 5.97413969039917 y_pred : [[7.3476467]] time :  5.644767999649048
# opti = SGD(lr=0.001) # default : 0.01
# loss : 1.8134666681289673 y_pred : [[9.939421]] time :  2.9710118770599365
# opti = Nadam(lr=0.001) # default : 0.002
# loss : 4.703295707702637 y_pred : [[15.112229]] time :  8.892017841339111

model.compile(loss='mse', optimizer=opti, metrics=['mae'])
import time
start_time = time.time()
model.fit(x, y, epochs=100, batch_size=1)
end_time = time.time() - start_time

# 4. predict eval
loss, mse = model.evaluate(x, y)
y_pred = model.predict([11])

print('loss :',loss  ,'y_pred :', y_pred, 'time : ', end_time)


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

# optimizer 일정한 변화가 없으면 값을 줄임 : simmilar with es, mcp ; callback!
