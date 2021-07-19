# example cifar10


from tensorflow.keras.datasets import cifar100

import numpy as np
import matplotlib.pyplot as plt

# 1. data
(x_train, y_train), (x_test, y_test) = cifar100.load_data() 

# preproccesing!!!!!!!!!!!

x_train = x_train.reshape(50000, 32*32*3) # (50000, 32, 32, 3)
x_test = x_test.reshape(10000, 32*32*3) # (10000, 32, 32, 3)

# print(x_train.shape, x_test.shape) # (50000, 3072) (10000, 3072)

from sklearn.preprocessing import MinMaxScaler, StandardScaler, RobustScaler, MaxAbsScaler, QuantileTransformer, PowerTransformer
scaler = StandardScaler()
# scaler.fit(x_train) 
# x_train = scaler.transform(x_train) 
x_train = scaler.fit_transform(x_train)
x_test = scaler.transform(x_test) 

x_train = x_train.reshape(50000, 32, 32, 3) # (50000, 32, 32, 3)
x_test = x_test.reshape(10000, 32, 32, 3) # (10000, 32, 32, 3)

# from sklearn.preprocessing import OneHotEncoder
# one = OneHotEncoder()
# y_train = y_train.reshape(-1,1)
# y_test = y_test.reshape(-1,1)
# one.fit(y_train)
# y_train = one.transform(y_train).toarray() # (50000, 100)
# y_test = one.transform(y_test).toarray() # (10000, 100)

from tensorflow.keras.utils import to_categorical
y_train = to_categorical(y_train)
y_test = to_categorical(y_test)

# 2. model
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Conv2D, Flatten, MaxPool2D

model = Sequential()
model.add(Conv2D(filters=128, kernel_size=(2, 2),                          
                        padding='valid', activation='relu', 
                        input_shape=(32, 32, 3))) 
model.add(Conv2D(128, (2, 2), padding='same', activation='relu'))                   
model.add(MaxPool2D())     

model.add(Conv2D(128, (2, 2), padding='valid', activation='relu'))                   
model.add(Conv2D(128, (2, 2), padding='same', activation='relu'))    
model.add(MaxPool2D())       

model.add(Conv2D(64, (2, 2), padding='valid', activation='relu'))                   
model.add(Conv2D(64, (2, 2), padding='same', activation='relu'))
model.add(MaxPool2D())       

model.add(Flatten())                                              
model.add(Dense(128, activation='relu'))
model.add(Dense(128, activation='relu'))
model.add(Dense(128, activation='relu'))
model.add(Dense(100, activation='softmax'))

# 3. comple fit // metrics 'acc'
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics='acc')

from tensorflow.keras.callbacks import EarlyStopping
es = EarlyStopping(monitor='val_loss', patience=7, mode='min', verbose=1)

import time 
start_time = time.time()
hist = model.fit(x_train, y_train, epochs=100, batch_size=64, verbose=1,
    validation_split=0.25, callbacks=[es])
end_time = time.time() - start_time

# 4. predict eval -> no need to

loss = model.evaluate(x_test, y_test, batch_size=64)
print("======================================")
print("total time : ", end_time)
print('loss : ', loss[0])
print('acc : ', loss[1])

# 5. plt visualize
import matplotlib.pyplot as plt
plt.figure(figsize=(9,5))

# 1 
plt.subplot(2,1,1)
plt.plot(hist.history["loss"], marker='.', c='red', label='loss')
plt.plot(hist.history["val_loss"], marker='.', c='blue', label='val_loss')
plt.grid()
plt.title("loss")
plt.ylabel("loss")
plt.xlabel("epochs")
plt.legend(loc='upper right')

# 2
plt.subplot(2,1,2)
plt.plot(hist.history["acc"])
plt.plot(hist.history["val_acc"])
plt.grid()
plt.title("acc")
plt.ylabel("acc")
plt.xlabel("epochs")
plt.legend(['acc', 'val_acc'])

plt.show()

'''
loss :  4.9466118812561035
acc :  0.4169999957084656

RobustScaler, val 0.005
total time :  77.44221878051758
loss :  5.040241241455078
acc :  0.4172999858856201

standard, val 0.2
total time :  58.47070121765137
loss :  3.875128984451294
acc :  0.3799999952316284

standard, batch = 64
total time :  111.92880010604858
loss :  3.409672975540161
acc :  0.3424000144004822

standard, val = 0.25, model edit


'''
