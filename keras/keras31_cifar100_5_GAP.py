# Global Average Pooling

# example cifar100

from tensorflow.keras.datasets import cifar100

import numpy as np
import matplotlib.pyplot as plt

# 1. data
(x_train, y_train), (x_test, y_test) = cifar100.load_data() 

x_train = x_train.reshape(50000, 32*32*3) # (50000, 32, 32, 3)
x_test = x_test.reshape(10000, 32*32*3) # (10000, 32, 32, 3)

from sklearn.preprocessing import MinMaxScaler, StandardScaler, RobustScaler, MaxAbsScaler, QuantileTransformer, PowerTransformer
scaler = StandardScaler()
# scaler.fit(x_train) 
# x_train = scaler.transform(x_train) 
x_train = scaler.fit_transform(x_train)
x_test = scaler.transform(x_test) 

x_train = x_train.reshape(50000, 32, 32, 3) # (50000, 32, 32, 3)
x_test = x_test.reshape(10000, 32, 32, 3) # (10000, 32, 32, 3)

from tensorflow.keras.utils import to_categorical
y_train = to_categorical(y_train)
y_test = to_categorical(y_test)

# 2. model / importing GAP2D 
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Conv2D, Flatten, MaxPool2D, Dropout, GlobalAveragePooling2D

model = Sequential()
model.add(Conv2D(filters=32, kernel_size=(2, 2),                          
                        padding='same', activation='relu', 
                        input_shape=(32, 32, 3))) 
model.add(Dropout(0.2))
model.add(Conv2D(32, (2, 2), padding='valid', activation='relu'))                   
model.add(MaxPool2D())     

model.add(Conv2D(64, (2, 2), padding='same', activation='relu'))                   
model.add(Dropout(0.2))
model.add(Conv2D(64, (2, 2), padding='valid', activation='relu'))    
model.add(MaxPool2D())  

model.add(Conv2D(128, (2, 2), padding='same', activation='relu'))                   
model.add(Dropout(0.2))
model.add(Conv2D(128, (2, 2), padding='same', activation='relu'))    
model.add(MaxPool2D())       

# model.add(Conv2D(256, (3, 3), padding='same', activation='relu'))                   
# model.add(Dropout(0.2))
# model.add(Conv2D(256, (3, 3), padding='same', activation='relu'))
# model.add(MaxPool2D())       

# model.add(Flatten())                                              
# model.add(Dense(512, activation='relu'))
# model.add(Dropout(0.1))
# model.add(Dense(256, activation='relu'))
# model.add(Dropout(0.1))
# model.add(Dense(128, activation='relu'))
# model.add(Dropout(0.1))

model.add(GlobalAveragePooling2D())
model.add(Dense(100, activation='softmax'))

# 3. comple fit // metrics 'acc'
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics='acc')

from tensorflow.keras.callbacks import EarlyStopping
es = EarlyStopping(monitor='val_loss', patience=10, mode='min', verbose=1)

import time 
start_time = time.time()
hist = model.fit(x_train, y_train, epochs=100, batch_size=256, verbose=2,
    validation_split=0.15, callbacks=[es])
end_time = time.time() - start_time

# 4. predict eval -> no need to

loss = model.evaluate(x_test, y_test, batch_size=256)
print("======================================")
print("total time : ", end_time)
print('loss : ', loss[0])
print('acc : ', loss[1])

# 5. plt visualize
import matplotlib.pyplot as plt
plt.figure(figsize=(9,5))

# plot 1 
plt.subplot(2,1,1)
plt.plot(hist.history["loss"], marker='.', c='red', label='loss')
plt.plot(hist.history["val_loss"], marker='.', c='blue', label='val_loss')
plt.grid()
plt.title("loss")
plt.ylabel("loss")
plt.xlabel("epochs")
plt.legend(loc='upper right')

# plot 2
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
loss :  2.268862247467041
acc :  0.4440999925136566

total time :  156.56675124168396
loss :  2.177590847015381
acc :  0.44670000672340393

total time :  122.86703252792358
loss :  2.1693880558013916
acc :  0.4564000070095062

total time :  158.07011938095093
loss :  2.1041903495788574
acc :  0.47209998965263367
'''
