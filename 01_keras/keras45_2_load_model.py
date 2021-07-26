# example cifar100

from tensorflow.keras.datasets import cifar100, mnist

import numpy as np
import matplotlib.pyplot as plt

# 1. data
(x_train, y_train), (x_test, y_test) = mnist.load_data() # (60000, 28, 28) (60000,) (10000, 28, 28) (10000,)

x_train = x_train.reshape(60000, 28*28*1)
x_test = x_test.reshape(10000, 28*28*1) 

from sklearn.preprocessing import MinMaxScaler, StandardScaler, RobustScaler, MaxAbsScaler, QuantileTransformer, PowerTransformer
scaler = MinMaxScaler()
x_train = scaler.fit_transform(x_train)
x_test = scaler.transform(x_test) 

x_train = x_train.reshape(60000, 28, 28, 1) 
x_test = x_test.reshape(10000, 28, 28, 1) 

from tensorflow.keras.utils import to_categorical
y_train = to_categorical(y_train)
y_test = to_categorical(y_test)

# 2. model
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import Dense, Conv2D, Flatten, MaxPool2D, Dropout
'''
model = Sequential()
model.add(Conv2D(filters=32, kernel_size=(2, 2),                          
                        padding='same', activation='relu', 
                        input_shape=(28, 28, 1))) 
model.add(Dropout(0.1))
model.add(Conv2D(32, (2, 2), padding='same', activation='relu'))                   
model.add(MaxPool2D())     

model.add(Conv2D(64, (2, 2), padding='same', activation='relu'))                   
model.add(Dropout(0.1))
model.add(Conv2D(64, (2, 2), padding='same', activation='relu'))    
model.add(MaxPool2D())       

model.add(Conv2D(128, (2, 2), padding='same', activation='relu'))                   
model.add(Dropout(0.1))
model.add(Conv2D(128, (2, 2), padding='same', activation='relu'))
model.add(MaxPool2D())       

model.add(Flatten())
model.add(Dense(512, activation='relu'))
model.add(Dropout(0.1))
model.add(Dense(256, activation='relu'))
model.add(Dropout(0.1))
model.add(Dense(128, activation='relu'))
model.add(Dropout(0.1))
model.add(Dense(10, activation='softmax'))
'''
model = load_model('./_save/keras45_1_save_model.h5')
model.summary()


# 3. compile fit // metrics 'acc'
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics='acc')

from tensorflow.keras.callbacks import EarlyStopping
es = EarlyStopping(monitor='val_loss', patience=5, mode='min', verbose=1)

import time 
start_time = time.time()
hist = model.fit(x_train, y_train, epochs=2, batch_size=128, verbose=2,
    validation_split=0.25, callbacks=[es])
end_time = time.time() - start_time

# 4. predict eval -> no need to

loss = model.evaluate(x_test, y_test, batch_size=128)
print("======================================")
print("total time : ", end_time)
print('loss : ', loss[0])
print('acc : ', loss[1])

# # 5. plt visualize
# import matplotlib.pyplot as plt
# plt.figure(figsize=(9,5))

# # plot 1 
# plt.subplot(2,1,1)
# plt.plot(hist.history["loss"], marker='.', c='red', label='loss')
# plt.plot(hist.history["val_loss"], marker='.', c='blue', label='val_loss')
# plt.grid()
# plt.title("loss")
# plt.ylabel("loss")
# plt.xlabel("epochs")
# plt.legend(loc='upper right')

# # plot 2
# plt.subplot(2,1,2)
# plt.plot(hist.history["acc"])
# plt.plot(hist.history["val_acc"])
# plt.grid()
# plt.title("acc")
# plt.ylabel("acc")
# plt.xlabel("epochs")
# plt.legend(['acc', 'val_acc'])

# plt.show()

'''
none load
total time :  9.25413990020752
loss :  0.036677632480859756
acc :  0.988099992275238

load_model
total time :  9.757547855377197
loss :  0.03656894341111183
acc :  0.9878000020980835
'''
