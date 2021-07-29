import numpy as np
from tensorflow.python.keras.layers.core import Dropout

# 1. data
x_train = np.load('./_save/_NPY/k59_cd_x_train.npy')
x_test = np.load('./_save/_NPY/k59_cd_x_test.npy')
y_train = np.load('./_save/_NPY/k59_cd_y_train.npy')
y_test = np.load('./_save/_NPY/k59_cd_y_test.npy')

# 2. model
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Conv2D, Flatten, MaxPooling2D, GlobalAveragePooling2D

model = Sequential()

model.add(Conv2D(filters = 32, kernel_size=(3,3), input_shape =(150,150,3), activation= 'relu'))
model.add(Conv2D(filters = 32, kernel_size=(3,3), activation= 'relu'))
model.add(MaxPooling2D(2,2))
model.add(Conv2D(filters = 16, kernel_size=(2,2), activation= 'relu'))
model.add(Conv2D(filters = 16, kernel_size=(2,2), activation= 'relu'))
model.add(MaxPooling2D(2,2))
# model.add(Conv2D(filters = 32, kernel_size=(3,3), activation= 'relu'))
# model.add(Conv2D(filters = 64, kernel_size=(2,2), activation= 'relu'))
# model.add(Conv2D(filters = 64, kernel_size=(3,3), activation= 'relu'))
# model.add(MaxPooling2D(2,2))
model.add(Flatten())
model.add(Dense(111, activation= 'relu'))
# model.add(Dropout(0.1))
# model.add(Dense(64, activation= 'relu'))
model.add(Dense(64, activation= 'relu'))
model.add(Dense(2, activation= 'sigmoid'))

# 3. compile train
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['acc'])

from tensorflow.keras.callbacks import EarlyStopping
es = EarlyStopping(monitor='val_acc', patience=10, mode='auto', verbose=1)

# hist = model.fit_generator(xy_train, epochs=50,
#  steps_per_epoch=32,
#  validation_data=xy_test,
#  validation_steps=4,
#  callbacks=[es]) # 32 -> 160/5

hist = model.fit(x_train, y_train, epochs=500,
                callbacks=[es],
                validation_split=0.05,
                steps_per_epoch=32,
                validation_steps=1)

acc = hist.history['acc']
val_acc = hist.history['val_acc']
loss = hist.history['loss']
val_loss = hist.history['val_loss']

# visualize upper data

# print('val_acc : ',val_acc[:-1])

loss = model.evaluate(x_test, y_test)
print('acc : ',acc[-10])
print('val_acc : ',val_acc[-10])
print('loss : ',loss[-10])

'''
acc :  0.9811215996742249
val_acc :  0.5755305886268616
loss :  2.8827121257781982

acc :  0.9895890951156616
val_acc :  0.5493133664131165
loss :  2.825904130935669

acc :  0.9951341152191162
val_acc :  0.5910224318504333
loss :  0.6094908714294434
'''