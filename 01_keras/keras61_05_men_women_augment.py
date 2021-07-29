# train data augment-> 120% of all dataset
# compare with banilar
# save_dir -> temp and delete

# train data augment-> 120% of all dataset
# compare with banilar
# save_dir -> temp and delete
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import numpy as np

# 1. data
train_datagen = ImageDataGenerator(
    rescale=1./255,
    horizontal_flip=True,
    vertical_flip=True,
    width_shift_range=0.1,
    height_shift_range=0.1,
    rotation_range=5,
    zoom_range=1.2,
    shear_range=0.7,
    fill_mode='nearest',
    validation_split=0.25
)

test_datagen = ImageDataGenerator(rescale=1./255)

x_train = np.load('./_save/_NPY/k59_mw_x_train.npy')
x_test = np.load('./_save/_NPY/k59_mw_x_test.npy')
y_train = np.load('./_save/_NPY/k59_mw_y_train.npy')
y_test = np.load('./_save/_NPY/k59_mw_y_test.npy')

# print(xy_train[0][0].shape) # (8005, 150, 150, 3)
# print(xy_train[0][1].shape) # (8005, 2)

augment_size = 400

randidx = np.random.randint(x_train.shape[0], size=augment_size) # take 40000 feature from train in random

x_argmented = x_train[randidx].copy()
y_argmented = y_train[randidx].copy()

x_argmented = x_argmented.reshape(x_argmented.shape[0], 150, 150, 3) # (32, 150, 150, 3)
x_train = x_train.reshape(x_train.shape[0], 150, 150, 3) # (160, 150, 150, 3)
x_test = x_test.reshape(x_test.shape[0], 150, 150, 3) # (120, 150, 150, 3)

x_argmented = train_datagen.flow(x_argmented, 
                                np.zeros(augment_size),
                                batch_size=augment_size,
                                shuffle=False).next()[0]

x_train = np.concatenate((x_train, x_argmented)) # (100000, 28, 28, 1) 
y_train = np.concatenate((y_train, y_argmented)) # (100000,)

# 2. model
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Conv2D, Flatten, MaxPooling2D

model = Sequential()
model.add(Conv2D(filters = 32, kernel_size=(3,3), input_shape =(150,150,3), activation= 'relu'))
model.add(Conv2D(filters = 32, kernel_size=(3,3),  activation= 'relu'))
model.add(MaxPooling2D(2,2))
model.add(Conv2D(filters = 128, kernel_size=(3,3),  activation= 'relu'))
model.add(Conv2D(filters = 128, kernel_size=(3,3),  activation= 'relu'))
model.add(MaxPooling2D(2,2))
model.add(Flatten())
model.add(Dense(128, activation= 'relu'))
model.add(Dense(32, activation= 'relu'))
model.add(Dense(1, activation= 'sigmoid'))

# 3. compile train
model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['acc'])

from tensorflow.keras.callbacks import EarlyStopping
es = EarlyStopping(monitor='val_acc', patience=10, mode='auto', verbose=1)

import time 

start_time = time.time()
hist = model.fit(x_train, y_train, epochs=10000, verbose=2,
    validation_split=0.2, callbacks=[es], steps_per_epoch=32,
                validation_steps=4)
end_time = time.time() - start_time

# 4. predict eval -> no need to

acc = hist.history['acc']
val_acc = hist.history['val_acc']
loss = hist.history['loss']
val_loss = hist.history['val_loss']

loss = model.evaluate(x_test, y_test)
print('acc : ',acc[-10])
print('val_acc : ',val_acc[-10])
# print('loss : ',loss[-10])
print('val_loss : ',val_loss[-10])                             

'''
with flow
acc :  0.887201726436615
val_acc :  0.5580589175224304

without flow
acc :  0.938035249710083
val_acc :  0.5513078570365906
'''