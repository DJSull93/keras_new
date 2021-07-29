# OpenCV -> 찾아볼것

import numpy as np
from tensorflow.keras.preprocessing.image import ImageDataGenerator

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
    fill_mode='nearest'
)
# 통상적으로 테스트셋은 증폭하지 않음
test_datagen = ImageDataGenerator(rescale=1./255)

# train 내부 폴더 -> y의 라벨로 자동 지정됨
# x, y 자동 생성
xy_train = train_datagen.flow_from_directory(
    '../_data/brain/train',
    target_size=(150, 150),
    batch_size=5,
    class_mode='binary',
    shuffle=True
)
# Found 160 images belonging to 2 classes.

xy_test = test_datagen.flow_from_directory(
    '../_data/brain/test',
    target_size=(150, 150),
    batch_size=5,
    class_mode='binary',
    shuffle=True
)
# Found 120 images belonging to 2 classes.
# <tensorflow.python.keras.preprocessing.image.DirectoryIterator object at 0x000001E1F3A15190>
# print(xy_test)

# print(xy_train)
# print(xy_train[0])
# print(xy_train[0][0].shape) # (5, 150, 150, 3) / 5 set (batch__size)
# print(xy_train[0][1].shape) # (5,)

# total160, batch=5 : 32 set
# print(xy_train[31][1].shape) # (5,)

# print(type(xy_train)) # <class 'tensorflow.python.keras.preprocessing.image.DirectoryIterator'>
# print(type(xy_train[0])) # <class 'tuple'>
# print(type(xy_train[0][0])) # <class 'numpy.ndarray'>
# print(type(xy_train[0][1])) # <class 'numpy.ndarray'>

# 2. model
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Conv2D, Flatten, MaxPool2D

model = Sequential()
model.add(Conv2D(16, (2,2), padding='same', activation='relu', input_shape=(150,150,3)))
# model.add(Conv2D(32, (2,2), padding='same', activation='relu'))
# model.add(MaxPool2D())
model.add(Flatten())
model.add(Dense(32, activation='relu'))
model.add(Dense(1, activation='sigmoid'))

# 3. compile train
model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['acc'])

from tensorflow.keras.callbacks import EarlyStopping
es = EarlyStopping(monitor='val_loss', patience=10, mode='min', verbose=1)

# model.fit(x_train, y_train)
hist = model.fit_generator(xy_train, epochs=50,
 steps_per_epoch=32,
 validation_data=xy_test,
 validation_steps=4,
 callbacks=[es]) # 32 -> 160/5

acc = hist.history['acc']
val_acc = hist.history['val_acc']
loss = hist.history['loss']
val_loss = hist.history['val_loss']

# visualize upper data

print('acc : ',acc[-1])
print('val_acc : ',val_acc[:-1])

'''
acc :  0.768750011920929
'''
