'''
캐글 뇌 CT사진 데이터 

ImageDataGenerator로 호출하여 x,y결합된 데이터셋 반환

'''


# OpenCV -> 찾아볼것

import numpy as np
from tensorflow.keras.preprocessing.image import ImageDataGenerator

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


