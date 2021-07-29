# 1. men wemen modeling
# 2. predict with my picture / HW
#  -> D:/_data

# need to traon / test split

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
    fill_mode='nearest',
    validation_split=0.25
)

# 통상적으로 테스트셋은 증폭하지 않음
test_datagen = ImageDataGenerator(rescale=1./255)

xy_train = train_datagen.flow_from_directory(
    '../_data/men_women',
    target_size=(150, 150),
    batch_size=2500,
    class_mode='binary',
    shuffle=True,
    subset='training'
)

xy_test = train_datagen.flow_from_directory(
    '../_data/men_women',
    target_size=(150, 150),
    batch_size=2500,
    class_mode='binary',
    shuffle=True,
    subset='validation'
)

np.save('./_save/_NPY/k59_mw_x_train', arr=xy_train[0][0])
np.save('./_save/_NPY/k59_mw_y_train', arr=xy_train[0][1])
np.save('./_save/_NPY/k59_mw_x_test', arr=xy_test[0][0])
np.save('./_save/_NPY/k59_mw_y_test', arr=xy_test[0][1])

# print(xy_train[0][0]) # 
# print(xy_train[0][1]) # 
# print(xy_train[0][0].shape) # (1920, 450, 450, 3)
# print(xy_train[0][1].shape) # 
