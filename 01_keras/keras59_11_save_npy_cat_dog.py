# use categorical but activation is sigmoid

import time
import numpy as np
from tensorflow.keras.preprocessing.image import ImageDataGenerator

start_time = time.time()

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

xy_train = train_datagen.flow_from_directory(
    '../_data/catdog/training_set',
    target_size=(150, 150),
    batch_size=8100,
    class_mode='categorical',
    shuffle=True
)

xy_test = test_datagen.flow_from_directory(
    '../_data/catdog/test_set',
    target_size=(150, 150),
    batch_size=8100,
    class_mode='categorical',
    shuffle=True
)

np.save('./_save/_NPY/k59_cd_x_train', arr=xy_train[0][0])
np.save('./_save/_NPY/k59_cd_x_test', arr=xy_test[0][0])
np.save('./_save/_NPY/k59_cd_y_train', arr=xy_train[0][1])
np.save('./_save/_NPY/k59_cd_y_test', arr=xy_test[0][1])

end_time = time.time() - start_time
print(end_time)

# print(xy_train[0][0]) # 
# print(xy_train[0][1]) # 
# print(xy_train[0][0].shape) # (8005, 150, 150, 3)
# print(xy_train[0][1].shape) # (8005, 2)

# total time : 90.78798818588257