# x_argmented x_train 각 10개 비교 이미지 출력
# 서브플롯 (2, 10, ?) 사용

from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.datasets import fashion_mnist
import numpy as np
from tensorflow.python.keras.backend import zeros

(x_train, y_train), (x_test, y_test) = fashion_mnist.load_data()

train_datagen = ImageDataGenerator(
    rescale=1./255,
    horizontal_flip=True,
    vertical_flip=True,
    width_shift_range=0.1,
    height_shift_range=0.1,
    rotation_range=10,
    zoom_range=0.2,
    shear_range=0.5,
    fill_mode='nearest'
)

# test_datagen = ImageDataGenerator(rescale=1./255)

augment_size = 40000

# randidx = np.random.randint(x_train.shape[0], size=augment_size) # take 40000 feature from train in random

# print(x_train.shape[0]) # 60000
# print(randidx) # [50653 24637 30472 ... 51686  3282 22404]
# print(randidx.shape) # (40000,)

# x_argmented = x_train[randidx].copy()
# y_argmented = y_train[randidx].copy()
x_argmented = x_train[:40000].copy()
y_argmented = y_train[:40000].copy()

x_argmented = x_argmented.reshape(x_argmented.shape[0], 28, 28, 1) # (40000, 28, 28, 1)
x_train = x_train.reshape(x_train.shape[0], 28, 28, 1) # (60000, 28, 28, 1)
x_test = x_test.reshape(x_test.shape[0], 28, 28, 1) # (10000, 28, 28, 1)

# print(x_argmented.shape, x_train.shape)

x_argmented = train_datagen.flow(x_argmented, 
                                np.zeros(augment_size),
                                batch_size=augment_size,
                                shuffle=False).next()[0]

# print(x_argmented.shape) # (40000, 28, 28, 1)

x_train = np.concatenate((x_train, x_argmented), axis=0) # (100000, 28, 28, 1) 
y_train = np.concatenate((y_train, y_argmented), axis=0) # (100000,)

# x_argmented x_train 각 10개 비교 이미지 출력
# 서브플롯 (2, 10, ?) 사용

import matplotlib.pyplot as plt

plt.figure(figsize=(2,10))
for i in range(20):
    if i <= 9:
        plt.subplot(2, 10, i+1)
        plt.axis('off')
        plt.imshow(x_train[i], cmap='gray')
    else:
        plt.subplot(2, 10, i+1)
        plt.axis('off')
        plt.imshow(x_train[i+60000], cmap='gray')

    
plt.show()


