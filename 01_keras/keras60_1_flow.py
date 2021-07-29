# flow one image to 100

from tensorflow.keras.datasets import fashion_mnist
import numpy as np
from tensorflow.python.keras.backend import zeros

(x_train, y_train), (x_test, y_test) = fashion_mnist.load_data()

from tensorflow.keras.preprocessing.image import ImageDataGenerator


train_datagen = ImageDataGenerator(
    rescale=1./255,
    horizontal_flip=True,
    vertical_flip=False,
    width_shift_range=0.1,
    height_shift_range=0.1,
    rotation_range=10,
    zoom_range=0.2,
    shear_range=0.5,
    fill_mode='nearest'
)

# test_datagen = ImageDataGenerator(rescale=1./255)

# xy_train = train_datagen.flow_from_directory(
#     '../_data/brain/train',
#     target_size=(150, 150),
#     batch_size=5,
#     class_mode='binary',
#     shuffle=True
# )

# 1. define ImageDataGenerator 
# 2. from directory : flow_from_directory() // x y are compiled as tuple
# 3. from dataset : flow()                  // x y are devided

# 배열을 반복하면서 새로운 축을 추가하기 : np.tile
# np.tile(arr, reps) method 는 'arr' 에는 배열을, 'reps'에는 반복하고자 하는 회수를 넣어줍니다.
# 'reps'에는 숫자를 넣을 수도 있고, 배열을 넣을 수도 있습니다.

augment_size = 100

x_data = train_datagen.flow(
    np.tile(x_train[0].reshape(28*28), augment_size).reshape(-1, 28, 28, 1), # x
    np.zeros(augment_size), # y
    batch_size=augment_size,
    shuffle=True
).next()
# flow/fromdirec result : Iterator -> 리스트를 순회할 수 있게 해줌
# -> 컬렉션으로부터 정보를 얻어내는 인터페이스 
# .next 사용시 한번에 출력

# .next 미사용, flow 실행 -> flow 1번 (리스트의 한개 단)
# .next 사용, flow 실행 -> flow 전체 (리스트의 전체 단)
# 리스트 for 문 사용 -> Iterator

# print(type(x_data)) # NumpyArrayIterator # .next() -> tuple
# print(type(x_data[0])) # tuple # .next() -> numpy.ndarray
# print(type(x_data[0][0])) # numpy.ndarray # .next -> numpy.ndarray
# print(x_data[0][0].shape) # (100, 28, 28, 1) -> x # .next -> (28, 28, 1)
# print(x_data[0][1].shape) # (100,)           -> y # .next -> (28, 28, 1)
# print(x_data[0].shape) #  # .next -> (100, 28, 28, 1)
# print(x_data[1].shape) #  # .next -> (100,)

import matplotlib.pyplot as plt

plt.figure(figsize=(10,10))
for i in range(100):
    plt.subplot(10, 10, i+1)
    plt.axis('off')
    plt.imshow(x_data[0][i], cmap='gray')

plt.show()