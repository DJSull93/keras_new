'''
이미지 -> (사진의 개수, 가로픽셀, 세로픽셀, 흑백(1)/컬러(3))
: 4차원 데이터

convolution layer -> CNN

Conv2D -> 2차원 이미지 작업에 사용되며, 입력 차원은 4차원

'''


# image process, CCN -> Conv2D (for image)

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Conv2D, Flatten
from tensorflow.python.keras.layers.core import Flatten

model = Sequential()
model.add(Conv2D(10, kernel_size=(2, 2), input_shape=(5, 5, 1))) # (N, 4, 4, 10), N = None
# input_shape -> ((batch, -hidden )height, width, color)
model.add(Conv2D(20, (2,2), activation='relu')) # (3, 3, 20) kernal_size can be hide
model.add(Conv2D(20, (2,2), activation='relu'))
# Conv2D -> multiple dimension -> output needs to 2D : Dense
# (None, 4, 4, 10) -> (None, 160) : reshape, flatten (make dimension lower level)
model.add(Flatten()) # (N, 160)
model.add(Dense(64, activation='relu'))
model.add(Dense(32, activation='relu'))
model.add(Dense(16, activation='relu'))
model.add(Dense(1, activation='sigmoid'))

model.summary()
'''
_________________________________________________________________
Layer (type)                 Output Shape              Param #
=================================================================
conv2d (Conv2D)              (None, 4, 4, 10)          50
_________________________________________________________________
conv2d_1 (Conv2D)            (None, 3, 3, 20)          820
_________________________________________________________________
flatten (Flatten)            (None, 180)               0
_________________________________________________________________
dense (Dense)                (None, 64)                11584
_________________________________________________________________
dense_1 (Dense)              (None, 32)                2080
_________________________________________________________________
dense_2 (Dense)              (None, 16)                528
_________________________________________________________________
dense_3 (Dense)              (None, 1)                 17
=================================================================
Total params: 15,079
Trainable params: 15,079
Non-trainable params: 0
_________________________________________________________________
'''

