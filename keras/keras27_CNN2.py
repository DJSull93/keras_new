# image process, CCN -> Conv2D (for image)
# maxpooling -> shape will de shrink by half -> only tskes max values in cut part
# 통상 2번 conv 하고 maxpool 1번 사용

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Conv2D, Flatten, MaxPool2D
from tensorflow.python.keras.layers.core import Flatten

model = Sequential()
model.add(Conv2D(10, kernel_size=(2, 2),                          # (N,10,10,1)
                         padding='same',input_shape=(10, 10, 1)))   # (N,10,10,10)
model.add(Conv2D(20, (2,2), activation='relu'))                   # (N,9,9,20)
model.add(Conv2D(30, (2,2), padding='valid', activation='relu'))  # (N,8,8,30)
model.add(MaxPool2D())                                            # (N,4,4,30)
model.add(Conv2D(15, (2,2)))                                      # (N,3,3,15)
model.add(Flatten())                                              # (N,135)
model.add(Dense(64, activation='relu'))
model.add(Dense(32, activation='relu'))
model.add(Dense(16, activation='relu'))
model.add(Dense(1, activation='sigmoid'))

model.summary()
'''
_________________________________________________________________
Layer (type)                 Output Shape              Param #
=================================================================
conv2d (Conv2D)              (None, 10, 10, 10)        50
_________________________________________________________________
conv2d_1 (Conv2D)            (None, 9, 9, 20)          820
_________________________________________________________________
conv2d_2 (Conv2D)            (None, 8, 8, 30)          2430
_________________________________________________________________
max_pooling2d (MaxPooling2D) (None, 4, 4, 30)          0
_________________________________________________________________
conv2d_3 (Conv2D)            (None, 3, 3, 15)          1815
_________________________________________________________________
flatten (Flatten)            (None, 135)               0
_________________________________________________________________
dense (Dense)                (None, 64)                8704
_________________________________________________________________
dense_1 (Dense)              (None, 32)                2080
_________________________________________________________________
dense_2 (Dense)              (None, 16)                528
_________________________________________________________________
dense_3 (Dense)              (None, 1)                 17
=================================================================
Total params: 16,444
Trainable params: 16,444
Non-trainable params: 0
_________________________________________________________________
'''

