# image process, CCN -> Conv2D (for image)

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Conv2D

model = Sequential()
model.add(Conv2D(10, kernel_size=(2, 2), input_shape=(28, 28, 1)))
