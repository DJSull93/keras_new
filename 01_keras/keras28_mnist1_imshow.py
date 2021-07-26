import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.datasets import mnist

(x_train, y_train), (x_test, y_test) = mnist.load_data() # (60000, 28, 28) (60000,) (10000, 28, 28) (10000,)

# print(x_train.shape, y_train.shape)
# print(x_test.shape, y_test.shape)
print(x_train[111])
print("y[0] value", y_train[111]) # y[0] value 5 ; means x_train[0] = 5

plt.imshow(x_train[111], 'gray')
plt.show()