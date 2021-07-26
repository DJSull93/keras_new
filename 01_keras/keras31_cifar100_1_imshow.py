# example cifar100
# imshow

from tensorflow.keras.datasets import cifar100

import numpy as np
import matplotlib.pyplot as plt
from icecream import ic

# 1. data
(x_train, y_train), (x_test, y_test) = cifar100.load_data() 

# ic(x_train.shape, x_test.shape) # (50000, 32, 32, 3) (10000, 32, 32, 3)
# ic(y_train.shape, y_test.shape) # (50000, 1) (10000, 1)

print(x_train[3])
print("y[0] value", y_train[3]) 

plt.imshow(x_train[35], 'gray')
plt.show()