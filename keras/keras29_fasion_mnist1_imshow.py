# example fasion_mnist  
# mnist imshow copy 

import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.datasets import fashion_mnist
from icecream import ic

# 1. data
(x_train, y_train), (x_test, y_test) = fashion_mnist.load_data() # (60000, 28, 28) (60000,) (10000, 28, 28) (10000,)

# ic(x_train.shape, x_test.shape)
# ic(y_train.shape, y_test.shape)

print(x_train[3])
print("y[0] value", y_train[3]) # y[0] value 5 ; means x_train[0] = 5

plt.imshow(x_train[3], 'gray')
plt.show()
