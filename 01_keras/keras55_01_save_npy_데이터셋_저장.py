from sklearn.datasets import load_iris, load_boston, load_diabetes, load_breast_cancer, load_wine
from tensorflow.keras.datasets import mnist, fashion_mnist, cifar10,cifar100

import numpy as np
import pandas as pd

######################iris######################
datasets = load_iris()

x_data = datasets.data
y_data = datasets.target

# print(type(x_data), type(y_data)) # <class 'numpy.ndarray'> <class 'numpy.ndarray'>

np.save('./_save/_NPY/k55_x_data_iris.npy', arr=x_data)
np.save('./_save/_NPY/k55_y_data_iris.npy', arr=y_data)

# boston, daibet, cancer, wine

######################boston######################
datasets = load_boston()

x_data = datasets.data
y_data = datasets.target


np.save('./_save/_NPY/k55_x_data_boston.npy', arr=x_data)
np.save('./_save/_NPY/k55_y_data_boston.npy', arr=y_data)

######################diabetes######################
datasets = load_diabetes()

x_data = datasets.data
y_data = datasets.target


np.save('./_save/_NPY/k55_x_data_diabetes.npy', arr=x_data)
np.save('./_save/_NPY/k55_y_data_diabetes.npy', arr=y_data)

######################cancer######################
datasets = load_breast_cancer()

x_data = datasets.data
y_data = datasets.target


np.save('./_save/_NPY/k55_x_data_cancer.npy', arr=x_data)
np.save('./_save/_NPY/k55_y_data_cancer.npy', arr=y_data)

######################wine######################
datasets = pd.read_csv('../_data/winequality-white.csv', sep=';',
                       index_col=None, header=0 ) # (4898, 12)

x_data = datasets.iloc[:,0:11] # (4898, 11)
y_data = datasets.iloc[:,[11]] # (4898, 10)

np.save('./_save/_NPY/k55_x_data_wine.npy', arr=x_data)
np.save('./_save/_NPY/k55_y_data_wine.npy', arr=y_data)

######################mnist######################
(x_train, y_train), (x_test, y_test) = mnist.load_data() 

np.save('./_save/_NPY/k55_x_data_mnist_train.npy', arr=x_train)
np.save('./_save/_NPY/k55_x_data_mnist_test.npy', arr=x_test)
np.save('./_save/_NPY/k55_y_data_mnist_train.npy', arr=y_train)
np.save('./_save/_NPY/k55_y_data_mnist_test.npy', arr=y_test)

######################fmnist######################
(x_train, y_train), (x_test, y_test) = fashion_mnist.load_data() 

np.save('./_save/_NPY/k55_x_data_fashion_mnist_train.npy', arr=x_train)
np.save('./_save/_NPY/k55_x_data_fashion_mnist_test.npy', arr=x_test)
np.save('./_save/_NPY/k55_y_data_fashion_mnist_train.npy', arr=y_train)
np.save('./_save/_NPY/k55_y_data_fashion_mnist_test.npy', arr=y_test)

######################cifar10######################
(x_train, y_train), (x_test, y_test) = cifar10.load_data() 

np.save('./_save/_NPY/k55_x_data_cifar10_train.npy', arr=x_train)
np.save('./_save/_NPY/k55_x_data_cifar10_test.npy', arr=x_test)
np.save('./_save/_NPY/k55_y_data_cifar10_train.npy', arr=y_train)
np.save('./_save/_NPY/k55_y_data_cifar10_test.npy', arr=y_test)

######################cifar100######################
(x_train, y_train), (x_test, y_test) = cifar100.load_data() 

np.save('./_save/_NPY/k55_x_data_cifar100_train.npy', arr=x_train)
np.save('./_save/_NPY/k55_x_data_cifar100_test.npy', arr=x_test)
np.save('./_save/_NPY/k55_y_data_cifar100_train.npy', arr=y_train)
np.save('./_save/_NPY/k55_y_data_cifar100_test.npy', arr=y_test)