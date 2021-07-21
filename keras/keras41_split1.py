# Timeseries data make train, test example function

import numpy as np

a = np.array(range(1, 11))
size = 5 # x , y total columns -> one data units

def split_x(dataset, size):
    aaa = []
    for i in range(len(dataset) - size + 2):
        subset = dataset[i : (i + size)]
        aaa.append(subset)
    return np.array(aaa)


dataset = split_x(a, size)

print(dataset)
'''
[[ 1  2  3  4  5]
 [ 2  3  4  5  6]
 [ 3  4  5  6  7]
 [ 4  5  6  7  8]
 [ 5  6  7  8  9]
 [ 6  7  8  9 10]]
'''

x = dataset[:, :-1]
y = dataset[:, -1]

print('x :\n', x)
print('y :\n', y)

'''
x : [[1 2 3 4]
 [2 3 4 5]
 [3 4 5 6]
 [4 5 6 7]
 [5 6 7 8]
 [6 7 8 9]]
y : [ 5  6  7  8  9 10]
'''