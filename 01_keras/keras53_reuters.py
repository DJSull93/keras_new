from tensorflow.keras.datasets import reuters
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

# 1. data
(x_train, y_train), (x_test, y_test) = reuters.load_data(
    num_words=10000, test_split=0.2)

# print(x_train[0], type(x_train[0]))
# [1, 2, 2, 8, 43, 10, 447, 5, 25, 207, 270, 5, 3095, 111, 16, 369, 186, 90, 67, 7, 89, 
# 5, 19, 102, 6, 19, 124, 15, 90, 67, 84, 22, 482, 26, 7, 48, 4, 49, 8, 864, 39, 209, 154, 6, 151, 6, 83, 11, 15, 22, 155, 11, 15, 7, 48, 9, 4579, 1005, 504, 6, 258, 6, 272, 
# 11, 15, 22, 134, 44, 11, 15, 16, 8, 197, 1245, 90, 67, 52, 29, 209, 30, 32, 132, 6, 109, 15, 17, 12] <class 'list'>
# print(x_train[1], type(x_train[1]))
# print(len(y_train)) # 8982
# print(len(x_train[0]), len(x_train[11])) # 87 59 -> padding
# print(len(np.unique(x_train))) # 8453

# print(x_train[0].shape) -> list has no shape
# print(x_train.shape, y_train.shape) # (8982,) (8982,)
# print(x_test.shape, y_test.shape) # (2246,) (2246,)

# print(type(x_train)) # <class 'numpy.ndarray'>

# print('max length :', max(len(i) for i in x_train)) # 2736
# print('avg length :', sum(map(len, x_train))/len(x_train)) # 145.5398574927633

# plt.hist([len(s) for s in x_train], bins=50)
# plt.show()

from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.utils import to_categorical

x_train = pad_sequences(x_train, padding='pre', maxlen=230) # (8982, 100)
x_test = pad_sequences(x_test, padding='pre', maxlen=230) # (2246, 100)

# print(x_train.shape, x_test.shape) 

# print(np.unique(y_train)) # 46 category

y_train = to_categorical(y_train) # (8982, 46) 
y_test = to_categorical(y_test) # (2246, 46)
# print(y_train.shape, y_test.shape)

# 2. model
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import Dense, Embedding, GRU, Input, Conv1D, GlobalAveragePooling1D

model = Sequential()
model.add(Embedding(input_dim=10000, output_dim=10, input_length=230))
model.add(GRU(32, activation='relu'))
model.add(Dense(100, activation='relu'))
model.add(Dense(64, activation='relu'))
model.add(Dense(64, activation='relu'))
model.add(Dense(46, activation='softmax'))

# 3. comple fit 
model.compile(loss='categorical_crossentropy', optimizer='adam'
                , metrics=['acc'])

from tensorflow.keras.callbacks import EarlyStopping
es = EarlyStopping(monitor='val_loss', patience=40, mode='min', verbose=1)

import time

start_time = time.time()
model.fit(x_train, y_train, epochs=1000, batch_size=4000, verbose=2,
    validation_split=0.175, callbacks=[es])
end_time = time.time() - start_time

# 4. predict eval 

loss = model.evaluate(x_test, y_test, batch_size=1024)
print("time : ", end_time)
print('loss : ', loss[0])
print('acc : ', loss[1])

'''
time :  191.13061475753784
loss :  2.0946764945983887
acc :  0.5307213068008423

time :  191.3090889453888
loss :  1.9916166067123413
acc :  0.5284950733184814
'''