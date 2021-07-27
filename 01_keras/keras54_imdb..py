from tensorflow.keras.datasets import imdb
import numpy as np

(x_train, y_train), (x_test, y_test) = imdb.load_data(
    num_words=10000)

# print(x_train.shape, y_train.shape) # (25000,) (25000,)
# print(x_test.shape, y_test.shape) # (25000,) (25000,)
# print(np.unique(y_train)) # [0 1]

# print('max length :', max(len(i) for i in x_train)) # 2494
# print('avg length :', sum(map(len, x_train))/len(x_train)) # 238.71364

from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.utils import to_categorical

x_train = pad_sequences(x_train, padding='pre', maxlen=240) # (25000, 240) 
x_test = pad_sequences(x_test, padding='pre', maxlen=240) # (25000, 240)

y_train = to_categorical(y_train) # (25000, 2) 
y_test = to_categorical(y_test) # (25000, 2)

# print(y_train.shape, y_test.shape)
# print(x_train.shape, x_test.shape)

# 2. model
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import Dense, Embedding, GRU, Input, Conv1D, GlobalAveragePooling1D

model = Sequential()
model.add(Embedding(input_dim=10000, output_dim=10, input_length=240))
model.add(GRU(32, activation='relu'))
model.add(Dense(100, activation='relu'))
model.add(Dense(64, activation='relu'))
model.add(Dense(32, activation='relu'))
model.add(Dense(2, activation='sigmoid'))

# 3. comple fit 
model.compile(loss='binary_crossentropy', optimizer='adam'
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
time :  197.96073698997498
loss :  1.1302212476730347
acc :  0.7832800149917603
'''