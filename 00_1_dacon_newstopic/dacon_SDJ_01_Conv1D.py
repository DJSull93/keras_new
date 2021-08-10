import re
import numpy as np
import pandas as pd
from sklearn import datasets
from sklearn.model_selection import train_test_split

# 1. data
path = './03_dacon_comp/_data/'
# data load
x_train = pd.read_csv(path+'train_data.csv') # (45654, 3) 
x_pred = pd.read_csv(path+'test_data.csv') # (9131, 2) 
label = pd.read_csv(path+'topic_dict.csv') # (7, 2)

# print(x_train.shape, x_test.shape, label.shape)

x_train = x_train.to_numpy() # (45654, 3) 
x_pred = x_pred.to_numpy() # (9131, 2)

# print(x_train.shape, x_test.shape)

# data cut
y_train = x_train[:,2] # (45654, )
x_train = x_train[:,1] # (45654, )
x_pred = x_pred[:,1] # (9131,)

for i, j in enumerate(x_train):
    text_clean = re.sub("[^가-힣ㄱ-ㅎㅏ-ㅣ\\s]", " ", j)
    x_train[i] = text_clean

for i, j in enumerate(x_pred):
    text_clean = re.sub("[^가-힣ㄱ-ㅎㅏ-ㅣ\\s]", " ", j)
    x_pred[i] = text_clean

from keras.preprocessing.text import Tokenizer

token = Tokenizer()
token.fit_on_texts(x_train)
x_train = token.texts_to_sequences(x_train) # (45654, )
x_pred = token.texts_to_sequences(x_pred) #  (9131, )

# print(len(np.unique(x_train))) # 45630

# print(len(x_train), len(x_test))
# print('max length :', max(len(i) for i in x_train)) # max length : 13
# print('avg length :', sum(map(len, x_train))/len(x_train)) # avg length : 6.623954089455469

# data padding / onehotencode
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.utils import to_categorical

x_train = pad_sequences(x_train, padding='pre', maxlen=13) # 'pre' <> 'post'
x_pred = pad_sequences(x_pred, padding='pre', maxlen=13) # 'pre' <> 'post'

y_train = to_categorical(y_train) # (45654, 7)

# print(x_train.shape, x_test.shape) # (45654, 10) (9131, 10)
# print(y_train.shape) # (45654, 7)

# train / test cut

x_train, x_test, y_train, y_test = train_test_split(x_train, y_train,
      test_size=0.15, shuffle=True, random_state=22)

# print(x_train.shape, x_test.shape) # (34240, 10) (11414, 10)
# print(y_train.shape, y_test.shape) # (34240, 7) (11414, 7)

# 2. model
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import Dense, Embedding, LSTM, GRU, Dropout, Conv1D, GlobalAveragePooling1D, MaxPool1D

model = Sequential()
model.add(Embedding(input_dim=101082, output_dim=64,
                     input_length=13))
model.add(Conv1D(100, 2, activation='relu', padding='same'))
# model.add(Conv1D(16, 2, activation='relu', padding='same'))
# model.add(MaxPool1D())
model.add(GlobalAveragePooling1D())
model.add(Dense(64, activation='relu')) # 54
# model.add(Dropout(0.2)) # 
# model.add(Dense(32, activation='relu')) # 
model.add(Dense(7, activation='softmax'))

# 3. compile train
model.compile(loss='categorical_crossentropy', optimizer='adam'
                , metrics=['acc'])

from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
es = EarlyStopping(monitor='val_loss', patience=5, 
                    mode='auto', verbose=2, restore_best_weights=True)

###################################################################
# file name auto change and save
import datetime
date = datetime.datetime.now()
date_time = date.strftime("%m%d_%H%M")

filepath = './03_dacon_comp/_MCP/'
filename = '{epoch:04d}_{val_loss:4f}.hdf5'
modelpath = "".join([filepath, "SDJ_DACON01_", date_time, "_", filename])
###################################################################
mcp = ModelCheckpoint(monitor='val_loss', mode='auto', verbose=2,
            save_best_only=True, filepath= modelpath)

import time

start_time = time.time()
model.fit(x_train, y_train, epochs=10000, batch_size=4096, verbose=2,
    validation_split=0.025, callbacks=[es, mcp])
end_time = time.time() - start_time

# 4. evaluate, predict
print('=================1. basic print=================')
loss = model.evaluate(x_test, y_test, batch_size=128)
print("time : ", end_time)
print('loss : ', loss[0])
print('acc : ', loss[1])

y_pred = model.predict(x_pred)
y_pred = np.argmax(y_pred, axis=1)
index = np.array([range(45654, 54785)])
index = index.reshape(9131, )
y_pred = np.column_stack([index, y_pred])
y_pred = pd.DataFrame(y_pred)
y_pred.to_csv('./03_dacon_comp/_save_csv/sample_submission.csv',
        index=False, header=['index', 'topic_idx'], encoding='UTF-8')

'''
print('=================2. load model=================')
model = load_model('./03_dacon_comp/_MCP/model_hist/SDJ_DACON01_0726_2234_0009_1.142526.hdf5')
loss = model.evaluate(x_test, y_test, batch_size=1024)
# print("time : ", end_time)
print('loss : ', loss[0])
print('acc : ', loss[1])
'''

'''
time :  41.61598348617554
loss :  1.1990669965744019
acc :  0.5895391702651978
'''