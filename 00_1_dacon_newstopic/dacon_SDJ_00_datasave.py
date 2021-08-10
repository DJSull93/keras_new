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

# preprocessing
from sklearn.feature_extraction.text import TfidfVectorizer
tfidf = TfidfVectorizer(analyzer='word', sublinear_tf=True, ngram_range=(1, 2), max_features=150000, binary=False)

tfidf.fit(x_train)

x_train = tfidf.transform(x_train).astype('float32') # (45654, 150000)
x_pred = tfidf.transform(x_pred).astype('float32') # (9131, 150000)

# print(x_train.shape, x_pred.shape)
# print(x_pred)

np.save('./_save/_NPY/DA_x_train.npy', arr=x_train)
np.save('./_save/_NPY/DA_y_train.npy', arr=y_train)
np.save('./_save/_NPY/DA_x_pred.npy', arr=x_pred)

print(x_pred.shape, y_train.shape)

'''
# save x_train, x_pred, y_train
'''
'''
# data padding / onehotencode
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.utils import to_categorical

x_train = pad_sequences(x_train, padding='pre', maxlen=9) # 'pre' <> 'post'
x_pred = pad_sequences(x_pred, padding='pre', maxlen=9) # 'pre' <> 'post'

y_train = to_categorical(y_train) # (45654, 7)

# print(x_train.shape, x_test.shape) # (45654, 10) (9131, 10)
# print(y_train.shape) # (45654, 7)

# train / test cut
x_train, x_test, y_train, y_test = train_test_split(x_train, y_train,
      test_size=0.05, shuffle=True, random_state=22)

# print(x_train.shape, x_test.shape) # (34240, 10) (11414, 10)
# print(y_train.shape, y_test.shape) # (34240, 7) (11414, 7)

# 2. model
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import Dense, Embedding, LSTM, GRU, Dropout, Conv1D, GlobalAveragePooling1D, MaxPool1D

model = Sequential()
model.add(Embedding(input_dim=101082, output_dim=9,
                     input_length=9))
model.add(LSTM(17))
model.add(Dense(54, activation='relu')) # 54
# model.add(Dropout(0.1)) # 
# model.add(Dense(10, activation='relu')) # 
model.add(Dense(7, activation='softmax'))

# 3. compile train
model.compile(loss='categorical_crossentropy', optimizer='adam'
                , metrics=['acc'])

from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
es = EarlyStopping(monitor='val_loss', patience=10, 
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
model.fit(x_train, y_train, epochs=1000, batch_size=512, verbose=2,
    validation_split=0.05, callbacks=[es, mcp])
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