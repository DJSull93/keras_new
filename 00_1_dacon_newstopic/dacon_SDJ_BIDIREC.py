import pandas as pd
import numpy as np
import re

import seaborn as sns
import matplotlib.pyplot as plt

from sklearn.metrics import accuracy_score, log_loss
from sklearn.model_selection import StratifiedKFold

import tensorflow as tf
from tensorflow.keras import Sequential
from tensorflow.keras.layers import Dense, Embedding, LSTM, Dropout, Bidirectional
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from tensorflow.keras.utils import plot_model, to_categorical
from tensorflow.keras.optimizers import Adam

from keras.utils import np_utils

import warnings 
warnings.filterwarnings(action='ignore')

path = './03_dacon_comp/_data/'

train      = pd.read_csv(path+'train_data.csv')
test       = pd.read_csv(path+'test_data.csv')
submission = pd.read_csv(path+'sample_submission.csv')
topic_dict = pd.read_csv(path+'topic_dict.csv')

X_train = np.array([x for x in train['title']])
X_test = np.array([x for x in test['title']])
Y_train = np.array([x for x in train['topic_idx']])

from keras.preprocessing.text import Tokenizer
vocab_size = 2000  

tokenizer = Tokenizer(num_words = vocab_size)  
tokenizer.fit_on_texts(X_train)
sequences_train = tokenizer.texts_to_sequences(X_train)
sequences_test = tokenizer.texts_to_sequences(X_test)

word_index = tokenizer.word_index

max_length = 11
padding_type='post'

train_x = pad_sequences(sequences_train, padding='post', maxlen=max_length)
test_x = pad_sequences(sequences_test, padding=padding_type, maxlen=max_length)

print(train_x.shape, test_x.shape)

train_y = np_utils.to_categorical(Y_train)
print(train_y)
print(train_y.shape)

vocab_size = 2000
embedding_dim = 200  
max_length = 11
padding_type='post'

model = Sequential()
model.add(Embedding(2000, 200, input_length=11))
model.add(Bidirectional(LSTM(128, return_sequences = True)))
model.add(Bidirectional(LSTM(128, return_sequences = True)))
model.add(Bidirectional(LSTM(64)))
model.add(Dense(7, activation='softmax'))

import tensorflow as tf

model.compile(loss='categorical_crossentropy', optimizer = 'adam', metrics=['acc'])

n_fold = 5  

cv = StratifiedKFold(n_splits = n_fold, shuffle=True, random_state=42)

test_y = np.zeros((test_x.shape[0], 7))
# print(test_x.shape, test_y.shape)

es = EarlyStopping(monitor='val_loss', min_delta=0.001, patience=7,
                   verbose=1, mode='min', baseline=None, restore_best_weights=True)

for i, (i_trn, i_val) in enumerate(cv.split(train_x, Y_train), 1):
    print(f'training model for CV #{i}')

    model.fit(train_x[i_trn], 
            to_categorical(Y_train[i_trn]),
            validation_data=(train_x[i_val], to_categorical(Y_train[i_val])),
            epochs=10,
            batch_size=4096,
            callbacks=[es])     # 조기 종료 옵션
                      
    test_y += model.predict(test_x) / n_fold

topic = []
for i in range(len(test_y)):
    topic.append(np.argmax(test_y[i]))

submission['topic_idx'] = topic

PATH = './03_dacon_comp/_save_csv/'
submission.to_csv(PATH + 'LSTM_sub.csv',index = False)
