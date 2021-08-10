import pandas as pd
import numpy as np
import re

from sklearn.model_selection import StratifiedKFold

import tensorflow as tf
from tensorflow.keras import Sequential
from tensorflow.keras.layers import Dense, Embedding, LSTM, Dropout, Bidirectional
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.optimizers import Adam
from konlpy.tag import Okt
from sklearn.model_selection import train_test_split

import warnings 
warnings.filterwarnings(action='ignore')

# 1. data load
path = './03_dacon_comp/_data/'
train_data  = pd.read_csv(path+'train_data.csv', header=0) # (45654, 3) 
test_data  = pd.read_csv(path+'test_data.csv', header=0) # (9131, 2) 
submission  = pd.read_csv(path+'sample_submission.csv') # (7, 2)

# 1-1.data cut
Y1_train = np.array([x for x in train_data['topic_idx']]) # (9131,)
X1_train = np.array([x for x in train_data['title']]) # (45654, )
X1_test = np.array([x for x in test_data['title']]) # (45654, )

for i, j in enumerate(X1_train):
    text_clean = re.sub("[^가-힣ㄱ-ㅎㅏ-ㅣ\\s]", " ", j)
    X1_train[i] = text_clean

for i, j in enumerate(X1_test):
    text_clean = re.sub("[^가-힣ㄱ-ㅎㅏ-ㅣ\\s]", " ", j)
    X1_test[i] = text_clean

def text_preprocessing(text_list):
    
    stopwords = ['을', '를', '이', '가', '은', '는', 'null'] 
    tokenizer = Okt() 
    token_list = []
    
    for text in text_list:
        txt = re.sub('[^가-힣a-z]', ' ', text) 
        token = tokenizer.morphs(txt) 
        token = [t for t in token if t not in stopwords or type(t) != float] 
        token_list.append(token)
        
    return token_list, tokenizer

x_train, okt = text_preprocessing(X1_train) 
x_test, okt = text_preprocessing(X1_test) 


from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences

vocab_size = 101082

tokenizer = Tokenizer(num_words = vocab_size)  
tokenizer.fit_on_texts(x_train) 
x_train = tokenizer.texts_to_sequences(x_train)
x_test = tokenizer.texts_to_sequences(x_test)

# data padding / onehotencode
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.utils import to_categorical

x_train = pad_sequences(x_train, padding='post', maxlen=14) # 'pre' <> 'post'
x_test = pad_sequences(x_test, padding='post', maxlen=14) # 'pre' <> 'post'

y_train = to_categorical(Y1_train) # (45654, 7)

x_train, x_test, y_train, y_test = train_test_split(x_train, y_train,
      test_size=0.2, shuffle=False)

# print(x_train.shape, x_test.shape) # (36523, 14) (9131, 14)
# print(y_train.shape, y_test.shape) # (36523, 7) (9131, 7)
# print(Y1_train.shape) # (45654,)

# 2. model
# params
vocab_size = 101082
embedding_dim = 200  
max_length = 14    
padding_type='post'

model = Sequential([Embedding(vocab_size, embedding_dim, input_length =max_length),
        tf.keras.layers.Bidirectional(LSTM(units = 32, activation='relu', return_sequences = True)),
        tf.keras.layers.Dropout(0.15),
        tf.keras.layers.Bidirectional(LSTM(units = 16, activation='relu', return_sequences = True)),
        tf.keras.layers.Dropout(0.15),
        tf.keras.layers.Bidirectional(LSTM(units = 8, activation='relu')),
        Dense(7, activation='softmax')    
    ])

# 3. compile
# params
BATCH = 512
EPOCHS = 8
VAL_SPL = 0.2

from tensorflow.keras.optimizers import Adam

optimizer = Adam(0.001)

lr = ReduceLROnPlateau(monitor='val_loss', patience=2,
    verbose=1, mode='auto', factor=0.8)

model.compile(loss= 'categorical_crossentropy', 
              optimizer= optimizer,
              metrics = ['acc']) 

from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
###################################################################
# file name auto change and save
import datetime
date = datetime.datetime.now()
date_time = date.strftime("%m%d_%H%M")

filepath = './03_dacon_comp/_MCP/'
filename = '{epoch:04d}_{val_loss:4f}.hdf5'
modelpath = "".join([filepath, "DA02_", date_time, "_", filename])
###################################################################
mcp = ModelCheckpoint(monitor='val_loss', mode='auto', verbose=2,
            save_best_only=True, filepath= modelpath)
es = EarlyStopping(monitor='val_loss', patience=3, 
                    mode='auto', verbose=2, restore_best_weights=True)

import time

start_time = time.time()
hist = model.fit(x_train, y_train, 
                epochs=EPOCHS, batch_size=BATCH, 
                validation_split= VAL_SPL,
                callbacks=[es],
                verbose=1) 

n_fold = 5  
seed = 1234

cv = StratifiedKFold(n_splits = n_fold, 
            shuffle=True, random_state=seed)

test_y = np.zeros((x_test.shape[0], 7))

for i, (i_trn, i_val) in enumerate(cv.split(x_train, Y1_train[:36523]), 1):
    print(f'training model for CV #{i}')

    model.fit(x_train[i_trn], 
            to_categorical(Y1_train[i_trn]),
            validation_data=(x_train[i_val], to_categorical(Y1_train[i_val])),
            epochs=5,
            batch_size=BATCH,
            callbacks=[es, lr, mcp])

    test_y += model.predict(x_test) / n_fold

end_time = time.time() - start_time
loss = model.evaluate(x_test, y_test, batch_size=128)
print('total time : ',round(end_time/60),'min')

topic = []

for i in range(len(test_y)):
    topic.append(np.argmax(test_y[i]))    



acc = hist.history['acc']
val_acc = hist.history['val_acc']
loss = hist.history['loss']
val_loss = hist.history['val_loss']


print('acc : ',acc[-3])
print('val_acc : ',val_acc[-3])
print('loss : ',loss[-3])
print('val_loss : ',val_loss[-3])


submission['topic_idx'] = topic
submission.to_csv('LSTM.csv',index = False)

'''
total time :  5 min
acc :  0.9602642059326172
val_acc :  0.8662559986114502
loss :  0.11688399314880371
val_loss :  0.5683019161224365

total time :  6 min
acc :  0.9048873782157898
val_acc :  0.8729637265205383
loss :  0.2956780791282654
val_loss :  0.41547220945358276
'''