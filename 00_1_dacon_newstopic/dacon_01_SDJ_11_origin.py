import re
import numpy as np
import pandas as pd
from sklearn import datasets
from sklearn.model_selection import train_test_split
from konlpy.tag import Okt

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

def text_preprocessing(text_list):
    
    stopwords = ['을', '를', '이', '가', '은', '는', 'null'] #불용어 설정
    tokenizer = Okt() #형태소 분석기 
    token_list = []
    
    for text in text_list:
        txt = re.sub('[^가-힣a-z]', ' ', text) #한글과 영어 소문자만 남기고 다른 글자 모두 제거
        token = tokenizer.morphs(txt) #형태소 분석
        token = [t for t in token if t not in stopwords or type(t) != float] #형태소 분석 결과 중 stopwords에 해당하지 않는 것만 추출
        token_list.append(token)
        
    return token_list, tokenizer

#형태소 분석기를 따로 저장한 이유는 후에 test 데이터 전처리를 진행할 때 이용해야 되기 때문입니다. 
x_train, okt = text_preprocessing(x_train) 
x_pred, okt = text_preprocessing(x_pred) 

from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences

def text2sequence(train_text, max_len=100):
    
    tokenizer = Tokenizer() #keras의 vectorizing 함수 호출
    tokenizer.fit_on_texts(train_text) #train 문장에 fit
    train_X_seq = tokenizer.texts_to_sequences(train_text) #각 토큰들에 정수 부여
    vocab_size = len(tokenizer.word_index) + 1 #모델에 알려줄 vocabulary의 크기 계산
    print('vocab_size : ', vocab_size)
    X_train = pad_sequences(train_X_seq, maxlen = max_len) #설정한 문장의 최대 길이만큼 padding
    
    return X_train, vocab_size, tokenizer

train_X, vocab_size, vectorizer = text2sequence(x_train['text'], max_len = 100)

# from keras.preprocessing.text import Tokenizer
# token = Tokenizer()
# token.fit_on_texts(x_train)
# x_train = token.texts_to_sequences(x_train) # (45654, )
# x_pred = token.texts_to_sequences(x_pred) #  (9131, )

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
model.add(Conv1D(64, 2, activation='relu', padding='same'))
model.add(GlobalAveragePooling1D())
model.add(Dense(64, activation='relu')) # 54
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
modelpath = "".join([filepath, "SDJ_DACON02_", date_time, "_", filename])
###################################################################
mcp = ModelCheckpoint(monitor='val_loss', mode='auto', verbose=2,
            save_best_only=True, filepath= modelpath)

import time

start_time = time.time()
model.fit(x_train, y_train, epochs=1000, batch_size=512, verbose=2,
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
model = load_model('./03_dacon_comp/_MCP/model_hist/SDJ_DACON01_0727_1407_0003_0.676241.hdf5')
loss = model.evaluate(x_test, y_test, batch_size=1024)
# print("time : ", end_time)
print('loss : ', loss[0])
print('acc : ', loss[1])
'''

'''
loss :  0.6284400820732117
acc :  0.7948605418205261
'''