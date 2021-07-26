# file name must be my predict number
import numpy as np
import pandas as pd
from pandas.core.arrays.sparse import dtype
from sklearn.model_selection import train_test_split

# 1. data
sam = pd.read_csv('./samsung_stock/_data/samsung20210721.csv', encoding='EUC-KR')
sk = pd.read_csv('./samsung_stock/_data/SK20210721.csv', encoding='EUC-KR')

# rename col
sam.columns=['date','st','hi','lo','en','6','7','8','9','10','ta','12','13','14','15','16']
sk.columns=['date','st','hi','lo','en','6','7','8','9','10','ta','12','13','14','15','16']

# drop 
sam.drop(['6','7','8','9','10','12','13','14','15','16'], axis='columns', inplace=True)
sam.drop(sam.index[2601:3604], axis=0, inplace=True)
sk.drop(['6','7','8','9','10','12','13','14','15','16'], axis='columns', inplace=True)
sk.drop(sk.index[2601:3604], axis=0, inplace=True)

# object to timestamp and sorting
sam['date'] = pd.to_datetime(sam.date)
sk['date'] = pd.to_datetime(sk.date)
sam = sam.sort_values(by='date', ascending=True)
sk = sk.sort_values(by='date', ascending=True)

sam.drop(['date'], axis='columns', inplace=True)
sk.drop(['date'], axis='columns', inplace=True)

# change df to np
sam = sam.to_numpy() # (2601, 5) 
sam = sam.astype('float32')
sk = sk.to_numpy() # (2601, 5)
sk = sk.astype('float32')

# timeseries split
size = 10

def split_x(dataset, size):
    st = []
    for i in range(len(dataset) - size + 1):
        subset = dataset[i : (i + size)]
        st.append(subset)
    return np.array(st)

sam = split_x(sam, size) # (2561, 40, 5) 
sk = split_x(sk, size) # (2561, 40, 5)

# sam ->, s 
xs = sam[:-1,:,[1,2,3,4]] # (2561, 40, 4)
ys = sam[:-1,:,0] # (2561, 40)
xs_pred = sam[-1,:,:] # 
xs_pred = np.delete(xs_pred, 0, 1)

xs = np.delete(xs, [-3,-2,-1], axis=0) # (2588, 10, 4) 
ys = np.delete(ys, [0,1,2], axis=0) # (2588, 10)

# print(xs.shape, ys.shape) #  
# print(xs[0,:,:]) # ~0715
# print(ys[0,:]) # ~0720
# print(ys) # ~0720
# print(xs_pred) # 0721

# sk -> k
xk = sk[:-1,:,[0,1,2,3,4]] # 
xk_pred = sk[-1,:,:] # 
xk = np.delete(xk, [-3,-2,-1], axis=0) # (2498, 40, 5)

# print(xs_pred.shape, xk_pred.shape)

# train test
xs_train, xs_test, xk_train, xk_test, y_train, y_test = train_test_split(xs, xk, ys,
      test_size=0.25, shuffle=True, random_state=9)

# print(xs_train.shape, xk_train.shape) # (1941, 10, 4) (1941, 10, 5)
# print(xs_test.shape, xk_test.shape) # (647, 10, 4) (647, 10, 5)

# preproccess
xs_train = xs_train.reshape(1941*10, 4)
xs_test = xs_test.reshape(647*10, 4)
xs_pred = xs_pred.reshape(10, 4)

xk_train = xk_train.reshape(1941*10, 5)
xk_test = xk_test.reshape(647*10, 5)
xk_pred = xk_pred.reshape(10, 5)

from sklearn.preprocessing import MinMaxScaler, StandardScaler, RobustScaler, MaxAbsScaler, QuantileTransformer, PowerTransformer
scaler = MinMaxScaler()

xs_train = scaler.fit_transform(xs_train)
xs_test = scaler.transform(xs_test)
xs_pred = scaler.transform(xs_pred)

xk_train = scaler.fit_transform(xk_train)
xk_test = scaler.transform(xk_test)
xk_pred = scaler.transform(xk_pred)

xs_train = xs_train.reshape(1941, 10, 4)
xs_test = xs_test.reshape(647, 10, 4)
xs_pred = xs_pred.reshape(1, 10, 4)

xk_train = xk_train.reshape(1941, 10, 5)
xk_test = xk_test.reshape(647, 10, 5)
xk_pred = xk_pred.reshape(1, 10, 5)

# 2. model
from tensorflow.keras.models import Sequential, Model, load_model
from tensorflow.keras.layers import Dense, GRU, Dropout, Input, LSTM, GlobalAveragePooling1D, Conv1D, Flatten, MaxPool1D

# 2-1. model1
input1 = Input(shape=(10,4))
xx = LSTM(units=64, activation='relu')(input1)
xx = Dense(100, activation='relu')(xx)
xx = Dense(64, activation='relu')(xx)
xx = Dropout(0.005)(xx)
xx = Dense(64, activation='relu')(xx)
xx = Dense(64, activation='relu')(xx)
xx = Dropout(0.005)(xx)
xx = Dense(64, activation='relu')(xx)
xx = Dense(64, activation='relu')(xx)
output1 = Dense(64)(xx)

# 2-2. model2
input2 = Input(shape=(10,5))
xx = LSTM(units=64, activation='relu')(input2)
xx = Dense(100, activation='relu')(xx)
xx = Dense(64, activation='relu')(xx)
xx = Dropout(0.005)(xx)
xx = Dense(64, activation='relu')(xx)
xx = Dense(64, activation='relu')(xx)
xx = Dropout(0.005)(xx)
xx = Dense(64, activation='relu')(xx)
xx = Dense(64, activation='relu')(xx)
output2 = Dense(64)(xx)

# 2-3. model 1, 2 merge
from tensorflow.keras.layers import concatenate

merge1 = concatenate([output1, output2]) # merge 도 layer 임
xx = Dense(100, activation='relu')(merge1)
xx = Dense(100, activation='relu')(xx)
# xx = Dropout(0.001)(xx)
xx = Dense(40, activation='relu')(merge1)
xx = Dense(40, activation='relu')(xx)
# xx = Dropout(0.001)(xx)
xx = Dense(20, activation='relu')(merge1)
xx = Dense(20, activation='relu')(xx)
# xx = Dropout(0.001)(xx)
xx = Dense(20, activation='relu')(xx)
# xx = Dense(8, activation='relu')(xx)
# xx = Dropout(0.001)(xx)
# xx = Dense(4, activation='relu')(xx)
# xx = Dense(4, activation='relu')(xx)
last_output = Dense(1)(xx)

model = Model(inputs=[input1, input2], outputs=last_output)

# 3. compile train

model.compile(loss='mse', optimizer='adam')

from keras.callbacks import EarlyStopping, ModelCheckpoint
es = EarlyStopping(monitor='val_loss', patience=100, 
            mode='auto', verbose=2, restore_best_weights=True)

###################################################################
# file name auto change and save
# 'filename(default)' + 'time' + 'loss'
# %m month %d day %H hour %M minute
import datetime
date = datetime.datetime.now()
date_time = date.strftime("%m%d_%H%M")

filepath = './_save/'
filename = '{epoch:04d}_{val_loss:4f}.hdf5'
modelpath = "".join([filepath, "SDJ_02_", date_time, "_", filename])
###################################################################

mcp = ModelCheckpoint(monitor='val_loss', mode='auto', verbose=2,
            save_best_only=True, 
            filepath= modelpath)

model.fit([xs_train, xk_train], y_train, epochs=10000, batch_size=1024, verbose=1, 
            callbacks=[es, mcp], validation_split=0.05)

print('=================1. basic print=================')
# 4. evaluate, predict
y1_predict= model.predict([xs_test, xk_test])
# print(y1_predict)

loss = model.evaluate([xs_test, xk_test], y_test, batch_size=32)
print('loss : ', loss)

result = model.predict([xs_pred, xk_pred])
print('26th start : ',result)

'''
print('=================2. load model=================')
model = load_model('./_save/MCP_M/SDJ_02_0723_1055_0801_4259067.500000.hdf5')

# 4. evaluate, predict
y1_predict= model.predict([xs_test, xk_test])
# print(y1_predict)

loss = model.evaluate([xs_test, xk_test], y_test)
print('loss : ', loss)

result = model.predict([xs_pred, xk_pred])
print('26th start : ',result)
'''

'''

'''