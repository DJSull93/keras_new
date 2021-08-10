# file name must be my predict number
import numpy as np
import pandas as pd
from pandas.core.arrays.sparse import dtype
from sklearn import datasets
from sklearn.metrics import r2_score
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
sam['date'] =pd.to_datetime(sam.date)
sk['date'] =pd.to_datetime(sk.date)
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
size = 20

def split_x(dataset, size):
    st = []
    for i in range(len(dataset) - size + 1):
        subset = dataset[i : (i + size)]
        st.append(subset)
    return np.array(st)

sam = split_x(sam, size) # (2581, 20, 5) 
sk = split_x(sk, size) # (2581, 20, 5)

# sam ->, s 
xs = sam[:-2,:,[0,1,2,4]] # (2580, 20, 4) 
ys = sam[:-2,:,3] # (2580, 20)
xs_pred = sam[-1,:,:] # (20, 4) 
xs_pred = np.delete(xs_pred, 3, 1)

xs_now = sam[-1,:,:] # (20, 4)
xs_now = np.delete(xs_now, 3, 1)

ys_answ = sam[-1,-1,3]

# print(xs.shape, ys.shape)
# print(xs[-1,:,:])
# print(ys[-1,:])
# print(ys)
# print(xs_pred)

# sk -> k
xk = sk[:-2,:,[0,1,2,3,4]] # (2580, 20, 5) 
xk_pred = sk[-1,:,:] # (20, 5)
xk_now = sk[-2,:,:] # (20, 5)


# print(xs_pred.shape, xk_pred.shape)

# train test
xs_train, xs_test, xk_train, xk_test, y_train, y_test = train_test_split(xs, xk, ys,
      test_size=0.25, shuffle=True, random_state=9)

# print(xs_train.shape, xk_train.shape) # (1935, 20, 4) (1935, 20, 5)
# print(xs_test.shape, xk_test.shape) # (645, 20, 4) (645, 20, 5)


# preproccess
xs_train = xs_train.reshape(1935*20, 4)
xs_test = xs_test.reshape(645*20, 4)
xs_pred = xs_pred.reshape(20, 4)
xs_now = xs_now.reshape(20, 4)
xk_train = xk_train.reshape(1935*20, 5)
xk_test = xk_test.reshape(645*20, 5)
xk_pred = xk_pred.reshape(20, 5)
xk_now = xk_now.reshape(20, 5)

from sklearn.preprocessing import MinMaxScaler, StandardScaler, RobustScaler, MaxAbsScaler, QuantileTransformer, PowerTransformer
scaler = MinMaxScaler()

xs_train = scaler.fit_transform(xs_train)
xs_test = scaler.transform(xs_test)
xs_pred = scaler.transform(xs_pred)
xs_now = scaler.transform(xs_now)
xk_train = scaler.fit_transform(xk_train)
xk_test = scaler.transform(xk_test)
xk_pred = scaler.transform(xk_pred)
xk_now = scaler.transform(xk_now)

xs_train = xs_train.reshape(1935, 20, 4)
xs_test = xs_test.reshape(645, 20, 4)
xs_pred = xs_pred.reshape(1, 20, 4)
xs_now = xs_now.reshape(1, 20, 4)

xk_train = xk_train.reshape(1935, 20, 5)
xk_test = xk_test.reshape(645, 20, 5)
xk_pred = xk_pred.reshape(1, 20, 5)
xk_now = xk_now.reshape(1, 20, 5)


# 2. model
from tensorflow.keras.models import Sequential, Model, load_model
from tensorflow.keras.layers import Dense, Conv1D, LSTM, Dropout, Input
'''
# 2-1. model1
input1 = Input(shape=(20,4))
xx = Conv1D(filters=32, kernel_size=5,
           padding="same",
           activation="relu")(input1)
xx = LSTM(units=16, activation='relu')(xx)
output1 = Dense(8)(xx)

# 2-2. model2
input2 = Input(shape=(20,5))
xx = Conv1D(filters=32, kernel_size=5,
           padding="same",
           activation="relu")(input2)
xx = LSTM(units=16, activation='relu')(xx)
output2 = Dense(8)(xx)

# 2-3. model 1, 2 merge
from tensorflow.keras.layers import concatenate

merge1 = concatenate([output1, output2]) # merge 도 layer 임
xx = Dense(4, activation='relu')(merge1)
last_output = Dense(1)(xx)

model = Model(inputs=[input1, input2], outputs=last_output)

# 3. compile train

model.compile(loss='mse', optimizer='adam')

from keras.callbacks import EarlyStopping, ModelCheckpoint
es = EarlyStopping(monitor='val_loss', patience=30, 
            mode='auto', verbose=1, restore_best_weights=True)

###################################################################
# file name auto change and save
# 'filename(default)' + 'time' + 'loss'
# %m month %d day %H hour %M minute
import datetime
date = datetime.datetime.now()
date_time = date.strftime("%m%d_%H%M")

filepath = './_save/'
filename = '{epoch:04d}_{val_loss:4f}.hdf5'
modelpath = "".join([filepath, "SDJ_S01_", date_time, "_", filename])
###################################################################

mcp = ModelCheckpoint(monitor='val_loss', mode='auto', verbose=1,
            save_best_only=True, 
            filepath= modelpath)

model.fit([xs_train, xk_train], y_train, epochs=1000, batch_size=2048, verbose=1, 
            callbacks=[es, mcp], validation_split=0.2)

print('=================1. basic print=================')
# 4. evaluate, predict
y1_predict= model.predict([xs_test, xk_test])
# print(y1_predict)

loss = model.evaluate([xs_test, xk_test], y_test)
print('loss : ', loss)

today = model.predict([xs_now, xk_now])
print('0722 : 79700 pred : ', today)

result = model.predict([xs_pred, xk_pred])
print(result)


'''
print('=================2. load model=================')
model = load_model('./_save/MCP/SDJ_S01_0722_1922_0450_3084131.250000.hdf5')

# 4. evaluate, predict
y1_predict= model.predict([xs_test, xk_test])
# print(y1_predict)

loss = model.evaluate([xs_test, xk_test], y_test)
print('loss : ', loss)

today = model.predict([xs_now, xk_now])
print('0722 : 79700 pred : ', today)

result = model.predict([xs_pred, xk_pred])
print(result)

'''
loss :  3013218.25
0722 : 79700 pred :  [[79082.87]]
[[79133.984]]
'''