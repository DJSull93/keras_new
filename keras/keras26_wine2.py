# wine2 예제
# acc 0.82 이상 만들 것

# wine csv -> sep ; 
# ./   : current folder
# ../ : upper folder

from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import Dense, Input
from sklearn.metrics import r2_score
from sklearn.model_selection import train_test_split
import numpy as np
import pandas as pd

# 1. data
datasets = pd.read_csv('../_data/winequality-white.csv', sep=';',
                       index_col=None, header=0 ) # (4898, 12)

# print(datasets[:5])
# print(datasets.shape)

x = datasets.iloc[:,0:11] # (4898, 11)
y = datasets.iloc[:,[11]] # (4898, 10)

# print(x[:5], y[:5])
# print(datasets.info)
# print(datasets.describe())
# print(np.unique(y))

# incoding -> One_Hot_Incoding 
# to_categorical 0, 1, 2 없으나 자동 생성
# -> keras OneHotencoder 사용
from sklearn.preprocessing import OneHotEncoder
one = OneHotEncoder()
one.fit(y)
y = one.transform(y).toarray()

x_train, x_test, y_train, y_test = train_test_split(x, y,
      test_size=0.005, shuffle=True, random_state=24)

from sklearn.preprocessing import MinMaxScaler, StandardScaler, RobustScaler, MaxAbsScaler, QuantileTransformer, PowerTransformer
scaler = RobustScaler()
scaler.fit(x_train) 
x_train = scaler.transform(x_train) 
x_test = scaler.transform(x_test) 

# 2. model 구성
model = Sequential()
model.add(Dense(240, input_dim=(11), activation="relu"))
model.add(Dense(240, activation="relu"))
model.add(Dense(200, activation="relu"))
model.add(Dense(124, activation="relu"))
model.add(Dense(60, activation="relu"))
model.add(Dense(7, activation="softmax"))

# 3. 컴파일 훈련
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['mse', 'accuracy'])

from tensorflow.keras.callbacks import EarlyStopping
es = EarlyStopping(monitor='val_loss', patience=240, mode='min', verbose=1)

model.fit(x_train, y_train, epochs=10000, batch_size=64, verbose=2,
    validation_split=0.002, callbacks=[es])

# 4. 평가 예측
# y_predict = model.predict([x_test])
# print('x의 예측값 : ', y_predict)
# r2 = r2_score(y_test, y_predict)
# print('R^2 score : ', r2)

loss = model.evaluate(x_test, y_test)
print('loss[category] : ', loss[0])
print('loss[accuracy] : ', loss[2])

# original
# print(y_test[:5])
# after softmax
# y_predict = model.predict(x_test[:5])
# print(y_predict)


'''
quantile
loss[category] :  2.3964686393737793
loss[accuracy] :  0.6432653069496155

robust
loss[category] :  1.6828268766403198
loss[accuracy] :  0.6653061509132385

loss[category] :  2.300713062286377
loss[accuracy] :  0.6938775777816772

loss[category] :  3.029759168624878
loss[accuracy] :  0.7162162065505981

loss[category] :  3.742452383041382
loss[accuracy] :  0.7200000286102295

loss[category] :  2.6221673488616943
loss[accuracy] :  0.7200000286102295

robust, randomstate = 24
loss[category] :  1.750553011894226
loss[accuracy] :  0.7599999904632568

loss[category] :  1.5601791143417358
loss[accuracy] :  0.800000011920929
'''

# dataset 라벨 , 셋 자체에 대한 처리로 정확도 올릴 수 있음 