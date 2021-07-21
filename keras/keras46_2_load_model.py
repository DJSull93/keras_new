import numpy as np
from sklearn import datasets
from sklearn.datasets import load_diabetes
from sklearn.metrics import r2_score
from sklearn.model_selection import train_test_split

# 1. data 구성
datasets = load_diabetes()
x = datasets.data # (442, 10)
y = datasets.target # (442,)

x_train, x_test, y_train, y_test = train_test_split(x, y,
      test_size=0.25, shuffle=True, random_state=9)

from sklearn.preprocessing import MinMaxScaler, StandardScaler, RobustScaler, MaxAbsScaler, QuantileTransformer, PowerTransformer
scaler = MaxAbsScaler()
x_train = scaler.fit_transform(x_train)
x_test = scaler.transform(x_test)

# 2. model 구성
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import Dense

# model = load_model('./_save/keras46_1_save_model_1.h5')
model = load_model('./_save/keras46_1_save_model_2.h5')

# model.summary()
'''
# 3. 컴파일 훈련
model.compile(loss='mse', optimizer='adam')

from tensorflow.keras.callbacks import EarlyStopping
es = EarlyStopping(monitor='val_loss', patience=30, mode='min', verbose=1)

import time 
start_time = time.time()
model.fit(x_train, y_train, epochs=1000, batch_size=32,
 verbose=2, validation_split=0.22, shuffle=True, 
 callbacks=[es])
end_time = time.time() - start_time
'''
# 4. 평가 예측
y_predict = model.predict([x_test])
# print('x의 예측값 : ', y_predict)

loss = model.evaluate(x_test, y_test)
# print("total time : ", end_time)

print('loss : ', loss)

r2 = r2_score(y_test, y_predict)
print('R^2 score : ', r2)

'''
total time :  3.239145040512085
loss :  2297.768798828125
R^2 score :  0.6017423424094249

loss :  9791.0556640625
R^2 score :  -0.6970212495685684
'''
