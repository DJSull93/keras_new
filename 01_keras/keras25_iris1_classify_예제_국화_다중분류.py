'''
이진 이후는 다중 분류임 
앞서 사용한 sigmoid, binary-crossentropy가 아닌 어떤 
활성함수와 지표를 쓰는지를 중점으로 학습할 것
'''

# iris 국화 예제 / 다중분류 / 변수에따른 종류

# labeling - indcoding 필요 
# 0, 1, 2 라벨 수 끼리의 관계가 동일해야 함
# scalar -> vector : ex) 0 = (1,0,0) 1=(0,1,0) 2=(0,0,1) 

# indcoding ex)
# [1, 2, 0, 1] (4, ) 
# [0,1,0]
# [0,0,1]
# [1,0,0]
# [0,1,0] -> (4, 3)

# 다중분류 데이터에 대해서는 다음이 강제됨
# output activation = softmax
# loss = categorical_crossentropy

from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import Dense, Input
import numpy as np
from sklearn.datasets import load_iris
from sklearn.metrics import r2_score
from sklearn.model_selection import train_test_split

# 1. data
datasets = load_iris()

# print(datasets.DESCR)
# print(datasets.feature_names) # ['sepal length (cm)', 'sepal width (cm)', 'petal length (cm)', 'petal width (cm)']     

x = datasets.data # (150, 4)
y = datasets.target # (150, ) - incoding-> (150,3)

# incoding -> One_Hot_Incoding 
from tensorflow.keras.utils import to_categorical
y = to_categorical(y)

# print(y[:5])
# print(y.shape)

x_train, x_test, y_train, y_test = train_test_split(x, y,
      test_size=0.1, shuffle=True, random_state=1)

from sklearn.preprocessing import MinMaxScaler, StandardScaler, RobustScaler, MaxAbsScaler, QuantileTransformer, PowerTransformer
scaler = QuantileTransformer()
scaler.fit(x_train) 
x_train = scaler.transform(x_train) 
x_test = scaler.transform(x_test) 

# 2. model 구성
# 결과가 0, 이므로 새로운 activation(활성화) func. -> sigmoid 사용
model = Sequential()
model.add(Dense(270, input_dim=(4), activation="relu"))
model.add(Dense(240, activation="relu"))
model.add(Dense(200, activation="relu"))
model.add(Dense(124, activation="relu"))
model.add(Dense(100, activation="relu"))
model.add(Dense(3, activation="softmax"))

# 3. 컴파일 훈련
# data 형태가 다르므로 mse 대신 categorical_crossentropy 사용
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['mse', 'accuracy'])

from tensorflow.keras.callbacks import EarlyStopping
es = EarlyStopping(monitor='loss', patience=20, mode='min', verbose=1)

model.fit(x_train, y_train, epochs=10000, batch_size=32, verbose=2,
    validation_split=0.5, callbacks=[es])

# 4. 평가 예측
# y_predict = model.predict([x_test])
# print('x의 예측값 : ', y_predict)
# r2 = r2_score(y_test, y_predict)
# print('R^2 score : ', r2)

loss = model.evaluate(x_test, y_test)
print('loss[binary] : ', loss[0])
print('loss[accuracy] : ', loss[2])

# original
print(y_test[:5])
# after softmax
y_predict = model.predict(x_test[:5])
print(y_predict)

'''
QuantileTransformer
Epoch 01330: early stopping
loss[binary] :  1.7642857983446447e-06
loss[accuracy] :  1.0

loss[binary] :  0.00010815416317200288
loss[accuracy] :  1.0

y_test[:5] =
[[1. 0. 0.]
 [0. 1. 0.]
 [0. 1. 0.]
 [1. 0. 0.]
 [0. 0. 1.]]
y_predict[:5] =
[[9.99999762e-01 1.79692137e-07 5.77463733e-29]
 [2.22764589e-04 9.99777138e-01 1.69286594e-07]
 [7.09510815e-08 9.99999881e-01 6.74925046e-11]
 [9.99999762e-01 2.95443215e-07 1.23214999e-27]
 [1.11267174e-17 5.99346473e-04 9.99400616e-01]]
'''