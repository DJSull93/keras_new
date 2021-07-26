# 보스턴 주택가격 완성할 것 
# 07. boston validation 적용 비교

# boston housing 
# loss, R2 print
# train 70%
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
import numpy as np
import matplotlib.pyplot as plt
from sklearn import datasets
from sklearn.datasets import load_boston
from sklearn.metrics import r2_score
from sklearn.model_selection import train_test_split

datasets = load_boston()

x = datasets.data # (506, 13) input_dim = 13
y = datasets.target # (506,) output_dim = 1

x_train, x_test, y_train, y_test = train_test_split(x, y,
      test_size=0.4, shuffle=True, random_state=12)

# print(datasets.feature_names) # ['CRIM' 'ZN' 'INDUS' 'CHAS' 'NOX' 'RM' 'AGE' 'DIS' 'RAD' 'TAX' 'PTRATIO' 'B' 'LSTAT'] B = 흑인 비율
# print(datasets.DESCR)


# 완성 한 뒤, 출력 결과 스샷

model = Sequential()
model.add(Dense(110, input_dim=13))
model.add(Dense(220))
model.add(Dense(10))
model.add(Dense(330))
model.add(Dense(20))
model.add(Dense(1))

model.compile(loss='mse', optimizer='adam')
model.fit(x_train, y_train, epochs=2100, batch_size=1,
 verbose=0, validation_split=0.3, shuffle=True)

y_predict = model.predict([x_test])
print('x의 예측값 : ', y_predict)

loss = model.evaluate(x_test, y_test)
print('loss : ', loss)

r2 = r2_score(y_test, y_predict)
print('R^2 score : ', r2)

'''
epo = 2100
loss :  25.138164520263672
R^2 score :  0.6805643002694967
x의 예측값 :  [[13.727425 ]
'''

