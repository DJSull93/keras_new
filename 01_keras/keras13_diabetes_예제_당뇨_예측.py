from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
import numpy as np
from sklearn import datasets
# import pandas as pd
from sklearn.datasets import load_diabetes
from sklearn.metrics import r2_score
from sklearn.model_selection import train_test_split

'''
두번째 예제 문제인 당뇨병 예측 데이터셋.

기본적인 사용법은 보스턴 집값예측과 유사하므로 코드를 천천히 보고

실습을 해보는 것을 추천함. 

마찬가지로 앞으로 계속 나옴
'''

# R2 0.62 이상
# 1. data 구성
datasets = load_diabetes()
x = datasets.data # (442, 10)
y = datasets.target # (442,)

#print(datasets.feature_names) # ['age', 'sex', 'bmi', 'bp', 's1', 's2', 's3', 's4', 's5', 's6']
#print(datasets.DESCR) #   - age     age in years
    #   - sex
    #   - bmi     body mass index
    #   - bp      average blood pressure
    #   - s1      tc, T-Cells (a type of white blood cells)
    #   - s2      ldl, low-density lipoproteins
    #   - s3      hdl, high-density lipoproteins
    #   - s4      tch, thyroid stimulating hormone
    #   - s5      ltg, lamotrigine
    #   - s6      glu, blood sugar level
#print(y[:30]) # [151.  75. 141. 206. 135.  97. 138.  63. 110. 310. 101.  69. 179. 185.
 #118. 171. 166. 144.  97. 168.  68.  49.  68. 245. 184. 202. 137.  85.
 #131. 283.]
#print(np.min(y), np.max(y)) # 25.0 346.0

x_train, x_test, y_train, y_test = train_test_split(x, y,
      test_size=0.5, shuffle=True, random_state=66)

# 2. model 구성
model = Sequential()
model.add(Dense(70,input_dim=10))
model.add(Dense(60))
model.add(Dense(50))
model.add(Dense(30))
model.add(Dense(20))
model.add(Dense(10))
model.add(Dense(1))

# 3. 컴파일 훈련
model.compile(loss='mse', optimizer='adam')
model.fit(x_train, y_train, epochs=100, batch_size=8,
 verbose=2, validation_split=0.2, shuffle=True)

# 4. 평가 예측
y_predict = model.predict([x_test])
# print('x의 예측값 : ', y_predict)

loss = model.evaluate(x_test, y_test)
print('loss : ', loss)

r2 = r2_score(y_test, y_predict)
print('R^2 score : ', r2)

'''
epo = 100
loss :  2876.104736328125
R^2 score :  0.5056882002869799
'''