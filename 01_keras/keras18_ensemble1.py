from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score
import numpy as np

# 1. data
# x1 기상청 (온도, 습도, 불쾌지수), x2 주가 (삼전, SK, Kakao)

x1 = np.array([range(100), range(301, 401), range(1, 101)])
x2 = np.array([range(101, 201), range(411, 511), range(100, 200)])
x1 = np.transpose(x1) # (100,3)
x2 = np.transpose(x2) # (100,3)
y1 = np.array(range(1001, 1101)) # (100,)

x1_train, x1_test, x2_train, x2_test, y1_train, y1_test = train_test_split(x1, x2, y1,
      train_size=0.9, shuffle=True, random_state=10)

# 2. model
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense, Input

# 2-1. model1
input1 = Input(shape=(3,))
dense1 = Dense(10, activation='relu', name='dense1')(input1)
dense2 = Dense(7, activation='relu', name='dense2')(dense1)
dense3 = Dense(5, activation='relu', name='dense3')(dense2)
output1 = Dense(11, name='output1')(dense3)

# 2-2. model2
input2 = Input(shape=(3,))
dense11 = Dense(10, activation='relu', name='dense11')(input2)
dense12 = Dense(10, activation='relu', name='dense12')(dense11)
dense13 = Dense(10, activation='relu', name='dense13')(dense12)
dense14 = Dense(10, activation='relu', name='dense14')(dense13)
output2 = Dense(12, name='output2')(dense14)

# 2-3. model 1, 2 merge
# concatenate 소문자 메소드 대문자 클래스 차이 없음, 버전에 따른 흔적임
from tensorflow.keras.layers import concatenate

merge1 = concatenate([output1, output2]) # merge 도 layer 임
merge2 = Dense(10)(merge1)
merge3 = Dense(5, activation='relu')(merge2)
last_output = Dense(1)(merge3)


model = Model(inputs=[input1, input2], outputs=last_output)

# model.summary()

# 3. compile, train
model.compile(loss='mse', optimizer='adam', metrics=['mae'])
model.fit([x1_train, x2_train], [y1_train], epochs=100, batch_size=8, verbose=2)

# 4. evaluate, predict
y1_predict= model.predict([x1_test, x2_test])
print(y1_predict)

R2_1 = r2_score(y1_test, y1_predict)
print("R2_1 : ", R2_1)

loss = model.evaluate([x1_test, x2_test], y1_test)
print('loss["mse"] : ', loss[0])
print('loss["mae"] : ', loss[1])

