# 1. data
# x1 기상청 (온도, 습도, 불쾌지수), x2 주가 (삼전, SK, Kakao)
import numpy as np

x1 = np.array([range(100), range(301, 401), range(1, 101)])
x2 = np.array([range(101, 201), range(411, 511), range(100, 200)])
x1 = np.transpose(x1) # (100,3)
x2 = np.transpose(x2) # (100,3)
y1 = np.array(range(1001, 1101)) # (100,)

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
from tensorflow.keras.layers import concatenate, Concatenate

merge1 = concatenate([output1, output2]) # merge 도 layer 임
last_output = Dense(1)(merge1)


model = Model(inputs=[input1, input2], outputs=last_output)

model.summary()




