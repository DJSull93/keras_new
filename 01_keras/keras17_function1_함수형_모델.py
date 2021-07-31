from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import Dense, Input
import numpy as np
from sklearn.metrics import r2_score
from sklearn.model_selection import train_test_split
'''
초반에 언급했듯, 모델은 시퀀셜, 함수형 두가지가 있으며 
성능은 동일

함수형만 가능한 기능이 있으므로 학습 필요
'''
# 1. data 
x = np.array ([range(100), range(301, 401), range(1, 101), 
               range(100), range(401, 501)])
x = np.transpose(x) # (100, 5)

y = np.array([range(701, 801), range(101, 201)]) 
y = np.transpose(y) # (100, 2)

# 2. model
# sequantial, model 성능은 동일, summary에서 input 명시만 차이가 있음
input1 = Input(shape=(5,))
dense1 = Dense(3)(input1)
dense2 = Dense(4)(dense1)
dense3 = Dense(10)(dense2)
output1 = Dense(2)(dense3)

model = Model(inputs=input1, outputs=output1)
'''
함수형 모델은 시퀀셜과 달리 첫 레이어를 받아들이는 shape만 지정 후
다음 층에서 이전 층을 계승하는 표기를 해야함

또한 시퀀셜과 차이점으로 앞에 model = model.Sequantial()로 지정하지 않고,
최종단 이후에 입력 출력 레이어를 각 모델의 시작과 끝으로 지정// 29번 라인
'''

# model = Sequential()
# model.add(Dense(3, input_dim=(5)))
# model.add(Dense(4))
# model.add(Dense(10))
# model.add(Dense(2))
model.summary()

# 3. compile, train

# 4. evaluate, predict

