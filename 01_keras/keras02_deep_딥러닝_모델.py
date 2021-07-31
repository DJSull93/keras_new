from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
import numpy as np
import matplotlib.pyplot as plt

# 1. 데이터
x = np.array([1,2,3,4,5])
y = np.array([1,2,4,3,5])

# 2. 모델 구성
model = Sequential()
model.add(Dense(5, input_dim=1))
model.add(Dense(3))
model.add(Dense(3))
model.add(Dense(3))
model.add(Dense(3))
model.add(Dense(3))
model.add(Dense(4))
model.add(Dense(1))
'''
앞서 살펴본 기초 덴스 모델을 여러층으로 엮어, 인간 신경계와 유사한 모양으로
구성, 레이어 구성은 앞으로 무수히 많이 하게 되며, 레이어와 노드의 관계를 
충분히 학습하여야 함.

딥러닝 모델을 summary 함수로 프린트하면 하단과 같음. 
-> 첫 레이어에는 들어가는 x 데이터의 차원값을 input_dim으로 넣어줌
-> 두번째 레이어 부터는 노드의 개수만 적어주면, 자동으로 상위 레이어의
    노드 수 만큼을 인식함
-> 최종 레이어에는 반환되는 y의 차원이 노드의 수와 같음
-->> 예시에서는 y 데이터가 스칼라(1차원)이므로 노드에 1을 입력


$$$$ 인공지능의 모델 종류
인공지능은 크게 DNN(일반 덴스), CNN(주로 이미지), LSTM(시계열)
세가지로 나뉘고, Dense 레이어는 기본이 되는 모델
'''

'''
model.summary()
_________________________________________________________________
Layer (type)                 Output Shape              Param #
=================================================================
dense (Dense)                (None, 5)                 10
_________________________________________________________________
dense_1 (Dense)              (None, 3)                 18
_________________________________________________________________
dense_2 (Dense)              (None, 3)                 12
_________________________________________________________________
dense_3 (Dense)              (None, 3)                 12
_________________________________________________________________
dense_4 (Dense)              (None, 3)                 12
_________________________________________________________________
dense_5 (Dense)              (None, 3)                 12
_________________________________________________________________
dense_6 (Dense)              (None, 4)                 16
_________________________________________________________________
dense_7 (Dense)              (None, 1)                 5
=================================================================
Total params: 97
Trainable params: 97
Non-trainable params: 0
_________________________________________________________________
PS D:\study> 
'''


# 3. 컴파일, 훈련
model.compile(loss='mse', optimizer='adam')
model.fit(x, y, epochs=1000, batch_size=1)

# 4. 평가 예측
loss = model.evaluate(x, y)
print('loss : ', loss)
result = model.predict([6])
print('6의 예측값 : ', result)


# 5. 데이터 시각화
y_predict = model.predict(x)
plt.scatter(x,y)
plt.plot(x,y_predict, color='red')
plt.show()

# scatter를 통해 x, y 축을 그리고
# plot으로 x와 y_predict를 x, y로 지정, 색상은 붉은색으로 지정
# show는 표를 출력. 해당 코드를 실행하여 확인 요망