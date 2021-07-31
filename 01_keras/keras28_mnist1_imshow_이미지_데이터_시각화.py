import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.datasets import mnist

(x_train, y_train), (x_test, y_test) = mnist.load_data() # (60000, 28, 28) (60000,) (10000, 28, 28) (10000,)

# print(x_train.shape, y_train.shape)
# print(x_test.shape, y_test.shape)
print(x_train[111])
print("y[0] value", y_train[111]) # y[0] value 5 ; means x_train[0] = 5

plt.imshow(x_train[111], 'gray')
plt.show()

'''
예제 : 수기 숫자 데이터 다중 분류

이미지 데이터 시각화로 확인
실행 시 x_train의 111번째 이미지 시각화
'''