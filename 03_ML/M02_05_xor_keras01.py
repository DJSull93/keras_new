'''
xor gate  - and, or, xor gate
'''

from sklearn.svm import SVC
import numpy as np
from sklearn.metrics import accuracy_score
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense


# 1. data
x_data = [[ 0, 0 ], [ 0, 1 ], [ 1, 0 ], [ 1, 1 ]]
y_data = [ 0, 1, 1, 0 ]

# 2. model
# model = SVC()
model = Sequential()
model.add(Dense(1, input_dim=2, activation='sigmoid'))
# single perceptron '=. LinearSVC

# 3. compile train
model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['acc'])
model.fit(x_data, y_data, batch_size=1, epochs=100)

# 4. predict
y_pred = model.predict(x_data)
print(x_data, 'result :', y_pred)

y_pred = np.round(y_pred, 0)
print(y_pred)

acc = accuracy_score(y_data, y_pred)
print('acc_score : ', acc)

result = model.evaluate(x_data, y_data)
print('model_score : ', result[1])

'''
[[0, 0], [0, 1], [1, 0], [1, 1]] result : [0 1 1 0]
acc_score :  1.0
model_score :  1.0
'''