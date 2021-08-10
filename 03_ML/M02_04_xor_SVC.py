'''
xor gate  - and, or, xor gate
SVC : 비선형 분포 데이터 구분 가능
'''

from sklearn.svm import SVC
import numpy as np
from sklearn.metrics import accuracy_score

# 1. data
x_data = [[ 0, 0 ], [ 0, 1 ], [ 1, 0 ], [ 1, 1 ]]
y_data = [ 0, 1, 1, 0 ]

# 2. model
model = SVC()

# 3. train
model.fit(x_data, y_data)

# 4. predict
y_pred = model.predict(x_data)
print(x_data, 'result :', y_pred)
acc = accuracy_score(y_data, y_pred)
print('acc_score : ', acc)

result = model.score(x_data, y_data)
print('model_score : ', result)

'''
[[0, 0], [0, 1], [1, 0], [1, 1]] result : [0 1 1 0]
acc_score :  1.0
model_score :  1.0
'''