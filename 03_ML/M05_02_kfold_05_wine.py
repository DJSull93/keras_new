import numpy as np
from sklearn.model_selection import train_test_split, KFold, cross_val_score
import warnings 
warnings.filterwarnings(action='ignore')

# 1. data
x_data = np.load('./_save/_NPY/k55_x_data_wine.npy')
y_data = np.load('./_save/_NPY/k55_y_data_wine.npy')

n_split = 5
kfold = KFold(n_splits=n_split, shuffle=True, 
                random_state=66)

# 2. model 구성
from sklearn.svm import LinearSVC, SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier

# model = LinearSVC()
# acc score :  [0.45918367 0.38265306 0.48877551 0.38508682 0.45250255]
# avg score : 0.4336
# model = SVC()
# acc score :  [0.4622449  0.4377551  0.44693878 0.46373851 0.4473953 ]
# avg score : 0.4516
# model = KNeighborsClassifier()
# acc score :  [0.48979592 0.48469388 0.4755102  0.46373851 0.45863126]
# avg score : 0.4745
# model = LogisticRegression()
# acc score :  [0.47142857 0.45204082 0.44795918 0.48723187 0.46578141]
# avg score : 0.4649
# model = DecisionTreeClassifier()
# acc score :  [0.63367347 0.58571429 0.60306122 0.60776302 0.59959142]
# avg score : 0.606
model = RandomForestClassifier()
# acc score :  [0.71938776 0.67959184 0.6877551  0.68947906 0.68335036]
# avg score : 0.6919

# 3. 컴파일 훈련
# 4. 평가 예측

score = cross_val_score(model, x_data, y_data, cv=kfold)
print('acc score : ', score)
# =. fit + model.score / training (n_splits) times
print('avg score :', np.round(np.mean(score), 4)) # get avg acc
# categorical -> score : accuracy