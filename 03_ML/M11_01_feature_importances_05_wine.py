from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
import numpy as np

# 1. data
x_data = np.load('./_save/_NPY/k55_x_data_wine.npy')
y_data = np.load('./_save/_NPY/k55_y_data_wine.npy')

x_train, x_test, y_train, y_test = train_test_split(x_data, y_data,
      test_size=0.25, shuffle=True, random_state=1234)

# 2. model
model = RandomForestClassifier()

# 3. fit
model.fit(x_train, y_train)

# 4. pred
acc = model.score(x_test, y_test)
print('acc : ', acc) 

print(model.feature_importances_)

'''
acc :  0.6710204081632654
[0.07364588 0.10232124 0.08157261 0.08736527 0.0834641  0.09156399
 0.09347665 0.1030764  0.08596974 0.08171103 0.11583309]
 '''