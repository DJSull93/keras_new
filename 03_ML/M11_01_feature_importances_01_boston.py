from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
import numpy as np

# 1. data
x_data = np.load('./_save/_NPY/k55_x_data_boston.npy')
y_data = np.load('./_save/_NPY/k55_y_data_boston.npy')

x_train, x_test, y_train, y_test = train_test_split(x_data, y_data,
      test_size=0.15, shuffle=True, random_state=1234)

# 2. model
model = RandomForestRegressor()

# 3. fit
model.fit(x_train, y_train)

# 4. pred
r2 = model.score(x_test, y_test)
print('r2 : ', r2) 

print(model.feature_importances_)

'''
r2 :  0.921791314508456
[0.03670381 0.00089329 0.00717647 0.00086608 0.02286844 0.38507632
 0.01371094 0.06777476 0.00397583 0.01388399 0.01765741 0.01394635
 0.41546631]
'''