from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
import numpy as np

# 1. data
x_data = np.load('./_save/_NPY/k55_x_data_diabetes.npy')
y_data = np.load('./_save/_NPY/k55_y_data_diabetes.npy')

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
r2 :  0.4075401641529469
[0.05601602 0.01245586 0.3406549  0.08600056 0.04979057 0.05747931
 0.06066375 0.01843002 0.2408341  0.0776749 ]
'''