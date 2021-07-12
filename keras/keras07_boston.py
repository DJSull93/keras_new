# 과제 3 0709
# boston housing 
# loss, R2 print
# train 70%
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
import numpy as np
import matplotlib.pyplot as plt
from sklearn import datasets
from sklearn.datasets import load_boston
from sklearn.metrics import r2_score
from sklearn.model_selection import train_test_split

datasets = load_boston()

x = datasets.data # (506, 13) input_dim = 13
y = datasets.target # (506,) output_dim = 1

x_train, x_test, y_train, y_test = train_test_split(x, y,
      test_size=0.2, shuffle=True, random_state=12)

print(x.shape)
print(y.shape)

print(datasets.feature_names) # ['CRIM' 'ZN' 'INDUS' 'CHAS' 'NOX' 'RM' 'AGE' 'DIS' 'RAD' 'TAX' 'PTRATIO' 'B' 'LSTAT'] B = 흑인 비율
print(datasets.DESCR)


# 완성 한 뒤, 출력 결과 스샷

model = Sequential()
model.add(Dense(15, input_dim=13))
model.add(Dense(22))
model.add(Dense(22))
model.add(Dense(11))
model.add(Dense(33))
model.add(Dense(33))
model.add(Dense(22))
model.add(Dense(21))
model.add(Dense(1))

model.compile(loss='mse', optimizer='adam')
model.fit(x_train, y_train, epochs=100, batch_size=1, verbose=0)

y_predict = model.predict([x_test])
print('x의 예측값 : ', y_predict)

loss = model.evaluate(x_test, y_test)
print('loss : ', loss)

r2 = r2_score(y_test, y_predict)
print('R^2 score : ', r2)
'''
epo = 100
loss :  21.215435028076172
R^2 score :  0.7432077221153058
x의 예측값 :  [[13.727425 ]

 [31.460625 ]
 [29.462017 ]
 [37.97054  ]
 [20.67062  ]
 [19.785673 ]
 [21.40147  ]
 [21.204853 ]
 [33.46039  ]
 [18.4698   ]
 [22.340975 ]
 [15.870033 ]
 [28.396297 ]
 [24.324862 ]
 [16.239511 ]
 [23.079227 ]
 [15.654264 ]
 [19.46925  ]
 [10.150562 ]
 [15.399672 ]
 [20.954195 ]
 [20.501232 ]
 [24.596119 ]
 [31.40215  ]
 [18.406984 ]
 [29.823757 ]
 [21.114483 ]
 [21.824589 ]
 [16.112556 ]
 [27.061815 ]
 [27.193691 ]
 [29.560966 ]
 [31.523226 ]
 [15.278336 ]
 [27.054487 ]
 [23.822706 ]
 [25.746239 ]
 [26.735952 ]
 [19.464134 ]
 [18.657856 ]
 [29.326239 ]
 [22.85563  ]
 [19.669683 ]
 [33.695133 ]
 [23.038404 ]
 [19.218225 ]
 [25.5608   ]
 [15.536123 ]
 [21.429308 ]
 [21.941397 ]
 [32.249573 ]
 [18.25261  ]
 [35.275963 ]
 [27.326262 ]
 [16.656984 ]
 [17.383284 ]
 [26.811275 ]
 [20.479567 ]
 [22.988443 ]
 [23.578468 ]
 [21.18969  ]
 [21.009584 ]
 [15.478912 ]
 [22.86363  ]
 [20.93922  ]
 [ 8.2966385]
 [17.993443 ]
 [24.806116 ]
 [26.673286 ]
 [19.743368 ]
 [ 8.720366 ]
 [22.169064 ]
 [23.92822  ]
 [19.048038 ]
 [24.425318 ]
 [28.81782  ]
 [31.918219 ]
 [10.025492 ]
 [20.92856  ]
 [ 9.518885 ]
 [30.024658 ]
 [15.8375435]
 [22.623035 ]
 [23.639479 ]
 [25.128923 ]
 [30.950989 ]
 [31.56254  ]
 [24.19822  ]
 [22.691206 ]
 [30.453712 ]
 [22.145906 ]
 [38.846115 ]
 [22.818884 ]
 [17.779346 ]
 [21.828072 ]
 [15.539738 ]
 [23.360723 ]
 [25.240965 ]
 [26.139498 ]
 [28.922892 ]
 [17.250015 ]
 [20.418272 ]
 [29.597979 ]
 [18.440685 ]
 [ 9.218425 ]
 [16.443604 ]
 [17.257309 ]
 [32.05423  ]
 [23.73814  ]
 [23.376444 ]
 [19.2076   ]
 [24.117626 ]
 [26.21147  ]
 [26.076258 ]
 [25.30545  ]
 [14.778086 ]
 [35.192974 ]
 [10.140741 ]
 [16.586578 ]
 [26.381649 ]
 [20.317995 ]
 [25.019262 ]
 [20.621115 ]
 [30.12685  ]
 [14.856068 ]
 [15.098553 ]
 [18.375044 ]
 [27.247972 ]
 [27.955778 ]
 [21.353909 ]
 [21.621513 ]
 [24.25497  ]
 [19.670334 ]
 [14.226711 ]
 [24.489338 ]
 [32.995712 ]
 [24.028927 ]
 [15.837959 ]
 [34.93743  ]
 [21.074192 ]
 [24.833515 ]
 [20.023647 ]
 [22.548431 ]
 [ 9.900106 ]
 [30.11632  ]
 [20.807121 ]
 [24.20659  ]
 [20.171562 ]
 [21.180166 ]
 [17.24137  ]
 [12.245539 ]
 [21.311924 ]]

'''



'''
.. _boston_dataset:

Boston house prices dataset
---------------------------

**Data Set Characteristics:**

    :Number of Instances: 506

    :Number of Attributes: 13 numeric/categorical predictive. Median Value (attribute 14) is usually the target.

    :Attribute Information (in order):
        - CRIM     per capita crime rate by town
        - ZN       proportion of residential land zoned for lots over 25,000 sq.ft.
        - INDUS    proportion of non-retail business acres per town
        - CHAS     Charles River dummy variable (= 1 if tract bounds river; 0 otherwise)
        - NOX      nitric oxides concentration (parts per 10 million)
        - RM       average number of rooms per dwelling
        - AGE      proportion of owner-occupied units built prior to 1940
        - DIS      weighted distances to five Boston employment 
centres
        - RAD      index of accessibility to radial highways    
        - TAX      full-value property-tax rate per $10,000     
        - PTRATIO  pupil-teacher ratio by town
        - B        1000(Bk - 0.63)^2 where Bk is the proportion 
of blacks by town
        - LSTAT    % lower status of the population
        - MEDV     Median value of owner-occupied homes in $1000's

    :Missing Attribute Values: None

    :Creator: Harrison, D. and Rubinfeld, D.L.

This is a copy of UCI ML housing dataset.
https://archive.ics.uci.edu/ml/machine-learning-databases/housing/


This dataset was taken from the StatLib library which is maintained at Carnegie Mellon University.

The Boston house-price data of Harrison, D. and Rubinfeld, D.L. 
'Hedonic
prices and the demand for clean air', J. Environ. Economics & Management,
vol.5, 81-102, 1978.   Used in Belsley, Kuh & Welsch, 'Regression diagnostics
...', Wiley, 1980.   N.B. Various transformations are used in the table on
pages 244-261 of the latter.

The Boston house-price data has been used in many machine learning papers that address regression
problems.

.. topic:: References

   - Belsley, Kuh & Welsch, 'Regression diagnostics: Identifying Influential Data and Sources of Collinearity', Wiley, 1980. 244-261.
   - Quinlan,R. (1993). Combining Instance-Based and Model-Based Learning. In Proceedings on the Tenth International Conference 
of Machine Learning, 236-243, University of Massachusetts, Amherst. Morgan Kaufmann.
'''