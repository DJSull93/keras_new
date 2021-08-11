import numpy as np
from sklearn.metrics import r2_score
from sklearn.model_selection import train_test_split

# 1. data 구성

x_data = np.load('./_save/_NPY/k55_x_data_diabetes.npy')
y_data = np.load('./_save/_NPY/k55_y_data_diabetes.npy')

# 2. model 구성
from sklearn.utils import all_estimators
import warnings 
warnings.filterwarnings(action='ignore')

allAlgorithms = all_estimators(type_filter='regressor') # or classifier
# print('Model num : ', len(allAlgorithms)) # Model num :  54

from sklearn.metrics import accuracy_score, r2_score
from sklearn.model_selection import KFold, cross_val_score

kfold = KFold(n_splits=5, shuffle=True, random_state=66)

for (name, algorithm) in allAlgorithms:
    try:
        model = algorithm()

        score = cross_val_score(model, x_data, y_data, cv=kfold)
        
        print(name, ' avg score : ', np.round(np.mean(score), 4))

    except:
        # continue
        print(name, 'is Null')

'''
ARDRegression  avg score :  0.4923
AdaBoostRegressor  avg score :  0.426
BaggingRegressor  avg score :  0.3813
BayesianRidge  avg score :  0.4893
CCA  avg score :  0.438
DecisionTreeRegressor  avg score :  -0.1343
DummyRegressor  avg score :  -0.0033
ElasticNet  avg score :  0.0054
ElasticNetCV  avg score :  0.4394
ExtraTreeRegressor  avg score :  -0.0772
ExtraTreesRegressor  avg score :  0.449
GammaRegressor  avg score :  0.0027
GaussianProcessRegressor  avg score :  -11.0753
GradientBoostingRegressor  avg score :  0.4372
HistGradientBoostingRegressor  avg score :  0.3947
HuberRegressor  avg score :  0.4822
IsotonicRegression  avg score :  nan
KNeighborsRegressor  avg score :  0.3673
KernelRidge  avg score :  -3.5938
Lars  avg score :  -0.1495
LarsCV  avg score :  0.4879
Lasso  avg score :  0.3518
LassoCV  avg score :  0.487
LassoLars  avg score :  0.3742
LassoLarsCV  avg score :  0.4866
LassoLarsIC  avg score :  0.4912
LinearRegression  avg score :  0.4876
LinearSVR  avg score :  -0.3681
MLPRegressor  avg score :  -2.9666
MultiOutputRegressor is Null
MultiTaskElasticNet  avg score :  nan
MultiTaskElasticNetCV  avg score :  nan
MultiTaskLasso  avg score :  nan
MultiTaskLassoCV  avg score :  nan
NuSVR  avg score :  0.1618
OrthogonalMatchingPursuit  avg score :  0.3121
OrthogonalMatchingPursuitCV  avg score :  0.4857
PLSCanonical  avg score :  -1.2086
PLSRegression  avg score :  0.4842
PassiveAggressiveRegressor  avg score :  0.4654
PoissonRegressor  avg score :  0.3341
RANSACRegressor  avg score :  -0.0236
RadiusNeighborsRegressor  avg score :  -0.0033
RandomForestRegressor  avg score :  0.4324
RegressorChain is Null
Ridge  avg score :  0.4212
RidgeCV  avg score :  0.4884
SGDRegressor  avg score :  0.4089
SVR  avg score :  0.1591
StackingRegressor is Null
TheilSenRegressor  avg score :  0.47
TransformedTargetRegressor  avg score :  0.4876
TweedieRegressor  avg score :  0.0032
VotingRegressor is Null
'''