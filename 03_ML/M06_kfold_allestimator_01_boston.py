import numpy as np
from sklearn.metrics import r2_score
from sklearn.model_selection import train_test_split

# 1. data 구성

x_data = np.load('./_save/_NPY/k55_x_data_boston.npy')
y_data = np.load('./_save/_NPY/k55_y_data_boston.npy')

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
ARDRegression  avg score :  0.6985
AdaBoostRegressor  avg score :  0.8399
BaggingRegressor  avg score :  0.865
BayesianRidge  avg score :  0.7038
CCA  avg score :  0.6471
DecisionTreeRegressor  avg score :  0.7475
DummyRegressor  avg score :  -0.0135
ElasticNet  avg score :  0.6708
ElasticNetCV  avg score :  0.6565
ExtraTreeRegressor  avg score :  0.6499
ExtraTreesRegressor  avg score :  0.8728
GammaRegressor  avg score :  -0.0136
GaussianProcessRegressor  avg score :  -5.9286
GradientBoostingRegressor  avg score :  0.885
HistGradientBoostingRegressor  avg score :  0.8581
HuberRegressor  avg score :  0.584
IsotonicRegression  avg score :  nan
KNeighborsRegressor  avg score :  0.5286
KernelRidge  avg score :  0.6854
Lars  avg score :  0.6977
LarsCV  avg score :  0.6928
Lasso  avg score :  0.6657
LassoCV  avg score :  0.6779
LassoLars  avg score :  -0.0135
LassoLarsCV  avg score :  0.6965
LassoLarsIC  avg score :  0.713
LinearRegression  avg score :  0.7128
LinearSVR  avg score :  0.5195
MLPRegressor  avg score :  0.4728
MultiOutputRegressor is Null
MultiTaskElasticNet  avg score :  nan
MultiTaskElasticNetCV  avg score :  nan
MultiTaskLasso  avg score :  nan
MultiTaskLassoCV  avg score :  nan
NuSVR  avg score :  0.2295
OrthogonalMatchingPursuit  avg score :  0.5343
OrthogonalMatchingPursuitCV  avg score :  0.6578
PLSCanonical  avg score :  -2.2096
PLSRegression  avg score :  0.6847
PassiveAggressiveRegressor  avg score :  0.0706
PoissonRegressor  avg score :  0.7549
RANSACRegressor  avg score :  0.2399
RadiusNeighborsRegressor  avg score :  nan
RandomForestRegressor  avg score :  0.8732
RegressorChain is Null
Ridge  avg score :  0.7109
RidgeCV  avg score :  0.7128
SGDRegressor  avg score :  -5.03378968850645e+26
SVR  avg score :  0.1963
StackingRegressor is Null
TheilSenRegressor  avg score :  0.6736
TransformedTargetRegressor  avg score :  0.7128
TweedieRegressor  avg score :  0.6558
VotingRegressor is Null
'''