'''
회귀 문제 : regressor 사용
싸이킷런 라이브러리 활용, 머신러닝 모델 다중 호출,

for loop 구성으로 각 모델별 성능 평가
'''

import numpy as np

x_data = np.load('./_save/_NPY/k55_x_data_boston.npy')
y_data = np.load('./_save/_NPY/k55_y_data_boston.npy')

from sklearn.metrics import r2_score
from sklearn.model_selection import train_test_split

# 1. data 구성

x_train, x_test, y_train, y_test = train_test_split(x_data, y_data,
      test_size=0.2, shuffle=True, random_state=66)

# print(x_train.shape, x_test.shape) # (404, 13) (102, 13)

from sklearn.preprocessing import MinMaxScaler, StandardScaler, RobustScaler, MaxAbsScaler, QuantileTransformer, PowerTransformer
scaler = RobustScaler()
x_train = scaler.fit_transform(x_train)
x_test = scaler.transform(x_test)

# 2. model 구성
from sklearn.utils import all_estimators
import warnings 
warnings.filterwarnings(action='ignore')

allAlgorithms = all_estimators(type_filter='regressor') # or classifier
# print(allAlgorithms)
print('Model num : ', len(allAlgorithms)) # Model num :  54

from sklearn.metrics import accuracy_score, r2_score

for (name, algorithm) in allAlgorithms:
    try:
        model = algorithm()

        model.fit(x_train, y_train)

        y_pred = model.predict(x_test)
        r2 = r2_score(y_test, y_pred)

        print(name, ' r2_score : ', r2)
    except:
        # continue
        print(name, 'is Null')

'''
ARDRegression  r2_score :  0.811900484682101
AdaBoostRegressor  r2_score :  0.894161226633045
BaggingRegressor  r2_score :  0.897925960589923
BayesianRidge  r2_score :  0.812462470639635
CCA  r2_score :  0.7913477184424631
DecisionTreeRegressor  r2_score :  0.8227544238659181
DummyRegressor  r2_score :  -0.0005370164400797517
ElasticNet  r2_score :  0.6752389310767979
ElasticNetCV  r2_score :  0.8086935754915119
ExtraTreeRegressor  r2_score :  0.7660020583330519
ExtraTreesRegressor  r2_score :  0.9352400041292697
GammaRegressor  r2_score :  0.6588547962085214
GaussianProcessRegressor  r2_score :  0.3671981056458581
GradientBoostingRegressor  r2_score :  0.9455541932943392
HistGradientBoostingRegressor  r2_score :  0.9323326124661162
HuberRegressor  r2_score :  0.7958267917603676
IsotonicRegression is Null
KNeighborsRegressor  r2_score :  0.825402539162744
KernelRidge  r2_score :  -1.6212370488952192
Lars  r2_score :  0.7746736096721591
LarsCV  r2_score :  0.7981576314184013
Lasso  r2_score :  0.7356524169152884
LassoCV  r2_score :  0.8123782845529534
LassoLars  r2_score :  -0.0005370164400797517
LassoLarsCV  r2_score :  0.8127604328474294
LassoLarsIC  r2_score :  0.8131423868817644
LinearRegression  r2_score :  0.8111288663608665
LinearSVR  r2_score :  0.7889826498560739
MLPRegressor  r2_score :  0.709694983386761
MultiOutputRegressor is Null
MultiTaskElasticNet is Null
MultiTaskElasticNetCV is Null
MultiTaskLasso is Null
MultiTaskLassoCV is Null
NuSVR  r2_score :  0.5055284475423076
OrthogonalMatchingPursuit  r2_score :  0.582761757138145
OrthogonalMatchingPursuitCV  r2_score :  0.7861744773872902
PLSCanonical  r2_score :  -2.231707974142581
PLSRegression  r2_score :  0.802731314200789
PassiveAggressiveRegressor  r2_score :  0.587721908723158
PoissonRegressor  r2_score :  0.86186804709122
RANSACRegressor  r2_score :  0.644848638686639
RadiusNeighborsRegressor is Null
RandomForestRegressor  r2_score :  0.9255051730682383
RegressorChain is Null
Ridge  r2_score :  0.8120405915660547
RidgeCV  r2_score :  0.812040591565488
SGDRegressor  r2_score :  0.7943805932966264
SVR  r2_score :  0.4943122225189971
StackingRegressor is Null
TheilSenRegressor  r2_score :  0.7867542900743018
TransformedTargetRegressor  r2_score :  0.8111288663608665
TweedieRegressor  r2_score :  0.6501917474297376
VotingRegressor is Null
'''