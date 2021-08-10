'''
회귀 문제 : regressor 사용
싸이킷런 라이브러리 활용, 머신러닝 모델 다중 호출,

for loop 구성으로 각 모델별 성능 평가
'''

import numpy as np

x_data = np.load('./_save/_NPY/k55_x_data_diabetes.npy')
y_data = np.load('./_save/_NPY/k55_y_data_diabetes.npy')

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
ARDRegression  r2_score :  0.4987485812817073
AdaBoostRegressor  r2_score :  0.3582543410493624
BaggingRegressor  r2_score :  0.3855930283473594
BayesianRidge  r2_score :  0.49956752296917606
CCA  r2_score :  0.4869640906496757
DecisionTreeRegressor  r2_score :  -0.23896841343139674
DummyRegressor  r2_score :  -0.00015425885559339214
ElasticNet  r2_score :  0.4169160025227353
ElasticNetCV  r2_score :  0.4971200605755439
ExtraTreeRegressor  r2_score :  -0.12861609399853458
ExtraTreesRegressor  r2_score :  0.3786318211817278
GammaRegressor  r2_score :  0.3616505326338155
GaussianProcessRegressor  r2_score :  -0.22951138312875585
GradientBoostingRegressor  r2_score :  0.3903548282636675
HistGradientBoostingRegressor  r2_score :  0.28899497703380905
HuberRegressor  r2_score :  0.5070148505109835
IsotonicRegression is Null
KNeighborsRegressor  r2_score :  0.3959429578616862
KernelRidge  r2_score :  -1.5495554456946001
Lars  r2_score :  0.4919866521464168
LarsCV  r2_score :  0.5010892359535759
Lasso  r2_score :  0.4919182290867833
LassoCV  r2_score :  0.49750113330389734
LassoLars  r2_score :  0.36543887418957965
LassoLarsCV  r2_score :  0.4951942790678263
LassoLarsIC  r2_score :  0.4994051517531072
LinearRegression  r2_score :  0.5063891053505039
LinearSVR  r2_score :  0.23177510238993193
MLPRegressor  r2_score :  -1.217600984789771
MultiOutputRegressor is Null
MultiTaskElasticNet is Null
MultiTaskElasticNetCV is Null
MultiTaskLasso is Null
MultiTaskLassoCV is Null
NuSVR  r2_score :  0.1519174519715223
OrthogonalMatchingPursuit  r2_score :  0.3293449115305742
OrthogonalMatchingPursuitCV  r2_score :  0.44354253337919747
PLSCanonical  r2_score :  -0.9750792277922924
PLSRegression  r2_score :  0.4766139460349792
PassiveAggressiveRegressor  r2_score :  0.5022694273882804
PoissonRegressor  r2_score :  0.4940498536933051
RANSACRegressor  r2_score :  0.0900960262472209
RadiusNeighborsRegressor is Null
RandomForestRegressor  r2_score :  0.38342651026375285
RegressorChain is Null
Ridge  r2_score :  0.5047469047003306
RidgeCV  r2_score :  0.49818373153525974
SGDRegressor  r2_score :  0.5020968929083192
SVR  r2_score :  0.1483247077641816
StackingRegressor is Null
TheilSenRegressor  r2_score :  0.5008845937521549
TransformedTargetRegressor  r2_score :  0.5063891053505039
TweedieRegressor  r2_score :  0.3652830573730501
VotingRegressor is Null
'''