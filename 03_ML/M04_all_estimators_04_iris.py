'''
분류 문제 : classifier 사용
싸이킷런 라이브러리 활용, 머신러닝 모델 다중 호출,

for loop 구성으로 각 모델별 성능 평가
'''

import numpy as np
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.utils import all_estimators
import warnings 
warnings.filterwarnings(action='ignore')

# 1. data
datasets = load_iris()

x = datasets.data 
y = datasets.target 

x_train, x_test, y_train, y_test = train_test_split(x, y,
      test_size=0.25, shuffle=True, random_state=1)

from sklearn.preprocessing import MinMaxScaler, StandardScaler, RobustScaler, MaxAbsScaler, QuantileTransformer, PowerTransformer
scaler = MinMaxScaler()
x_train = scaler.fit_transform(x_train) 
x_test = scaler.transform(x_test) 

# 2. model 구성

allAlgorithms = all_estimators(type_filter='classifier') # or regressor
# print(allAlgorithms)
# print('Model num : ', len(allAlgorithms)) # Model num :  41

from sklearn.metrics import accuracy_score

for (name, algorithm) in allAlgorithms:
    try:
        model = algorithm()

        model.fit(x_train, y_train)

        y_pred = model.predict(x_test)
        acc = accuracy_score(y_test, y_pred)

        print(name, ' accuracy : ', acc)
    except:
        # continue
        print(name, 'is Null')

'''
AdaBoostClassifier  accuracy :  0.9736842105263158
BaggingClassifier  accuracy :  0.9736842105263158
BernoulliNB  accuracy :  0.23684210526315788
CalibratedClassifierCV  accuracy :  0.8157894736842105
CategoricalNB is Null
ClassifierChain is Null
ComplementNB  accuracy :  0.5789473684210527
DecisionTreeClassifier  accuracy :  0.9736842105263158
DummyClassifier  accuracy :  0.23684210526315788
ExtraTreeClassifier  accuracy :  0.9210526315789473
ExtraTreesClassifier  accuracy :  0.9736842105263158
GaussianNB  accuracy :  0.9736842105263158
GaussianProcessClassifier  accuracy :  0.9210526315789473
GradientBoostingClassifier  accuracy :  0.9736842105263158
HistGradientBoostingClassifier  accuracy :  0.9736842105263158
KNeighborsClassifier  accuracy :  0.9736842105263158
LabelPropagation  accuracy :  0.9736842105263158
LabelSpreading  accuracy :  0.9736842105263158
LinearDiscriminantAnalysis  accuracy :  1.0
LinearSVC  accuracy :  0.8421052631578947
LogisticRegression  accuracy :  0.9210526315789473
LogisticRegressionCV  accuracy :  0.9736842105263158
MLPClassifier  accuracy :  0.9473684210526315
MultiOutputClassifier is Null
MultinomialNB  accuracy :  0.5789473684210527
NearestCentroid  accuracy :  0.9473684210526315
NuSVC  accuracy :  0.9736842105263158
OneVsOneClassifier is Null
OneVsRestClassifier is Null
OutputCodeClassifier is Null
PassiveAggressiveClassifier  accuracy :  0.9473684210526315
Perceptron  accuracy :  0.6052631578947368
QuadraticDiscriminantAnalysis  accuracy :  1.0
RadiusNeighborsClassifier  accuracy :  0.5789473684210527
RandomForestClassifier  accuracy :  0.9736842105263158
RidgeClassifier  accuracy :  0.7631578947368421
RidgeClassifierCV  accuracy :  0.7631578947368421
SGDClassifier  accuracy :  0.9473684210526315
SVC  accuracy :  0.9736842105263158
StackingClassifier is Null
VotingClassifier is Null
'''

'''
CV : cross validation 
LogisticRegressionCV  accuracy :  0.9736842105263158
'''