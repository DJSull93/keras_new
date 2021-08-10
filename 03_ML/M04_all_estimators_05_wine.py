'''
분류 문제 : classifier 사용
싸이킷런 라이브러리 활용, 머신러닝 모델 다중 호출,

for loop 구성으로 각 모델별 성능 평가
'''

import numpy as np
from sklearn.model_selection import train_test_split

# 1. data
x_data = np.load('./_save/_NPY/k55_x_data_wine.npy')
y_data = np.load('./_save/_NPY/k55_y_data_wine.npy')

x_train, x_test, y_train, y_test = train_test_split(x_data, y_data,
      test_size=0.25, shuffle=True, random_state=66)

from sklearn.preprocessing import MinMaxScaler, StandardScaler, RobustScaler, MaxAbsScaler, QuantileTransformer, PowerTransformer
scaler = MaxAbsScaler()
scaler.fit(x_train) 
x_train = scaler.transform(x_train) 
x_test = scaler.transform(x_test) 

# 2. model 구성
from sklearn.utils import all_estimators
import warnings 
warnings.filterwarnings(action='ignore')

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
AdaBoostClassifier  accuracy :  0.41959183673469386
BaggingClassifier  accuracy :  0.6587755102040816
BernoulliNB  accuracy :  0.4546938775510204
CalibratedClassifierCV  accuracy :  0.5379591836734694
CategoricalNB  accuracy :  0.4555102040816327
ClassifierChain is Null
ComplementNB  accuracy :  0.3983673469387755
DecisionTreeClassifier  accuracy :  0.6212244897959184
DummyClassifier  accuracy :  0.4546938775510204
ExtraTreeClassifier  accuracy :  0.5893877551020408
ExtraTreesClassifier  accuracy :  0.7036734693877551
GaussianNB  accuracy :  0.45877551020408164
GaussianProcessClassifier  accuracy :  0.5330612244897959 - takes time to long
GradientBoostingClassifier  accuracy :  0.5918367346938775
HistGradientBoostingClassifier  accuracy :  0.689795918367347
KNeighborsClassifier  accuracy :  0.5820408163265306
LabelPropagation  accuracy :  0.4775510204081633
LabelSpreading  accuracy :  0.4685714285714286
LinearDiscriminantAnalysis  accuracy :  0.5248979591836734
LinearSVC  accuracy :  0.5314285714285715
LogisticRegression  accuracy :  0.5395918367346939
LogisticRegressionCV  accuracy :  0.5412244897959184
MLPClassifier  accuracy :  0.5346938775510204
MultiOutputClassifier is Null
MultinomialNB  accuracy :  0.4546938775510204
NearestCentroid  accuracy :  0.2897959183673469
NuSVC is Null
OneVsOneClassifier is Null
OneVsRestClassifier is Null
OutputCodeClassifier is Null
PassiveAggressiveClassifier  accuracy :  0.4579591836734694
Perceptron  accuracy :  0.5134693877551021
QuadraticDiscriminantAnalysis  accuracy :  0.4857142857142857
RadiusNeighborsClassifier  accuracy :  0.4546938775510204
RandomForestClassifier  accuracy :  0.6922448979591836
RidgeClassifier  accuracy :  0.5330612244897959
RidgeClassifierCV  accuracy :  0.5330612244897959
SGDClassifier  accuracy :  0.5061224489795918
SVC  accuracy :  0.5224489795918368
StackingClassifier is Null
VotingClassifier is Null
'''