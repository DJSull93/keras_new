'''
분류 문제 : classifier 사용
싸이킷런 라이브러리 활용, 머신러닝 모델 다중 호출,

for loop 구성으로 각 모델별 성능 평가
'''

import numpy as np
from sklearn.model_selection import train_test_split

# 1. data
x_data = np.load('./_save/_NPY/k55_x_data_cancer.npy')
y_data = np.load('./_save/_NPY/k55_y_data_cancer.npy')

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
AdaBoostClassifier  accuracy :  0.972027972027972
BaggingClassifier  accuracy :  0.9440559440559441
BernoulliNB  accuracy :  0.6433566433566433
CalibratedClassifierCV  accuracy :  0.972027972027972
CategoricalNB  accuracy :  0.6503496503496503
ClassifierChain is Null
ComplementNB  accuracy :  0.9090909090909091
DecisionTreeClassifier  accuracy :  0.951048951048951
DummyClassifier  accuracy :  0.6433566433566433
ExtraTreeClassifier  accuracy :  0.9370629370629371
ExtraTreesClassifier  accuracy :  0.972027972027972
GaussianNB  accuracy :  0.9440559440559441
GaussianProcessClassifier  accuracy :  0.965034965034965
GradientBoostingClassifier  accuracy :  0.958041958041958
HistGradientBoostingClassifier  accuracy :  0.972027972027972
KNeighborsClassifier  accuracy :  0.972027972027972
LabelPropagation  accuracy :  0.972027972027972
LabelSpreading  accuracy :  0.965034965034965
LinearDiscriminantAnalysis  accuracy :  0.958041958041958
LinearSVC  accuracy :  0.972027972027972
LogisticRegression  accuracy :  0.965034965034965
LogisticRegressionCV  accuracy :  0.9790209790209791
MLPClassifier  accuracy :  0.951048951048951
MultiOutputClassifier is Null
MultinomialNB  accuracy :  0.8951048951048951
NearestCentroid  accuracy :  0.9370629370629371
NuSVC  accuracy :  0.9440559440559441
OneVsOneClassifier is Null
OneVsRestClassifier is Null
OutputCodeClassifier is Null
PassiveAggressiveClassifier  accuracy :  0.972027972027972
Perceptron  accuracy :  0.972027972027972
QuadraticDiscriminantAnalysis  accuracy :  0.9440559440559441
RadiusNeighborsClassifier is Null
RandomForestClassifier  accuracy :  0.972027972027972
RidgeClassifier  accuracy :  0.951048951048951
RidgeClassifierCV  accuracy :  0.958041958041958
SGDClassifier  accuracy :  0.972027972027972
SVC  accuracy :  0.9790209790209791
StackingClassifier is Null
VotingClassifier is Null
'''