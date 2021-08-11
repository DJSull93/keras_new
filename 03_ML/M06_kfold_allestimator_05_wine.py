import numpy as np
from sklearn.metrics import r2_score
from sklearn.model_selection import train_test_split

# 1. data 구성

x_data = np.load('./_save/_NPY/k55_x_data_wine.npy')
y_data = np.load('./_save/_NPY/k55_y_data_wine.npy')

# 2. model 구성
from sklearn.utils import all_estimators
import warnings 
warnings.filterwarnings(action='ignore')

allAlgorithms = all_estimators(type_filter='classifier') # or classifier
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
AdaBoostClassifier  avg score :  0.4171
BaggingClassifier  avg score :  0.6515
BernoulliNB  avg score :  0.4488
CalibratedClassifierCV  avg score :  0.5035
CategoricalNB  avg score :  nan
ClassifierChain is Null
ComplementNB  avg score :  0.3659
DecisionTreeClassifier  avg score :  0.6076
DummyClassifier  avg score :  0.4488
ExtraTreeClassifier  avg score :  0.6072
ExtraTreesClassifier  avg score :  0.6929
GaussianNB  avg score :  0.4485
GaussianProcessClassifier  avg score :  0.5792
GradientBoostingClassifier  avg score :  0.5972
HistGradientBoostingClassifier  avg score :  0.6731
KNeighborsClassifier  avg score :  0.4745
LabelPropagation  avg score :  0.5731
LabelSpreading  avg score :  0.5733
LinearDiscriminantAnalysis  avg score :  0.5308
LinearSVC  avg score :  0.4555
LogisticRegression  avg score :  0.4649
LogisticRegressionCV  avg score :  0.5029
MLPClassifier  avg score :  0.4998
MultiOutputClassifier is Null
MultinomialNB  avg score :  0.3975
NearestCentroid  avg score :  0.1072
NuSVC  avg score :  nan
OneVsOneClassifier is Null
OneVsRestClassifier is Null
OutputCodeClassifier is Null
PassiveAggressiveClassifier  avg score :  0.3597
Perceptron  avg score :  0.328
QuadraticDiscriminantAnalysis  avg score :  0.4771
RadiusNeighborsClassifier  avg score :  nan
RandomForestClassifier  avg score :  0.6897
RidgeClassifier  avg score :  0.5255
RidgeClassifierCV  avg score :  0.5255
SGDClassifier  avg score :  0.2656
SVC  avg score :  0.4516
StackingClassifier is Null
VotingClassifier is Null
'''