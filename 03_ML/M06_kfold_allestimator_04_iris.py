import numpy as np
from sklearn.datasets import load_iris
from sklearn.utils import all_estimators
import warnings 
warnings.filterwarnings(action='ignore')

# 1. data
x_data = np.load('./_save/_NPY/k55_x_data_iris.npy')
y_data = np.load('./_save/_NPY/k55_y_data_iris.npy')

# 2. model 구성

allAlgorithms = all_estimators(type_filter='classifier') # or regressor

from sklearn.metrics import accuracy_score
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
AdaBoostClassifier  avg score :  0.8867
BaggingClassifier  avg score :  0.9533
BernoulliNB  avg score :  0.2933
CalibratedClassifierCV  avg score :  0.9133
CategoricalNB  avg score :  0.9333
ClassifierChain is Null
ComplementNB  avg score :  0.6667
DecisionTreeClassifier  avg score :  0.9533
DummyClassifier  avg score :  0.2933
ExtraTreeClassifier  avg score :  0.92
ExtraTreesClassifier  avg score :  0.9533
GaussianNB  avg score :  0.9467
GaussianProcessClassifier  avg score :  0.96
GradientBoostingClassifier  avg score :  0.9667
HistGradientBoostingClassifier  avg score :  0.94
KNeighborsClassifier  avg score :  0.96
LabelPropagation  avg score :  0.96
LabelSpreading  avg score :  0.96
LinearDiscriminantAnalysis  avg score :  0.98
LinearSVC  avg score :  0.9667
LogisticRegression  avg score :  0.9667
LogisticRegressionCV  avg score :  0.9733
MLPClassifier  avg score :  0.9867
MultiOutputClassifier is Null
MultinomialNB  avg score :  0.9667
NearestCentroid  avg score :  0.9333
NuSVC  avg score :  0.9733
OneVsOneClassifier is Null
OneVsRestClassifier is Null
OutputCodeClassifier is Null
PassiveAggressiveClassifier  avg score :  0.8
Perceptron  avg score :  0.78
QuadraticDiscriminantAnalysis  avg score :  0.98
RadiusNeighborsClassifier  avg score :  0.9533
RandomForestClassifier  avg score :  0.9467
RidgeClassifier  avg score :  0.84
RidgeClassifierCV  avg score :  0.84
SGDClassifier  avg score :  0.8067
SVC  avg score :  0.9667
StackingClassifier is Null
VotingClassifier is Null
'''