import numpy as np
from sklearn.metrics import r2_score
from sklearn.model_selection import train_test_split

# 1. data 구성

x_data = np.load('./_save/_NPY/k55_x_data_cancer.npy')
y_data = np.load('./_save/_NPY/k55_y_data_cancer.npy')

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
AdaBoostClassifier  avg score :  0.9649
BaggingClassifier  avg score :  0.942
BernoulliNB  avg score :  0.6274
CalibratedClassifierCV  avg score :  0.9263
CategoricalNB  avg score :  nan
ClassifierChain is Null
ComplementNB  avg score :  0.8963
DecisionTreeClassifier  avg score :  0.928
DummyClassifier  avg score :  0.6274
ExtraTreeClassifier  avg score :  0.9086
ExtraTreesClassifier  avg score :  0.9719
GaussianNB  avg score :  0.942
GaussianProcessClassifier  avg score :  0.9122
GradientBoostingClassifier  avg score :  0.9578
HistGradientBoostingClassifier  avg score :  0.9737
KNeighborsClassifier  avg score :  0.928
LabelPropagation  avg score :  0.3902
LabelSpreading  avg score :  0.3902
LinearDiscriminantAnalysis  avg score :  0.9614
LinearSVC  avg score :  0.9227
LogisticRegression  avg score :  0.9385
LogisticRegressionCV  avg score :  0.9578
MLPClassifier  avg score :  0.9403
MultiOutputClassifier is Null
MultinomialNB  avg score :  0.8928
NearestCentroid  avg score :  0.8893
NuSVC  avg score :  0.8735
OneVsOneClassifier is Null
OneVsRestClassifier is Null
OutputCodeClassifier is Null
PassiveAggressiveClassifier  avg score :  0.898
Perceptron  avg score :  0.7771
QuadraticDiscriminantAnalysis  avg score :  0.9525
RadiusNeighborsClassifier  avg score :  nan
RandomForestClassifier  avg score :  0.9684
RidgeClassifier  avg score :  0.9543
RidgeClassifierCV  avg score :  0.9561
SGDClassifier  avg score :  0.9122
SVC  avg score :  0.921
StackingClassifier is Null
VotingClassifier is Null
'''