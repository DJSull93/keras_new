# Practice : n_comp upper than 0.95

from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from xgboost import XGBClassifier
from sklearn.model_selection import train_test_split
import numpy as np
from tensorflow.keras.datasets import mnist

from sklearn.decomposition import PCA

(x_train, _), (x_test, _) = mnist.load_data() # (60000, 28, 28), (10000, 28, 28)

x = np.append(x_train, x_test, axis=0) # (70000, 28, 28)

x = x.reshape(x.shape[0], x.shape[1]*x.shape[2])

pca = PCA(n_components=784)
x = pca.fit_transform(x)

pca_EVR = pca.explained_variance_ratio_
print(pca_EVR)

cumsum = np.cumsum(pca_EVR)
print(cumsum)

print(np.argmax(cumsum >= 1.0)+1) 
# 0.95 : 154
# 0.99 : 331
# 0.999 : 486
# 1.0 : 713
# total 784 -> lot of dummy data 