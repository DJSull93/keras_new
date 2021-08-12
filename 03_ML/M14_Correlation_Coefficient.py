# correlation coefficient 상관계수

import numpy as np
import pandas as pd
from sklearn.datasets import load_iris

datasets = load_iris()
# print(datasets.keys())
# dict_keys(['data', 'target', 'frame', 'target_names', 'DESCR', 'feature_names', 'filename'])

# print(datasets.target_names)
# ['setosa' 'versicolor' 'virginica']

x = datasets.data # (150, 4) 
y = datasets.target # (150,)

# df = pd.DataFrame(x, columns=datasets.feature_names)
df = pd.DataFrame(x, columns=datasets['feature_names'])

# print(df)

# add y columns
df["Target"] = y

print(df.head())

print('================correlation coefficient hitmap================')

print(df.corr())

import matplotlib.pyplot as plt
import seaborn as sns

sns.set(font_scale=1.2)
sns.heatmap(data=df.corr(), square=True, annot=True, cbar=True)

plt.show()