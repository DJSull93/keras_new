# mnist PCA and CNN layer
# compare with banila

# Practice : n_comp upper than 0.95 : 154 -> 13 * 13 = 169
# make model -> Tensorflow DNN, compare with banila

from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from xgboost import XGBClassifier
from sklearn.model_selection import train_test_split
import numpy as np
from tensorflow.keras.datasets import mnist

from sklearn.decomposition import PCA
import warnings 
warnings.filterwarnings(action='ignore')

# 1. data
(x_train, y_train), (x_test, y_test) = mnist.load_data() # (60000, 28, 28) (60000,) (10000, 28, 28) (10000,)

x = np.append(x_train, x_test, axis=0) # (70000, 28, 28)
y = np.append(y_train, y_test, axis=0) # (70000,)

x = x.reshape(x.shape[0], x.shape[1]*x.shape[2])

cut = 20

pca = PCA(n_components=cut*cut)
x = pca.fit_transform(x)

x = x.reshape(x.shape[0], cut, cut)

x_train, x_test, y_train, y_test = train_test_split(x, y,
      test_size=0.14, shuffle=True, random_state=77)

x_train = x_train.reshape(x_train.shape[0], cut*cut)
x_test = x_test.reshape(x_test.shape[0], cut*cut)

from sklearn.preprocessing import MinMaxScaler, StandardScaler, RobustScaler, MaxAbsScaler, QuantileTransformer, PowerTransformer
scaler = QuantileTransformer()
x_train = scaler.fit_transform(x_train)
x_test = scaler.transform(x_test) 

x_train = x_train.reshape(x_train.shape[0], cut, cut)
x_test = x_test.reshape(x_test.shape[0], cut, cut)

from sklearn.preprocessing import OneHotEncoder
one = OneHotEncoder()
y_train = y_train.reshape(-1,1)
y_test = y_test.reshape(-1,1)
one.fit(y_train)
y_train = one.transform(y_train).toarray() # (60000, 10)
y_test = one.transform(y_test).toarray() # (10000, 10)

# 2. model
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Conv1D, Flatten, MaxPool1D, GlobalAveragePooling1D, Dropout
from sklearn.model_selection import RandomizedSearchCV, GridSearchCV, KFold, cross_val_score, train_test_split

model = Sequential()
model.add(Conv1D(filters=32, kernel_size=2, padding='same',                        
                        activation='relu' ,input_shape=(cut, cut))) 
model.add(Conv1D(32, 2, padding='same', activation='relu'))                   
model.add(MaxPool1D())                                         
model.add(Conv1D(64, 2, padding='same', activation='relu'))                   
model.add(Conv1D(64, 2, padding='same', activation='relu'))    
model.add(Flatten())                                              
model.add(Dense(256, activation='relu'))
model.add(Dense(128, activation='relu'))
model.add(Dense(64, activation='relu'))
model.add(Dense(10, activation='softmax'))


# 3. comple fit // metrics 'acc'
from tensorflow.keras.optimizers import Adam

op = Adam(lr = 0.001)
model.compile(loss='categorical_crossentropy', optimizer=op, metrics='acc')

from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
es = EarlyStopping(monitor='val_loss', patience=20, 
                mode='min', verbose=1)
lr = ReduceLROnPlateau(monitor='val_loss', patience=5, 
                mode='auto', verbose=1, factor=0.8)

import time
start_time = time.time()
hist = model.fit(x_train, y_train, epochs=100, batch_size=512, verbose=2,
    validation_split=0.15, callbacks=[es, lr])
end_time = time.time() - start_time

# 4. predict eval -> no need to

loss = model.evaluate(x_test, y_test)
acc = hist.history['acc']
val_acc = hist.history['val_acc']
loss = hist.history['loss']
val_loss = hist.history['val_loss']

print("total time : ", end_time)
print('acc : ',acc[-20])
print('val_acc : ',val_acc[-20])
print('loss : ',loss[-20])
print('val_loss : ',val_loss[-20])

'''
CNN
time =  12.263086080551147
loss :  0.08342713862657547
acc :  0.9821000099182129

DNN
time :  20.324632167816162
loss :  0.09825558215379715
acc :  0.9785000085830688

PCA_DNN 0.95
time :  55.58410167694092
loss :  0.0827394351363182
acc :  0.9757142663002014

PCA_DNN 0.999
time :  36.505455017089844
loss :  0.2318371683359146
acc :  0.9433731436729431

PCA_CNN 0.95
total time :  30.180605173110962
acc :  0.966131865978241
val_acc :  0.8970099687576294
loss :  0.10952125489711761
val_loss :  0.37102603912353516
'''