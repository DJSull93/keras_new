# flow to 100,000 
# make model and compare with banila
# save_dir -> temp and delete

from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.datasets import cifar10
import numpy as np

(x_train, y_train), (x_test, y_test) = cifar10.load_data()

train_datagen = ImageDataGenerator(
    rescale=1./255,
    horizontal_flip=True,
    vertical_flip=False,
    width_shift_range=0.1,
    height_shift_range=0.1,
    rotation_range=10,
    zoom_range=0.2,
    shear_range=0.5,
    fill_mode='nearest'
)

# test_datagen = ImageDataGenerator(rescale=1./255)

augment_size = 50000

randidx = np.random.randint(x_train.shape[0], size=augment_size) # take 40000 feature from train in random

# print(x_train.shape[0]) # 60000
# print(randidx) # [50653 24637 30472 ... 51686  3282 22404]
# print(randidx.shape) # (40000,)

x_argmented = x_train[randidx].copy()
y_argmented = y_train[randidx].copy()

x_argmented = x_argmented.reshape(x_argmented.shape[0], 32, 32, 3) # (40000, 28, 28, 1)
x_train = x_train.reshape(x_train.shape[0], 32, 32, 3) # (60000, 28, 28, 1)
x_test = x_test.reshape(x_test.shape[0], 32, 32, 3) # (10000, 28, 28, 1)

# print(x_argmented.shape, x_train.shape)

x_argmented = train_datagen.flow(x_argmented, 
                                np.zeros(augment_size),
                                batch_size=augment_size,
                                # save_to_dir='d:/temp/',
                                shuffle=False).next()[0]

# print(x_argmented.shape) # (40000, 28, 28, 1)

x_train = np.concatenate((x_train, x_argmented)) # (100000, 28, 28, 1) 
y_train = np.concatenate((y_train, y_argmented)) # (100000,)

# print(x_train.shape, y_train.shape)

# from 44_7
x_train = x_train.reshape(100000, 32*32*3) # (100000, 28, 28, 1)
x_test = x_test.reshape(10000, 32*32*3) # (10000, 28, 28, 1)

from sklearn.preprocessing import MinMaxScaler, StandardScaler, RobustScaler, MaxAbsScaler, QuantileTransformer, PowerTransformer
scaler = QuantileTransformer()
x_train = scaler.fit_transform(x_train)
x_test = scaler.transform(x_test) 

x_train = x_train.reshape(100000, 32*32, 3) # (100000, 28, 28, 1)
x_test = x_test.reshape(10000, 32*32, 3) # (10000, 28, 28, 1)

from sklearn.preprocessing import OneHotEncoder
one = OneHotEncoder()
y_train = y_train.reshape(-1,1)
y_test = y_test.reshape(-1,1)
one.fit(y_train)
y_train = one.transform(y_train).toarray() # (100000, 10)
y_test = one.transform(y_test).toarray() # (10000, 10)

# 2. model
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Conv1D, Flatten, MaxPool1D, GlobalAveragePooling1D, Dropout

model = Sequential()
model.add(Conv1D(filters=32, kernel_size=2, padding='same',                        
                        activation='relu' ,input_shape=(32*32, 3))) 
model.add(Conv1D(32, 2, padding='same', activation='relu'))                   
model.add(MaxPool1D())                                         
model.add(Conv1D(64, 2, padding='same', activation='relu'))                   
model.add(Conv1D(64, 2, padding='same', activation='relu'))    
model.add(Flatten())                                              
model.add(Dense(256, activation='relu'))
model.add(Dense(124, activation='relu'))
model.add(Dense(84, activation='relu'))
model.add(Dense(10, activation='softmax'))

# 3. comple fit // metrics 'acc'
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics='acc')

from tensorflow.keras.callbacks import EarlyStopping
es = EarlyStopping(monitor='val_loss', patience=10, mode='min', verbose=1)

import time 

start_time = time.time()
hist = model.fit(x_train, y_train, epochs=10000, batch_size=576, verbose=2,
    validation_split=0.05, callbacks=[es])
end_time = time.time() - start_time

# 4. predict eval -> no need to

acc = hist.history['acc']
val_acc = hist.history['val_acc']
loss = hist.history['loss']
val_loss = hist.history['val_loss']

loss = model.evaluate(x_test, y_test)
print('acc : ',acc[-10])
print('val_acc : ',val_acc[-10])
# print('loss : ',loss[-10])
print('val_loss : ',val_loss[-10])

'''
with flow
acc :  0.7419999837875366
val_acc :  0.6362000107765198
val_loss :  1.054795503616333

without flow
acc :  0.7980841994285583
val_acc :  0.6344000101089478
val_loss :  1.1285284757614136
'''