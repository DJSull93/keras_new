import numpy as np

'''
np.save('./_save/_NPY/k59_mw_x_train', arr=xy_train[0][0])
np.save('./_save/_NPY/k59_mw_y_train', arr=xy_train[0][1])
np.save('./_save/_NPY/k59_mw_x_test', arr=xy_test[0][0])
np.save('./_save/_NPY/k59_mw_y_test', arr=xy_test[0][1])
'''

# 1. data
x_train = np.load('./_save/_NPY/k59_mw_x_train.npy')
x_test = np.load('./_save/_NPY/k59_mw_x_test.npy')
y_train = np.load('./_save/_NPY/k59_mw_y_train.npy')
y_test = np.load('./_save/_NPY/k59_mw_y_test.npy')
x_pred = np.load('./_save/_NPY/k59_mw_x_pred.npy')

# 2. model
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Conv2D, Flatten, MaxPooling2D

model = Sequential()

model.add(Conv2D(filters = 32, kernel_size=(3,3), input_shape =(150,150,3), activation= 'relu'))
model.add(Conv2D(filters = 32, kernel_size=(3,3),  activation= 'relu'))
# model.add(MaxPooling2D(2,2))
model.add(Conv2D(filters = 64, kernel_size=(2,2),  activation= 'relu'))
model.add(Conv2D(filters = 64, kernel_size=(3,3),  activation= 'relu'))
# model.add(MaxPooling2D(2,2))
model.add(Flatten())
model.add(Dense(128, activation= 'relu'))
model.add(Dense(64, activation= 'relu'))
model.add(Dense(1, activation= 'sigmoid'))

# 3. compile train
model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['acc'])

from tensorflow.keras.callbacks import EarlyStopping
es = EarlyStopping(monitor='val_loss', patience=10, mode='min', verbose=1)

# hist = model.fit_generator(xy_train, epochs=50,
#  steps_per_epoch=32,
#  validation_data=xy_test,
#  validation_steps=4,
#  callbacks=[es]) # 32 -> 160/5

hist = model.fit(x_train, y_train, epochs=500,
                callbacks=[es],
                validation_split=0.2,
                steps_per_epoch=32,
                validation_steps=4)

acc = hist.history['acc']
val_acc = hist.history['val_acc']
loss = hist.history['loss']
val_loss = hist.history['val_loss']

# visualize upper data

# print('val_acc : ',val_acc[:-1])

loss = model.evaluate(x_test, y_test)
print('acc : ',acc[-1])
print('loss : ',loss[0])

y_predict = model.predict([x_pred])
res = (1-y_predict) * 100
print('남자일 확률 : ',res, '%')


'''
for pic 586
acc :  0.982367753982544
loss :  3.6602091789245605
남자일 확률 :  [[98.27359]] %

for pic 601
acc :  0.9743072986602783
loss :  3.1328744888305664
남자일 확률 :  [[99.99987]] %
'''