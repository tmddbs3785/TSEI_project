from keras.models import Sequential
from keras.datasets import mnist
from keras.layers import Dense, Conv2D, MaxPooling2D, Dropout, Flatten, Conv1D, MaxPooling1D, Embedding
from keras.utils import np_utils
from keras.callbacks import ModelCheckpoint,EarlyStopping

import os
import matplotlib.pyplot as plt
import sys
import numpy
import tensorflow as tf
from tensorflow.contrib.learn.python.learn.estimators._sklearn import train_test_split

seed = 0
numpy.random.seed(seed)
tf.set_random_seed(seed)

npz_file = numpy.load('NPInter.npz')
npz_x_list = numpy.hstack([npz_file['XP'],npz_file['XR']])
X_train, X_test, Y_train, Y_test = train_test_split(npz_x_list,npz_file['Y'], test_size=0.1, random_state=seed)


model = Sequential()
model.add(Conv1D(32, kernel_size=4, input_shape=(739,1) ,activation='relu'))
model.add(Conv1D(64, 4, activation = 'relu'))
model.add(MaxPooling1D(pool_size=2))
model.add(Dropout(0.25))
model.add(Flatten())   # 2D -> 1D
model.add(Dense(128, activation='relu'))
model.add(Dense(32, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(1, activation='sigmoid'))

model.compile(loss='binary_crossentropy',optimizer='adam',metrics=['accuracy'])

history = model.fit(X_train, Y_train, validation_data=(X_test, Y_test), epochs=30, batch_size=128)

print("\n Test Accuracy: %.4f" % (model.evaluate(X_test, Y_test)[1]))

y_vloss = history.history['val_loss']
y_loss = history.history['loss']

x_len = numpy.arange(len(y_loss))

plt.plot(x_len, y_vloss, marker='.', c="red", label = 'Testset_loss')
plt.plot(x_len, y_loss, marker='.', c="blue", label = 'Trainset_loss')

plt.legend(loc='upper right')
plt.grid()
plt.xlabel('epoch')
plt.ylabel('loss')
plt.show()