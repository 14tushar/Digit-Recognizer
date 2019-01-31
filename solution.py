# -*- coding: utf-8 -*-
"""
Created on Wed Jan  9 02:06:41 2019

@author: Tushar
"""

import numpy as np
import pandas as pd
data1 = pd.read_csv('train.csv')
data2 = pd.read_csv('test.csv')
data3 = pd.read_csv('sample_submission.csv')
from keras.models import Sequential
from keras.layers import Dense, Conv2D, Flatten, AveragePooling2D

y_train = data1.iloc[:,[0]].values
X_train = (data1.iloc[:,1:].values)
X_test = (data2.iloc[:,:].values)

X_train = X_train.reshape(42000,28,28,1)
X_test = X_test.reshape(28000,28,28,1)

from keras.utils import to_categorical as tc
y_train = tc(y_train)

model = Sequential()
model.add(Conv2D(64, kernel_size=3, activation='relu', input_shape=(28,28,1)))
model.add(Conv2D(32, kernel_size=3, activation='relu'))
model.add(Flatten())
model.add(Dense(10, activation='softmax'))
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
model.fit(X_train, y_train, epochs=3)

y_pred = model.predict(X_train)
y_pred2 = model.predict(X_test)

max1=np.amax(y_pred2,axis=1)
y_pred3 = np.ndarray(28000)
for i in range(0,28000):
    y_pred3[i] = np.where(y_pred2[i]==max1[i])[0][0]

y_ = data3.iloc[:,0].values
df = pd.DataFrame(data={'ImageId':y_,'Label':y_pred3})
df.to_csv('Submission3.csv')

model2 = Sequential()
model2.add(Conv2D(16, kernel_size=3, activation='relu', input_shape=(28,28,1)))
model2.add(AveragePooling2D(pool_size=(2,2), strides=2, padding='valid', data_format='channels_last'))
model2.add(Conv2D(32, kernel_size=3, activation='relu'))
model2.add(AveragePooling2D(pool_size=(2,2), strides=2, padding='valid', data_format='channels_last'))
model2.add(Conv2D(64, kernel_size=3, activation='relu'))
model2.add(Flatten())
model2.add(Dense(100, activation='relu'))
model2.add(Dense(10, activation='softmax'))
model2.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
model2.fit(X_train, y_train, epochs=3)
model.fit(X_train, y_train, epochs=3)

y_pred = model.predict(X_train)
y_pred2 = model.predict(X_test)

max1=np.amax(y_pred2,axis=1)
y_pred3 = np.ndarray(28000)
for i in range(0,28000):
    y_pred3[i] = np.where(y_pred2[i]==max1[i])[0][0]

y_ = data3.iloc[:,0].values
df = pd.DataFrame(data={'ImageId':y_,'Label':y_pred3})
df.to_csv('Submission4.csv')
