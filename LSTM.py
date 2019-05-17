#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu May 16 14:03:55 2019

@author: william
"""

from pandas_datareader import data, wb
import pandas as pd
import numpy as np

import seaborn as sns
import librosa
import os
import keras
import os
import pandas as pd
import librosa.display
import glob 

from numpy import random
import numpy
from sklearn.preprocessing import LabelEncoder
import numpy as np
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation, Flatten, LSTM, CuDNNLSTM, BatchNormalization
from keras.layers import Convolution2D, MaxPooling2D
from keras.optimizers import Adam
from keras.utils import np_utils
from sklearn import metrics 
import tensorflow as tf
from keras.backend.tensorflow_backend import set_session
import pickle
import matplotlib.pyplot as plt
from keras.optimizers import SGD
import keras.backend as K
import csv
K.set_image_dim_ordering('th')
e = np.array([])
listfile_name = []
mfcc_value=100
max_range=9000# array will have the shape (max_range+1,mfcc_value,number depending on duration)
duration=4
"""
file0 = os.path.join('./Train/0.wav')
X0, sr0 = librosa.load(file0)
f = librosa.feature.mfcc(y=X0, sr=sr0, n_mfcc=mfcc_value)
a = []

for i in range(1,max_range):
    try:
        file_name = os.path.join('./Train1',str(i)+'.wav')
        X, sr = librosa.load(file_name) 
        mfccs = librosa.feature.mfcc(y=X, sr=sr, n_mfcc=mfcc_value)
        if mfccs.shape[1]!=173:
            a.append(i)
        f = np.r_[f,mfccs]
        print(f.shape,"shape mfccs")#duration =2 =>87, duration=3 =>130, duration=4 =>173
        listfile_name.append(file_name)
    except:
        pass

with open('a.pickle', 'wb') as handle:
     pickle.dump(a, handle, protocol=pickle.HIGHEST_PROTOCOL)
with open('f.pickle', 'wb') as handle:
     pickle.dump(f, handle, protocol=pickle.HIGHEST_PROTOCOL)

with open("train.csv",'r') as files:
    with open("traintest.csv",'w') as f1:
        next(files) # skip header line
        for line in files:
            f1.write(line)

train = pd.read_csv(os.path.join( './train.csv'))

def parser(row):
   ID = row.ID
   label = row.Class
   return pd.Series([ID, label])

temp = train.apply(parser, axis=1)
temp.columns = ['ID','label']
y = np.array(temp.label.tolist())

with open('y.pickle', 'wb') as handle:
     pickle.dump(temp, handle, protocol=pickle.HIGHEST_PROTOCOL)

with open('y.pickle','rb') as handle:
     y=pickle.load(handle)
with open("train.csv",'r') as files:
    with open("traintest.csv",'w') as f1:
        n
with open('a.pickle','rb') as handle:
     a=pickle.load(handle)

tab =[]
with open('traintest.csv', 'r') as reader:
    reader = csv.reader(reader) 
    for row in reader:
        tab.append(row)
        for i in range(len(a)):
            row=np.asarray(row)
p=[]
tab1=np.asarray(tab)
for i in range(5434):
    for j in range(len(a)):
        if str(a[j])==tab1[i,0]:
            p.append(i)   
tab1=tab1.tolist()
p=np.asarray(p)
for index in sorted(p, reverse=True):
    del tab1[index]
tab1=np.asarray(tab1)
d={'ID':tab1[:,0], 'Class':tab1[:,1]}
df = pd.DataFrame(data=d)
df.to_csv('traintest.csv',index=False)
train = pd.read_csv(os.path.join( './traintest.csv'))
def parser(row):
   label = row.Class
   return pd.Series([label])
temp = train.apply(parser, axis=1)
temp.columns = ['label']
y = np.array(temp.label.tolist())
lb = LabelEncoder()
y = np_utils.to_categorical(lb.fit_transform(y))

with open('yalt.pickle', 'wb') as handle:
     pickle.dump(y, handle, protocol=pickle.HIGHEST_PROTOCOL)
"""
with open('f.pickle','rb') as handle:
     f=pickle.load(handle)

g= np.split(f, 4561)
g = np.asarray(g)
#g = np.expand_dims(g, axis=0)
#g = g.reshape(g.shape[0], 1, 100, 173)
with open('yalt.pickle','rb') as handle:
     y=pickle.load(handle)
ytrain=y[:4561]
g = g - g.mean(axis=0)       #normalization
g = g / np.abs(g).max(axis=0)#normalization
opt = SGD(lr=0.001)
opt1 = keras.optimizers.Adam(lr=0.008, decay=1e-3)
model = Sequential()
model.add(CuDNNLSTM(400, input_shape=(100,173), return_sequences=True))
model.add(Dropout(0.2))
model.add(CuDNNLSTM(800, return_sequences=False))
model.add(Dropout(0.5))
model.add(Dense(800, activation='relu'))
model.add(Dense(600, activation='relu'))
model.add(Dense(200, activation='relu'))
model.add(Dense(10, activation='softmax'))
print(model.summary())
# 8. Compile model
model.compile(loss='categorical_crossentropy',
              optimizer=opt1,
              metrics=['accuracy'])
# 9. Fit model on training data
model.fit(g, ytrain,batch_size=12, nb_epoch=300, verbose=1, validation_split=0.09)



