#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Apr 25 13:48:07 2019

@author: william
"""

import keras
import os
import pandas as pd
import librosa.display
import glob 
import IPython.display as ipd
from numpy import random
import numpy
from sklearn.preprocessing import LabelEncoder
import numpy as np
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation, Flatten
from keras.layers import Convolution2D, MaxPooling2D
from keras.optimizers import Adam
from keras.utils import np_utils
from sklearn import metrics 
import tensorflow as tf
from keras.backend.tensorflow_backend import set_session
import pickle
import matplotlib.pyplot as plt
config = tf.ConfigProto()
#limit at 20% gpu memory
#config.gpu_options.per_process_gpu_memory_fraction = 0.2
set_session(tf.Session(config=config))
#allouer + de mémoire sur le gpu"
config = tf.ConfigProto()
config.gpu_options.allow_growth = True
sess = tf.Session(config=config)




train = pd.read_csv(os.path.join( './train.csv'))
test = pd.read_csv(os.path.join('./test.csv'))
i = random.choice(train.index)
audio_name = train.ID[i]
path = os.path.join('./Train', str(audio_name) + '.wav')

"""
def parser(row):
   # function to load files and extract features
   file_name = os.path.join('./Train', str(row.ID) + '.wav')
   # handle exception to check if there isn't a file which is corrupted
   try:
      # here kaiser_fast is a technique used for faster extraction
      X, sr = librosa.load(file_name, res_type='kaiser_fast') 

      
      
      mfccs = np.mean(librosa.feature.mfcc(y=X, sr=sr, n_mfcc=40).T,axis=0) 
   except Exception as e:
      print("Error encountered while parsing file: ", file_name)
      return None, None
   
   feature = mfccs
   label = row.Class
 
   return pd.Series([feature, label])

temp = train.apply(parser, axis=1)
temp.columns = ['feature', 'label']
X = np.array(temp.feature.tolist())
y = np.array(temp.label.tolist())
lb = LabelEncoder()
print(X.shape,"X")
y = np_utils.to_categorical(lb.fit_transform(y)) # to use crossentropy loss fct

with open('temptotx.pickle', 'wb') as handle:
     pickle.dump(X, handle, protocol=pickle.HIGHEST_PROTOCOL)
        
with open('temptoty.pickle', 'wb') as handle:
     pickle.dump(y, handle, protocol=pickle.HIGHEST_PROTOCOL)
"""
with open('tempx.pickle','rb') as handle:
     X=pickle.load(handle)
with open('tempy.pickle','rb') as handle:
     y=pickle.load(handle)



num_labels = y.shape[1]

filter_size = 2
#adam = keras.optimizers.Adam(lr=0.001, beta_1=0.9, beta_2=0.999, epsilon=None, decay=0.0, amsgrad=False)
input0=(40,)#for temp
input1=(174,)#for temptot
# build model
model = Sequential()

model.add(Dense(800, input_shape=input0))
model.add(Activation('relu'))
model.add(Dropout(0.6))
#ajouter 2 layers si perds nulles
model.add(Dense(1500))
model.add(Activation('relu'))
#model.add(Dropout(0.1))

model.add(Dense(num_labels))
model.add(Activation('sigmoid'))
model.compile(loss='binary_crossentropy', metrics=['accuracy'], optimizer='adam')#categorical_crossentropy: basic, binary : better results, adam better results than SGD for lower eopch
callback=keras.callbacks.EarlyStopping(monitor='val_loss', min_delta=0, patience=15, verbose=0, mode='auto', baseline=None, restore_best_weights=False)

print(model.summary())
history=model.fit(X, y, batch_size=174, epochs=600, verbose=1, validation_split=0.1, callbacks=[callback])#174 batch perfect for acc
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train', 'validation'], loc='upper left')
plt.show()

"""
def parser_test(row):
   # function to load files and extract features
   file_name = os.path.join('./Test', str(row.ID) + '.wav')
   # handle exception to check if there isn't a file which is corrupted
   try:
     
      # here kaiser_fast is a technique used for faster extraction
      X_test, sr = librosa.load(file_name, res_type='kaiser_fast') 

      mfccs = np.mean(librosa.feature.mfcc(y=X_test, sr=sr, n_mfcc=40).T,axis=0) 
   except Exception as e:
      print("Error encountered while parsing file: ", file_name)
      return None, None
   #feature = np.hstack(([stdt_amp],mfccs,chroma,mel,contrast,tonnetz))
   feature = mfccs
   i = row.ID

   return pd.Series([i,feature])

temp2=test.apply(parser_test,axis=1)
temp2.columns = ['ID','feature']

test_ID = np.array(temp2.ID.tolist())

test_feature = np.array(temp2.feature.tolist())

lb = LabelEncoder()
#test_ID = np_utils.to_categorical(lb.fit_transform(test_ID))

with open('temptestmfcx.pickle', 'wb') as handle:
     pickle.dump(test_ID, handle, protocol=pickle.HIGHEST_PROTOCOL)
        
with open('temptestmfcy.pickle', 'wb') as handle:
     pickle.dump(test_feature, handle, protocol=pickle.HIGHEST_PROTOCOL)
"""
with open('temptestmfcx.pickle','rb') as handle:
     test_ID=pickle.load(handle)
with open('temptestmfcy.pickle','rb') as handle:
     test_feature=pickle.load(handle)

#mn=np.mean( test_feature ,axis=0)
#test_feature-=mn
#sd=np.std( test_feature ,axis=0)
#test_feature/=sd

result=model.predict(test_feature,verbose=1)
result=result.argmax(axis=-1)
d={
    0:'air_conditioner',
    1:'car_horn',
    2:'children_playing',
    3:'dog_bark',
    4:'drilling',
    5:'engine_idling',
    6:'gun_shot',
    7:'jackhammer',
    8:'siren',
    9:'street_music',
}
resultL=[]
for i in range(len(result)):
    resultL.append(d[result[i]])
resultL= np.asarray(resultL)
resultL=np.hstack(resultL)
data ={'Class': resultL,
       'ID': test_ID}

temp3 = pd.DataFrame(data=data)

temp3.to_csv('result.csv',index=False)