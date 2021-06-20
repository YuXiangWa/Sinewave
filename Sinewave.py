# -*- coding: utf-8 -*-
"""
Created on Fri Jun 18 14:19:48 2021

@author: user
"""

import os
import json
import time
import math
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from keras.models import Sequential
from keras.layers import Dense, Activation, Dropout, LSTM
from keras.utils import plot_model
from keras.callbacks import EarlyStopping, ModelCheckpoint
import datetime as dt


def plot_results(predicted_data, true_data):
    fig = plt.figure(facecolor='white')
    ax = fig.add_subplot(111)
    ax.plot(true_data, label='True Data')
    plt.plot(predicted_data, label='Prediction')
    plt.legend()
    plt.show()


def plot_results_multiple(predicted_data, true_data, prediction_len):
    fig = plt.figure(facecolor='white')
    ax = fig.add_subplot(111)
    ax.plot(true_data, label='True Data')
    plt.legend()
    for i, data in enumerate(predicted_data):
        padding = [None for p in range(i * prediction_len)]
        plt.plot(padding + data, label='Prediction')
    plt.show()

def predict_sequences_multiple(data, window_size, prediction_len,debug=False):
	if debug == False:
		print('[Model] Predicting Sequences Multiple...')
		prediction_seqs = []
		for i in range(int(len(data)/prediction_len)):
			curr_frame = data[i*prediction_len]
			predicted = []
			for j in range(prediction_len):
				predicted.append(model.predict(curr_frame[np.newaxis,:,:])[0,0])
				curr_frame = curr_frame[1:]
				curr_frame = np.insert(curr_frame, [window_size-2], predicted[-1], axis=0)
			prediction_seqs.append(predicted)
		return prediction_seqs


    
dataframe = pd.read_csv('sinewave.csv')
split = int(len(dataframe) * 0.8)
data_train = dataframe.get("sinewave").values[:split]
data_test  = dataframe.get("sinewave").values[split:]

seq_len = 50
train=[]
for i in range(len(data_train)-seq_len):
   x.append([data_train[i:i+seq_len]])
X_train = np.array(train).reshape(len(train),seq_len,1)
y_train = data_train[seq_len:]
test=[]
for i in range(len(data_test)-seq_len):
   test.append([data_test[i:i+seq_len]]) 
X_test = np.array(test).reshape(len(test),seq_len,1)
y_test = data_test[seq_len:]

model = Sequential()
model.add(LSTM(50, input_shape=(50, 1), return_sequences=True))
model.add(Dropout(0.05))
model.add(LSTM(100, return_sequences=False))
model.add(Dropout(0.05))
model.add(Dense(1))
model.compile(loss="mse", optimizer="adam")
plot_model(model, to_file='model_test.png',show_shapes=True)
save_fname = os.path.join("saved_models", '%s-e%s.h5' % (dt.datetime.now().strftime('%d%m%Y-%H%M%S'), str(2)))
callbacks = [EarlyStopping(monitor='val_loss', patience=2),ModelCheckpoint(filepath=save_fname, monitor='val_loss', save_best_only=True)]
model.fit(X_train,y_train,epochs=2,batch_size=32,callbacks=callbacks)
save_fname = os.path.join("saved_models", '%s-e%s.h5' % (dt.datetime.now().strftime('%d%m%Y-%H%M%S'), str(2)))
model.save(save_fname)

predictions = predict_sequences_multiple(X_test, seq_len, seq_len,debug=False)
print (np.array(predictions).shape)

plot_results_multiple(predictions, y_test, seq_len)