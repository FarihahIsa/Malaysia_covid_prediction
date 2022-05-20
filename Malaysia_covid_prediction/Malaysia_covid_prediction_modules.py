#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri May 20 16:44:47 2022

@author: farihahisa
"""

import pickle
import pandas as pd
import os
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
import numpy as np
import missingno as msno
from sklearn.preprocessing import LabelEncoder
from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import IterativeImputer
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from sklearn.metrics import mean_absolute_error
from tensorflow.keras.callbacks import TensorBoard
import datetime

#%% Path
DATASET_TRAIN_PATH = os.path.join(os.getcwd(), 'datasets', 'cases_malaysia_train.csv')
DATASET_TEST_PATH = os.path.join(os.getcwd(), 'datasets','cases_malaysia_test.csv')
LOG_PATH = os.path.join(os.getcwd(), 'logs')
MMS_SCALER_PATH = os.path.join(os.getcwd(),'saved_models', 'MinMaxScaler.pkl')
MODEL_SAVE_PATH = os.path.join(os.getcwd(), 'saved_models','model.h5')

#%%
df_train = pd.read_csv(DATASET_TRAIN_PATH)
df_test = pd.read_csv(DATASET_TEST_PATH)

#%%
class ExploratoryDataAnalysis():
    
    def CleaningData(self,data):
        
        # remove not important columns in X_train

        df_train = df_train.drop(columns='date')

        # convert object to string in X_train

        df_train['cases_new'] = LabelEncoder().fit_transform(df_train['cases_new'])

        # filling missing values in X_train

        imputer = IterativeImputer()
        df_train = imputer.fit_transform(df_train)

        # remove date column in X_test

        df_test = df_test.drop(columns='date')

        df_test['cases_new'] = LabelEncoder().fit_transform(df_test['cases_new'])

        # fill in NaN values in X_test

        imputer = IterativeImputer()
        
        df_test = imputer.fit_transform(df_test)
        
        return data
        
    def DataProcessing(self,data):
        
        mms = MinMaxScaler()

        df_train_scaled = mms.fit_transform(df_train, -1)

        # saving mms

        pickle.dump(mms, open(MMS_SCALER_PATH,'wb')) 
        
        return data
        
    def TrainingDataset(self,data):
        
        # Window size 

        window_size = 30  # 30 days is used to predict what next value

        X_train=[]
        Y_train=[]

        for i in range(window_size,len(df_train)): # window_size, max number of row
            X_train.append(df_train_scaled[i-window_size:i,0])
            Y_train.append(df_train[i,0])

        # Convert to array
        X_train=np.array(X_train)
        Y_train=np.array(Y_train)

        # Testing Dataset

        temp = np.concatenate((df_train_scaled,df_test))
        length_window = window_size+len(df_test)
        temp = temp[-length_window:]

        X_test=[]
        Y_test=[]


        for i in range(window_size,len(temp)):
            X_test.append(temp[i-window_size:i,0])
            Y_test.append(temp[i,0])


        X_test = np.array(X_test)
        Y_test = np.array(Y_test)


        X_train = np.expand_dims(X_train, axis=-1)
        X_test = np.expand_dims(X_test, axis=-1)
        
        return data
        
class ModelCreation():
    
    def ModelNN(self,data):
        
        model = Sequential()
        model.add(LSTM(64,activation='tanh',
        return_sequences=(True),
        input_shape=(X_train.shape[1],1)))
        model.add(Dropout(0.2))
        model.add(LSTM(64))
        model.add(Dropout(0.2))
        model.add(Dense(1))
        model.summary()
        
        return data
    
    def ModelCompile(self,data):
        
        model.compile(optimizer='adam',
        loss='mse',
        metrics=['mse'])

        hist = model.fit(X_train,Y_train,epochs=50, 
                         batch_size=128,
                         validation_data=(X_test,Y_test),
                         callbacks=[tensorboard_callback])

        print(hist.history.keys())
        
        return data
        
class ModelEvaluation():
    
    def ModelPredict(self,data):
        
        predicted = [] 

        for test in X_test:
            predicted.append(model.predict(X_test))
            
        predicted = np.array(predicted)
        
        return data
    
class ModelAnalysis():
    
    def Predicted (self,data):
        
        plt.figure()
        plt.plot(predicted.reshape(len(predicted),1)) # reshape into 100,1
        plt.plot(Y_test)
        plt.legend(['Predicted','Actual']) 
        plt.show() 

        y_true = Y_test
        y_pred = predicted.reshape(len(predicted),1)
        print(mean_absolute_error(y_true,y_pred)/sum(abs(y_true))*100) #--> MAE = 0.597
        
        return data
    
#%% 

eda = ExploratoryDataAnalysis()
mc = ModelCreation()
me = ModelEvaluation()
ma = ModelAnalysis()    