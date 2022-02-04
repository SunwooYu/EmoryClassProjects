# -*- coding: utf-8 -*-
"""
Created on Mon Dec 13 00:51:40 2021

@author: sunwo
"""


import numpy as np
import pandas as pd
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Activation
from tensorflow.keras import optimizers
from sklearn.model_selection import train_test_split
from keras.models import load_model
from sklearn.metrics import log_loss
from sklearn.metrics import accuracy_score


NEpochs = 1000
BatchSize=1000
Optimizer=optimizers.RMSprop(lr=0.001)

# Read in the data

TotalDF = pd.read_csv('sample_rs24.csv',sep=',',header=0,quotechar='"')
list(TotalDF)


# weight of evidence encoding

TotalDF.head()
TotalDF_enc = TotalDF.copy()

df1 = pd.DataFrame(TotalDF_enc.groupby('site_domain_pareto')['click'].mean())
df1['nonclick'] = 1-df1.click
df1['nonclick'] = np.where(df1['nonclick']==0, 00000.1, df1['nonclick'])
df1['WoE'] = np.log(df1.click/df1.nonclick)
TotalDF_enc.loc[:, 'WoE_site_domain'] = TotalDF_enc['site_domain_pareto'].map(df1['WoE'])
TotalDF_enc.drop(columns= 'site_domain_pareto')

df2 = pd.DataFrame(TotalDF_enc.groupby('site_id_pareto')['click'].mean())
df2['nonclick'] = 1-df2.click
df2['nonclick'] = np.where(df2['nonclick']==0, 00000.1, df2['nonclick'])
df2['WoE'] = np.log(df2.click/df2.nonclick)
TotalDF_enc.loc[:, 'WoE_site_id'] = TotalDF_enc['site_domain_pareto'].map(df2['WoE'])
TotalDF_enc.drop(columns= 'site_id_pareto')


df3 = pd.DataFrame(TotalDF_enc.groupby('device_model_pareto')['click'].mean())
df3['nonclick'] = 1-df3.click
df3['nonclick'] = np.where(df3['nonclick']==0, 00000.1, df3['nonclick'])
df3['WoE'] = np.log(df3.click/df3.nonclick)
TotalDF_enc.loc[:, 'WoE_device_model'] = TotalDF_enc['device_model_pareto'].map(df3['WoE'])
TotalDF_enc.drop(columns= 'device_model_pareto')



df4 = pd.DataFrame(TotalDF_enc.groupby('C14_pareto')['click'].mean())
df4['nonclick'] = 1-df4.click
df4['nonclick'] = np.where(df4['nonclick']==0, 00000.1, df4['nonclick'])
df4['WoE'] = np.log(df4.click/df4.nonclick)
TotalDF_enc.loc[:, 'WoE_C14'] = TotalDF_enc['C14_pareto'].map(df4['WoE'])
TotalDF_enc.drop(columns= 'C14_pareto')



df5 = pd.DataFrame(TotalDF_enc.groupby('C17_pareto')['click'].mean())
df5['nonclick'] = 1-df5.click
df5['nonclick'] = np.where(df5['nonclick']==0, 00000.1, df5['nonclick'])
df5['WoE'] = np.log(df5.click/df5.nonclick)
TotalDF_enc.loc[:, 'WoE_C17'] = TotalDF_enc['C17_pareto'].map(df5['WoE'])
TotalDF_enc.drop(columns= 'C17_pareto')

TotalDF_enc.drop(columns= ['Unnamed: 0', 'id'])

list(TotalDF_enc)

TotalDF_enc.to_csv('WoEencoding.csv')

# split the data frame into train and test dataset
train, t_and_v = train_test_split(TotalDF_enc, test_size=0.6, random_state=5, shuffle=True)

TrIsclick = np.array(train['click'])

TrX = np.array(train.iloc[:,3:])

TrXrsc = (TrX - TrX.min(axis=0))/TrX.ptp(axis=0)
print(TrXrsc.shape)
print(TrXrsc.min(axis=0))
print(TrXrsc.max(axis=0))

# No need to rescale the Y because it is already 0 and 1. But check
print(TrIsclick.min())
print(TrIsclick.max())

# Rescale the validation data
val, test = train_test_split(t_and_v, test_size=0.5, random_state=5, shuffle=True)

valisclick = np.array(val['click'])

valX = np.array(val.iloc[:,3:])

valXrsc = (valX - TrX.min(axis=0))/TrX.ptp(axis=0)
print(valXrsc.shape)
print(valXrsc.min(axis=0))
print(valXrsc.max(axis=0))

print(valisclick.min())
print(valisclick.max())

# Rescale the test data

testisclick = np.array(test['click'])

TestX = np.array(test.iloc[:,3:])

TestXrsc = (TestX - TrX.min(axis=0))/TrX.ptp(axis=0)
print(TestXrsc.shape)
print(TestXrsc.min(axis=0))
print(TestXrsc.max(axis=0))

print(testisclick.min())
print(testisclick.max())


#%% Set up Neural Net Model

NN = Sequential()

NN.add(Dense(units=40,input_shape=(TrXrsc.shape[1],),activation="relu",use_bias=True))
NN.add(Dense(units=40,activation="relu",use_bias=True))
NN.add(Dense(units=20,activation="relu",use_bias=True))
NN.add(Dense(units=10,activation="relu",use_bias=True))
NN.add(Dense(units=5,activation="relu",use_bias=True))
NN.add(Dense(units=1,activation="sigmoid",use_bias=True))

NN.compile(loss='binary_crossentropy', optimizer=Optimizer,metrics=['binary_crossentropy'])
print(NN.summary())

#%% Fit NN Model

from keras.callbacks import EarlyStopping

StopRule = EarlyStopping(monitor='val_loss',mode='min',verbose=0,patience=100,min_delta=0.0)

FitHist = NN.fit(TrXrsc,TrIsclick,validation_data=(valXrsc,valisclick), \
                         epochs=NEpochs,batch_size=BatchSize,verbose=0, \
                             callbacks=[StopRule])
    
#FitHist = SpiralNN.fit(TrXrsc,TrColorCode,epochs=NEpochs,batch_size=BatchSize,verbose=0)

print("Number of Epochs = "+str(len(FitHist.history['binary_crossentropy'])))
print("Final training binary_crossentropy: "+str(FitHist.history['binary_crossentropy'][-1]))
print("Recent history for training binary_crossentropy: "+str(FitHist.history['binary_crossentropy'][-10:-1]))
print("Final validation binary_crossentropy: "+str(FitHist.history['binary_crossentropy'][-1]))
print("Recent history for validation binary_crossentropy: "+str(FitHist.history['binary_crossentropy'][-10:-1]))


#%% Prediction on the test data
TestDF = pd.read_csv('ProjectTestDataEncoding.csv',sep=',',header=0,quotechar='"')
list(TotalDF)
list(TestDF)

TestDF.loc[:, 'WoE_site_domain'] = TestDF['site_domain_encoding'].map(df1['WoE'])
TestDF.drop(columns= 'site_domain_encoding')

TestDF.loc[:, 'WoE_site_id'] = TestDF['site_id_pareto'].map(df2['WoE'])
TestDF.drop(columns= 'site_id_pareto')

TestDF.loc[:, 'WoE_device_model'] = TestDF['device_model_pareto'].map(df3['WoE'])
TestDF.drop(columns= 'device_model_pareto')

TestDF.loc[:, 'WoE_C14'] = TestDF['C14_pareto'].map(df4['WoE'])
TestDF.drop(columns= 'C14_pareto')

TestDF.loc[:, 'WoE_C17'] = TestDF['C17_pareto'].map(df5['WoE'])
TestDF.drop(columns= 'C17_pareto')

id_column = TestDF['id']

AllTestX = np.array(TestDF.iloc[:,1:])

AllTestXrsc = (AllTestX - TrX.min(axis=0))/TrX.ptp(axis=0)
print(AllTestXrsc.shape)
print(AllTestXrsc.min(axis=0))
print(AllTestXrsc.max(axis=0))

AllTestP = NN.predict(AllTestXrsc,batch_size=AllTestXrsc.shape[0])
TestDF['TestP'] = AllTestP.reshape(-1)

predict = TestDF[['id', 'TestP']]

submission = pd.read_csv('ProjectSubmission-TeamX.csv',sep=',',header=0,quotechar='"')

submission = submission.join(predict.set_index('id'), on='id')
submission = submission.drop(columns = 'P(click)')

submission = submission.rename(columns={"TestP": "P(click)"})

submission.to_csv('ProjectSubmission-Team5.csv')
