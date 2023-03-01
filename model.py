import pandas as pd
import math
import matplotlib.pyplot as plt
import numpy as np
import os
import seaborn as sns


import keras
import tensorflow as tf
from sklearn.preprocessing import MinMaxScaler

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import LSTM


from sklearn.metrics import mean_squared_error
from sklearn.metrics import mean_absolute_error as MAE

pwd = os.getcwd()
filepath = os.path.join(pwd,"DATA/try_data.csv")

df = pd.read_csv(filepath)

df1 = df.reset_index()['cases']

scaler = MinMaxScaler(feature_range=(0,1))
df1 = scaler.fit_transform(np.array(df1).reshape(-1,1))

training_size = int(len(df1)*0.8)
test_size= len(df1)-training_size

train_data,test_data = df1[0:training_size,:],df1[training_size:len(df1),:1]

#Using a time series approach. This makes the next value in the dataset dependent upon the preceeding values.

def create_dataset (dataset,time_step=1):
    dataX, dataY = [], []
    for i in range(len(dataset)-time_step-1):
        a = dataset[i:(i+time_step),0]
        dataX.append(a)
        dataY.append(dataset[i+time_step, 0])
    return np.array(dataX), np.array(dataY)


time_step = 2
X_train, y_train = create_dataset(train_data, time_step)
X_test, y_test = create_dataset(test_data, time_step)


#Adding a third axis(reshaping) to the dataset.

X_train = X_train.reshape(X_train.shape[0], X_train.shape[1], 1) 
X_test = X_test.reshape(X_test.shape[0], X_test.shape[1], 1)

model = Sequential()
model.add(LSTM(11,return_sequences=True, input_shape=(2,1)))
model.add(LSTM(11,return_sequences=True))
model.add(LSTM(11))
model.add(Dense(1))
model.compile(loss='mean_squared_error', optimizer = "adam")

print(model.summary())

model.fit(X_train, y_train, validation_data=(X_test,y_test), epochs=400, batch_size=38, verbose=1)

train_predict=model.predict(X_train)
test_predict=model.predict(X_test)


train_predict = scaler.inverse_transform(train_predict)
test_predict = scaler.inverse_transform(test_predict)

math.sqrt(mean_squared_error(y_train, train_predict))
math.sqrt(mean_squared_error(y_test,test_predict))

y_true, y_pred = y_test, model.predict(X_test).astype(int)
print("Mean Absolute Error(MAE): %f" %MAE(y_true, y_pred))


# pickling the model
import pickle
pickle_out = open("classifier.pkl", "wb")
pickle.dump(model, pickle_out)
pickle_out.close()




