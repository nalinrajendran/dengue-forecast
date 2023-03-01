import pandas as pd
import numpy as np
import pickle
import streamlit as st
from PIL import Image


import datetime as dt
from dateutil import relativedelta as rdt
import numpy as np
import io
import os

# plotting
import matplotlib.pyplot as plt
import matplotlib.dates as mdat
import plotly.express as px

# time series analysis and forecasting

import statsmodels.api as sm
from statsmodels.graphics.tsaplots import plot_acf
from scipy.stats import normaltest

# web app
import streamlit as st

from PIL import Image

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

import warnings
warnings.filterwarnings("ignore")




# loading in the model to predict on the data
pickle_in = open('classifier.pkl', 'rb')
classifier = pickle.load(pickle_in)

def welcome():
	return 'welcome all'






def main():
	# giving the webpage a title
	
	
	# # here we define some of the front end elements of the web page like
	# # the font and background color, the padding and the text to be displayed
	# html_temp = """
	# <div style ="background-color:yellow;padding:13px">
	# <h1 style ="color:black;text-align:center;">Dengue Prediction Using <br> TIME-SERIES </h1>
	# </div>
	# """
	
	# this line allows us to display the front end aspects we have
	# defined in the above code
	# st.markdown(html_temp, unsafe_allow_html = True)
	

	

	
	result =""
	






wpath = os.path.dirname(__file__)






st.header('DENGUE CASES PREDICTION USING :blue[RNN]')


st.write("**PREDICTED CASES**")
# read the source data file
df = pd.read_csv(wpath + "/DATA/try_data.csv")



# %%
# enable the end user to upload a csv file:
st.sidebar.write("_" * 30)
st.sidebar.write("**file uploader:**")

uploaded_file = st.sidebar.file_uploader(
    label="Upload the csv file containing the time series:",
    type="csv",
    accept_multiple_files=False,
    help='''Upload a csv file that contains your time series.     
        ''')   
if uploaded_file is not None:
    if uploaded_file.size <= 10000:    # size < 10 kB
        df = pd.read_csv(uploaded_file)
st.sidebar.write("_" * 30)

# convert objects/strings to datetime and numbers; set datetime index
# df = df.dropna()


df2 = df
df2 = df2.reset_index()['cases']

scaler = MinMaxScaler(feature_range=(0,1))
df2 = scaler.fit_transform(np.array(df2).reshape(-1,1))

training_size = int(len(df2)*1)
test_size= len(df2)-training_size

valid_data,test_data = df2[0:training_size,:],df2[training_size:len(df2),:1]

def create_dataset (dataset,time_step=1):
    dataX, dataY = [], []
    for i in range(len(dataset)-time_step-1):
        a = dataset[i:(i+time_step),0]
        dataX.append(a)
        dataY.append(dataset[i+time_step, 0])
    return np.array(dataX), np.array(dataY)


time_step = 2
X_train, y_train = create_dataset(valid_data, time_step)

X_train = X_train.reshape(X_train.shape[0], X_train.shape[1], 1)

train_predict= classifier.predict(X_train)
train_predict = scaler.inverse_transform(train_predict)



st.write(train_predict)
st.write("PREDICTED CASES")

train_predict_year = train_predict[0:12]

st.line_chart(train_predict_year)
# st.write(math.sqrt(mean_squared_error(y_train, train_predict)))




# fig = plt.plot(train_predict)

# st.pyplot(fig)

# fig = px.line(train_predict, x="Date", 
#     title="chart: sweatpants popularity", width=1000)
# st.plotly_chart(fig, use_container_width=False)

st.write("USER INPUT FOR THE TIME SERIES DATA")
st.dataframe(df.style.highlight_max(axis=0))
st.write("Cases in the User Input")
st.line_chart(df[["cases"]])











	
if __name__=='__main__':
	main()
