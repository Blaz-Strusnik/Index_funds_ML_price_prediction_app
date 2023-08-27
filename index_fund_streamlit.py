import streamlit as st
import math
import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from keras.models import Sequential
from keras.layers import Dense,LSTM
import matplotlib.pyplot as plt
plt.style.use('bmh')
import tensorflow as tf

#Index funds prices
st.header(":red[The index funds ML price prediction app is not intended for real world application. The index funds ML price prediction app is strictly for educational purposes!]")

st.header("Index funds price prediction")

df = pd.read_csv('./IXIC.csv')
#100 rows
df = df.head(100)

#Display data
st.header("All raw data")
df
#Get the number of rows and columns in the dataset 
st.header("Number of rows and columns")
df.shape

st.header("Data for training and prediction")
 
data = {"Date":df['Date'], "Close":df['Close']}
 
df = pd.DataFrame(data)
df
st.header("Ploted data for training and prediction")
st.line_chart(data=df, x='Date', y='Close')

#Create dataframe 'Close' column
data = df.filter(['Close'])
#Convert the dataframe to a np array
dt_set = data.values
#Number of rows for training
st.header("Number of rows for training")
training_data_len = math.ceil( len(dt_set) *.8)

training_data_len

#Scale the data
st.header("Scaled data")
scaler = MinMaxScaler(feature_range=(0,1))
scaled_data = scaler.fit_transform(dt_set)
scaled_data

#Training dt_set
#Scaled_training dt_set
train_dt = scaled_data[0:training_data_len , :]
#Split data
x_train = []
y_train = []
for i in range(60, len(train_dt)):
    x_train.append(train_dt[i-60:i, 0])
    y_train.append(train_dt[i, 0])
    if i<= 61:
        print(x_train)
        print(y_train)
        print()
        
#Convert data to np arrays
x_train, y_train = np.array(x_train), np.array(y_train)

#Restructure the data
st.header("Split the data")
x_train = np.reshape(x_train, (x_train.shape[0], x_train.shape[1], 1))
x_train.shape

#LSTM model
model = Sequential()
model.add(LSTM(50, return_sequences=True, input_shape= (x_train.shape[1], 1)))
model.add(LSTM(50, return_sequences= False))
model.add(Dense(15))
model.add(Dense(1))

#Compile
model.compile(optimizer='adam', loss='mean_squared_error')

#Train
model.fit(x_train, y_train, batch_size=1, epochs=1)

#Testing the dt_set
#Array with scaled values
test_data = scaled_data[training_data_len - 60: , :]
#Test data x_test and y_test
x_test = []
y_test = dt_set[training_data_len:, :]
for i in range(60, len(test_data)):
    x_test.append(test_data[i-60:i, 0])
    
#Convert data to a np array
x_test = np.array(x_test)
#Restructure the data
x_test = np.reshape(x_test, (x_test.shape[0], x_test.shape[1], 1 ))
#Predicted price values
pred = model.predict(x_test)
pred = scaler.inverse_transform(pred)
st.header("Root mean squared error")
#Root mean squared error
rmse=np.sqrt(np.mean(((pred- y_test)**2)))
rmse

#Plot the data
train = data[:training_data_len]
m_validation = data[training_data_len:]
m_validation['Predict'] = pred

#Model validation and predicted prices
st.header("Validated model")
m_validation
 
data = {"Trained plot": train['Close']}
 
df = pd.DataFrame(data)
st.header("Ploted Trained Data")

st.line_chart(data=df)


#Get data
nasdaq_c = pd.read_csv('./IXIC.csv')
#New Dataframe
new_df = nasdaq_c.filter(['Close'])
#60 day history
history_60_days = new_df[-60:].values
#Scale the data  0 && 1
history_60_days_scaled = scaler.transform(history_60_days)
#List
X_test = []
#Add 60 day history
X_test.append(history_60_days_scaled)
#Convert the X_test data set to a np array
X_test = np.array(X_test)
#Restructure the data
X_test = np.reshape(X_test, (X_test.shape[0], X_test.shape[1], 1))
#Predicted scaled price
predicted_price = model.predict(X_test)
#Revert the scaling
predicted_price = scaler.inverse_transform(predicted_price)
print(predicted_price)


nasdaq_c = pd.read_csv('./IXIC.csv')
print(nasdaq_c['Date'])