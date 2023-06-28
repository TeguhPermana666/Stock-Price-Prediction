import time  # to simulate a real time data, time loop
import numpy as np  # np mean, np random
import pandas as pd  # read csv, df manipulation
# import plotly.express as px  # interactive charts
import streamlit as st  # 🎈 data web app development
import yfinance as yf
from pandas_datareader import data as pdr
from datetime import datetime
import joblib
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import MinMaxScaler
yf.pdr_override()
sns.set_style('whitegrid')
plt.style.use("fivethirtyeight")

# Load dataset
df_AAPL = pdr.get_data_yahoo('AAPL', start='2012-01-01', end=datetime.now())
df_GOOG= pdr.get_data_yahoo('GOOG', start='2012-01-01', end=datetime.now())
df_MSFT= pdr.get_data_yahoo('MSFT', start='2012-01-01', end=datetime.now())
df_AMZN= pdr.get_data_yahoo('AMZN', start='2012-01-01', end=datetime.now())


# Load model
model_appl = joblib.load(r"model/model_appl.pkl")
model_goog = joblib.load(r"model/model_goog.pkl")
model_msft = joblib.load(r"model/model_msft.pkl")
model_amzn = joblib.load(r"model/model_amzn.pkl")

# dashboard title
st.title("Real-Time Stock Market Dashboard")

# top-level filters
company_list = ["Apple","Google","Miscrosoft","Amazon"]
company_filter = st.selectbox("Select the company", pd.unique(company_list))

if company_filter == "Apple":    
    st.dataframe(df_AAPL)
    data = df_AAPL.filter(['Close'])
    dataset = data.values
    training_data_len = int(np.ceil( len(dataset) * .95 ))
    scaler = MinMaxScaler(feature_range=(0,1))
    scaled_data = scaler.fit_transform(dataset)
    test_data = scaled_data[training_data_len - 60: , :]
    x_test = []
    y_test = dataset[training_data_len:, :]
    for i in range(60, len(test_data)):
        x_test.append(test_data[i-60:i, 0])
    x_test = np.array(x_test)
    x_test = np.reshape(x_test, (x_test.shape[0], x_test.shape[1], 1 )) 
    predictions = model_appl.predict(x_test)
    predictions = scaler.inverse_transform(predictions)
    train = data[:training_data_len]
    valid = data[training_data_len:]
    valid['Predictions'] = predictions
    fig,ax=plt.subplots()
    plt.title('Close Price History Apple')
    plt.xlabel('Date', fontsize=18)
    plt.ylabel('Close Price USD ($)', fontsize=18)
    ax.plot(train['Close'],linewidth=1.5)
    st.pyplot(fig)
    st.header("Predicting Apple Stock Market 2023 - 2024 :")
    fig,ax = plt.subplots()
    plt.title('Prediction Stock Market Apple')
    plt.figure(figsize=(16,6))
    plt.xlabel('Date', fontsize=18)
    plt.ylabel('Close Price USD ($)', fontsize=18)
    ax.plot(train['Close'],linewidth=1.5)
    ax.plot(valid[['Predictions']],linewidth=1.5)
    ax.legend(['Stock Apple','Prediction Stock'], loc='lower right')
    st.pyplot(fig)

elif company_filter == "Google":    
    st.dataframe(df_GOOG)
    data = df_GOOG.filter(['Close'])
    dataset = data.values
    training_data_len = int(np.ceil( len(dataset) * .95 ))
    scaler = MinMaxScaler(feature_range=(0,1))
    scaled_data = scaler.fit_transform(dataset)
    test_data = scaled_data[training_data_len - 60: , :]
    x_test = []
    y_test = dataset[training_data_len:, :]
    for i in range(60, len(test_data)):
        x_test.append(test_data[i-60:i, 0])
    x_test = np.array(x_test)
    x_test = np.reshape(x_test, (x_test.shape[0], x_test.shape[1], 1 )) 
    predictions = model_goog.predict(x_test)
    predictions = scaler.inverse_transform(predictions)
    train = data[:training_data_len]
    valid = data[training_data_len:]
    valid['Predictions'] = predictions
    fig,ax=plt.subplots()
    plt.title('Close Price History Google')
    plt.xlabel('Date', fontsize=18)
    plt.ylabel('Close Price USD ($)', fontsize=18)
    ax.plot(train['Close'],linewidth=1.5)
    st.pyplot(fig)
    st.header("Predicting Google Stock Market 2023 - 2024 :")
    fig,ax = plt.subplots()
    plt.title('Prediction Stock Market Google')
    plt.figure(figsize=(16,6))
    plt.xlabel('Date', fontsize=18)
    plt.ylabel('Close Price USD ($)', fontsize=18)
    ax.plot(train['Close'],linewidth=1.5)
    ax.plot(valid[['Predictions']],linewidth=1.5)
    ax.legend(['Stock Apple','Prediction Stock'], loc='lower right')
    st.pyplot(fig)            

elif company_filter == "Miscrosoft":    
    st.dataframe(df_MSFT)
    data = df_MSFT.filter(['Close'])
    dataset = data.values
    training_data_len = int(np.ceil( len(dataset) * .95 ))
    scaler = MinMaxScaler(feature_range=(0,1))
    scaled_data = scaler.fit_transform(dataset)
    test_data = scaled_data[training_data_len - 60: , :]
    x_test = []
    y_test = dataset[training_data_len:, :]
    for i in range(60, len(test_data)):
        x_test.append(test_data[i-60:i, 0])
    x_test = np.array(x_test)
    x_test = np.reshape(x_test, (x_test.shape[0], x_test.shape[1], 1 )) 
    predictions = model_msft.predict(x_test)
    predictions = scaler.inverse_transform(predictions)
    train = data[:training_data_len]
    valid = data[training_data_len:]
    valid['Predictions'] = predictions
    fig,ax=plt.subplots()
    plt.title('Close Price History Miscrosoft')
    plt.xlabel('Date', fontsize=18)
    plt.ylabel('Close Price USD ($)', fontsize=18)
    ax.plot(train['Close'],linewidth=1.5)
    st.pyplot(fig)
    st.header("Predicting Miscrosoft Stock Market 2023 - 2024 :")
    fig,ax = plt.subplots()
    plt.title('Prediction Stock Market Miscrosoft')
    plt.figure(figsize=(16,6))
    plt.xlabel('Date', fontsize=18)
    plt.ylabel('Close Price USD ($)', fontsize=18)
    ax.plot(train['Close'],linewidth=1.5)
    ax.plot(valid[['Predictions']],linewidth=1.5)
    ax.legend(['Stock Miscrosoft','Prediction Stock'], loc='lower right')
    st.pyplot(fig)
    
elif company_filter == "Amazon":    
    st.dataframe(df_AMZN)
    data = df_AMZN.filter(['Close'])
    dataset = data.values
    training_data_len = int(np.ceil( len(dataset) * .95 ))
    scaler = MinMaxScaler(feature_range=(0,1))
    scaled_data = scaler.fit_transform(dataset)
    test_data = scaled_data[training_data_len - 60: , :]
    x_test = []
    y_test = dataset[training_data_len:, :]
    for i in range(60, len(test_data)):
        x_test.append(test_data[i-60:i, 0])
    x_test = np.array(x_test)
    x_test = np.reshape(x_test, (x_test.shape[0], x_test.shape[1], 1 )) 
    predictions = model_amzn.predict(x_test)
    predictions = scaler.inverse_transform(predictions)
    train = data[:training_data_len]
    valid = data[training_data_len:]
    valid['Predictions'] = predictions
    fig,ax=plt.subplots()
    plt.title('Close Price History Amazon')
    plt.xlabel('Date', fontsize=18)
    plt.ylabel('Close Price USD ($)', fontsize=18)
    ax.plot(train['Close'],linewidth=1.5)
    st.pyplot(fig)
    st.header("Predicting Amazon Stock Market 2023 - 2024 :")
    fig,ax = plt.subplots()
    plt.title('Prediction Stock Market Amazon')
    plt.figure(figsize=(16,6))
    plt.xlabel('Date', fontsize=18)
    plt.ylabel('Close Price USD ($)', fontsize=18)
    ax.plot(train['Close'],linewidth=1.5)
    ax.plot(valid[['Predictions']],linewidth=1.5)
    ax.legend(['Stock Amazon','Prediction Stock'], loc='lower right')
    st.pyplot(fig)   