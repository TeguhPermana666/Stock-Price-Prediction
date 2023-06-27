import time  # to simulate a real time data, time loop
import numpy as np  # np mean, np random
import pandas as pd  # read csv, df manipulation
import plotly.express as px  # interactive charts
import streamlit as st  # 🎈 data web app development
from pandas_datareader import data as pdr
from datetime import datetime

# Get the stock quote
df_AAPL = pdr.get_data_yahoo('AAPL', start='2012-01-01', end=datetime.now())
df_GOOG= pdr.get_data_yahoo('GOOG', start='2012-01-01', end=datetime.now())
df_MSFT= pdr.get_data_yahoo('MSFT', start='2012-01-01', end=datetime.now())
df_AMZN= pdr.get_data_yahoo('AMZN', start='2012-01-01', end=datetime.now())

