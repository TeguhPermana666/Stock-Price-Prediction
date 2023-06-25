import streamlit as st
import datetime
from plotly import graph_objects as go
import numpy as np
import plotly.graph_objects as go
from stock import Stock

st.set_page_config(layout="wide", initial_sidebar_state="expanded")
st.title('Stock forecast dashboard')

      
# ------ layout setting---------------------------
window_selection_c = st.sidebar.container() # create an empty container in the sidebar
window_selection_c.markdown("## Insights") # add a title to the sidebar container
sub_columns = window_selection_c.columns(2) #Split the container into two columns for start and end date

# ----------Time window selection-----------------
YESTERDAY=datetime.date.today()-datetime.timedelta(days=1)
YESTERDAY = Stock.nearest_business_day(YESTERDAY) #Round to business day

DEFAULT_START=YESTERDAY - datetime.timedelta(days=700)
DEFAULT_START = Stock.nearest_business_day(DEFAULT_START)

START = sub_columns[0].date_input("From", value=DEFAULT_START, max_value=YESTERDAY - datetime.timedelta(days=1))
END = sub_columns[1].date_input("To", value=YESTERDAY, max_value=YESTERDAY, min_value=START)

START = Stock.nearest_business_day(START)
END = Stock.nearest_business_day(END)
# ---------------stock selection------------------
STOCKS = np.array([ "GOOG", "GME","AAPL",'TSLA'])  # TODO : include all stocks
SYMB = window_selection_c.selectbox("select stock", STOCKS)

chart_width = st.expander(label="chart width").slider("", 1000, 2800, 1400)


# # # ------------------------Plot stock linechart--------------------

fig=go.Figure()
stock = Stock(symbol=SYMB)
stock.load_data(START, END, inplace=True)
fig = stock.plot_raw_data(fig)

#---------------styling for plotly-------------------------
fig.update_layout(
            width=chart_width,
            margin=dict(l=0, r=0, t=0, b=0, pad=0),
            legend=dict(
                x=0,
                y=0.99,
                traceorder="normal",
                font=dict(size=12),
            ),
            autosize=False,
            template="plotly_dark",
)

st.write(fig)

change_c = st.sidebar.container()
with change_c:
    stock.show_delta()

#----part-1--------------------------------Session state intializations---------------------------------------------------------------

if "TEST_INTERVAL_LENGTH" not in st.session_state:
    # set the initial default value of test interval
    st.session_state.TEST_INTERVAL_LENGTH = 60

if "TRAIN_INTERVAL_LENGTH" not in st.session_state:
    # set the initial default value of the training length widget
    st.session_state.TRAIN_INTERVAL_LENGTH = 500

if "HORIZON" not in st.session_state:
    # set the initial default value of horizon length widget
    st.session_state.HORIZON = 60
    
if "TRAIN_JOB" not in st.session_state:
    # set the initial default value of horizon length widget
    st.session_state.TRAIN_JOB = True

if "TRAINED" not in st.session_state:
    # set the initial default value of horizon length widget
    st.session_state.TRAINED = True

#---------------------------------------------------------Train_test_forecast_splits---------------------------------------------------
st.sidebar.markdown("## Forecasts")
train_test_forecast_c = st.sidebar.container()

train_test_forecast_c.markdown("## Select interval lengths")
HORIZON = train_test_forecast_c.number_input(
    "Inference horizon", min_value=7, max_value=200, key="HORIZON"
)
TEST_INTERVAL_LENGTH = train_test_forecast_c.number_input(
    "number of days to test on and visualize",   
    min_value=7,
    key="TEST_INTERVAL_LENGTH",
)

TRAIN_INTERVAL_LENGTH = train_test_forecast_c.number_input(
    "number of  day to use for training",
    min_value=60,
    key="TRAIN_INTERVAL_LENGTH",
)

# train_test_forecast_c.button(
#     label="Train",
#     key='TRAIN_JOB'
# )
"""
part-3
Calling a static method in the Stock class to create a stock object, train prophet,test , forecast and plot. 
It's behavior depends on the session_state variables  linked to the widgets above. 
"""
Stock.train_test_forecast_report(SYMB)