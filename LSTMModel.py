import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import yfinance as yf
from tensorflow.keras.models import load_model
from sklearn.preprocessing import MinMaxScaler
import streamlit as st

start = '2010-01-01'
end = '2024-12-31'

st.title('Stock Trend Prediction')

user_input = st.text_input('Enter Stock Ticker', 'AAPL')
df = yf.download(user_input, start=start, end=end, auto_adjust=False)


# Describing data
st.subheader('Data from 2010 - 2024')
st.write(df.describe())

# Visualization
st.


# df = df.reset_index()
# df = df.drop(('Date', ''), axis=1)
# df = df.drop(('Adj Close', 'AAPL'), axis=1)
# ma100 = df.Close.rolling(100).mean()
# ma200 = df.Close.rolling(200).mean()
# plt.figure(figsize= (12, 6))
# plt.plot(df[('Close', 'AAPL')])
# plt.plot(ma100, 'r')
# plt.plot(ma200, 'g')
# plt.show()

# #Splitting Data into training model
# def train_test():
#     data_training = pd.DataFrame(df['Close'][0:int(len(df) * 0.70)])
#     data_testing = pd.DataFrame(df['Close'][int(len(df) * 0.70) : int(len(df))])
#     # print(data_training.shape)
#     # print(data_testing.shape)
#     scaler = MinMaxScaler(feature_range=(0, 1))
#     data_training_array = scaler.fit_transform(data_training)
#     print(data_training_array)

# train_test()