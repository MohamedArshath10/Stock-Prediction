import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import yfinance as yf

start = '2010-01-01'
end = '2022-12-31'

df = yf.download('AAPL', start=start, end=end, auto_adjust=False)
def graph():
    df = df.reset_index()
    df = df.drop(('Date', ''), axis=1)
    df = df.drop(('Adj Close', 'AAPL'), axis=1)
    ma100 = df.Close.rolling(100).mean()
    ma200 = df.Close.rolling(200).mean()
    plt.figure(figsize= (12, 6))
    plt.plot(df[('Close', 'AAPL')])
    plt.plot(ma100, 'r')
    plt.plot(ma200, 'g')
    plt.show()

#Splitting Data into training model
def train_test():
    data_training = pd.DataFrame(df['Close'][0:int(len(df) * 0.70)])
    data_testing = pd.DataFrame(df['Close'][int(len(df) * 0.70) : int(len(df))])
    print(data_training.shape)
    print(data_testing.shape)

train_test()