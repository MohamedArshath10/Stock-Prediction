from flask import Flask, request, send_file
import yfinance as yf
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import os
from sklearn.preprocessing import MinMaxScaler
from model import predict_with_jax

app = Flask(__name__)

@app.route('/', methods=['GET', 'POST'])
def index():
    result_html = ''

    if request.method == 'POST':
        ticker = request.form['ticker'].upper()
        df = yf.download(ticker, start='2010-01-01', end='2024-12-31')

        # Save Closing Price Chart
        plt.figure(figsize=(12, 6))
        plt.plot(df['Close'])
        plt.title(f"{ticker} Closing Price")
        plt.savefig('closing.png')
        plt.close()

        # Data Prep
        data_training = df['Close'][0:int(len(df)*0.7)]
        data_testing = df['Close'][int(len(df)*0.7):]
        scaler = MinMaxScaler(feature_range=(0, 1))
        data_training_array = scaler.fit_transform(data_training.values.reshape(-1, 1))

        # Training Set
        x_train, y_train = [], []
        for i in range(100, len(data_training_array)):
            x_train.append(data_training_array[i-100:i])
            y_train.append(data_training_array[i, 0])
        x_train, y_train = np.array(x_train), np.array(y_train)

        # Testing Set
        past_100_days = data_training.tail(100)
        final_df = pd.concat([past_100_days, data_testing], ignore_index=True)
        input_data = scaler.transform(final_df.values.reshape(-1, 1))

        x_test, y_test = [], []
        for i in range(100, len(input_data)):
            x_test.append(input_data[i-100:i])
            y_test.append(input_data[i, 0])
        x_test, y_test = np.array(x_test), np.array(y_test)

        # JAX Prediction
        y_predicted = predict_with_jax(x_test, x_train, y_train)
        scale_factor = 1 / scaler.scale_[0]
        y_predicted *= scale_factor
        y_test *= scale_factor

        # Plot Prediction vs Actual
        plt.figure(figsize=(12, 6))
        plt.plot(y_test, 'b', label='Original')
        plt.plot(y_predicted, 'r', label='Predicted')
        plt.xlabel('Time')
        plt.ylabel('Price')
        plt.legend()
        plt.title('Predicted vs Actual')
        plt.savefig('prediction.png')
        plt.close()

        result_html = f"""
            <h3>Closing Price Chart</h3>
            <img src='/chart/closing' width='800'><br>
            <h3>Predicted vs Actual</h3>
            <img src='/chart/prediction' width='800'><br>
        """

    return f'''
        <h1>Stock Trend Prediction (Flask + JAX)</h1>
        <form method="post">
            Enter Stock Ticker: <input type="text" name="ticker" value="{ticker}" required>
            <input type="submit" value="Predict">
        </form>
        <br>
        {result_html}
    '''

@app.route('/chart/<name>')
def chart(name):
    path = f"{name}.png"
    if os.path.exists(path):
        return send_file(path, mimetype='image/png')
    return "Chart not found", 404

if __name__ == '__main__':
    app.run(debug=True)
