from flask import Flask, request, send_file
import yfinance as yf
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import os
import time
from datetime import datetime
from sklearn.preprocessing import MinMaxScaler
from model import predict_with_jax, predict_next_days, train_state_loop, create_train_state, MLP  # extended imports

app = Flask(__name__)

@app.route('/', methods=['GET', 'POST'])
def index():
    result_html = ''
    ticker_value = 'AAPL'

    if request.method == 'POST':
        ticker = request.form['ticker'].upper()
        ticker_value = ticker

        # Download stock data
        df = yf.download(ticker, start='2010-01-01', end=datetime.today().strftime('%Y-%m-%d'), auto_adjust=False)

        # Save closing price chart
        plt.figure(figsize=(12, 6))
        plt.plot(df['Close'])
        plt.title(f"{ticker} Closing Price")
        plt.xlabel('Date')
        plt.ylabel('Price')
        plt.grid(True)
        plt.tight_layout()
        plt.savefig('closing.png')
        plt.close()

        # Prepare training and testing data
        data_training = df['Close'][0:int(len(df)*0.7)]
        data_testing = df['Close'][int(len(df)*0.7):]
        scaler = MinMaxScaler(feature_range=(0, 1))
        data_training_array = scaler.fit_transform(data_training.values.reshape(-1, 1))

        # Training set
        x_train, y_train = [], []
        for i in range(100, len(data_training_array)):
            x_train.append(data_training_array[i-100:i])
            y_train.append(data_training_array[i, 0])
        x_train, y_train = np.array(x_train), np.array(y_train)

        # Testing set
        past_100_days = data_training.tail(100)
        final_df = pd.concat([past_100_days, data_testing], ignore_index=True)
        input_data = scaler.transform(final_df.values.reshape(-1, 1))

        x_test, y_test = [], []
        for i in range(100, len(input_data)):
            x_test.append(input_data[i-100:i])
            y_test.append(input_data[i, 0])
        x_test, y_test = np.array(x_test), np.array(y_test)

        # Predict with JAX
        y_predicted = predict_with_jax(x_test, x_train, y_train, ticker)
        scale_factor = 1 / scaler.scale_[0]
        y_predicted = y_predicted * scale_factor
        y_test = y_test * scale_factor

        # Save prediction chart
        plt.figure(figsize=(12, 6))
        plt.plot(y_test, 'b', label='Original')
        plt.plot(y_predicted, 'r', label='Predicted')
        plt.xlabel('Time')
        plt.ylabel('Price')
        plt.title('Predicted vs Actual')
        plt.legend()
        plt.grid(True)
        plt.tight_layout()
        plt.savefig('prediction.png')
        plt.close()

        # --------------- Predict next 30 days ----------------
        import jax
        import jax.numpy as jnp
        model = MLP()
        rng = jax.random.PRNGKey(0)
        from model import create_train_state

        # Get trained state
        from model import train_state_loop
        state = train_state_loop(model, rng, x_train, y_train)

        # Use last window
        x_last_seq = jnp.array(x_test[-1])
        next_30_scaled = predict_next_days(x_last_seq, state, 30)
        next_30 = next_30_scaled * scale_factor

        # Plot next 30 days
        plt.figure(figsize=(12, 6))
        plt.plot(next_30, marker='o', label='Next 30 Days')
        plt.title(f"{ticker} — Forecast Next 30 Days")
        plt.xlabel("Days from today")
        plt.ylabel("Price")
        plt.grid(True)
        plt.legend()
        plt.tight_layout()
        plt.savefig('forecast.png')
        plt.close()
        # ------------------------------------------------------

        # Cache-busting
        timestamp = int(time.time())
        result_html = f"""
            <h2>Results for <span style='color:green'>{ticker}</span></h2>
            <h3>Closing Price Chart</h3>
            <img src='/chart/closing?t={timestamp}' width='800'><br>
            <h3>Predicted vs Actual</h3>
            <img src='/chart/prediction?t={timestamp}' width='800'><br>
            <h3>📅 Forecast — Next 30 Days</h3>
            <img src='/chart/forecast?t={timestamp}' width='800'><br>
        """

    return f'''
        <h1>📈 Stock Trend Prediction (Flask + JAX)</h1>
        <form method="post">
            Enter Stock Ticker: <input type="text" name="ticker" value="{ticker_value}" required>
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
