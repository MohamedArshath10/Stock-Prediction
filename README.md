<div align="center">

#  Stock Market Predictor

### A machine learning web app that predicts stock prices and forecasts the next 30 days using JAX neural networks.

[![Python](https://img.shields.io/badge/Python-3.10-3670A0?style=for-the-badge&logo=python&logoColor=ffdd54)](https://python.org)
[![Flask](https://img.shields.io/badge/Flask-black?style=for-the-badge&logo=flask&logoColor=white)](https://flask.palletsprojects.com)
[![JAX](https://img.shields.io/badge/JAX-Neural%20Network-412991?style=for-the-badge&logo=google&logoColor=white)](https://jax.readthedocs.io)
[![Render](https://img.shields.io/badge/Deployed-Render-46E3B7?style=for-the-badge&logo=render&logoColor=white)](https://render.com)

</div>

---

## ✨ Features

- 📡 **Live Stock Data** — Fetches real-time historical data using yFinance for any valid ticker (AAPL, TSLA, GOOGL, etc.)
- 🧠 **JAX Neural Network** — Custom MLP model built with Flax and trained using Optax's Adam optimizer
- 📊 **Predicted vs Actual Chart** — Visual comparison of model predictions against real test prices
- 📅 **30-Day Forecast** — Autoregressive prediction of the next 30 days using a sliding window
- 💾 **Model Caching** — Trained models are saved per ticker in msgpack format and reloaded on repeat requests
- 📉 **Closing Price Chart** — Historical price trend visualization from 2010 to today
- ⚡ **JIT Compilation** — Training step compiled with `@jax.jit` for fast repeated execution
- 🌐 **Flask Web Interface** — Simple form-based UI to enter any stock ticker and view results instantly

---

## 🛠️ Tech Stack

| Technology | Purpose |
|------------|---------|
| Python | Core programming language |
| Flask | Web framework and routing |
| JAX | High-performance numerical computing and automatic differentiation |
| Flax (linen) | Neural network architecture definition |
| Optax | Gradient-based optimization (Adam optimizer) |
| yFinance | Fetching historical stock market data |
| Scikit-learn | MinMaxScaler for data normalization |
| NumPy + Pandas | Data manipulation and preprocessing |
| Matplotlib | Chart generation (closing price, prediction, forecast) |
| Gunicorn | Production WSGI server for deployment |

---

## 🏗️ Project Structure

```
Stock-Prediction/
│
├── model.py              # Flask app, data pipeline, training trigger, chart generation
├── LSTMModel.py          # MLP model architecture, training loop, forecasting logic
├── requirements.txt      # Python dependencies
│
└── saved_models/         # Auto-created — cached trained models per ticker
    └── AAPL.msgpack      # Example: saved model for AAPL ticker
```

---

## 🧠 How the Model Works

### Architecture — MLP (Multi-Layer Perceptron)

```
Input: 100-day price sequence  →  shape (batch, 100, 1)
        ↓
Flatten  →  shape (batch, 100)
        ↓
Dense(64)  →  ReLU activation
        ↓
Dense(1)   →  predicted next-day price (scaled)
```

### Training

- **Loss function:** Mean Squared Error (MSE)
- **Optimizer:** Adam with learning rate `0.001` via Optax
- **Epochs:** 100 iterations per ticker
- **Gradient computation:** `jax.grad()` — automatic differentiation
- **Speed:** `@jax.jit` compiles the training step via XLA for fast repeated calls
- **Random seed:** `jax.random.PRNGKey(0)` — fully reproducible initialization

---

## 🔄 Full Prediction Pipeline

```
User submits stock ticker (e.g. AAPL)
        ↓
yFinance downloads historical data from 2010-01-01 to today
        ↓
Extract closing prices → split 70% train / 30% test
        ↓
MinMaxScaler normalizes values to range [0, 1]
(fit on train only — no data leakage)
        ↓
Sliding window: 100 days input → 1 day output
        ↓
Check saved_models/ for cached model
   ├── Found → load and skip training
   └── Not found → train 100 epochs → save to disk
        ↓
Predict on test set → inverse scale → plot vs actual
        ↓
Autoregressive 30-day forecast using last test window
        ↓
Three charts rendered and served to the browser
```

---

## 📊 Output Charts

| Chart | Description |
|-------|-------------|
| **Closing Price** | Full historical closing price from 2010 to today |
| **Predicted vs Actual** | Model predictions overlaid on real test prices |
| **30-Day Forecast** | Next 30 days predicted autoregressively from the last known window |

---

## 💾 Model Caching

Trained models are saved per ticker to avoid retraining on every request:

- First request for `AAPL` → trains for 100 epochs → saves to `saved_models/AAPL.msgpack`
- Second request for `AAPL` → loads saved model instantly → skips training
- If the saved file is corrupt → automatically deleted → model retrained fresh
- Different tickers get separate models since each stock has unique price patterns

---

## ⚙️ Getting Started

### Prerequisites
- Python 3.10+
- pip
- Git

### 1. Clone the repository

```bash
git clone https://github.com/MohamedArshath10/Stock-Prediction.git
cd Stock-Prediction
```

### 2. Install dependencies

```bash
pip install -r requirements.txt
```

### 3. Run the app

```bash
python model.py
```

Open `http://localhost:5000` in your browser, enter any stock ticker, and click Predict.

---

## 📦 Requirements

```
Flask
yfinance
pandas
numpy
matplotlib
scikit-learn
jax
jaxlib
flax
optax
gunicorn
```

---

## 🚀 Deployment

### Render
1. Push the repo to GitHub
2. Create a new Web Service on [render.com](https://render.com)
3. Set the start command:
```bash
gunicorn model:app
```
4. Set Python environment and deploy

> **Note:** The `saved_models/` folder is ephemeral on Render — models are retrained on first request after each deployment.

---

## ⚠️ Limitations

- The model uses only historical closing prices — no sentiment, news, or fundamental data
- Multi-step forecasting accumulates error over 30 days — treat the forecast as a trend indicator, not an exact price
- MLP flattens the time sequence — an LSTM or Transformer would better capture temporal dependencies
- Predictions are based on learned patterns and cannot account for unexpected market events

---

## 🌱 Future Improvements

- [ ] Replace MLP with LSTM or Transformer for better sequential learning
- [ ] Add technical indicators as features — RSI, MACD, Bollinger Bands
- [ ] Display RMSE and MAPE evaluation metrics on the results page
- [ ] Build a React frontend to replace the raw HTML Flask response
- [ ] Add user-selectable date ranges and forecast horizons
- [ ] Add confidence intervals around the 30-day forecast
- [ ] Persist cached models to cloud storage so they survive server restarts

---

## 👨‍💻 Author

**Mohamed Arshath**
- LinkedIn: [mohamedarshathm](https://www.linkedin.com/in/mohamedarshathm)
- GitHub: [MohamedArshath10](https://github.com/MohamedArshath10)
- Email: arshath.m2003@gmail.com

---

<div align="center">
⭐ If you found this project useful, consider giving it a star!
</div>
