import yfinance as yf
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from datetime import datetime

from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from tensorflow.keras.callbacks import EarlyStopping


# CONFIG


TICKER = "TSLA"          # <- change to AAPL, TSLA, etc.
START_DATE = "2021-01-01"
END_DATE = datetime.today().strftime("%Y-%m-%d")

LOOKBACK = 60            # how many past days to use to predict the next day
TEST_SIZE = 0.2          # last 20% of samples used for testing


# DOWNLOAD DATA


print(f"Downloading {TICKER} data...")
data = yf.download(TICKER, start=START_DATE, end=END_DATE)

if data.empty:
    raise ValueError("No data downloaded. Check ticker or dates.")

# Use Adj Close if available, otherwise Close
price_col = "Adj Close" if "Adj Close" in data.columns else "Close"

df = data[[price_col]].copy()
df.dropna(inplace=True)

print(f"Data points: {len(df)}")


# 1. FEATURE ENGINEERING


df = data.copy()

# use adjusted close if available, otherwise close
price_col = "Adj Close" if "Adj Close" in df.columns else "Close"

# momentum features
df["Return_1d"] = df[price_col].pct_change(1)
df["Return_5d"] = df[price_col].pct_change(5)

# trend features (moving averages)
df["MA_10"] = df[price_col].rolling(window=10).mean()
df["MA_20"] = df[price_col].rolling(window=20).mean()

# volatility feature (10-day rolling std of daily returns)
df["Volatility_10d"] = df["Return_1d"].rolling(window=10).std()

# drop rows with NaNs created by pct_change / rolling
df = df.dropna().copy()

# choose the feature columns for the LSTM
feature_cols = [
    price_col,           # main price
    "Open", "High", "Low", "Volume",
    "Return_1d", "Return_5d",
    "MA_10", "MA_20", "Volatility_10d",
]

# some tickers might not have all OHLCV columns – keep only existing ones
feature_cols = [c for c in feature_cols if c in df.columns]

print("Using feature columns:", feature_cols)

# scale all features to [0, 1]
scaler = MinMaxScaler()
scaled_features = scaler.fit_transform(df[feature_cols].values)

# index of the price column inside feature_cols
price_idx = feature_cols.index(price_col)


# 2. BUILD SEQUENCES


def create_sequences(features, target_index, lookback):
    X, y = [], []
    for i in range(lookback, len(features)):
        # past LOOKBACK rows of all features
        X.append(features[i - lookback:i])
        # next-day price (scaled), from the same row i
        y.append(features[i, target_index])
    return np.array(X), np.array(y)

X, y = create_sequences(scaled_features, price_idx, LOOKBACK)

print("Sequence shape:", X.shape, y.shape)  # (samples, LOOKBACK, num_features)

# time-based train / test split (no shuffling!)
split_index = int(len(X) * (1 - TEST_SIZE))

X_train, X_test = X[:split_index], X[split_index:]
y_train, y_test = y[:split_index], y[split_index:]

print(f"Train samples: {len(X_train)}, Test samples: {len(X_test)}")


# HYPERPARAMETERS

EPOCHS = 100          # increased from 60
BATCH_SIZE = 16       # smaller batches can help convergence
DROPOUT_RATE = 0.3    # regularization strength


# IMPROVED LSTM MODEL

model = Sequential([
    LSTM(
        units=128,
        return_sequences=True,
        input_shape=(X_train.shape[1], X_train.shape[2])
    ),
    Dropout(DROPOUT_RATE),

    LSTM(
        units=128,
        return_sequences=False
    ),
    Dropout(DROPOUT_RATE),

    Dense(64, activation="relu"),
    Dense(32, activation="relu"),
    Dense(1)  # final price output
])

model.compile(
    optimizer="adam",
    loss="mae"      # MAE works nicely for price prediction
)


# EARLY STOPPING

early_stop = EarlyStopping(
    monitor="val_loss",
    patience=8,
    restore_best_weights=True
)

history = model.fit(
    X_train,
    y_train,
    epochs=EPOCHS,
    batch_size=BATCH_SIZE,
    validation_split=0.1,   # use 10% of training data as validation
    callbacks=[early_stop],
    shuffle=False,
    verbose=1
)

# 3. EVALUATE MODEL


# model outputs scaled prices (in feature space)
y_pred_scaled = model.predict(X_test)

# helper: inverse-transform ONLY the price column
def inverse_price(scaled_price_vector):
    """
    scaled_price_vector: shape (n_samples, 1)
    We rebuild a dummy feature matrix, put the scaled price in the correct
    column, inverse-transform, then read back the true price.
    """
    n = scaled_price_vector.shape[0]
    dummy = np.zeros((n, len(feature_cols)))
    dummy[:, price_idx] = scaled_price_vector.ravel()
    inv = scaler.inverse_transform(dummy)
    return inv[:, price_idx]

y_test_unscaled = inverse_price(y_test.reshape(-1, 1))
y_pred_unscaled = inverse_price(y_pred_scaled.reshape(-1, 1))

from sklearn.metrics import mean_absolute_error, mean_squared_error

mae = mean_absolute_error(y_test_unscaled, y_pred_unscaled)
mse = mean_squared_error(y_test_unscaled, y_pred_unscaled)
rmse = np.sqrt(mse)

print("\nModel Performance (LSTM with 10 features):")
print(f"MAE  : {mae:.4f}")
print(f"RMSE : {rmse:.4f}")


# 4. PERCENT ERROR & WITHIN-10% ACCURACY CHECK


# % error for each test prediction
pct_errors = np.abs(y_pred_unscaled - y_test_unscaled) / y_test_unscaled * 100

# success rate: predictions within ±10%
within_10_accuracy = np.mean(pct_errors <= 10) * 100

print("\nError Analysis:")
print(f"Average % Error: {pct_errors.mean():.2f}%")
print(f"Predictions within ±10% of actual: {within_10_accuracy:.1f}% of test days")


# PLOT ACTUAL VS PREDICTED


# Build matching date index for the y values
all_target_dates = df.index[LOOKBACK:]
test_dates = all_target_dates[split_index:]

plt.figure(figsize=(12, 5))
plt.plot(test_dates, y_test_unscaled, label="Actual")
plt.plot(test_dates, y_pred_unscaled, label="Predicted")
plt.title(f"{TICKER} LSTM Stock Price Prediction")
plt.xlabel("Date")
plt.ylabel("Price")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()


# 4. NEXT-DAY PRICE PREDICTION


print("\nNext-Day Price Prediction:")

# use the LAST LOOKBACK rows from the scaled feature matrix
last_sequence = scaled_features[-LOOKBACK:]              # shape (LOOKBACK, num_features)
last_sequence = last_sequence.reshape(1, LOOKBACK, -1)   # shape (1, LOOKBACK, num_features)

next_price_scaled = model.predict(last_sequence)[0][0]
next_price = inverse_price(np.array([[next_price_scaled]]))[0]

# last actual close from df

print("\nLast few rows from Yahoo Finance:")
print(df[[price_col]].tail())
print("Last date in dataset:", df.index[-1])

last_close = float(df[price_col].iloc[-1])
print(f"Last closing ({price_col}): {last_close:.2f}")

last_close = float(df[price_col].iloc[-1])

print(f"Last closing ({price_col}): {last_close:.2f}")
print(f"Predicted next-day ({price_col}): {next_price:.2f}")
print("----------------------------------------")