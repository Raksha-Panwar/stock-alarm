# lstm_app.py
# nrsy mmin ofqs rudr -- app password

import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import time

from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense

# Import your helper utilities
from helpers import fetch_history, period_from_inputs, beep, notify_desktop, send_email


# ============================================================
# STREAMLIT UI
# ============================================================
st.set_page_config(page_title="Stock Alarm - LSTM", layout="wide")  # Sets the browser tab title
st.title("📉 Stock Alarm — LSTM Forecasting (Deep Learning)")   # title of page


# ------------------------------------------------------------
# Sidebar Inputs
# ------------------------------------------------------------
market = st.sidebar.selectbox("Market", ["India", "US"])      # sidebar selectbox named market to choose among US , India
asset_type = st.sidebar.selectbox("Asset Type", ["Stock", "Index", "FnO", "Crypto"])       # same as above

default_ticker = "INFY" if market == "India" else "AAPL"        # sets value of variable default_ticker based on maket selection
ticker = st.sidebar.text_input("Ticker", default_ticker)        # a text_input box named ticker 

years = st.sidebar.number_input("Years", 0, 50, 1)      # number_input year, month and day that increases/ decreases by 1 on click of - and + from min to max and default_value is 1
months = st.sidebar.number_input("Months", 0, 11, 0)
days = st.sidebar.number_input("Days", 0, 365, 0)

period = period_from_inputs(
    years=int(years),
    months=int(months),
    days=int(days)
)

# Real-time alert settings
target_price = st.sidebar.number_input("Alarm when price <=", value=0.0, step=0.1)
check_interval = st.sidebar.number_input("Polling interval (seconds)", value=10, min_value=1)

# Email settings
with st.sidebar.expander("Email Settings (optional)"):
    smtp_server = st.text_input("SMTP Server", "smtp.gmail.com")
    smtp_port = st.number_input("SMTP Port", value=587)
    smtp_user = st.text_input("SMTP Username / Email", value="raksha26panwar@gmail.com")
    smtp_pass = st.text_input("SMTP Password", type="password")
    alert_email = st.text_input("Send Alerts To Email", value="rakshapanwar@zohomail.in")


# ============================================================
# LSTM Model Preparation
# ============================================================

# series -> a 1D array of values & window -> LSTM model looks at last 60 days to predict the next value
def prepare_lstm_data(series, window=60):
    """Creates sliding window data for LSTM"""
    # X → input sequences (features) & y → output values (labels / targets)
    X, y = [], []
    for i in range(window, len(series)):
        # Given the last 60 values in X, predict the 61st value - y
        X.append(series[i - window:i])      # 0 to 59th (previous 60 values)
        y.append(series[i])     # 60th (61st value i.e., to be predicted)
    return np.array(X), np.array(y)     # final array


def train_lstm(close_prices):
    """Builds and trains a small LSTM model"""

    scaler = MinMaxScaler()
    scaled = scaler.fit_transform(close_prices.reshape(-1, 1))

    # Create data windows
    X, y = prepare_lstm_data(scaled, window=60)

    X = X.reshape(X.shape[0], X.shape[1], 1)

    # Build model
    model = Sequential()
    model.add(LSTM(50, return_sequences=True, input_shape=(60, 1)))
    model.add(LSTM(50))
    model.add(Dense(1))

    model.compile(optimizer="adam", loss="mse")

    # Train
    model.fit(X, y, epochs=5, batch_size=32, verbose=0)

    return model, scaler, scaled


# def lstm_forecast(model, scaler, scaled_values, days=30):
#     """Forecast future values using last known window"""

#     window = scaled_values[-60:]
#     prediction_list = []

#     for _ in range(days):
#         pred = model.predict(window.reshape(1, 60, 1), verbose=0)
#         prediction_list.append(pred[0, 0])

#         # Update window
#         window = np.vstack((window[1:], pred))

#     return scaler.inverse_transform(np.array(prediction_list).reshape(-1, 1))


# # ============================================================
# # FETCH + FORECAST (LSTM)
# # ============================================================
# if st.button("Fetch & Forecast"):

#     try:
#         df = fetch_history("US" if market == "US" else "India", asset_type, ticker, period)
#         st.write("Downloaded:", len(df), "rows")

#         if df.empty or "Close" not in df.columns:
#             st.error("❌ No valid Close price data found.")
#         else:
#             close_prices = df["Close"].values

#             # Train LSTM
#             model, scaler, scaled = train_lstm(close_prices)

#             # 30-day forecast
#             forecast = lstm_forecast(model, scaler, scaled, days=30)

#             # Show forecast
#             st.subheader("LSTM Forecast (Next 30 Days)")
#             st.line_chart(forecast)

#             # Combine with historical for full chart
#             combined = np.concatenate((close_prices[-120:], forecast.flatten()))
#             st.subheader("Historical + Forecast Chart")
#             st.line_chart(combined)

#             st.success("Forecast generated successfully.")

#     except Exception as e:
#         st.error(f"Error: {e}")


# # ============================================================
# # REAL-TIME MONITORING
# # ============================================================
# if st.button("Start Real-time Monitoring"):

#     if target_price <= 0:
#         st.error("Please enter a valid alert price.")
#     else:
#         st.success("⏳ Monitoring price...")

#         status = st.empty()

#         try:
#             while True:
#                 df_live = fetch_history(
#                     "US" if market == "US" else "India",
#                     asset_type, ticker,
#                     period="1d"
#                 )

#                 if df_live.empty:
#                     status.write("No data available...")
#                     time.sleep(check_interval)
#                     continue

#                 price = df_live["Close"].iloc[-1]
#                 status.write(f"Live Price: {price}")
#                 status.write(f"DEBUG → price={price}, target={target_price}")

#                 if price <= target_price:

#                     msg = f"🚨 ALERT: {ticker} reached {price}"

#                     beep()
#                     notify_desktop(msg)

#                     if alert_email:
#                         send_email(
#                             smtp_server, smtp_port,
#                             smtp_user, smtp_pass,
#                             alert_email,
#                             "Price Alert Triggered", msg
#                         )

#                     st.warning(msg)
#                     break

#                 time.sleep(check_interval)

#         except Exception as e:
#             st.error(f"Monitoring error: {e}")
