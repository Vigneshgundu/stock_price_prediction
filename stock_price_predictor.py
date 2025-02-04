import streamlit as st
import pandas as pd
import numpy as np
from keras.models import load_model
import matplotlib.pyplot as plt
import yfinance as yf
from datetime import datetime
from sklearn.preprocessing import MinMaxScaler

st.title("Stock Price Predictor App")

# Get stock ID from user
stock = st.text_input("Enter the Stock ID", "GOOG")

# Define time range
end = datetime.now()
start = datetime(end.year - 20, end.month, end.day)

# Fetch stock data
google_data = yf.download(stock, start, end)
if google_data.empty:
    st.error("Failed to fetch stock data. Please check the stock symbol and try again.")
    st.stop()

google_data = google_data[['Close']].copy()  # Ensure only 'Close' column is used

# Load pre-trained model
model_path = r"C:\Users\gundu\OneDrive\Desktop\web-development\projects\python_stock_price_prediciton\Latest_stock_price_model.keras"
model = load_model(model_path)

st.subheader("Stock Data")
st.write(google_data)

# Splitting data
splitting_len = int(len(google_data) * 0.7)
x_test = google_data.iloc[splitting_len:].copy()

def plot_graph(figsize, values, full_data, extra_data=0, extra_dataset=None):
    fig = plt.figure(figsize=figsize)
    plt.plot(values, 'Orange', label='Moving Average')
    plt.plot(full_data['Close'], 'b', label='Original Close Price')
    if extra_data:
        plt.plot(extra_dataset, label='Additional MA')
    plt.legend()
    return fig

# Moving Averages and Plots
for days in [250, 200, 100]:
    google_data[f'MA_for_{days}_days'] = google_data['Close'].rolling(days).mean()
    st.subheader(f'Original Close Price and MA for {days} days')
    st.pyplot(plot_graph((15, 6), google_data[f'MA_for_{days}_days'], google_data))

st.subheader('Original Close Price and MA for 100 days and MA for 250 days')
st.pyplot(plot_graph((15, 6), google_data['MA_for_100_days'], google_data, 1, google_data['MA_for_250_days']))

# Data Scaling
scaler = MinMaxScaler(feature_range=(0, 1))
scaled_data = scaler.fit_transform(x_test[['Close']])

x_data, y_data = [], []
for i in range(100, len(scaled_data)):
    x_data.append(scaled_data[i - 100:i])
    y_data.append(scaled_data[i])

x_data, y_data = np.array(x_data), np.array(y_data)

# Model Predictions
predictions = model.predict(x_data)
inv_pre = scaler.inverse_transform(predictions)
inv_y_test = scaler.inverse_transform(y_data)

# Create DataFrame for comparison
ploting_data = pd.DataFrame({
    'original_test_data': inv_y_test.reshape(-1),
    'predictions': inv_pre.reshape(-1)
}, index=google_data.index[splitting_len + 100:])

st.subheader("Original values vs Predicted values")
st.write(ploting_data)

# Plot Original vs Predicted Prices
st.subheader('Original Close Price vs Predicted Close Price')
fig = plt.figure(figsize=(15, 6))
plt.plot(pd.concat([google_data['Close'][:splitting_len + 100], ploting_data], axis=0))
plt.legend(["Data - not used", "Original Test Data", "Predicted Test Data"])
st.pyplot(fig)
