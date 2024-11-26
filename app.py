import streamlit as st
import joblib
import yfinance as yf
import pandas as pd

# Load the pre-trained model
model = joblib.load('banknifty_option_model.pkl')

# Define the function to fetch data
def get_data(ticker):
    data = yf.download(ticker, period='5d', interval='1h')  # Last 5 days, hourly data
    data['Next_Day_Close'] = data['Close'].shift(-1)
    data['Daily_Change'] = data['Close'] - data['Open']
    data['Price_Range'] = data['High'] - data['Low']
    data['Percentage_Change'] = (data['Close'] - data['Open']) / data['Open'] * 100
    data.dropna(inplace=True)
    return data

# Set up the Streamlit UI
st.title('Bank Nifty Option Price Prediction')
ticker = st.text_input('Enter Ticker (e.g., ^NSEBANK or BANKNIFTY23NOV43000CE.NFO)', '^NSEBANK')

if ticker:
    # Get new data
    data = get_data(ticker)
    
    # Prepare features for prediction
    X_new = data[['Open', 'High', 'Low', 'Close', 'Daily_Change', 'Price_Range', 'Percentage_Change']]
    
    # Make predictions
    predictions = model.predict(X_new)
    data['Predicted_Close'] = predictions
    
    st.write("Predicted Close Prices for the Next Day")
    st.dataframe(data[['Close', 'Predicted_Close']])
