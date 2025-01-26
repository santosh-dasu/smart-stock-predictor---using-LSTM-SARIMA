# Import necessary libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import yfinance as yf
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout, Bidirectional
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from tensorflow.keras.optimizers import Adam
from datetime import datetime, timedelta
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import sys

# Fetch stock data with error handling
def fetch_stock_data(ticker, start_date, end_date):
    try:
        stock_data = yf.download(ticker, start=start_date, end=end_date)
        if stock_data.empty:
            raise ValueError(f"No data found for ticker: {ticker}")
        return stock_data
    except Exception as e:
        print(f"Error fetching data for {ticker}: {e}")
        sys.exit(1)

# Feature engineering: Add technical indicators
def add_technical_indicators(data):
    try:
        # Moving Averages
        data['MA_50'] = data['Close'].rolling(window=50).mean()
        data['MA_200'] = data['Close'].rolling(window=200).mean()
        
        # Relative Strength Index (RSI)
        delta = data['Close'].diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
        rs = gain / loss
        data['RSI'] = 100 - (100 / (1 + rs))
        
        # Moving Average Convergence Divergence (MACD)
        data['MACD'] = data['Close'].ewm(span=12, adjust=False).mean() - data['Close'].ewm(span=26, adjust=False).mean()
        
        # Drop NaN values created by rolling windows
        data.dropna(inplace=True)
        return data
    except Exception as e:
        print(f"Error adding technical indicators: {e}")
        sys.exit(1)

# Preprocess data with error handling
def preprocess_data(stock_data):
    try:
        # Add technical indicators
        stock_data = add_technical_indicators(stock_data)
        
        # Use multiple features
        features = ['Open', 'High', 'Low', 'Close', 'Volume', 'MA_50', 'MA_200', 'RSI', 'MACD']
        data = stock_data[features].values
        
        # Normalize data
        scaler = MinMaxScaler(feature_range=(0, 1))
        scaled_data = scaler.fit_transform(data)
        
        # Split into training (80%) and validation (20%) data
        training_data_len = int(len(scaled_data) * 0.8)
        train_data = scaled_data[:training_data_len, :]
        val_data = scaled_data[training_data_len - 60:, :]  # Use last 60 days for sequence
        
        # Create sequences
        def create_sequences(data, seq_length):
            X = []
            y = []
            for i in range(seq_length, len(data)):
                X.append(data[i-seq_length:i, :])
                y.append(data[i, 3])  # 'Close' is the 4th column
            return np.array(X), np.array(y)
        
        seq_length = 60
        X_train, y_train = create_sequences(train_data, seq_length)
        X_val, y_val = create_sequences(val_data, seq_length)
        
        return X_train, y_train, X_val, y_val, scaler, seq_length, scaled_data, features
    except Exception as e:
        print(f"Error preprocessing data: {e}")
        sys.exit(1)

# Build LSTM model with error handling
def build_lstm_model(input_shape):
    try:
        model = Sequential()
        model.add(Bidirectional(LSTM(units=100, return_sequences=True), input_shape=input_shape))
        model.add(Dropout(0.3))
        model.add(Bidirectional(LSTM(units=100, return_sequences=False)))
        model.add(Dropout(0.3))
        model.add(Dense(units=50, activation='relu'))
        model.add(Dense(units=1))
        
        # Use a learning rate scheduler
        optimizer = Adam(learning_rate=0.001)
        model.compile(optimizer=optimizer, loss='mean_squared_error')
        return model
    except Exception as e:
        print(f"Error building LSTM model: {e}")
        sys.exit(1)

# Predict future prices for 6 months ahead with error handling
def predict_future_price(model, scaler, scaled_data, seq_length, days_ahead=120):
    try:
        # Use the last seq_length days of data to make predictions
        test_data = scaled_data[-seq_length:, :]
        
        predictions = []
        for _ in range(days_ahead):
            X_test = np.array([test_data[-seq_length:]])
            X_test = np.reshape(X_test, (X_test.shape[0], X_test.shape[1], X_test.shape[2]))
            
            pred_price = model.predict(X_test)
            predictions.append(pred_price[0, 0])
            
            # Append the prediction to the test data for the next prediction
            new_row = np.append(test_data[-1, 1:], pred_price[0, 0])  # Shift features and add new prediction
            test_data = np.vstack([test_data, new_row])
        
        # Inverse transform the predictions
        dummy_array = np.zeros((len(predictions), len(features)))
        dummy_array[:, 3] = predictions  # Place predictions in the 'Close' column
        predictions = scaler.inverse_transform(dummy_array)[:, 3]
        
        # Generate future dates
        last_date = stock_data.index[-1]
        future_dates = pd.date_range(start=last_date + timedelta(days=1), periods=days_ahead, freq='B')
        
        return predictions, future_dates
    except Exception as e:
        print(f"Error predicting future prices: {e}")
        sys.exit(1)

# Plot historical data and predictions with error handling
def plot_history_prediction(stock_data, predictions, future_dates, ticker):
    try:
        plt.figure(figsize=(16, 8))
        plt.title(f'{ticker} Stock Price Prediction')
        plt.xlabel('Date')
        plt.ylabel('Close Price USD ($)')
        
        # Plot historical data
        plt.plot(stock_data.index, stock_data['Close'], label='Historical Data', color='blue')
        
        # Plot predictions starting from the last historical date
        plt.plot([stock_data.index[-1]] + list(future_dates), [stock_data['Close'].iloc[-1]] + list(predictions), label='Predictions for Next 6 Months', color='red', linestyle='--')
        
        plt.legend()
        plt.grid()
        plt.show()
    except Exception as e:
        print(f"Error plotting data: {e}")
        sys.exit(1)

# Main execution with error handling
if __name__ == "__main__":
    try:
        # Input ticker symbol
        ticker = input("Enter the ticker symbol: ").strip().upper()
        if not ticker:
            raise ValueError("Ticker symbol cannot be empty.")
        
        # Define date range (fetch data up to the current date)
        end_date = datetime.now().strftime('%Y-%m-%d')
        start_date = (datetime.now() - timedelta(days=5*365)).strftime('%Y-%m-%d')
        
        # Fetch stock data
        stock_data = fetch_stock_data(ticker, start_date, end_date)
        
        # Preprocess data
        X_train, y_train, X_val, y_val, scaler, seq_length, scaled_data, features = preprocess_data(stock_data)
        
        # Build and train the model
        model = build_lstm_model((X_train.shape[1], X_train.shape[2]))
        early_stopping = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)
        reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.2, patience=5, min_lr=0.0001)
        history = model.fit(X_train, y_train, batch_size=32, epochs=100, validation_data=(X_val, y_val), callbacks=[early_stopping, reduce_lr])
        
        # Predict the stock prices for 6 months ahead
        predictions, future_dates = predict_future_price(model, scaler, scaled_data, seq_length, days_ahead=120)
        
        # Display the predicted stock price for 6 months ahead
        print(f"Predicted Stock Price for {ticker} in 6 months: ${predictions[-1]:.2f}")
        
        # Plot historical data and predictions
        plot_history_prediction(stock_data, predictions, future_dates, ticker)
    except KeyboardInterrupt:
        print("\nProgram interrupted by user.")
        sys.exit(0)
    except Exception as e:
        print(f"An error occurred: {e}")
        sys.exit(1)