import yfinance as yf
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import logging
from statsmodels.tsa.statespace.sarimax import SARIMAX
from sklearn.metrics import mean_squared_error, mean_absolute_error, mean_absolute_percentage_error
from pmdarima import auto_arima
from datetime import timedelta

# Configure logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

# Function to fetch historical data
def fetch_stock_data(ticker, start_date, end_date):
    """
    Fetch historical stock data for a given ticker and date range.
    """
    try:
        stock_data = yf.download(ticker, start=start_date, end=end_date)
        if stock_data.empty:
            logging.error(f"No data found for ticker {ticker}")
            return None
        stock_data = stock_data['Close'].asfreq('D', method='pad')  # Fill missing dates
        logging.info(f"Successfully fetched data for {ticker}")
        return stock_data
    except Exception as e:
        logging.error(f"Error fetching data: {e}")
        return None

# Function to impute missing data
def impute_missing_data(data):
    """
    Impute missing values in the data using forward and backward fill.
    """
    if data.isnull().sum() > 0:
        logging.info(f"Imputing {data.isnull().sum()} missing values")
        data = data.fillna(method='ffill').fillna(method='bfill')  # Forward and backward fill
    return data

# Function to optimize SARIMA parameters using auto_arima
def optimize_sarima_params(data):
    """
    Automatically optimize SARIMA parameters using the auto_arima function.
    """
    try:
        model = auto_arima(data, seasonal=True, m=7, stepwise=True, trace=True, error_action='ignore', suppress_warnings=True)
        order = model.order
        seasonal_order = model.seasonal_order
        logging.info(f"Optimized SARIMA parameters - Order: {order}, Seasonal Order: {seasonal_order}")
        return order, seasonal_order
    except Exception as e:
        logging.error(f"Error optimizing SARIMA parameters: {e}")
        return (1, 1, 1), (1, 1, 1, 7)  # Default values

# Function to train SARIMA model and make predictions
def train_sarima_and_predict(data, order, seasonal_order):
    """
    Train a SARIMA model and make predictions for the next day.
    """
    try:
        # Split data into training and testing sets
        train_data = data.iloc[:-1]
        test_data = data.iloc[-1:]

        # Fit SARIMAX model
        model = SARIMAX(train_data, order=order, seasonal_order=seasonal_order)
        model_fit = model.fit(disp=False)

        # Predict the next day's stock price
        prediction = model_fit.forecast(steps=1)

        # Evaluate model
        rmse = np.sqrt(mean_squared_error(test_data, prediction))
        mae = mean_absolute_error(test_data, prediction)
        mape = mean_absolute_percentage_error(test_data, prediction)
        logging.info(f"Model Evaluation - RMSE: {rmse}, MAE: {mae}, MAPE: {mape}")

        return prediction[0], model_fit
    except Exception as e:
        logging.error(f"Error training SARIMA model: {e}")
        return None, None

# Function to plot historical data and prediction
def plot_data_and_prediction(data, prediction):
    """
    Plot historical stock data and the predicted value for the next day.
    """
    try:
        plt.figure(figsize=(12, 6))
        plt.plot(data.index, data, label='Historical Data', color='blue')

        # Plot the prediction point
        prediction_date = data.index[-1] + timedelta(days=1)
        plt.scatter(prediction_date, prediction, color='red', label='Next Day Prediction')

        # Add title and labels
        plt.title('1 Year Historical Data and Next Day Prediction')
        plt.xlabel('Date')
        plt.ylabel('Stock Price')
        plt.legend()
        plt.grid(True)
        plt.show()
    except Exception as e:
        logging.error(f"Error plotting data: {e}")

# Main function
def main():
    """
    Main function to execute the stock prediction pipeline.
    """
    try:
        # Input stock ticker from user
        ticker = input("Enter the stock ticker: ").strip().upper()

        # Fetch 1 year of historical data
        end_date = pd.Timestamp.now()
        start_date = end_date - pd.DateOffset(years=1)
        stock_data = fetch_stock_data(ticker, start_date, end_date)

        if stock_data is None:
            return

        # Impute missing data
        stock_data = impute_missing_data(stock_data)

        # Optimize SARIMA parameters
        order, seasonal_order = optimize_sarima_params(stock_data)

        # Train SARIMA model and make prediction
        prediction, model_fit = train_sarima_and_predict(stock_data, order, seasonal_order)

        if prediction is not None:
            print(f"Predicted stock price for the next day: {prediction}")

            # Plot historical data and prediction
            plot_data_and_prediction(stock_data, prediction)
    except Exception as e:
        logging.error(f"Error in main function: {e}")

if __name__ == "__main__":
    main()