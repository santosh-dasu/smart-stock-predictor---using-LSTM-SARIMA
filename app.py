import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import logging
from datetime import datetime, timedelta
import tensorflow as tf
from sklearn.preprocessing import MinMaxScaler
from statsmodels.tsa.statespace.sarimax import SARIMAX
from sklearn.metrics import mean_squared_error, mean_absolute_error, mean_absolute_percentage_error
from pmdarima import auto_arima
import plotly.graph_objects as go

# Configure logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

# Cache stock data fetching to avoid redundant requests
@st.cache_data(ttl=86400)  # Cache for 24 hours
def fetch_stock_data(ticker, start_date, end_date):
    try:
        logging.info(f"Fetching data for ticker: {ticker}, start_date: {start_date}, end_date: {end_date}")
        stock_data = yf.download(ticker, start=start_date, end=end_date)
        if stock_data.empty:
            logging.error(f"No data found for ticker {ticker}")
            return None
        stock_data = stock_data['Close'].asfreq('B', method='ffill')  # Use 'B' for business days
        logging.info(f"Successfully fetched data for {ticker}")
        return stock_data
    except Exception as e:
        logging.error(f"Error fetching data: {e}")
        return None

# Function to impute missing data
def impute_missing_data(data):
    if data.isnull().sum() > 0:
        logging.info(f"Imputing {data.isnull().sum()} missing values")
        data = data.fillna(method='ffill').fillna(method='bfill')
    return data

# Function to optimize SARIMA parameters
def optimize_sarima_params(data):
    try:
        model = auto_arima(data, seasonal=True, m=21, stepwise=True, trace=True, error_action='ignore', suppress_warnings=True)
        order = model.order
        seasonal_order = model.seasonal_order
        logging.info(f"Optimized SARIMA parameters - Order: {order}, Seasonal Order: {seasonal_order}")
        return order, seasonal_order
    except Exception as e:
        logging.error(f"Error optimizing SARIMA parameters: {e}")
        return (1, 1, 1), (1, 1, 1, 21)  # Default values

# Function to train SARIMA model and make predictions
def train_sarima_and_predict(data, order, seasonal_order, days=1):
    try:
        model = SARIMAX(data, order=order, seasonal_order=seasonal_order)
        model_fit = model.fit(disp=False)
        forecast = model_fit.forecast(steps=days)
        return forecast, model_fit
    except Exception as e:
        logging.error(f"Error training SARIMA model: {e}")
        return None, None

# Function to evaluate model performance
def evaluate_model(actual, predicted):
    rmse = np.sqrt(mean_squared_error(actual, predicted))
    mae = mean_absolute_error(actual, predicted)
    mape = mean_absolute_percentage_error(actual, predicted)
    return rmse, mae, mape

# Function to plot SARIMA prediction using Plotly
def plot_sarima_prediction(data, forecast, selected_dates):
    try:
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=data.index, y=data, mode='lines', name='Historical Data', line=dict(color='blue', width=2)))
        fig.add_trace(go.Scatter(x=selected_dates, y=forecast, mode='lines+markers', name='Predicted Price', line=dict(color='red', width=2)))
        fig.update_layout(
            title=dict(text='SARIMA: Historical Data & Predicted Stock Prices', x=0.5, xanchor='center'),
            xaxis_title='Year',
            yaxis_title='Stock Price',
            legend=dict(font=dict(size=12)),
            template='plotly_white',
            hovermode='x unified'
        )
        st.plotly_chart(fig, use_container_width=True)  # Make the plot responsive
    except Exception as e:
        logging.error(f"Error plotting SARIMA data: {e}")
        st.error(f"Error plotting SARIMA data: {e}")

# LSTM Model: Data Preprocessing, Training, and Prediction
def preprocess_lstm_data(data, look_back=60):
    if len(data) < look_back:
        logging.error("Not enough data to preprocess for LSTM.")
        return None, None, None

    scaler = MinMaxScaler(feature_range=(0, 1))
    scaled_data = scaler.fit_transform(data.values.reshape(-1, 1))

    x, y = [], []
    for i in range(look_back, len(scaled_data)):
        x.append(scaled_data[i-look_back:i, 0])
        y.append(scaled_data[i, 0])

    x, y = np.array(x), np.array(y)
    x = np.reshape(x, (x.shape[0], x.shape[1], 1))

    return x, y, scaler

def build_lstm_model(input_shape):
    model = tf.keras.models.Sequential()
    model.add(tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(units=100, return_sequences=True, input_shape=input_shape)))
    model.add(tf.keras.layers.Dropout(0.3))
    model.add(tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(units=50, return_sequences=False)))
    model.add(tf.keras.layers.Dropout(0.2))
    model.add(tf.keras.layers.Dense(units=1))  # Output layer

    model.compile(optimizer='adam', loss='mean_squared_error')
    return model

def train_lstm_and_predict(data, months):
    x, y, scaler = preprocess_lstm_data(data)
    if x is None:
        return [], [], None

    model = build_lstm_model((x.shape[1], 1))

    # EarlyStopping callback to stop training if no improvement
    early_stopping = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)

    # Splitting into training and validation sets (80% train, 20% validation)
    train_size = int(len(x) * 0.8)
    x_train, x_val = x[:train_size], x[train_size:]
    y_train, y_val = y[:train_size], y[train_size:]

    # Train the model with validation data
    model.fit(x_train, y_train, epochs=30, batch_size=64, validation_data=(x_val, y_val), callbacks=[early_stopping], verbose=0)

    # Predict the stock price for the selected number of months (business days)
    total_days = months * 21  # Average of 21 business days per month
    future_data = data[-60:].values.reshape(-1, 1)
    future_scaled = scaler.transform(future_data)
    future_x = np.array([future_scaled])

    predictions = []
    for _ in range(total_days):
        predicted_price = model.predict(future_x)
        predictions.append(predicted_price[0, 0])
        # Append the predicted price to future_data and remove the first value
        future_scaled = np.append(future_scaled[1:], predicted_price, axis=0)
        future_x = np.array([future_scaled])

    # Inverse transform the predictions
    predictions = scaler.inverse_transform(np.array(predictions).reshape(-1, 1)).flatten()
    predicted_dates = pd.date_range(start=data.index[-1] + timedelta(days=1), periods=total_days, freq='B')

    return predictions, predicted_dates, scaler

# Function to plot LSTM prediction using Plotly
def plot_lstm_prediction(data, predicted_dates, predicted_prices):
    try:
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=data.index, y=data, mode='lines', name='Historical Data', line=dict(color='blue', width=2)))
        fig.add_trace(go.Scatter(x=predicted_dates, y=predicted_prices, mode='lines', name='Predicted Price (Next Period)', line=dict(color='orange', width=2)))
        fig.update_layout(
            title=dict(text='LSTM: Historical Data & Predicted Stock Prices', x=0.5, xanchor='center'),
            xaxis_title='Year',
            yaxis_title='Stock Price',
            legend=dict(font=dict(size=12)),
            template='plotly_white',
            hovermode='x unified'
        )
        st.plotly_chart(fig, use_container_width=True)  # Make the plot responsive
    except Exception as e:
        logging.error(f"Error plotting data: {e}")
        st.error(f"Error plotting data: {e}")

# Streamlit UI
st.set_page_config(page_title="Stock Price Prediction", page_icon="📈", layout="wide")

# Custom CSS for better styling
st.markdown(
    """
    <style>
    .stButton>button {
        background-color: #4CAF50;
        color: white;
        font-size: 16px;
        padding: 10px 24px;
        border-radius: 8px;
        border: none;
    }
    .stButton>button:hover {
        background-color: #45a049;
    }
    .stSelectbox>div>div>div>div {
        font-size: 16px;
    }
    .stDateInput>div>div>input {
        font-size: 16px;
    }
    .stNumberInput>div>div>input {
        font-size: 16px;
    }
    .stMarkdown h1 {
        color: #1E90FF;
        text-align: center;
    }
    .stMarkdown h2 {
        color: #87CEEB;
        text-align: center;
    }
    .stMarkdown h3 {
        color: #D35400;
    }
    .css-1d391kg {
        padding-top: 1rem;
        padding-bottom: 1rem;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

# Main title and description
st.markdown("<h1>SMART STOCK PREDICTOR</h1>", unsafe_allow_html=True)
st.markdown(
    """
    Welcome to the SMART STOCK PREDICTOR! Use our SARIMA-LSTM hybrid model for accurate stock predictions. Select a stock ticker and prediction method to begin.
    """
)
st.divider()

# Sidebar for stock ticker input and prediction settings
with st.sidebar:
    st.header("⚡ Configuration")
    st.markdown("Fine-tune your prediction settings here")

    # Ticker input with examples
    ticker = st.text_input(
        "Provide Stock Ticker:",  
        help="Examples: AAPL (Apple), MSFT (Microsoft), GOOGL (Google), TSLA (Tesla), AMZN (Amazon)"
    )

    # Prediction type selection
    prediction_type = st.selectbox(
        "Select Forecasting Model", 
        ["Short-Term (SARIMA)", "Long-Term (LSTM)"]
    )

    # Dynamic settings based on prediction type
    if prediction_type == "Short-Term (SARIMA)":
        start_date = (datetime.now() - timedelta(days=2*365)).date()  # 2 years of data
        end_date = datetime.now().date()
        stock_data_2y = fetch_stock_data(ticker, start_date, end_date)
        if stock_data_2y is not None and not stock_data_2y.empty:
            last_data_date = stock_data_2y.index[-1].date()
            min_predict_date = last_data_date + timedelta(days=1)  # Predict from the next day onward
            max_predict_date = last_data_date + timedelta(days=7)
            selected_date = st.date_input(
                "Select prediction date", 
                min_value=min_predict_date, 
                max_value=max_predict_date, 
                value=min_predict_date,
                help="Select a date within the next 7 business days."
            )
            days = (selected_date - last_data_date).days
            predict_button = st.button("Predict", help="Click to generate predictions.")
        else:
            st.error("No data available for the selected ticker.")
            selected_date = None
            days = 1
            predict_button = False
    elif prediction_type == "Long-Term (LSTM)":
        months = st.number_input(
            "Predict stock price for next (1-12 months)", 
            min_value=1, 
            max_value=12, 
            value=1,
            help="Select the number of months for long-term prediction."
        )
        predict_button = st.button("Predict", help="Click to generate predictions.")

# When the user clicks the button
if predict_button:
    if prediction_type == "Short-Term (SARIMA)":
        if stock_data_2y is not None and selected_date is not None:
            st.subheader("Short-Term Prediction (SARIMA)")
            st.write(f"Predicting stock price for *{selected_date}* using the SARIMA model.")
            
            with st.spinner("Fetching and processing data..."):
                stock_data_2y = impute_missing_data(stock_data_2y)
                order, seasonal_order = optimize_sarima_params(stock_data_2y)
                forecast, model_fit = train_sarima_and_predict(stock_data_2y, order, seasonal_order, days)
                
                if forecast is not None:
                    selected_dates = pd.date_range(start=stock_data_2y.index[-1] + timedelta(days=1), periods=days, freq='B')
                    st.success(f" Predicted Stock Price is *${forecast[-1]:.2f}*")
                    plot_sarima_prediction(stock_data_2y, forecast, selected_dates)
                else:
                    st.error("❌ Failed to generate SARIMA prediction. Please check the logs for details.")
        else:
            st.warning("⚠ No data available for prediction.")
    elif prediction_type == "Long-Term (LSTM)":
        st.subheader("Long-Term Prediction (LSTM)")
        st.write(f"Predicting stock price for the next *{months} month(s)* using the LSTM model.")
        
        with st.spinner("Fetching and processing data..."):
            start_date = (datetime.now() - timedelta(days=5*365)).date()  # 5 years of data
            end_date = datetime.now().date()
            stock_data_5y = fetch_stock_data(ticker, start_date, end_date)

            if stock_data_5y is not None:
                
                stock_data_5y = impute_missing_data(stock_data_5y)
                st.write("🔄 Imputed missing data.")

                # Check if there is enough data for LSTM
                if len(stock_data_5y) < 60:
                    st.error("❌ Not enough historical data for LSTM prediction. At least 60 days of data are required.")
                else:
                    predictions, predicted_dates, scaler = train_lstm_and_predict(stock_data_5y, months)
                    
                    if len(predictions) > 0:
                        st.success(f" Predicted Stock Price after *{months} month(s)* is *${predictions[-1]:.2f}*")
                        plot_lstm_prediction(stock_data_5y, predicted_dates, predictions)
                    else:
                        st.error("❌ Failed to generate predictions. Please check the logs for errors.")
            else:
                st.warning("⚠ No data available for prediction.")