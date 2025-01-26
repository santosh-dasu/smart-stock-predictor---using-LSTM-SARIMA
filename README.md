# Stock Price Prediction with SARIMA and LSTM Models

This repository hosts a robust and interactive stock price prediction system that leverages advanced time series forecasting techniques. The system combines SARIMA (Seasonal AutoRegressive Integrated Moving Average) for short-term predictions and LSTM (Long Short-Term Memory) for long-term forecasts. Built with Streamlit, the application provides an intuitive interface for users to analyze and predict stock prices using historical data from Yahoo Finance.


### 1. Short-Term Prediction (SARIMA)
- **Purpose**: Predicts stock prices for the next 1-7 business days.
- **Key Features**:
  - Automatically optimizes SARIMA parameters using `auto_arima`.
  - Handles missing data with forward and backward imputation.
  - Evaluates model performance using RMSE, MAE, and MAPE.
- **Use Case**: Ideal for traders and investors focusing on short-term price movements.

### 2. Long-Term Prediction (LSTM)
- **Purpose**: Predicts stock prices for the next 1-12 months.
- **Key Features**:
  - Utilizes a Bidirectional LSTM architecture with dropout layers to prevent overfitting.
  - Incorporates technical indicators such as Moving Averages, RSI, and MACD for enhanced accuracy.
  - Evaluates predictions using RMSE, MAE, and MAPE.
- **Use Case**: Suitable for long-term investors and portfolio managers.

### 3. Interactive Web Application
Built with Streamlit, the app provides a user-friendly interface for:
- Inputting stock tickers (e.g., AAPL, TSLA, MSFT).
- Selecting prediction models (SARIMA or LSTM).
- Configuring prediction settings (e.g., prediction date or number of months).
- Visualizing historical data and predictions using Plotly.
- Responsive and visually appealing design.


## Installation

### To set up the project locally, follow these steps:

1. **Clone the Repository**:
   ```bash
   git clone https://github.com/your-username/stock-price-prediction.git
   cd stock-price-prediction
   ```

2. **Install Dependencies**:
   Ensure you have Python 3.8 or higher installed. Install the required packages using:
   ```bash
   pip install -r requirements.txt
   ```

3. **Run the Streamlit App**:
   Launch the interactive web application:
   ```bash
   streamlit run app.py
   ```

4. **Run Individual Models**:
   - For SARIMA:
     ```bash
     python sarima_model.py
     ```
   - For LSTM:
     ```bash
     python lstm_model.py
     ```


## Usage

### 1. Streamlit App
- Open the app in your browser after running `streamlit run app.py`.
- Enter a stock ticker (e.g., AAPL, MSFT, TSLA) in the sidebar.
- Select the prediction model:
  - **SARIMA** for short-term predictions (1-7 business days).
  - **LSTM** for long-term predictions (1-12 months).
- Configure the prediction settings (e.g., prediction date or number of months).
- Click **Predict** to generate and visualize the stock price forecast.

### 2. Command-Line Interface
- Run `sarima_model.py` or `lstm_model.py` directly for standalone predictions.
- Follow the prompts to input the stock ticker and view predictions.


## File Structure

```plaintext
stock-price-prediction/
├── app.py                  # Streamlit app for stock price prediction
├── lstm_model.py           # LSTM model implementation
├── sarima_model.py         # SARIMA model implementation
├── README.md               # Project documentation
├── requirements.txt        # List of dependencies
└── data/                   # Directory for storing stock data (optional)
```


## Dependencies

- **Python** 3.8+
- **Streamlit** (for the web app)
- **yfinance** (for fetching stock data)
- **pandas** and **numpy** (for data manipulation)
- **tensorflow** (for LSTM model)
- **scikit-learn** (for evaluation metrics)
- **statsmodels** and **pmdarima** (for SARIMA model)
- **plotly** (for interactive visualizations)


## Contributing

We welcome contributions to improve the project! If you'd like to contribute, follow these steps:

1. Fork the repository.
2. Create a new branch for your feature or bugfix.
3. Commit your changes and push to the branch.
4. Submit a pull request with a detailed description of your changes.


## License

This project is licensed under the MIT License. See the LICENSE file for details.


## Acknowledgments

- **Yahoo Finance** for providing historical stock data.
- **Streamlit** for enabling the creation of interactive web apps.
- **TensorFlow** and **statsmodels** for providing robust machine learning and statistical modeling tools.
