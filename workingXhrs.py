import pandas as pd
import yfinance as yf
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt

def predict_cryptocurrency_price(symbol):
    # Download historical price data for the last 5 hours
    start_date = pd.Timestamp.today() - pd.Timedelta(hours=2)
    end_date = pd.Timestamp.today()
    df = yf.download(symbol, start=start_date, end=end_date)

    # Preprocess data
    X = df[['Open', 'High', 'Low']]
    y = df['Close']

    # Train model
    model = LinearRegression()
    model.fit(X, y)

    # Make predictions
    last_row = df.tail(1)
    X_pred = last_row[['Open', 'High', 'Low']]
    date_pred = last_row.index[0] + pd.Timedelta(hours=1)  # predict the next hour's price
    y_pred = model.predict(X_pred)
    print('Predicted price for', symbol, 'at', date_pred.strftime('%Y-%m-%d %H:%M:%S'), ':', y_pred[0])

    # Plotting
    plt.figure(figsize=(10, 6))
    plt.plot(df.index, df['Close'], label='Actual Prices', marker='o')
    plt.plot(date_pred, y_pred, 'ro', label='Predicted Price')

    plt.title(f'{symbol} Cryptocurrency Prices and Prediction')
    plt.xlabel('Date')
    plt.ylabel('Closing Price (USD)')
    plt.legend()
    plt.grid(True)
    #plt.show()

# Cryptocurrency menu
cryptocurrencies = {'EGLD': 'Elrond', 'BTC': 'Bitcoin', "ETH": "Etherum"}  # Add more cryptocurrencies as needed

print('Select a cryptocurrency:')
for i, (symbol, name) in enumerate(cryptocurrencies.items(), 1):
    print(f"{i}. {name} ({symbol})")

user_choice = int(input('Enter the number corresponding to the cryptocurrency: '))
selected_symbol = list(cryptocurrencies.keys())[user_choice - 1]

# Update symbol and make prediction
symbol = f"{selected_symbol}-USD"
predict_cryptocurrency_price(symbol)
