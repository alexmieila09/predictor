import pandas as pd
import yfinance as yf
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt
import mysql.connector


def predict_cryptocurrency_price(symbol):
    #CONFIG
    # Open - the first traded price
    # High - the highest traded price 
    # Low - the lowest traded price 
    # Close - the final traded price
    
    historicalPrice = pd.Timedelta(hours = 24)     # Config historicalPriceInterval
    predictingPeriodConfig = 2
    predictingPeriod = pd.Timedelta (hours = predictingPeriodConfig)     # Config predictingPeriod

    start_date = pd.Timestamp.today() - historicalPrice  # Download historical price data for the last <historicalPrice> hours
    end_date = pd.Timestamp.today()
    df = yf.download(symbol, start=start_date, end=end_date)

    # Preprocess data
    X = df[['Open', 'High', 'Low']]
    y = df['Close']

    # Train model
    model = LinearRegression()
    model.fit(X, y)

    db = mysql.connector.connect(
        host = "slyperformance.ro",
        user = "slyperf1_tradingbot",
        password ="tradingbot",
        database = "slyperf1_tradingbot"
    )

    print('Successfully connected to: ', db)
    # Make predictions
    last_row = df.tail(1)
    X_pred = last_row[['Open', 'High', 'Low']]
    # predict the next hour's price
    date_pred = last_row.index[0] + predictingPeriod # Predicting period price data based on <historicalpriceInterval>   
    # predict the next hour's price
    y_pred = model.predict(X_pred)
    print()
    print('Df.index Additional informations data: ', df.index)
    print()
    print('X = ', X_pred)
    print('Open:', X_pred['Open'].values[0])
    print('High:', X_pred['High'].values[0])
    print('Low:', X_pred['Low'].values[0])
    print()
    print('Y = CLOSE')      
    print()
    print('Predicted price for', symbol, 'For', predictingPeriodConfig, ' Hours', ': CLOSE @ ', y_pred[0])
    
    openPrice = X_pred['Open'].values[0]
    lowPrice = X_pred['Low'].values[0]
    highPrice = X_pred['High'].values[0]

    mycursor = db.cursor()
    sql = "INSERT INTO main_predictor (predicted_close_price, currency, open, low, high, interval_time, prediction_hour) VALUES (%s, %s, %s, %s, %s, %s, %s)"
    val = (y_pred[0], symbol, openPrice, lowPrice, highPrice, predictingPeriodConfig, date_pred) 
    mycursor.execute(sql, val)
    db.commit()

    print('Starting date:', start_date)
    print('Ending date: ', end_date)
    print('Prediction Date:', date_pred)

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

while True:
    print()
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
print("Exiting the script.")
