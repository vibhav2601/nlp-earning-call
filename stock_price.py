from datetime import datetime, timedelta
import yfinance as yf
def get_stock_prices(ticker, dt):
    #account for close markets
    if dt.hour >16:
        first_day = dt
        second_day = dt + timedelta(days=3)
    else:
        first_day = dt - timedelta(days= 3)
        second_day = dt + timedelta(days=1)


    stock = yf.Ticker(ticker)
    data = stock.history(start=first_day.strftime("%Y-%m-%d"), end=second_day.strftime("%Y-%m-%d"))

    close_prices = data['Close'].tolist()
    if len(data) == 0:
      print('skipping stock')
      return None
    dates = data.index.strftime('%Y-%m-%d').tolist()

    return list(zip(dates, close_prices))
