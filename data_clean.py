import jsonlines
from datetime import datetime, timedelta
import yfinance as yf

# will prob have date functions
def get_stock_prices(ticker, dt):
    next_date = dt + timedelta(days=2)  # include buffer for weekend/holiday

    stock = yf.Ticker(ticker)
    data = stock.history(start=dt.strftime("%Y-%m-%d"), end=next_date.strftime("%Y-%m-%d"))

    close_prices = data['Close'].tolist()
    dates = data.index.strftime('%Y-%m-%d').tolist()

    return list(zip(dates, close_prices))



with jsonlines.open("earnings_transcript_short.jsonl") as reader:
    for obj in reader:
        ticker = obj.get("ticker", "")
        date = obj.get("date", "")
        cleaned = date.replace("p.m.", "PM").replace("a.m.", "AM").replace("ET", "").strip()
        # print(cleaned)
        dt = datetime.strptime(cleaned, "%b %d, %Y, %I:%M %p")
        prices_on_earnings_day = get_stock_prices(ticker, dt)
        print(f"{ticker} | {date} => {prices_on_earnings_day}")