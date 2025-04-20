import jsonlines
from datetime import datetime, timedelta
import yfinance as yf
from stock_price import get_stock_prices 
from transformers import AutoTokenizer, AutoModel




with jsonlines.open("data/earnings_transcript_short.jsonl") as reader:
    for obj in reader:
        ticker = obj.get("ticker", "")
        date = obj.get("date", "")
        cleaned = date.replace("p.m.", "PM").replace("a.m.", "AM").replace("ET", "").strip()
        dt = datetime.strptime(cleaned, "%b %d, %Y, %I:%M %p")
        prices_compiled = get_stock_prices(ticker, dt)
        change = (prices_compiled[-1][1] - prices_compiled[0][1])/(prices_compiled[0][1])
        if change > 0.025:
            truth = 1
        elif change < -0.025:
            truth = -1
        else:
            truth = 0
        print(f"{ticker} | {date} => {prices_compiled, truth}")

        # finbert encodings
        



# finbert model - encodings
# transformers pass
# loss
# passback