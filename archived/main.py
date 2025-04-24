import jsonlines
from datetime import datetime, timedelta
import yfinance as yf
from stock_price import get_stock_prices 
from finbert_embeddings import get_finbert_embedding





def main():
    with jsonlines.open("data/earning_transcript_single.jsonl") as reader:
        for obj in reader:
            transcript = obj.get("transcript", "")
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
            encoding = get_finbert_embedding(transcript= transcript)
            print(encoding)



# finbert model - encodings
# transformers pass
# loss
# passback

main()