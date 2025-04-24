import jsonlines
from datetime import datetime
from stock_price import get_stock_prices  # Assumes you already have this file
import os

input_path = "data/earning_call.jsonl"
output_path = "data/labeled_earning_call.jsonl"

os.makedirs("data", exist_ok=True)
count = 0

with jsonlines.open(input_path) as reader, jsonlines.open(output_path, mode='w') as writer:
    for obj in reader:
        ticker = obj.get("ticker", "")
        date_str = obj.get("date", "")
        transcript = obj.get("transcript", "")

        if isinstance(date_str, list):
            date_str = date_str[-1]
        try:
            cleaned = date_str.replace(",", "").replace(".", "").replace("ET", "").strip()
            cleaned = cleaned.replace("p.m.", "PM").replace("a.m.", "AM")
            cleaned = cleaned.replace("April", "Apr").replace("March", "Mar")
            dt = datetime.strptime(cleaned, "%b %d %Y %I:%M %p")
            prices = get_stock_prices(ticker, dt)
            if not prices:
                continue
        except Exception as e:
            continue

        change = (prices[-1][1] - prices[0][1]) / prices[0][1]
        label = 1 if change < 0 else 0  # 1 = stock down, 0 = stock up

        writer.write({
            "transcript": transcript,
            "label": label
        })
        count += 1

print(f"Wrote {count} labeled examples to {output_path}")