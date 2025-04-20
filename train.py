# train.py
import torch
import torch.nn as nn
import torch.optim as optim
from model import HierarchicalFinBERTModel
import jsonlines
from datetime import datetime
from stock_price import get_stock_prices
import numpy as np
from tqdm import tqdm

def load_data(jsonl_path):
    data = []
    with jsonlines.open(jsonl_path) as reader:
        for obj in reader:
            ticker = obj.get("ticker", "")
            date = obj.get("date", "")
            transcript = obj.get("transcript", "")

            cleaned = date.replace("p.m.", "PM").replace("a.m.", "AM").replace("ET", "").strip()
            dt = datetime.strptime(cleaned, "%b %d, %Y, %I:%M %p")
            prices_compiled = get_stock_prices(ticker, dt)
            change = (prices_compiled[-1][1] - prices_compiled[0][1])/(prices_compiled[0][1])

            if change > 0.025:
                label = 2  # positive
            elif change < -0.025:
                label = 0  # negative
            else:
                label = 1  # neutral

            data.append({
                'transcript': transcript,
                'label': label
            })
    return data

def train_model(model, train_data, val_data, batch_size=1, num_epochs=5, learning_rate=1e-5):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.AdamW(model.parameters(), lr=learning_rate)

    for epoch in range(num_epochs):
        model.train()
        total_loss = 0

        for i in tqdm(range(0, len(train_data), batch_size)):
            batch_data = train_data[i:i + batch_size]
            optimizer.zero_grad()

            for item in batch_data:
                transcript = item['transcript']
                label = torch.tensor([item['label']], dtype=torch.long).to(device)
                output = model(transcript).unsqueeze(0)  # logits
                loss = criterion(output, label)
                loss.backward()
                total_loss += loss.item()

            optimizer.step()

        avg_loss = total_loss / len(train_data)
        print(f"Epoch {epoch + 1}/{num_epochs}, Loss: {avg_loss:.4f}")

        # Validation
        model.eval()
        correct = 0
        total = 0
        with torch.no_grad():
            for item in val_data:
                transcript = item['transcript']
                label = item['label']
                logits = model(transcript)
                pred_class = torch.argmax(logits).item()
                correct += (pred_class == label)
                total += 1

        accuracy = correct / total
        print(f"Validation Accuracy: {accuracy:.4f}")

    return model

if __name__ == "__main__":
    data = load_data("data/earnings_transcript_short.jsonl")
    np.random.shuffle(data)
    split_idx = int(0.8 * len(data))
    train_data = data[:split_idx]
    val_data = data[split_idx:]

    model = HierarchicalFinBERTModel()

    trained_model = train_model(
        model,
        train_data,
        val_data,
        batch_size=1,
        num_epochs=5,
        learning_rate=1e-5
    )

    torch.save(trained_model.state_dict(), 'earnings_call_model.pth')
