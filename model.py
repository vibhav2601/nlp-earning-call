import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from tqdm import tqdm
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from transformers import AutoModel, AutoTokenizer
from dataset_utils import load_gold_and_weak_data
from focal_loss import FocalLoss

class FinBERTToneEmbeddingClassifier(nn.Module):
    def __init__(self, model_name='ProsusAI/finbert'):
        super().__init__()
        self.bert = AutoModel.from_pretrained(model_name)
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.dropout = nn.Dropout(0.3)
        self.classifier = nn.Linear(self.bert.config.hidden_size, 1)

    def forward(self, text_batch):
        device = self.classifier.weight.device
        inputs = self.tokenizer(text_batch, return_tensors="pt", padding=True, truncation=True, max_length=512).to(device)
        outputs = self.bert(**inputs)
        cls_embedding = outputs.last_hidden_state[:, 0, :]  # CLS token
        x = self.dropout(cls_embedding)
        return self.classifier(x).squeeze(-1)

def train_model(model, train_data, val_data, batch_size=4, num_epochs=5, learning_rate=3e-5):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)
    criterion = FocalLoss(gamma=2.0)
    optimizer = optim.AdamW(model.parameters(), lr=learning_rate)

    for epoch in range(num_epochs):
        model.train()
        total_loss = 0

        for i in tqdm(range(0, len(train_data), batch_size), desc=f"Epoch {epoch+1}"):
            batch = train_data[i:i + batch_size]
            texts = [item['transcript'] for item in batch if item.get('label') is not None]
            labels = [item['label'] for item in batch if item.get('label') is not None]

            if not texts:
                continue

            targets = torch.tensor(labels, dtype=torch.float).to(device)
            optimizer.zero_grad()
            outputs = model(texts)
            loss = criterion(outputs, targets)
            loss.backward()
            total_loss += loss.item()
            optimizer.step()

        print(f"Epoch {epoch + 1}/{num_epochs}, Loss: {total_loss / len(train_data):.4f}")

        model.eval()
        all_preds = []
        all_labels = []
        with torch.no_grad():
          for item in val_data:
              if item.get('label') is None:
                  continue
              text = item['transcript']
              label = item['label']
              output = model([text])
              prob = torch.sigmoid(output).item()
              pred = int(prob > 0.5)

              # ✅ Print predicted vs true label
              print(f"Pred: {pred} | Prob: {prob:.4f} | True: {label}")

              all_preds.append(pred)
              all_labels.append(label)


        accuracy = accuracy_score(all_labels, all_preds)
        precision = precision_score(all_labels, all_preds)
        recall = recall_score(all_labels, all_preds)
        f1 = f1_score(all_labels, all_preds)
        print(f"Validation Metrics: Acc={accuracy:.4f}, Precision={precision:.4f}, Recall={recall:.4f}, F1={f1:.4f}")

    return model

if __name__ == "__main__":
    print("Loading data...")
    data = load_gold_and_weak_data("data/labeled_earning_call.jsonl", "data/weak_labeled_10k.tsv")
    print(f"Loaded {len(data)} total samples")

    np.random.shuffle(data)
    split_idx = int(0.8 * len(data))
    train_data = data[:split_idx]
    val_data = data[split_idx:]

    model = FinBERTToneEmbeddingClassifier()
    trained_model = train_model(model, train_data, val_data, batch_size=4, num_epochs=5, learning_rate=3e-5)

    torch.save(trained_model.state_dict(), "binary_fingertone_embedding_model.pth")