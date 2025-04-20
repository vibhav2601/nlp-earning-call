
from transformers import AutoTokenizer, AutoModel
import torch

tokenizer = AutoTokenizer.from_pretrained("yiyanghkust/finbert-tone")
model = AutoModel.from_pretrained("yiyanghkust/finbert-tone")

def get_finbert_embedding(transcript):
    # Tokenize without truncation
    tokens = tokenizer(transcript, return_tensors="pt", truncation=False, padding=False)
    input_ids = tokens["input_ids"][0]

    # Split into 512-token chunks
    chunk_size = 510
    cls_embeddings = []

    for i in range(0, len(input_ids), chunk_size):
        chunk_ids = input_ids[i:i+chunk_size]

        # Add special tokens [CLS] and [SEP]
        chunk_ids = torch.cat([
            torch.tensor([tokenizer.cls_token_id]),
            chunk_ids,
            torch.tensor([tokenizer.sep_token_id])
        ])

        # Prepare input for model
        chunk = {
            "input_ids": chunk_ids.unsqueeze(0),
            "attention_mask": torch.ones_like(chunk_ids).unsqueeze(0)
        }

        # Forward pass
        with torch.no_grad():
            output = model(**chunk)
        
        # Take [CLS] embedding
        cls_emb = output.last_hidden_state[0, 0]  # shape: (768,)
        cls_embeddings.append(cls_emb)

    # Average all CLS embeddings
    final_embedding = torch.stack(cls_embeddings).mean(dim=0)
    return final_embedding  # shape: (768,)
