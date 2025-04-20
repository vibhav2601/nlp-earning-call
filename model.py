# model.py
import torch
import torch.nn as nn
from transformers import AutoModel, AutoTokenizer

class HierarchicalFinBERTModel(nn.Module):
    def __init__(self, model_name='yiyanghkust/finbert-tone', num_layers=3, num_heads=6):
        super().__init__()
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.segment_encoder = AutoModel.from_pretrained(model_name)

        self.doc_encoder = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(d_model=768, nhead=num_heads),
            num_layers=num_layers
        )
        self.classifier = nn.Linear(768 * 2, 3)  # 3-class classification: -1, 0, +1

    def chunk_and_embed(self, transcript):
        tokens = self.tokenizer(transcript, return_tensors="pt", truncation=False, padding=False)
        input_ids = tokens["input_ids"][0]

        chunk_size = 510
        cls_embeddings = []

        for i in range(0, len(input_ids), chunk_size):
            chunk_ids = input_ids[i:i + chunk_size]
            chunk_ids = torch.cat([
                torch.tensor([self.tokenizer.cls_token_id]),
                chunk_ids,
                torch.tensor([self.tokenizer.sep_token_id])
            ])
            chunk = {
                "input_ids": chunk_ids.unsqueeze(0).to(self.classifier.weight.device),
                "attention_mask": torch.ones_like(chunk_ids).unsqueeze(0).to(self.classifier.weight.device)
            }
            with torch.no_grad():
                output = self.segment_encoder(**chunk)
            cls_emb = output.last_hidden_state[0, 0]
            cls_embeddings.append(cls_emb)

        return torch.stack(cls_embeddings)  # [n_chunks, 768]

    def forward(self, transcript):
        segment_embeddings = self.chunk_and_embed(transcript)  # [n, 768]
        encoded = self.doc_encoder(segment_embeddings.unsqueeze(1)).squeeze(1)  # [n, 768]

        first = encoded[0]
        max_pool = torch.max(encoded, dim=0).values
        doc_vector = torch.cat([first, max_pool], dim=-1)  # [1536]

        return self.classifier(doc_vector)  # logits (no activation here)