import torch
import torch.nn as nn
from transformers import AutoModel, AutoTokenizer

class HierarchicalFinBERTModel(nn.Module):
    def __init__(self, model_name='yiyanghkust/finbert-tone', num_layers=3, num_heads=6, device=None):
        super().__init__()
        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.segment_encoder = AutoModel.from_pretrained(model_name).to(self.device)
        self.hidden_size = self.segment_encoder.config.hidden_size

        self.doc_encoder = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(d_model=self.hidden_size, nhead=num_heads),
            num_layers=num_layers
        ).to(self.device)

        self.attn_pool = nn.Sequential(
            nn.Linear(self.hidden_size, 1),
            nn.Softmax(dim=0)
        )
        self.norm = nn.LayerNorm(self.hidden_size * 2)
        self.dropout = nn.Dropout(0.3)
        self.classifier = nn.Linear(self.hidden_size * 2, 2).to(self.device)

    def chunk_and_embed(self, transcript, max_chunks=150):
        tokens = self.tokenizer(transcript, return_tensors="pt", truncation=False, padding=False, add_special_tokens=False)
        input_ids = tokens["input_ids"][0]
        chunk_size = 510
        cls_embeddings = []

        for i in range(0, len(input_ids), chunk_size):
            if len(cls_embeddings) >= max_chunks:
                break
            chunk_ids = input_ids[i:i + chunk_size]
            chunk_text = self.tokenizer.decode(chunk_ids)
            chunk = self.tokenizer(chunk_text, return_tensors="pt", padding="max_length", truncation=True, max_length=512)
            chunk = {k: v.to(self.device) for k, v in chunk.items()}
            output = self.segment_encoder(**chunk)
            cls_emb = output.last_hidden_state[:, 0, :]
            cls_embeddings.append(cls_emb.squeeze(0))

        return torch.stack(cls_embeddings).to(self.device)

    def forward(self, transcript):
        segment_embeddings = self.chunk_and_embed(transcript)
        encoded = self.doc_encoder(segment_embeddings.unsqueeze(1)).squeeze(1)
        attn_weights = self.attn_pool(encoded)
        attn_vector = torch.sum(attn_weights * encoded, dim=0)
        doc_vector = torch.cat([encoded[0], attn_vector], dim=-1)
        doc_vector = self.dropout(self.norm(doc_vector))
        return self.classifier(doc_vector)