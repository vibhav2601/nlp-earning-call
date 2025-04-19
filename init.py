

git remote set-url origin https://vibhav2601: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@huggingface.co/jlh-ibm/earnings_call
$: git pull origin
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
GIT_LFS_SKIP_SMUDGE=1 git clone https://vibhav2601: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@huggingface.co/datasets/jlh-ibm/earnings_call

git filter-repo --replace-text <(echo "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==REMOVED_TOKEN")


from datasets import load_dataset
from transformers import pipeline
import pandas as pd

# # Load dataset
dataset = load_dataset("./earnings_call")

transcripts_df = dataset['train'].to_pandas()
# print(transcripts_df)

sample_texts = transcripts_df['text'].dropna().head(3).tolist()


sentiment_pipe = pipeline(
    "sentiment-analysis",
    model="cardiffnlp/twitter-roberta-base-sentiment-latest",
    tokenizer="cardiffnlp/twitter-roberta-base-sentiment-latest"
)

ner_pipe = pipeline("ner", grouped_entities=True)
aspect_keywords = ["growth", "revenue", "profit", "loss", "market", "guidance"]

# Run NLP Tasks
for idx, text in enumerate(sample_texts):
    print(f"\n--- Transcript {idx+1} ---")
    
    # Sentiment Analysis (full text)
    sentiment = sentiment_pipe(text)
    print("Sentiment:", sentiment)

    # Aspect-Based Analysis
    print("Aspects Mentioned:", [kw for kw in aspect_keywords if kw in text.lower()])

    # Named Entity Recognition
    entities = ner_pipe(text)
    named_entities = [(ent['word'], ent['entity_group']) for ent in entities]
    print("Named Entities:", named_entities)
