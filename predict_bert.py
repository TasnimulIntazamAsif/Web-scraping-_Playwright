import sys
from pathlib import Path

import pandas as pd
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification



# --- Make project root importable
BASE_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(BASE_DIR))

from utils.preprocess import clean_text

MODEL_PATH = BASE_DIR / "bert_model" / "saved_model"
REAL_NEWS_PATH = BASE_DIR / "data" / "real_news.csv"
OUT_PATH = BASE_DIR / "data" / "predicted_news.csv"

# Load tokenizer and model
tokenizer = AutoTokenizer.from_pretrained(str(MODEL_PATH))
model = AutoModelForSequenceClassification.from_pretrained(str(MODEL_PATH))
model.eval()

df = pd.read_csv(REAL_NEWS_PATH, encoding="utf-8")
df["text"] = (df["headline"] + " " + df["content"]).apply(clean_text)

predictions = []
for text in df["text"]:
    inputs = tokenizer(text, return_tensors="pt", truncation=True, padding=True, max_length=256)
    with torch.no_grad():
        outputs = model(**inputs)
        pred = torch.argmax(outputs.logits, dim=1).item()
        predictions.append(pred)

df["predicted_label"] = predictions
df["prediction"] = df["predicted_label"].map({1: "Real News", 0: "Fake News"})

df.to_csv(OUT_PATH, index=False, encoding="utf-8")
print(f"✅ BERT prediction saved to {OUT_PATH}")