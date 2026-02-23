# train_bert.py
import sys
from pathlib import Path

import pandas as pd
from sklearn.model_selection import train_test_split
from datasets import Dataset
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    Trainer,
    TrainingArguments
)

BASE_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(BASE_DIR))

from utils.preprocess import clean_text
from config import FAKE_NEWS_FILE

MODEL_NAME = "bert-base-multilingual-cased"

DATA_DIR = BASE_DIR / "data"
LOGS_DIR = BASE_DIR / "logs"
SAVED_MODEL_DIR = BASE_DIR / "bert_model" / "saved_model"

REAL_NEWS_PATH = DATA_DIR / "real_news.csv"
FAKE_NEWS_PATH = BASE_DIR / FAKE_NEWS_FILE

DATA_DIR.mkdir(exist_ok=True)
LOGS_DIR.mkdir(exist_ok=True)
SAVED_MODEL_DIR.mkdir(parents=True, exist_ok=True)

# Auto-find fake csv if missing
if not FAKE_NEWS_PATH.exists():
    candidates = sorted(DATA_DIR.glob("*fake*.csv")) + sorted(DATA_DIR.glob("*Fake*.csv"))
    if candidates:
        FAKE_NEWS_PATH = candidates[0]

if not FAKE_NEWS_PATH.exists():
    raise FileNotFoundError(
        f"Fake news CSV not found.\n"
        f"Expected: {BASE_DIR / FAKE_NEWS_FILE}\n"
        f"Also searched in: {DATA_DIR}\n"
        f"Fix: Put Fake_news.csv inside {DATA_DIR} or update FAKE_NEWS_FILE in config.py"
    )

# Read fake news safely
try:
    fake_df = pd.read_csv(FAKE_NEWS_PATH, encoding="utf-8")
except UnicodeDecodeError:
    fake_df = pd.read_csv(FAKE_NEWS_PATH, encoding="ISO-8859-1")

# Read real news
real_df = pd.read_csv(REAL_NEWS_PATH, encoding="utf-8")

real_df["label"] = 1
fake_df["label"] = 0

df = pd.concat(
    [
        real_df[["headline", "content", "label"]],
        fake_df[["headline", "content", "label"]],
    ],
    ignore_index=True
)

df["text"] = (df["headline"] + " " + df["content"]).apply(clean_text)
df.to_csv(DATA_DIR / "merged_news.csv", index=False, encoding="utf-8")

train_df, test_df = train_test_split(df[["text", "label"]], test_size=0.2, random_state=42)

train_ds = Dataset.from_pandas(train_df.reset_index(drop=True))
test_ds = Dataset.from_pandas(test_df.reset_index(drop=True))

tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

def tokenize(batch):
    return tokenizer(batch["text"], padding="max_length", truncation=True, max_length=256)

train_ds = train_ds.map(tokenize, batched=True)
test_ds = test_ds.map(tokenize, batched=True)

train_ds.set_format("torch", columns=["input_ids", "attention_mask", "label"])
test_ds.set_format("torch", columns=["input_ids", "attention_mask", "label"])

model = AutoModelForSequenceClassification.from_pretrained(MODEL_NAME, num_labels=2)

common_args = dict(
    output_dir=str(SAVED_MODEL_DIR),
    save_strategy="epoch",
    learning_rate=2e-5,
    per_device_train_batch_size=8,
    per_device_eval_batch_size=8,
    num_train_epochs=3,
    weight_decay=0.01,
    logging_dir=str(LOGS_DIR),
    load_best_model_at_end=True,
)

# transformers compatibility: eval_strategy vs evaluation_strategy
try:
    training_args = TrainingArguments(eval_strategy="epoch", **common_args)
except TypeError:
    training_args = TrainingArguments(evaluation_strategy="epoch", **common_args)

# ✅ Trainer init compatibility (tokenizer argument removed in newer transformers)
trainer_kwargs = dict(
    model=model,
    args=training_args,
    train_dataset=train_ds,
    eval_dataset=test_ds,
)

try:
    trainer = Trainer(**trainer_kwargs, processing_class=tokenizer)
except TypeError:
    trainer = Trainer(**trainer_kwargs, tokenizer=tokenizer)

trainer.train()

trainer.save_model(str(SAVED_MODEL_DIR))
tokenizer.save_pretrained(str(SAVED_MODEL_DIR))

print("✅ BERT model trained and saved")
print(f"✅ Fake news file used: {FAKE_NEWS_PATH}")