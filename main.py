# main.py
import os
import sys
import subprocess
from pathlib import Path

from scraper import scrape_news
from config import OUTPUT_FILE

ROOT = Path(__file__).resolve().parent

def ensure_directories():
    (ROOT / "data").mkdir(exist_ok=True)
    (ROOT / "logs").mkdir(exist_ok=True)
    (ROOT / "bert_model" / "saved_model").mkdir(parents=True, exist_ok=True)
    (ROOT / "utils").mkdir(exist_ok=True)

if __name__ == "__main__":
    ensure_directories()

    print("🚀 Starting News Scraper...")
    df = scrape_news()

    if df.empty:
        print("❌ No data scraped.")
        raise SystemExit(1)

    out_path = ROOT / OUTPUT_FILE
    df.to_csv(out_path, index=False, encoding="utf-8")
    print(f"✅ Scraped news saved to {out_path}")

    print("\n📊 Training BERT Fake News Detection Model...")

    subprocess.run([sys.executable, str(ROOT / "train_bert.py")], check=True)

    #subprocess.run([sys.executable, str(ROOT / "bert_model" / "train_bert.py")], check=True)

    print("\n🔍 Predicting Fake / Real News...")
    #subprocess.run([sys.executable, str(ROOT / "bert_model" / "predict_bert.py")], check=True)
    subprocess.run([sys.executable, str(ROOT / "predict_bert.py")], check=True)