import os
from scraper import scrape_news
from config import OUTPUT_FILE

def ensure_directories():
    os.makedirs("data", exist_ok=True)
    os.makedirs("logs", exist_ok=True)

if __name__ == "__main__":

    ensure_directories()

    print("Starting News Scraper...")
    df = scrape_news()

    if df.empty:
        print("❌ No data scraped. Check selectors.")
    else:
        df.to_csv(OUTPUT_FILE, index=False, encoding="utf-8")
        print(f"✅ Scraping completed. Data saved to {OUTPUT_FILE}")