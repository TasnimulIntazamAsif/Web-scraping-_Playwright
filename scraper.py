# scraper.py
from playwright.sync_api import sync_playwright
import pandas as pd
import logging
import os
from urllib.parse import urljoin
from config import BASE_URL, HEADLESS, MAX_ARTICLES

os.makedirs("logs", exist_ok=True)

logging.basicConfig(
    filename="logs/scraper.log",
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s"
)

def detect_content_type(url, page):
    url_lower = url.lower()
    if "video" in url_lower:
        return "Video"
    elif "live" in url_lower:
        return "Live Update"
    elif "sport" in url_lower:
        return "Sports"
    elif "business" in url_lower:
        return "Business"
    elif "world" in url_lower:
        return "International"
    elif "bangladesh" in url_lower:
        return "Regional (Bangladesh)"
    else:
        if page.locator("video").count() > 0:
            return "Video"
        return "News Article"

def scrape_news():
    articles_data = []
    with sync_playwright() as p:
        browser = p.chromium.launch(headless=HEADLESS)
        page = browser.new_page()
        page.goto(BASE_URL)
        page.wait_for_load_state("networkidle")

        links = page.locator("a")
        article_urls = []

        for i in range(links.count()):
            href = links.nth(i).get_attribute("href")
            if href:
                full_url = urljoin(BASE_URL, href)
                if BASE_URL.split("/")[2] in full_url:
                    article_urls.append(full_url)

        article_urls = list(set(article_urls))

        # Safe slicing: if MAX_ARTICLES is None -> keep all
        if MAX_ARTICLES is not None:
            article_urls = article_urls[:MAX_ARTICLES]

        article_id = 1

        for link in article_urls:
            try:
                page.goto(link)
                page.wait_for_load_state("networkidle")

                try:
                    headline = page.locator("h1").first.inner_text().strip()
                except Exception:
                    continue

                content = ""
                paragraphs = page.locator("p")
                for i in range(paragraphs.count()):
                    text = paragraphs.nth(i).inner_text().strip()
                    if text:
                        content += text + " "
                if not content:
                    continue

                date = "Not Found"
                try:
                    time_el = page.locator("time").first
                    date = time_el.get_attribute("datetime") or time_el.inner_text()
                except Exception:
                    pass

                category = detect_content_type(link, page)

                articles_data.append({
                    "article_id": article_id,
                    "domain": link,
                    "date": date,
                    "category": category,
                    "headline": headline,
                    "content": content,
                    "label": 1
                })

                article_id += 1

            except Exception as e:
                logging.error(f"{link} - {e}")

        browser.close()

    return pd.DataFrame(articles_data)