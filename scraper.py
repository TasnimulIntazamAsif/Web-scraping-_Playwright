from playwright.sync_api import sync_playwright
import pandas as pd
import logging
import os
from urllib.parse import urljoin
from config import BASE_URL, HEADLESS, MAX_ARTICLES

# Ensure logs directory exists
os.makedirs("logs", exist_ok=True)

logging.basicConfig(
    filename="logs/scraper.log",
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s"
)

def detect_content_type(url, page):
    """
    Detect article type specifically: International, Regional, Sports, Business, Live, Video
    """

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
        # Check page for <video> tag
        if page.locator("video").count() > 0:
            return "Video"
        return "News Article"

def scrape_news():

    articles_data = []

    with sync_playwright() as p:
        browser = p.chromium.launch(headless=HEADLESS)
        page = browser.new_page()

        print("Opening website...")
        page.goto(BASE_URL)
        page.wait_for_load_state("networkidle")

        # Collect all anchor links
        links = page.locator("a")
        article_urls = []

        for i in range(links.count()):
            href = links.nth(i).get_attribute("href")

            if href:
                full_url = urljoin(BASE_URL, href)

                if BASE_URL.split("/")[2] in full_url:
                    article_urls.append(full_url)

        article_urls = list(set(article_urls))
        article_urls = article_urls[:MAX_ARTICLES]

        print(f"Found {len(article_urls)} article links")

        for link in article_urls:
            try:
                print(f"Scraping: {link}")
                page.goto(link)
                page.wait_for_load_state("networkidle")

                # -------- TITLE --------
                try:
                    title = page.locator("h1").first.inner_text().strip()
                except:
                    continue

                # -------- CONTENT --------
                content = ""
                paragraphs = page.locator("p")
                for i in range(paragraphs.count()):
                    text = paragraphs.nth(i).inner_text().strip()
                    if text:
                        content += text + " "
                content = content.strip()

                # -------- TIMEFRAME --------
                timeframe = "Not Found"
                try:
                    time_element = page.locator("time").first
                    timeframe_text = time_element.get_attribute("datetime") or time_element.inner_text().strip()
                    if timeframe_text:
                        timeframe = timeframe_text
                except:
                    pass

                # -------- CONTENT TYPE --------
                content_type = detect_content_type(link, page)

                # -------- IMAGE URLs --------
                image_links = []

                # Hero image
                try:
                    hero_img = page.locator("article img").first
                    src = (
                        hero_img.get_attribute("src") or
                        hero_img.get_attribute("data-src") or
                        hero_img.get_attribute("data-original") or
                        hero_img.get_attribute("data-lazy-src")
                    )
                    srcset = hero_img.get_attribute("srcset")
                    if srcset:
                        src = srcset.split(",")[-1].split(" ")[0]

                    if src and not src.startswith("data:"):
                        image_links.append(urljoin(link, src))
                except:
                    pass

                # All other images
                images = page.locator("img")
                for i in range(images.count()):
                    img = images.nth(i)
                    src = (
                        img.get_attribute("src") or
                        img.get_attribute("data-src") or
                        img.get_attribute("data-original") or
                        img.get_attribute("data-lazy-src")
                    )
                    srcset = img.get_attribute("srcset")
                    if srcset:
                        src = srcset.split(",")[-1].split(" ")[0]

                    if not src or src.startswith("data:"):
                        continue

                    full_image_url = urljoin(link, src)

                    if full_image_url not in image_links:
                        image_links.append(full_image_url)

                if not content:
                    continue

                articles_data.append({
                    "Title": title,
                    "Content": content,
                    "Timeframe": timeframe,
                    "Content_Type": content_type,
                    "Image_URLs": ", ".join(image_links) if image_links else "No Image",
                    "URL": link
                })

            except Exception as e:
                print("Error:", e)
                logging.error(f"Error scraping {link}: {e}")

        browser.close()

    print(f"Scraped {len(articles_data)} articles successfully")

    return pd.DataFrame(articles_data)