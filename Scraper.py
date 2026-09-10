import os
import re
import json
import asyncio
from bs4 import BeautifulSoup
from playwright.async_api import async_playwright

DATA_DIR = "./raw_scheme_data"
IMAGES_DIR = os.path.join(DATA_DIR, "images")
os.makedirs(IMAGES_DIR, exist_ok=True)

TARGET_URL = "https://www.tn.gov.in/scheme_list.php?dep_id=Mg=="

async def scrape_all_schemes():
    async with async_playwright() as p:
        browser = await p.chromium.launch(headless=True)
        context = await browser.new_context()
        page = await context.new_page()
        
        print(f"Loading root scheme list from {TARGET_URL}...")
        await page.goto(TARGET_URL, wait_until="networkidle")
        
        # Parse table links
        content = await page.content()
        soup = BeautifulSoup(content, 'html.parser')
        
        scheme_links = []
        for a_tag in soup.find_all('a', href=True):
            href = a_tag['href']
            # Target detail pages (e.g., scheme_view.php or department links)
            if 'scheme' in href.lower() or 'view' in href.lower():
                full_url = href if href.startswith('http') else f"https://www.tn.gov.in/{href.lstrip('/')}"
                title = a_tag.get_text(strip=True)
                if title and full_url not in [item['url'] for item in scheme_links]:
                    scheme_links.append({"title": title, "url": full_url})
        
        print(f"Found {len(scheme_links)} individual scheme pages.")
        
        scraped_dataset = []
        
        # Crawl each detail link
        for idx, item in enumerate(scheme_links):
            print(f"[{idx+1}/{len(scheme_links)}] Scraping: {item['title']}")
            try:
                detail_page = await context.new_page()
                await detail_page.goto(item['url'], wait_until="domcontentloaded", timeout=30000)
                
                html_body = await detail_page.content()
                detail_soup = BeautifulSoup(html_body, 'html.parser')
                
                # Extract main content container or text
                main_text = detail_soup.get_text(separator="\n", strip=True)
                
                # Download inline images (if scheme contains infographic/flowcharts)
                img_paths = []
                img_tags = detail_soup.find_all('img')
                for img_idx, img in enumerate(img_tags):
                    src = img.get('src')
                    if src and not any(x in src.lower() for x in ['logo', 'banner', 'icon', 'header']):
                        img_url = src if src.startswith('http') else f"https://www.tn.gov.in/{src.lstrip('/')}"
                        try:
                            img_response = await detail_page.request.get(img_url)
                            img_filename = f"scheme_{idx+1}_img_{img_idx+1}.png"
                            save_path = os.path.join(IMAGES_DIR, img_filename)
                            with open(save_path, "wb") as f:
                                f.write(await img_response.body())
                            img_paths.append(save_path)
                        except Exception as img_err:
                            pass
                
                scraped_dataset.append({
                    "scheme_id": f"scheme_{idx+1}",
                    "title": item['title'],
                    "url": item['url'],
                    "raw_text": main_text,
                    "image_paths": img_paths
                })
                await detail_page.close()
            except Exception as e:
                print(f"Error scraping {item['url']}: {e}")
                
        await browser.close()
        
        # Save raw dump locally
        with open(os.path.join(DATA_DIR, "scraped_schemes.json"), "w", encoding="utf-8") as f:
            json.dump(scraped_dataset, f, indent=2, ensure_ascii=False)
            
        print("Scraping completed successfully.")

if __name__ == "__main__":
    asyncio.run(scrape_all_schemes())