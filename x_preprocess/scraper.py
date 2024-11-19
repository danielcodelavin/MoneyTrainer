import requests
from bs4 import BeautifulSoup
from datetime import datetime, timedelta
import time
from typing import List, Set
import random
from urllib.parse import quote, urlparse, urlencode
from tqdm import tqdm

def scrape(keywords: List[str], end_date: datetime, days_back: int,
          articles_per_keyword: int = 5, max_attempts_per_keyword: int = 30) -> List[str]:
    """
    Scrape news articles using Bing News search with date range support.
    """
    failed_urls = set()
    processed_urls = set()
    
    BLOCKED_SITES = {
        'msn.com', 'microsoft.com', 'linkedin.com', 'facebook.com', 'twitter.com',
        'medium.com', 'forbes.com', 'washingtonpost.com', 'nytimes.com',
        'reuters.com', 'bloomberg.com', 'wsj.com', 'yahoo.com',
        'financial-news.co.uk', 'seattletimes.com', 'businessinsider.com'
    }
    
    headers = {
        'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36',
        'Accept-Language': 'en-US,en;q=0.9'
    }

    def is_allowed_site(url: str) -> bool:
        try:
            domain = urlparse(url).netloc.lower()
            return not any(blocked in domain for blocked in BLOCKED_SITES)
        except:
            return False

    def get_article_content(url: str) -> str:
        if url in failed_urls:
            return ""
        try:
            response = requests.get(url, headers=headers, timeout=15)
            if response.status_code != 200:
                failed_urls.add(url)
                return ""

            soup = BeautifulSoup(response.text, 'html.parser')
            
            for element in soup(['script', 'style', 'nav', 'header', 'footer', 'aside', 'iframe']):
                element.decompose()
            
            main_content = soup.find(['article', 'main', 'div'], 
                                   class_=lambda x: x and any(term in str(x).lower() 
                                   for term in ['article', 'content', 'story', 'body', 'main']))
            
            paragraphs = (main_content.find_all('p') if main_content else soup.find_all('p'))
            
            content_parts = []
            for p in paragraphs:
                text = p.get_text().strip()
                if (len(text) > 50 and 
                    not any(phrase in text.lower() for phrase in 
                        ['cookie', 'subscribe', 'sign up', 'newsletter', 'registration'])):
                    content_parts.append(text)
            
            content = ' '.join(content_parts)
            return content if len(content) > 200 else ""

        except Exception:
            failed_urls.add(url)
            return ""

    def search_bing_news(keyword: str, pbar: tqdm) -> List[str]:
        articles = []
        attempts = 0
        page = 1
        
        while attempts < max_attempts_per_keyword and len(articles) < articles_per_keyword:
            try:
                params = {
                    'q': keyword,
                    'first': (page - 1) * 10,
                    'qft': f'interval=7&FORM=PTFTNR',  # Last 7 days
                    'setlang': 'en-US'
                }
                
                search_url = f"https://www.bing.com/news/search?{urlencode(params)}"
                
                response = requests.get(search_url, headers=headers, timeout=15)
                if response.status_code != 200:
                    time.sleep(random.uniform(3, 5))
                    continue

                soup = BeautifulSoup(response.text, 'html.parser')
                news_items = soup.find_all(['div', 'article'], 
                                     class_=lambda x: x and any(term in str(x).lower() 
                                     for term in ['news-card', 'newsitem', 'news-item', 'card']))
                
                if not news_items and page > 5:
                    break
                    
                for item in news_items:
                    links = item.find_all('a', href=True)
                    valid_links = [
                        link['href'] for link in links 
                        if link['href'].startswith('http') 
                        and link['href'] not in processed_urls 
                        and link['href'] not in failed_urls 
                        and is_allowed_site(link['href'])
                    ]
                    
                    for url in valid_links:
                        attempts += 1
                        processed_urls.add(url)
                        content = get_article_content(url)
                        
                        if content:
                            articles.append(content)
                            pbar.update(1)
                            break
                            
                        time.sleep(random.uniform(2, 4))
                        if attempts >= max_attempts_per_keyword or len(articles) >= articles_per_keyword:
                            break
                            
                    if attempts >= max_attempts_per_keyword or len(articles) >= articles_per_keyword:
                        break
                
                page += 1
                time.sleep(random.uniform(2, 4))

            except Exception as e:
                if page > 5:
                    break
                time.sleep(random.uniform(2, 4))
                continue
                
        return articles

    all_articles = []
    total_articles = len(keywords) * articles_per_keyword
    
    with tqdm(total=total_articles, desc="Fetching articles") as pbar:
        for keyword in keywords:
            keyword_articles = search_bing_news(keyword, pbar)
            all_articles.extend(keyword_articles)
            time.sleep(random.uniform(4, 6))
    
    print(f"\nTotal articles fetched: {len(all_articles)}")
    return all_articles

if __name__ == "__main__":
    keywords = ["artificial intelligence", "climate change"]
    end_date = datetime(2024, 1, 1)
    days_back = 7
    
    articles = scrape(
        keywords=keywords,
        end_date=end_date,
        days_back=days_back,
        articles_per_keyword=5,
        max_attempts_per_keyword=30
    )

    for i, article in enumerate(articles, 1):
        print(f"\nArticle {i}:")
        print(article[:400] + "..." if len(article) > 200 else article)
        print("-" * 80)