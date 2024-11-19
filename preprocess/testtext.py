import requests
from bs4 import BeautifulSoup
from datetime import datetime, timedelta
import time
from typing import List, Set
import random
from urllib.parse import quote, urlparse

def scrape(keywords: List[str], end_date: datetime, days_back: int, articles_per_keyword: int, 
           max_attempts_per_keyword: int = 20) -> List[str]:
    """
    Scrape news articles using Bing News search.
    
    Parameters:
    - keywords: List of search terms
    - end_date: End date for articles (currently unused)
    - days_back: Number of days to look back (currently unused)
    - articles_per_keyword: Maximum number of articles to collect per keyword
    - max_attempts_per_keyword: Maximum number of URLs to try per keyword before giving up
    """
    
    # Sites that are known to work well
    PREFERRED_SITES = {
        'forbes.com',
        'reuters.com',
        'bloomberg.com',
        'techcrunch.com',
        'venturebeat.com',
        'theverge.com',
        'wired.com',
        'zdnet.com',
        'businessinsider.com',
        'cnbc.com',
        'bbc.com',
        'theguardian.com',
    }
    
    # Sites to avoid due to scraping difficulties
    BLOCKED_SITES = {
    'msn.com',
    'microsoft.com',
    'linkedin.com',
    'facebook.com',
    'twitter.com',
    'medium.com',
    'financial-news.co.uk',
    'govtech.com',
    'thesun.co.uk',
    'seattletimes.com',
    'worldathletics.org',
    'washingtonpost.com',
    'business-standard.com',
    'nytimes.com',
    'reuters.com'
}
    
    # Track failed URLs across all searches
    failed_urls = set()
    
    def is_allowed_site(url: str) -> bool:
        """Check if the URL is from an allowed site."""
        domain = urlparse(url).netloc.lower()
        base_domain = '.'.join(domain.split('.')[-2:])
        
        if url in failed_urls:
            return False
            
        if any(blocked in domain for blocked in BLOCKED_SITES):
            return False
        
        return True
    
    def get_article_content(url: str) -> str:
        """Extract main content from article URL."""
        try:
            headers = {
                'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36',
                'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8',
                'Accept-Language': 'en-US,en;q=0.5',
                'DNT': '1',
                'Connection': 'keep-alive',
                'Upgrade-Insecure-Requests': '1',
            }
            
            response = requests.get(url, headers=headers, timeout=10)
            soup = BeautifulSoup(response.text, 'html.parser')
            
            # Remove unwanted elements
            for element in soup(['script', 'style', 'nav', 'header', 'footer', 'aside', 'iframe']):
                element.decompose()
            
            content_paragraphs = []
            
            main_content = soup.find(['article', 'main', 'div'], 
                                   class_=lambda x: x and any(term in str(x).lower() 
                                   for term in ['article', 'content', 'story', 'body']))
            
            if main_content:
                paragraphs = main_content.find_all('p')
            else:
                paragraphs = soup.find_all('p')
            
            for p in paragraphs:
                text = p.get_text().strip()
                if (len(text) > 50 and 
                    not any(phrase in text.lower() for phrase in 
                        ['cookie', 'subscribe', 'sign up', 'newsletter', 'registration', 
                         'sign in', 'log in', 'privacy policy', 'terms of service',
                         'advertisement', 'sponsored', 'share this article'])):
                    content_paragraphs.append(text)
            
            content = ' '.join(content_paragraphs)
            
            if len(content) > 200:
                return content
                
            failed_urls.add(url)
            return ""
            
        except Exception as e:
            print(f"Error fetching article from {url}: {str(e)}")
            failed_urls.add(url)
            return ""

    def search_bing_news(keyword: str) -> List[str]:
        """Search for news articles using Bing News."""
        articles = []
        processed_urls = set()
        attempts = 0
        
        try:
            search_url = f"https://www.bing.com/news/search?q={quote(keyword)}"
            
            headers = {
                'User-Agent': 'Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36',
                'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8',
                'Accept-Language': 'en-US,en;q=0.5',
            }
            
            print(f"Fetching news from: {search_url}")
            response = requests.get(search_url, headers=headers, timeout=10)
            soup = BeautifulSoup(response.text, 'html.parser')
            
            news_items = soup.find_all(['div', 'article'], 
                                     class_=lambda x: x and any(term in str(x).lower() 
                                     for term in ['news-card', 'newsitem', 'news-item', 'card']))
            
            print(f"Found {len(news_items)} news items")
            
            for item in news_items:
                # Check if we've exceeded our attempts limit
                if attempts >= max_attempts_per_keyword:
                    print(f"Reached maximum attempts ({max_attempts_per_keyword}) for keyword: {keyword}")
                    return articles
                    
                try:
                    links = item.find_all('a', href=True)
                    for link in links:
                        url = link['href']
                        
                        # Skip if we've already processed or failed this URL
                        if url in processed_urls or url in failed_urls:
                            continue
                            
                        if url.startswith('http') and is_allowed_site(url):
                            attempts += 1
                            print(f"Processing: {url} (Attempt {attempts}/{max_attempts_per_keyword})")
                            
                            content = get_article_content(url)
                            processed_urls.add(url)
                            
                            if content:
                                print(f"Successfully extracted content from {url}")
                                articles.append(content)
                                if len(articles) >= articles_per_keyword:
                                    return articles
                            
                            if attempts >= max_attempts_per_keyword:
                                print(f"Reached maximum attempts ({max_attempts_per_keyword}) for keyword: {keyword}")
                                return articles
                                
                            time.sleep(random.uniform(2, 4))
                except Exception as e:
                    print(f"Error processing item: {str(e)}")
                    continue
            
        except Exception as e:
            print(f"Error in Bing search: {str(e)}")
            
        return articles

    # Collect articles for all keywords
    all_articles = []
    
    for keyword in keywords:
        print(f"\nSearching articles for keyword: {keyword}")
        articles = search_bing_news(keyword)
        print(f"Found {len(articles)} articles for keyword: {keyword}")
        all_articles.extend(articles)
        
        # Add delay between keyword searches
        time.sleep(random.uniform(3, 6))
    
    print(f"\nTotal Articles Found: {len(all_articles)}")
    print(f"Total URLs that failed: {len(failed_urls)}")
    return all_articles

if __name__ == "__main__":
    # Test with realistic keywords
    keywords = ["artificial intelligence", "climate change", "space exploration"]
    end_date = datetime.now()
    
    # Define clear limits
    articles_per_keyword = 4
    max_attempts_per_keyword = 20
    
    print(f"Searching for up to {articles_per_keyword} articles per keyword")
    print(f"Will try maximum {max_attempts_per_keyword} URLs per keyword before giving up")
    
    articles = scrape(keywords, end_date, 3, articles_per_keyword, max_attempts_per_keyword)
    
    for i, article in enumerate(articles, 1):
        print(f"\nArticle {i}:")
        print(article[:500] + "..." if len(article) > 500 else article)
        print("-" * 80)
    
    print("\nTotal Articles:", len(articles))