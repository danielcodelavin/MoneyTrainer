from datetime import datetime, timedelta
import time
from typing import List
from tqdm import tqdm
from duckduckgo_search import DDGS

def scrape(keywords: List[str], target_date: datetime, articles_per_keyword: int = 5) -> List[dict]:
    """
    Get news articles from a specific date using DuckDuckGo News search.
    
    Args:
        keywords: List of keywords to search for
        target_date: The specific date to search for
        articles_per_keyword: Number of articles to fetch per keyword
    """
    all_articles = []
    date_str = target_date.strftime("%Y-%m-%d")
    
    with tqdm(total=len(keywords) * articles_per_keyword, desc="Fetching articles") as pbar:
        with DDGS() as ddgs:
            for keyword in keywords:
                print(f"\nSearching for: {keyword}")
                
                try:
                    # Include date directly in search query
                    results = list(ddgs.news(
                        keywords=f"{keyword} {date_str}",
                        region="us-en",
                        safesearch="moderate",
                        max_results=articles_per_keyword
                    ))
                    
                    print(f"Found {len(results)} results")
                    
                    # Add results to articles list
                    for result in results:
                        article = {
                            'keyword': keyword,
                            'title': result.get('title', ''),
                            'body': result.get('body', ''),
                            'source': result.get('source', ''),
                            'date': result.get('date', ''),
                            'url': result.get('link', '')
                        }
                        
                        all_articles.append(article)
                        pbar.update(1)
                        print(f"Added article: {article['title']}")
                    
                    # Small delay between keywords
                    time.sleep(1)
                    
                except Exception as e:
                    print(f"Error searching for keyword '{keyword}': {str(e)}")
                    continue
    
    print(f"\nTotal articles fetched: {len(all_articles)}")
    return all_articles

if __name__ == "__main__":
    keywords = ["artificial intelligence", "climate change"]
    target_date = datetime(2024, 1, 1)  # January 1, 2024
    
    articles = scrape(
        keywords=keywords,
        target_date=target_date,
        articles_per_keyword=5
    )

    # Print results
    for i, article in enumerate(articles, 1):
        print(f"\nArticle {i}:")
        print(f"Keyword: {article['keyword']}")
        print(f"Title: {article['title']}")
        print(f"Source: {article['source']}")
        print(f"Date: {article['date']}")
        print(f"URL: {article['url']}")
        print(f"Body: {article['body']}")
        print("-" * 80)