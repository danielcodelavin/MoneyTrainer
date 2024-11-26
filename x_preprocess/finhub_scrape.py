import requests
import time
from datetime import datetime, timedelta
import finnhub
def get_stock_news(symbol: str, date: str) -> list:
    """
    Get news headlines and content for a stock symbol for given date and previous day.
    Validates data before adding to results to avoid NULL/empty values.
    
    Args:
        symbol (str): Stock symbol (e.g., 'AAPL')
        date (str): Date in format 'YYYY-MM-DD'
    
    Returns:
        list: List of news headlines and content in sequential order
    """
    api_key = "ct25q59r01qoprggvv3gct25q59r01qoprggvv40"
    results = []

    # Convert date string to datetime
    end_date = datetime.strptime(date, '%Y-%m-%d')
    start_date = end_date - timedelta(days=1)
    
    # Make separate request for each day
    for current_date in [start_date, end_date]:
        date_str = current_date.strftime('%Y-%m-%d')
        
        # Make API request for single day
        url = f'https://finnhub.io/api/v1/company-news?symbol={symbol}&from={date_str}&to={date_str}&token={api_key}'
        response = requests.get(url)
        
        # Handle different status codes
        if response.status_code == 200:
            news_items = response.json()
            for item in news_items:
                # Get headline and summary
                headline = item.get('headline', '')
                summary = item.get('summary', '')
                
                # Validate both fields exist and aren't empty
                if (headline and summary and  # Check if both are non-empty
                    headline.strip() and summary.strip() and  # Check if they're not just whitespace
                    headline.lower() != 'null' and summary.lower() != 'null'):  # Check for "NULL" strings
                    
                    results.append(headline)
                    results.append(summary)
        
        elif response.status_code == 429:
            print(f"Rate limit hit for date {date_str}. Waiting and trying again...")
            time.sleep(60)
            continue
        elif response.status_code == 401:
            print("Invalid API key")
            return results
        else:
            print(f"Error {response.status_code} when fetching news for {date_str}")
        
        time.sleep(1)
    
    return results