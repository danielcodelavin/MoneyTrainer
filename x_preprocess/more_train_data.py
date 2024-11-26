import csv
import random
from datetime import datetime, timedelta
import time
from typing import List
from duckduckgo_search import DDGS
from tqdm import tqdm
from finhub_scrape import get_stock_news
class DataCollector:
    def __init__(self):
        # File paths
        self.csv_stockscreen_filepath = '/Users/daniellavin/Desktop/proj/MoneyTrainer/cleaned_stockscreen.csv'
        self.output_filepath = '/Users/daniellavin/Desktop/proj/MoneyTrainer/x_preprocess/training_headlines.txt'
        
        # Date range
        self.earliest_date = datetime(2023, 12, 1)
        self.latest_date = datetime(2024, 11, 17)
        
        # Load stock data
        self.Stock_Symbols = []
        self.Stock_Names = []
        self.Stock_Sectors = []
        self.Stock_Industries = []
        self._load_stock_data()

    def _load_stock_data(self):
        """Load stock data from CSV"""
        with open(self.csv_stockscreen_filepath, 'r') as file:
            csv_reader = csv.DictReader(file)
            for row in csv_reader:
                self.Stock_Symbols.append(row['Symbol'])
                self.Stock_Names.append(row['Name'])
                self.Stock_Sectors.append(row['Sector'])
                self.Stock_Industries.append(row['Industry'])
    
    def ddg_scrape(self, keywords: List[str], date_str, max_articles_per_keyword: int = 5) -> List[str]:
        """Get news articles from DuckDuckGo"""
        all_bodies = []
        
        with DDGS() as ddgs:
            for keyword in keywords:
                try:
                    results = list(ddgs.news(
                        keywords=f"{keyword} {date_str}",
                        region="us-en",
                        safesearch="moderate",
                        max_results=max_articles_per_keyword
                    ))
                    
                    bodies = [result.get('body', '').strip() for result in results if result.get('body')]
                    all_bodies.extend(bodies)
                    
                    time.sleep(1)
                    
                except Exception as e:
                    print(f"Error searching for keyword '{keyword}': {str(e)}")
                    continue
        
        all_bodies = [body for body in dict.fromkeys(all_bodies) if body]
        return all_bodies

    def get_random_date(self) -> str:
        """Generate random weekday date"""
        while True:
            random_date = self.earliest_date + timedelta(
                days=random.randint(0, (self.latest_date - self.earliest_date).days)
            )
            if random_date.weekday() < 5:  # Monday = 0, Friday = 4
                return random_date.strftime('%Y-%m-%d')

    def collect_data(self, iterations_per_stock: int = 3, articles_per_keyword: int = 10):
        """Collect data and append to existing file"""
        print(f"Starting data collection at {datetime.now()}")
        print(f"Will process {len(self.Stock_Symbols)} stocks, {iterations_per_stock} times each")
        
        total_iterations = len(self.Stock_Symbols) * iterations_per_stock
        articles_collected = 0
        
        with tqdm(total=total_iterations, desc="Collecting Data") as pbar:
            for _ in range(iterations_per_stock):
                for i in range(len(self.Stock_Symbols)):
                    # Generate random date
                    date_str = self.get_random_date()
                    
                    # Create keywords for this stock
                    keywords = [
                        self.Stock_Names[i],
                        self.Stock_Industries[i],
                        f"{self.Stock_Industries[i]} Industry",
                        f"{self.Stock_Sectors[i]} Stocks"
                    ]
                    time.sleep(1.5)
                    # Get articles
                  #  headlines = self.ddg_scrape( keywords=keywords, date_str=date_str, max_articles_per_keyword=articles_per_keyword)
                    headlines = get_stock_news(symbol=self.Stock_Symbols[i], date=date_str)
                    print(headlines)
                    # Append to file if we got any headlines
                    if headlines:
                        with open(self.output_filepath, 'a', encoding='utf-8') as f:
                            line = ';'.join(headlines)
                            f.write(line + '\n')
                        articles_collected += len(headlines)
                    
                    pbar.update(1)
                    pbar.set_postfix({'Articles': articles_collected})
                    
                    # Random delay between stocks
                    time.sleep(random.uniform(1.0, 2.0))
        
        print(f"\nData collection completed at {datetime.now()}")
        print(f"Total articles collected: {articles_collected}")
        
        # Print file statistics
        with open(self.output_filepath, 'r', encoding='utf-8') as f:
            total_lines = sum(1 for line in f)
            f.seek(0)
            total_articles = sum(len(line.strip().split(';')) for line in f)
        
        print(f"Final file statistics:")
        print(f"Total lines in file: {total_lines}")
        print(f"Total articles in file: {total_articles}")

def main():
    collector = DataCollector()
    
    # Collect new data
    # Parameters to adjust:
    # - iterations_per_stock: how many times to process each stock
    # - articles_per_keyword: how many articles to fetch per keyword
    collector.collect_data(
        iterations_per_stock=3,  # Process each stock 3 times
        articles_per_keyword=10  # Get up to 10 articles per keyword
    )

if __name__ == "__main__":
    main()