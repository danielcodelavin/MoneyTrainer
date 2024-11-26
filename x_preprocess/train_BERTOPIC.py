import numpy as np
from bertopic import BERTopic
from sentence_transformers import SentenceTransformer
import logging
import torch
from typing import List, Tuple
import os
import argparse
from pygooglenews import GoogleNews
from datetime import datetime, timedelta
import time
import csv
import random
from tqdm import tqdm
from duckduckgo_search import DDGS

class FinancialTopicModel:
    def __init__(
        self,
        n_gram_range: Tuple[int, int] = (1, 3),
        min_topic_size: int = 8,    
        nr_topics: int = 50          
    ):
        logging.basicConfig(level=logging.INFO)
        self.logger = logging.getLogger(__name__)
        
        # Comprehensive financial/stock market vocabulary
        self.finance_vocab = [
            # Price Movements - Positive
            "stock up", "shares rise", "market surge", "price soar", "stock climb",
            "upward trend", "price jump", "stocks advance", "market rebound", "bounce back", "up"
            
            # Price Movements - Negative
            "stock down", "shares fall", "market plunge", "price drop", "stock sink",
            "downward trend", "price slump", "stocks retreat", "market tumble", "sell off", "down"
            
            # Performance - Positive
            "earnings beat", "revenue growth", "profit surge", "sales increase", 
            "strong performance", "record profit", "better than expected",
            "exceeds forecast", "margin expansion", "positive guidance",
            
            # Performance - Negative
            "earnings miss", "revenue decline", "profit drop", "sales decrease",
            "weak performance", "record loss", "worse than expected",
            "misses forecast", "margin compression", "negative guidance",
            
            # Market Sentiment - Positive
            "investor confidence", "market optimism", "bullish outlook",
            "strong momentum", "buying opportunity", "market recovery",
            "positive sentiment", "investor enthusiasm", "market strength",
            "confidence boost",
            
            # Market Sentiment - Negative
            "investor concern", "market pessimism", "bearish outlook",
            "losing momentum", "selling pressure", "market weakness",
            "negative sentiment", "investor fear", "market stress",
            "confidence crisis",
            
            # Corporate Events - Positive
            "successful merger", "strategic acquisition", "expansion plan",
            "partnership deal", "successful launch", "contract win",
            "major breakthrough", "positive restructuring", "innovation success",
            
            # Corporate Events - Negative
            "failed merger", "acquisition collapse", "bankruptcy risk",
            "partnership termination", "failed launch", "contract loss",
            "major setback", "forced restructuring", "innovation failure",
            
            # Financial Health - Positive
            "strong balance", "cash rich", "debt reduction",
            "dividend increase", "capital strength", "healthy liquidity",
            "credit upgrade", "cost efficiency", "profitable growth",
            
            # Financial Health - Negative
            "weak balance", "cash poor", "debt burden",
            "dividend cut", "capital crisis", "liquidity crisis",
            "credit downgrade", "cost pressure", "profit warning",
            
            # Market Analysis - Positive
            "buy signal", "breakout level", "support holding",
            "upside potential", "growth opportunity", "market leader",
            "outperform peer", "upgrade rating", "price target raised",
            
            # Market Analysis - Negative
            "sell signal", "breakdown level", "support broken",
            "downside risk", "growth concerns", "market laggard",
            "underperform peer", "downgrade rating", "price target cut",
            
            # Economic Context - Positive
            "economic growth", "sector strength", "industry recovery",
            "demand increase", "supply improvement", "market stability",
            "positive outlook", "growth forecast", "recovery path",
            
            # Economic Context - Negative
            "economic slowdown", "sector weakness", "industry decline",
            "demand decrease", "supply disruption", "market instability",
            "negative outlook", "recession fear", "recovery doubt",
            
            # Core Terms - Positive
            "uptrend", "profitable", "outperform", "upgrade",
            "bullish", "growth", "beat", "rally", "gain", "increase",
            "strong", "positive", "rise", "surge", "soar", "climb", "rally", "beat"
            
            # Core Terms - Negative
            "downtrend", "unprofitable", "underperform", "downgrade",
            "bearish", "decline", "miss", "crash", "loss", "decrease",
            "weak", "negative", "fall", "plunge", "sink", "selloff", "miss"
        ]
        
        self.n_gram_range = n_gram_range
        self.min_topic_size = min_topic_size
        self.nr_topics = nr_topics
        
        self.sentence_model = SentenceTransformer('all-MiniLM-L6-v2')
        
        self.topic_model = BERTopic(
            embedding_model=self.sentence_model,
            n_gram_range=n_gram_range,
            min_topic_size=min_topic_size,
            nr_topics=nr_topics,
            verbose=True
        )
    
    def ddg_scrape(self,keywords: List[str], date_str, max_articles_per_keyword: int = 5) -> List[dict]:
        """
        Get news articles from a specific date using DuckDuckGo News search.
        
        Args:
            keywords: List of keywords to search for
            target_date: The specific date to search for
            articles_per_keyword: Number of articles to fetch per keyword
        """
        all_bodies = []  # We'll just store article bodies in a 1D array
        
    # Format date for search

        
        with DDGS() as ddgs:
            for keyword in keywords:
                try:
                    # Search for articles
                    results = list(ddgs.news(
                        keywords=f"{keyword} {date_str}",
                        region="us-en",
                        safesearch="moderate",
                        max_results=max_articles_per_keyword
                    ))
                    
                    # Extract just the bodies and add to our list
                    bodies = [result.get('body', '').strip() for result in results if result.get('body')]
                    all_bodies.extend(bodies)
                    
                    # Be nice to the API
                    time.sleep(1)
                    
                except Exception as e:
                    print(f"Error searching for keyword '{keyword}': {str(e)}")
                    continue
        
        # Remove any empty strings and duplicates while preserving order
        all_bodies = [body for body in dict.fromkeys(all_bodies) if body]
        
        return all_bodies

    # Just change your existing function to this:
    def new_scrape_articles(self, keywords: List[str], end_date: str, end_time: str, days_back: int = 3, max_articles_per_keyword: int = 10) -> List[str]:
        print(f"Starting scrape at {datetime.now()}")
        end_datetime = datetime.strptime(f"{end_date} {end_time}", '%Y-%m-%d %H:%M:%S')
        start_datetime = end_datetime - timedelta(days=days_back)
        
        # Create single instance of GoogleNews with shorter timeout
        gn = GoogleNews(lang='en', country='US')
        all_sentences = []
        
        for i, keyword in enumerate(keywords):
            print(f"Processing keyword {i+1}/{len(keywords)}: {keyword}")
            start_time = time.time()
            
            try:
                # Reduce the time window to minimize data
                from_date = (end_datetime - timedelta(days=1)).strftime('%Y-%m-%d')
                to_date = end_datetime.strftime('%Y-%m-%d')
                
                search = gn.search(keyword, from_=from_date, to_=to_date)
                print(f"Search completed in {time.time() - start_time:.2f} seconds")
                
                if not search.get('entries'):
                    print(f"No entries found for {keyword}")
                    continue
                    
                article_count = 0
                for entry in search['entries'][:max_articles_per_keyword]:  # Limit the loop directly
                    try:
                        pub_date = datetime.strptime(entry.published, '%a, %d %b %Y %H:%M:%S %Z')
                        if start_datetime <= pub_date <= end_datetime:
                            sentences = [s.strip() for s in 
                                    entry.title.replace('! ', '!SPLIT')\
                                                .replace('? ', '?SPLIT')\
                                                .replace('. ', '.SPLIT')\
                                                .split('SPLIT') 
                                    if s.strip()]
                            all_sentences.extend(sentences)
                            article_count += 1
                    except Exception as e:
                        print(f"Error processing entry: {e}")
                        continue
                
                print(f"Processed {article_count} articles for {keyword}")
                
            except Exception as e:
                print(f"Failed on keyword '{keyword}': {e}")
                continue
            
            # Shorter delay between keywords
            time.sleep(1)
        
        print(f"Scraping completed at {datetime.now()}")
        return all_sentences
            
    
    def generate_and_train(self, headlines_file_path: str) -> None:
        """
        Generates training data and trains the model in one go.
        This is a placeholder - implement your data generation logic here.
        """
        self.logger.info("Starting data generation and training process...")
        csv_stockscreen_filepath = '/Users/daniellavin/Desktop/proj/MoneyTrainer/cleaned_stockscreen.csv'
        
        Stock_Symbols = []
        Stock_Names = []
        Stock_Sectors = []
        Stock_Industries = []
        # YYYY MM DD
        earliest_date = datetime(2022, 12, 1)
        latest_date = datetime(2024, 11, 15)
    
        # 
        with open(csv_stockscreen_filepath, 'r') as file:
            csv_reader = csv.DictReader(file)
            
            # Iterate through each row and append to respective lists
            for row in csv_reader:
                Stock_Symbols.append(row['Symbol'])
                Stock_Names.append(row['Name'])
                Stock_Sectors.append(row['Sector'])
                Stock_Industries.append(row['Industry'])
        
        ### We will iterate through all stock symbols 4 times and generate data for each with a random datetime. this  will give us around 1600 news headlines sets so 1600000 headlines to train on
        headlines_sets = []
        total_iterations = 2 * len(Stock_Symbols)
        with tqdm(total=total_iterations, desc="Processing") as pbar:
            for _ in range(2):
                for i in range(len(Stock_Symbols)):
                    # Generating random date and time here
                    
                    while True:
                        random_date = earliest_date + timedelta(days=random.randint(0, (latest_date - earliest_date).days))
                        if random_date.weekday() < 5:  # 0 = Monday, 4 = Friday
                            break
                    
                    random_time = timedelta(
                            hours=random.randint(9, 20),  # 9 AM to 9 PM (20 = 8:59 PM, end of the day)
                            minutes=random.randint(0, 59),
                            seconds=random.randint(0, 59))
                    str_date = random_date.strftime('%Y-%m-%d')
                    # Convert random_time (timedelta) to a string in 'HH:MM:SS' format
                    total_seconds = int(random_time.total_seconds())
                    hours, remainder = divmod(total_seconds, 3600)
                    minutes, seconds = divmod(remainder, 60)
                    str_time = f"{hours:02}:{minutes:02}:{seconds:02}"
                    
                    keywords = [Stock_Names[i], Stock_Industries[i], Stock_Industries[i] + ' Industry', Stock_Sectors[i] + 'Stocks']
                    
                    headline_set = self.new_scrape_articles(keywords=keywords, end_date=str_date, end_time=str_time, days_back=4, max_articles_per_keyword=25)
                    #headline_set = self.ddg_scrape(keywords=keywords, date_str=str_date, max_articles_per_keyword=5)
                    headlines_sets.append(headline_set)
                    
                    pbar.update(1)
        self.logger.info("Training model on generated data...")
        
        with open(headlines_file_path, 'w', encoding='utf-8') as f:
            for headline_set in headlines_sets:
                line = ';'.join(headline_set)  # join sentences with semicolons
                f.write(line + '\n')  # write line to file
        
        with open(headlines_file_path, 'r', encoding='utf-8') as f:
            headlines_sets = [line.strip().split(';') for line in f]
            headlines = [h for headline_set in headlines_sets for h in headline_set]
        
        self.logger.info(f"Training on {len(headlines)} headlines...")
        
        topics, _ = self.topic_model.fit_transform(headlines)
        
        # Log topics information
        topic_info = self.topic_model.get_topic_info()
        n_topics = len(topic_info)
        self.logger.info(f"Model trained with {n_topics} topics")
        
        # Log top topics and their key terms
        self.logger.info("\nTop 10 largest topics and their terms:")
        for idx, row in topic_info.head(10).iterrows():
            if idx != -1:  # Skip the outlier topic
                terms = self.topic_model.get_topic(idx)[:5]  # Get top 5 terms
                term_str = ", ".join([term[0] for term in terms])
                self.logger.info(f"Topic {idx}: {term_str}")
    
    def save_model(self, output_dir: str) -> None:
        """Save the trained model to the specified directory"""
        os.makedirs(output_dir, exist_ok=True)
        self.topic_model.save(os.path.join(output_dir, "financial_topic_model"))
        self.logger.info(f"Model saved to {output_dir}")
    def process_headlines(self, headlines: List[str]) -> np.ndarray:
        """Process a batch of headlines and return their mean-pooled topic vector"""
        topics, probs = self.topic_model.transform(headlines)
        return np.mean(probs, axis=0)


def main():
    # Define file paths directly
    headlines_file = '/Users/daniellavin/Desktop/proj/MoneyTrainer/x_preprocess/training_headlines.txt'
    model_output = '/Users/daniellavin/Desktop/proj/MoneyTrainer/x_preprocess/BERTOPIC_MODEL_40.pt'
    # Initialize model
    model = FinancialTopicModel(
        n_gram_range=(1, 3),
        min_topic_size=8,
        nr_topics=50
    )
    
    # Generate data and train model
    model.generate_and_train(headlines_file)
    
    # Save model
    model.save_model(model_output)
if __name__ == "__main__":
    main()