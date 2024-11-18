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

class FinancialTopicModel:
    def __init__(
        self,
        n_gram_range: Tuple[int, int] = (1, 2),
        min_topic_size: int = 12,    
        nr_topics: int = 40          
    ):
        logging.basicConfig(level=logging.INFO)
        self.logger = logging.getLogger(__name__)
        
        # Comprehensive financial/stock market vocabulary
        self.finance_vocab = [
            # Price movements
            "up", "rise", "fall", "drop", "surge", "plunge", "soar", "decline",
            "increase", "decrease", "gain", "loss", "jump", "sink", "climb",
            
            # Performance indicators
            "profit", "revenue", "earnings", "growth", "performance", "sales",
            "margin", "return", "dividend", "yield", "target", "forecast",
            
            # Market sentiment
            "bullish", "bearish", "optimistic", "pessimistic", "confidence",
            "uncertainty", "volatile", "stable", "risk", "opportunity",
            "momentum", "sentiment",
            
            # Corporate actions
            "merger", "acquisition", "IPO", "spinoff", "restructure",
            "expansion", "layoff", "partnership", "investment", "launch",
            "deal", "contract",
            
            # Economic indicators
            "inflation", "GDP", "interest-rate", "outlook", "economy",
            "sector", "industry", "market-share", "demand", "supply",
            
            # Trading terms
            "volume", "trading", "stock", "shares", "market", "index",
            "position", "portfolio", "valuation", "price",
            
            # Company health
            "debt", "cash-flow", "assets", "liabilities", "balance-sheet",
            "quarterly", "annual", "guidance", "estimate",
            
            # Market events
            "announcement", "report", "news", "update", "release",
            "conference", "meeting", "presentation"
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
    
    def new_scrape_articles(self,keywords: List[str], end_date: str, end_time: str, days_back: int = 3, max_articles_per_keyword: int = 10) -> List[str]:
        end_datetime = datetime.strptime(f"{end_date} {end_time}", '%Y-%m-%d %H:%M:%S')
        start_datetime = end_datetime - timedelta(days=days_back)
        
        gn = GoogleNews(lang='en', country='US')
        all_sentences = []
        
        for keyword in keywords:
            search = gn.search(keyword, from_=start_datetime.strftime('%Y-%m-%d'), to_=end_datetime.strftime('%Y-%m-%d'))
            
            article_count = 0
            for entry in search['entries']:
                pub_date = datetime.strptime(entry.published, '%a, %d %b %Y %H:%M:%S %Z')
                if start_datetime <= pub_date <= end_datetime:
                    # Split title into sentences by common sentence endings
                    sentences = [s.strip() for s in 
                               entry.title.replace('! ', '!SPLIT')\
                                        .replace('? ', '?SPLIT')\
                                        .replace('. ', '.SPLIT')\
                                        .split('SPLIT') 
                               if s.strip()]
                    all_sentences.extend(sentences)
                    article_count += 1
                    if article_count >= max_articles_per_keyword:
                        break
                time.sleep(1)  # Be polite, wait a second between requests
        
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
        for i in range(4):
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

                # i is our iterator for stock symbols and we need it also for industry and sector so this part is very important
                ##### IMPORTANT HERE AS WE ARE DEFININING OUR KEYWORD APPROACH
                ################################################
                keywords = [Stock_Names[i], Stock_Names[i] + ' Stock', Stock_Industries[i], Stock_Industries[i] + ' Industry', Stock_Sectors[i] + 'Sector']
                headline_set = self.new_scrape_articles(keywords=keywords, end_date=str_date, end_time=str_time, days_back=4, max_articles_per_keyword=25)
                headlines_sets.append(headline_set)

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
        n_gram_range=(1, 2),
        min_topic_size=12,
        nr_topics=40
    )
    
    # Generate data and train model
    model.generate_and_train(headlines_file)
    
    # Save model
    model.save_model(model_output)


if __name__ == "__main__":
    main()