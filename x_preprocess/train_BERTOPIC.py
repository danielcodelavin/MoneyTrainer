import numpy as np
from bertopic import BERTopic
from sentence_transformers import SentenceTransformer
import logging
import torch
from typing import List, Tuple
import os
import argparse
from sklearn.feature_extraction.text import CountVectorizer
from pygooglenews import GoogleNews
from datetime import datetime, timedelta
import time
import csv
import random
from tqdm import tqdm

class FinancialTopicModel:
    def __init__(
        self,
        n_gram_range: Tuple[int, int] = (1, 3),
        min_topic_size: int = 12,    
        nr_topics: int = 50          
    ):
        logging.basicConfig(level=logging.INFO)
        self.logger = logging.getLogger(__name__)
        
        # Refined financial vocabulary with explicit phrases
        self.finance_vocab = [
        # Price Movements - Positive
        "stock up", "shares rise", "market surge", "price soar", "stock climb",
        "upward trend", "price jump", "stocks advance", "market rebound", "bounce back", "uptrend",
        
        # Price Movements - Negative
        "stock down", "shares fall", "market plunge", "price drop", "stock sink",
        "downward trend", "price slump", "stocks retreat", "market tumble", "selloff", "downtrend",
        
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
        "confidence boost", "renewed optimism",
        
        # Market Sentiment - Negative
        "investor concern", "market pessimism", "bearish outlook",
        "losing momentum", "selling pressure", "market weakness",
        "negative sentiment", "investor fear", "market stress",
        "confidence crisis", "deepening uncertainty",
        
        # Corporate Events - Positive
        "successful merger", "strategic acquisition", "expansion plan",
        "partnership deal", "successful launch", "contract win",
        "major breakthrough", "positive restructuring", "innovation success",
        "new market entry", "game-changing development",
        
        # Corporate Events - Negative
        "failed merger", "acquisition collapse", "bankruptcy risk",
        "partnership termination", "failed launch", "contract loss",
        "major setback", "forced restructuring", "innovation failure",
        "executive shakeup", "regulatory crackdown",
        
        # Financial Health - Positive
        "strong balance", "cash rich", "debt reduction",
        "dividend increase", "capital strength", "healthy liquidity",
        "credit upgrade", "cost efficiency", "profitable growth",
        "balance sheet improvement",
        
        # Financial Health - Negative
        "weak balance", "cash poor", "debt burden",
        "dividend cut", "capital crisis", "liquidity crisis",
        "credit downgrade", "cost pressure", "profit warning",
        "cash flow issues",
        
        # Market Analysis - Positive
        "buy signal", "breakout level", "support holding",
        "upside potential", "growth opportunity", "market leader",
        "outperform peer", "upgrade rating", "price target raised",
        "sector leadership",
        
        # Market Analysis - Negative
        "sell signal", "breakdown level", "support broken",
        "downside risk", "growth concerns", "market laggard",
        "underperform peer", "downgrade rating", "price target cut",
        "sector underperformance",
        
        # Economic Context - Positive
        "economic growth", "sector strength", "industry recovery",
        "demand increase", "supply improvement", "market stability",
        "positive outlook", "growth forecast", "recovery path",
        "fiscal stimulus impact",
        
        # Economic Context - Negative
        "economic slowdown", "sector weakness", "industry decline",
        "demand decrease", "supply disruption", "market instability",
        "negative outlook", "recession fear", "recovery doubt",
        "stagflation risk",
        
        # Core Terms - Positive
        "profitable", "outperform", "upgrade", "bullish", "growth",
        "beat", "rally", "gain", "increase", "strong", "positive", "rise", "surge", "soar", "climb",
        
        # Core Terms - Negative
        "unprofitable", "underperform", "downgrade", "bearish", "decline",
        "miss", "crash", "loss", "decrease", "weak", "negative", "fall", "plunge", "sink", "losses"
    ]

        
        self.vectorizer = CountVectorizer(
            vocabulary=self.finance_vocab,
            ngram_range=n_gram_range,
            stop_words='english'
        )
        
        self.n_gram_range = n_gram_range
        self.min_topic_size = min_topic_size
        self.nr_topics = nr_topics
        
        self.sentence_model = SentenceTransformer('all-MiniLM-L6-v2')
        
        self.topic_model = BERTopic(
            embedding_model=self.sentence_model,
            vectorizer_model=self.vectorizer,
            n_gram_range=n_gram_range,
            min_topic_size=min_topic_size,
            nr_topics=nr_topics,
            top_n_words=10,
            verbose=True,
            calculate_probabilities=True,
            seed_topic_list=[self.finance_vocab]
        )

    def analyze_topics(self, topic_info, topics, preprocessed_headlines, model_info_path: str):
        """Analyze all topics with sentiment grouping and save to file"""
        
        # Define sentiment categories
        positive_terms = {
            "rise", "surge", "soar", "climb", "rally", "beat", "bullish", "growth",
            "profit", "uptrend", "outperform", "upgrade", "strong", "positive"
        }
        
        negative_terms = {
            "fall", "drop", "plunge", "sink", "selloff", "miss", "bearish", "decline",
            "loss", "downtrend", "underperform", "downgrade", "weak", "negative",
            "crash", "crisis", "warning", "bankruptcy", "default", "distress"
        }
        
        # Initialize sentiment buckets
        positive_topics = []
        negative_topics = []
        neutral_topics = []
        
        # Open file for writing
        with open(model_info_path, 'w', encoding='utf-8') as f:
            f.write("FINANCIAL TOPIC MODEL ANALYSIS\n")
            f.write("=" * 50 + "\n\n")
            f.write(f"Analysis Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write(f"Total Headlines Analyzed: {len(preprocessed_headlines)}\n")
            f.write(f"Number of Topics: {self.nr_topics}\n\n")
            
            f.write("DETAILED TOPIC ANALYSIS\n")
            f.write("=" * 50 + "\n\n")
            
            # Analyze all topics
            for idx, row in topic_info.iterrows():
                if idx != -1:  # Skip outlier topic
                    terms = self.topic_model.get_topic(idx)
                    term_str = ", ".join([f"{term[0]} ({term[1]:.3f})" for term in terms[:5]])
                    
                    # Count sentiment terms
                    pos_count = sum(1 for term, _ in terms if any(pos in term.lower() for pos in positive_terms))
                    neg_count = sum(1 for term, _ in terms if any(neg in term.lower() for neg in negative_terms))
                    
                    # Classify topic sentiment
                    if pos_count > neg_count:
                        sentiment = "POSITIVE"
                        positive_topics.append(idx)
                    elif neg_count > pos_count:
                        sentiment = "NEGATIVE"
                        negative_topics.append(idx)
                    else:
                        sentiment = "NEUTRAL"
                        neutral_topics.append(idx)
                    
                    # Get example headlines
                    topic_docs = [doc for doc, topic in zip(preprocessed_headlines, topics) if topic == idx][:2]
                    
                    # Write topic details to file
                    f.write(f"\nTopic {idx} (Size: {row['Count']}) - {sentiment}\n")
                    f.write(f"Terms: {term_str}\n")
                    f.write("Example Headlines:\n")
                    for doc in topic_docs:
                        f.write(f"  • {doc}\n")
            
            # Write sentiment distribution
            total_topics = len(positive_topics) + len(negative_topics) + len(neutral_topics)
            f.write("\nSENTIMENT DISTRIBUTION\n")
            f.write("=" * 50 + "\n\n")
            f.write(f"Positive Topics: {len(positive_topics)}/{total_topics} ({len(positive_topics)/total_topics*100:.1f}%)\n")
            f.write(f"Negative Topics: {len(negative_topics)}/{total_topics} ({len(negative_topics)/total_topics*100:.1f}%)\n")
            f.write(f"Neutral Topics: {len(neutral_topics)}/{total_topics} ({len(neutral_topics)/total_topics*100:.1f}%)\n")
            
            # Write model parameters
            f.write("\nMODEL PARAMETERS\n")
            f.write("=" * 50 + "\n\n")
            f.write(f"N-gram Range: {self.n_gram_range}\n")
            f.write(f"Minimum Topic Size: {self.min_topic_size}\n")
            f.write(f"Number of Topics: {self.nr_topics}\n")
            f.write(f"Embedding Model: all-MiniLM-L6-v2\n")
            
            # Write vocabulary information
            f.write("\nVOCABULARY STATISTICS\n")
            f.write("=" * 50 + "\n\n")
            f.write(f"Total Financial Terms: {len(self.finance_vocab)}\n")
            
        self.logger.info(f"Model information saved to {model_info_path}")

    def new_scrape_articles(self, keywords: List[str], end_date: str, end_time: str, days_back: int = 3, max_articles_per_keyword: int = 10) -> List[str]:
        end_datetime = datetime.strptime(f"{end_date} {end_time}", '%Y-%m-%d %H:%M:%S')
        start_datetime = end_datetime - timedelta(days=days_back)
        
        gn = GoogleNews(lang='en', country='US')
        all_sentences = []
        
        for keyword in tqdm(keywords, desc="Processing keywords", leave=False):
            search = gn.search(keyword, from_=start_datetime.strftime('%Y-%m-%d'), to_=end_datetime.strftime('%Y-%m-%d'))
            
            article_count = 0
            for entry in search['entries']:
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
                    if article_count >= max_articles_per_keyword:
                        break
                time.sleep(0.05) # Be polite, wait a second between requests
        
        return all_sentences

    def generate_and_train(self, headlines_file_path: str, model_info_path: str) -> None:
        self.logger.info("Starting data generation and training process...")
        csv_stockscreen_filepath = '/Users/daniellavin/Desktop/proj/MoneyTrainer/cleaned_stockscreen.csv'
        
        # Load stock data
        Stock_Symbols = []
        Stock_Names = []
        Stock_Sectors = []
        Stock_Industries = []

        with open(csv_stockscreen_filepath, 'r') as file:
            csv_reader = csv.DictReader(file)
            for row in csv_reader:
                Stock_Symbols.append(row['Symbol'])
                Stock_Names.append(row['Name'])
                Stock_Sectors.append(row['Sector'])
                Stock_Industries.append(row['Industry'])
        
        earliest_date = datetime(2022, 12, 1)
        latest_date = datetime(2024, 11, 15)
        
        headlines_sets = []
        total_iterations = 8 * len(Stock_Symbols)  # Increased from 8 to 10
        
        with tqdm(total=total_iterations, desc="Generating headlines data") as pbar:
            for iteration in range(8):  # Increased from 8 to 10
                for i in range(len(Stock_Symbols)):
                    while True:
                        random_date = earliest_date + timedelta(days=random.randint(0, (latest_date - earliest_date).days))
                        if random_date.weekday() < 5:  # 0-4 = Monday-Friday
                            break
                    
                    random_time = timedelta(
                        hours=random.randint(9, 20),
                        minutes=random.randint(0, 59),
                        seconds=random.randint(0, 59))
                    
                    str_date = random_date.strftime('%Y-%m-%d')
                    total_seconds = int(random_time.total_seconds())
                    hours, remainder = divmod(total_seconds, 3600)
                    minutes, seconds = divmod(remainder, 60)
                    str_time = f"{hours:02}:{minutes:02}:{seconds:02}"

                    keywords = [
                        Stock_Names[i],
                        Stock_Names[i] + ' Stock',
                        Stock_Industries[i] + ' Industry',
                        Stock_Sectors[i] + ' Stocks'
                    ]
                    
                    headline_set = self.new_scrape_articles(
                        keywords=keywords,
                        end_date=str_date,
                        end_time=str_time,
                        days_back=4,
                        max_articles_per_keyword=25
                    )
                    headlines_sets.append(headline_set)
                    pbar.update(1)
                    
                    completion = (iteration * len(Stock_Symbols) + i + 1) / total_iterations * 100
                    pbar.set_postfix({'Completion': f'{completion:.1f}%'})

        self.logger.info("Writing headlines to file...")
        with open(headlines_file_path, 'w', encoding='utf-8') as f:
            for headline_set in tqdm(headlines_sets, desc="Writing to file"):
                line = ';'.join(headline_set)
                f.write(line + '\n')
        
        self.logger.info("Loading headlines for training...")
        with open(headlines_file_path, 'r', encoding='utf-8') as f:
            headlines_sets = [line.strip().split(';') for line in f]
            headlines = [h for headline_set in headlines_sets for h in headline_set]
        
        # Preprocess headlines to focus on financial content
        preprocessed_headlines = [
            h for h in headlines 
            if any(term.lower() in h.lower() for term in self.finance_vocab)
        ]
        
        self.logger.info(f"Training on {len(preprocessed_headlines)} headlines after financial content filtering...")
        
        # Train the model
        topics, probs = self.topic_model.fit_transform(preprocessed_headlines)
        
        # Get topics information and analyze
        topic_info = self.topic_model.get_topic_info()
        self.analyze_topics(topic_info, topics, preprocessed_headlines, model_info_path)

    def save_model(self, output_dir: str) -> None:
        os.makedirs(output_dir, exist_ok=True)
        self.topic_model.save(os.path.join(output_dir, "financial_topic_model"))
        self.logger.info(f"Model saved to {output_dir}")
    
    def process_headlines(self, headlines: List[str]) -> np.ndarray:
        topics, probs = self.topic_model.transform(headlines)
        return np.mean(probs, axis=0)

def main():
    headlines_file = '/Users/daniellavin/Desktop/proj/MoneyTrainer/x_preprocess/training_headlines.txt'
    model_output = '/Users/daniellavin/Desktop/proj/MoneyTrainer/x_preprocess/BERTOPIC_MODEL_40.pt'
    model_info_file = '/Users/daniellavin/Desktop/proj/MoneyTrainer/x_preprocess/MODELINFO.txt'

    model = FinancialTopicModel(
        n_gram_range=(1, 3),
        min_topic_size=12,
        nr_topics=50
    )
    
    model.generate_and_train(headlines_file, model_info_file)
    model.save_model(model_output)

if __name__ == "__main__":
    main()