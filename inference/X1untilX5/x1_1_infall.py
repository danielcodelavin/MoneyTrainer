import torch
import pandas as pd
import yfinance as yf
from datetime import datetime, timedelta
from sklearn.metrics.pairwise import cosine_similarity
from typing import List, Tuple, Dict, Optional
import os
from typing import List, Optional
import numpy as np
from bertopic import BERTopic
import logging
import random
from tqdm import tqdm
from pygooglenews import GoogleNews
import time
from sentence_transformers import SentenceTransformer
import csv
from finhub_scrape import get_stock_news
import traceback
from pathlib import Path
import torch.nn as nn

class ResidualBlock(nn.Module):
    def __init__(self, in_features):
        super(ResidualBlock, self).__init__()
        self.block = nn.Sequential(
            nn.Linear(in_features, in_features * 2),
            nn.LeakyReLU(),
            nn.Dropout(0.2),
            nn.Linear(in_features * 2, in_features)
        )
        self.leaky_relu = nn.LeakyReLU()
        self.norm = nn.LayerNorm(in_features)
        
    def forward(self, x):
        residual = x
        out = self.block(x)
        out += residual
        out = self.norm(out)
        return self.leaky_relu(out)

class MultiModalMLP(nn.Module):
    def __init__(self, device='cuda'):
        super(MultiModalMLP, self).__init__()
        
        # Stock data path with increased complexity
        self.stock_path = nn.Sequential(
            nn.Linear(80, 256),
            nn.LeakyReLU(),
            nn.Dropout(0.2),
            ResidualBlock(256),
            nn.Linear(256, 384),
            nn.LeakyReLU(),
            nn.Dropout(0.2),
            ResidualBlock(384),
            nn.Linear(384, 512),
            nn.LeakyReLU(),
            nn.Dropout(0.2),
            ResidualBlock(512),
            nn.Linear(512, 384),
            nn.LeakyReLU(),
            nn.Dropout(0.2),
            ResidualBlock(384),
            nn.Linear(384, 256),
            nn.LeakyReLU(),
            nn.Dropout(0.2)
        )
        
        # Topic vector path with increased complexity
        self.topic_path = nn.Sequential(
            nn.Linear(30, 128),
            nn.LeakyReLU(),
            nn.Dropout(0.2),
            nn.Linear(128, 256),
            nn.LeakyReLU(),
            nn.Dropout(0.2),
            ResidualBlock(256),
            nn.Linear(256, 192),
            nn.LeakyReLU(),
            nn.Dropout(0.2),
            ResidualBlock(192),
            nn.Linear(192, 128),
            nn.LeakyReLU(),
            nn.Dropout(0.2)
        )
        
        # Combined path with increased complexity
        self.combined_path = nn.Sequential(
            nn.Linear(384, 512),
            nn.BatchNorm1d(512),
            nn.LeakyReLU(),
            nn.Dropout(0.2),
            ResidualBlock(512),
            nn.Linear(512, 384),
            nn.BatchNorm1d(384),
            nn.LeakyReLU(),
            nn.Dropout(0.2),
            ResidualBlock(384),
            nn.Linear(384, 256),
            nn.BatchNorm1d(256),
            nn.LeakyReLU(),
            nn.Dropout(0.2),
            ResidualBlock(256),
            nn.Linear(256, 128),
            nn.BatchNorm1d(128),
            nn.LeakyReLU(),
            nn.Dropout(0.2),
            nn.Linear(128, 1)
        )
        
        self.to(device)
        
    def forward(self, x):
        # Split the input tensor into stock data and topic vector
        stock_data = x[:80]
        topic_vector = x[80:]
        
        # Process through respective paths
        stock_features = self.stock_path(stock_data)
        topic_features = self.topic_path(topic_vector)
        
        # Combine features
        combined = torch.cat((stock_features, topic_features))
        return self.combined_path(combined.unsqueeze(0)).squeeze()

# Using all original functions exactly as they were
def validate_raw_data(data: pd.Series) -> bool:
    if data is None or data.empty:
        return False
    if np.isinf(data).any():
        return False
    if np.isnan(data).any():
        return False
    if (data <= 0).any():
        return False
    return True

def prepare_single_stock_data(ticker_symbol: str, start_datetime: datetime, days: int = 7, min_points: int = 80) -> Optional[Tuple[torch.Tensor, float]]:
    ticker = yf.Ticker(ticker_symbol)
    all_data = []
    days_checked = 0
    print(f"Fetching data for {ticker_symbol}")
    
    while len(all_data) < min_points and days_checked < days:
        try:
            day_end = start_datetime - timedelta(days=days_checked)
            day_start = day_end - timedelta(days=1)
            data = ticker.history(start=day_start, end=day_end, interval="60m", prepost=True, auto_adjust=False)
            
            if not data.empty and 'Close' in data.columns:
                valid_data = data['Close'].dropna()
                if validate_raw_data(valid_data):
                    all_data.extend(valid_data.values)
                    print(f"Added {len(valid_data)} points for {ticker_symbol} - Total: {len(all_data)}")
        except Exception as e:
            print(f"Error on {ticker_symbol} for {day_start}: {str(e)}")
        
        days_checked += 1
        time.sleep(0.1)  # Rate limiting
    
    if len(all_data) < min_points:
        print(f"Insufficient data points ({len(all_data)}/{min_points}) for {ticker_symbol}")
        if len(all_data) <= (min_points*0.7):
            with open('problematic_tickers.txt', 'a') as f:
                f.write(f"{ticker_symbol}\n")
        return None
        
    try:
        if len(all_data) > min_points:
            all_data = all_data[:min_points]
            
        tensor_data = torch.tensor(all_data, dtype=torch.float32)
        mean = tensor_data.mean()
        std = tensor_data.std()
        if std == 0:
            return None
        normalized_tensor = (tensor_data - mean) / std
        return normalized_tensor, all_data[-1]  # Return normalized data and last raw price
    except Exception as e:
        print(f"Error processing {ticker_symbol} data: {str(e)}")
        return None

def process_headlines_with_bertopic(headlines: List[str], topic_model: BERTopic) -> Optional[torch.Tensor]:
    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger(__name__)
    
    try:
        if not headlines or not isinstance(headlines, list):
            logger.error("Invalid headlines input: must be non-empty list of strings")
            return None
            
        if not all(isinstance(h, str) for h in headlines):
            logger.error("All headlines must be strings")
            return None
            
        headlines = [h.strip() for h in headlines if h.strip()]
        
        if not headlines:
            logger.error("No valid headlines after cleaning")
            return None
            
        nr_topics = len(topic_model.get_topic_info())
        
        topics, probs = topic_model.transform(headlines)
        
        full_probs = np.zeros((len(headlines), nr_topics))
        
        if isinstance(probs, (float, np.float64)):
            topic_idx = topics if isinstance(topics, (int, np.integer)) else topics[0]
            full_probs[0, topic_idx] = probs
        else:
            probs = np.array(probs)
            for i, (topic, prob) in enumerate(zip(topics, probs)):
                if isinstance(prob, np.ndarray):
                    for t, p in zip(topic, prob):
                        if 0 <= t < nr_topics:
                            full_probs[i, t] = p
                else:
                    if 0 <= topic < nr_topics:
                        full_probs[i, topic] = prob
                
        mean_probs = np.mean(full_probs, axis=0)
        
        if mean_probs.shape[0] != nr_topics:
            logger.error(f"Wrong number of topics in output: got {mean_probs.shape[0]}, expected {nr_topics}")
            return None
        
        mean_probs_tensor = torch.from_numpy(mean_probs).float()
        
        logger.info(f"Successfully processed {len(headlines)} headlines")
        
        return mean_probs_tensor
        
    except Exception as e:
        logger.error(f"Error processing headlines: {str(e)}")
        return None

def new_scrape_articles(keywords: List[str], end_date: str, end_time: str, days_back: int = 3, max_articles_per_keyword: int = 10) -> List[str]:
    print(f"Starting scrape at {datetime.now()}")
    end_datetime = datetime.strptime(f"{end_date} {end_time}", '%Y-%m-%d %H:%M:%S')
    start_datetime = end_datetime - timedelta(days=days_back)
    
    gn = GoogleNews(lang='en', country='US')
    all_titles = []
    
    for i, keyword in enumerate(keywords):
        print(f"Processing keyword {i+1}/{len(keywords)}: {keyword}")
        start_time = time.time()
        
        try:
            from_date = (end_datetime - timedelta(days=1)).strftime('%Y-%m-%d')
            to_date = end_datetime.strftime('%Y-%m-%d')
            
            search = gn.search(keyword, from_=from_date, to_=to_date)
            print(f"Search completed in {time.time() - start_time:.2f} seconds")
            
            if not search.get('entries'):
                print(f"No entries found for {keyword}")
                continue
                
            article_count = 0
            for entry in search['entries'][:max_articles_per_keyword]:
                try:
                    pub_date = datetime.strptime(entry.published, '%a, %d %b %Y %H:%M:%S %Z')
                    if start_datetime <= pub_date <= end_datetime:
                        if entry.title.strip():
                            all_titles.append(entry.title.strip())
                            article_count += 1
                except Exception as e:
                    print(f"Error processing entry: {e}")
                    continue
            
            print(f"Processed {article_count} articles for {keyword}")
            
        except Exception as e:
            print(f"Failed on keyword '{keyword}': {e}")
            continue
        
      #  time.sleep(1)
    
    print(f"Scraping completed at {datetime.now()}")
    return all_titles

def datagenerator(
    stock_symbol: str,
    stock_name: str,
    stock_industry: str,
    stock_sector: str,
    bertopic_model: BERTopic
    ,current_datetime
) -> Optional[torch.Tensor]:
    try:
        #current_datetime = datetime.now()
        #current_datetime = datetime(2024, 11, 4, 13, 0, 0)
        str_date = current_datetime.strftime('%Y-%m-%d')
        str_time = current_datetime.strftime('%H:%M:%S')
        
        stock_data_result = prepare_single_stock_data(
            ticker_symbol=stock_symbol,
            start_datetime=current_datetime,
            days=10,
            min_points=80
        )
        
        if stock_data_result is None:
            return None
            
        stock_data, raw_latest_price = stock_data_result
        
        keywords = [stock_name, stock_industry + ' Industry']
        print("[    CHECKUP   ]  :" + stock_symbol + stock_name + stock_industry)
        second_headlines = new_scrape_articles(
            keywords=keywords,
            end_date=str_date,
            end_time=str_time,
            days_back=1,
            max_articles_per_keyword=10
        )
        
        headlines = get_stock_news(symbol=stock_symbol, date=str_date)
        headlines.extend(second_headlines)
        clean_headlines = [x for x in headlines if x]
        clean_headlines = list(set(clean_headlines))
        
        if not clean_headlines:
            print(f"No headlines found for {stock_symbol}, skipping tensor creation")
            return None
            
        news_embedding = process_headlines_with_bertopic(clean_headlines, bertopic_model)
        if news_embedding is None:
            print(f"BERTopic processing failed for {stock_symbol}, skipping tensor creation")
            return None
            
        news_embedding = news_embedding[:30] if news_embedding is not None else None
        
        print(f"\nTensor Components for {stock_symbol}:")
        print(f"Stock Data (first 5): {stock_data[:10].tolist()}")
        print(f"Topic Vector (first 5): {news_embedding[:10].tolist()}\n")
        
        integrated_tensor = torch.cat([
            stock_data,
            news_embedding
        ])
        
        return integrated_tensor
        
    except Exception as e:
        print(f"Error in datagenerator for {stock_symbol}: {str(e)}")
        traceback.print_exc()
        return None

def create_directory_structure(current_date,base_path: str) -> Path:
    #current_date = datetime.now()
    path = Path(base_path) / str(current_date.year) / f"{current_date.month:02d}" / f"{current_date.day:02d}"
    path.mkdir(parents=True, exist_ok=True)
    return path

def main():
    # Configuration paths
    csv_path = "/Users/daniellavin/Desktop/proj/MoneyTrainer/Hybrid_stockscreen.csv"  # Update this
    bertopic_model_path = "/Users/daniellavin/Desktop/proj/MoneyTrainer/x_preprocess/2_BERTOPIC_MODEL_50.pt"  # Update this
    checkpoint_path = "/Users/daniellavin/Desktop/proj/MoneyTrainer/checkpoints/checkpx1/checkpoint_epoch_43.pt"  # Update this
    results_base_path = "/Users/daniellavin/Desktop/proj/MoneyTrainer/results"  # Update this
    



    # Load models
    model = MultiModalMLP(device='cuda' if torch.cuda.is_available() else 'cpu')
    checkpoint = torch.load(checkpoint_path)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    
    topic_model = BERTopic.load(bertopic_model_path)
    
    # Read stock data from CSV
    Stock_Symbols = []
    Stock_Names = []
    Stock_Sectors = []
    Stock_Industries = []


    current_time = datetime.now().strftime("%Y%m%d_%H%M%S")
    current_time = datetime(2024, 12, 2, 13, 0, 0)
    current_string_time = current_time.strftime("%Y%m%d_%H%M%S")

    
    with open(csv_path, 'r') as file:
        csv_reader = csv.DictReader(file)
        for row in csv_reader:
            Stock_Symbols.append(row['Symbol'])
            Stock_Names.append(row['Name'])
            Stock_Sectors.append(row['Sector'])
            Stock_Industries.append(row['Industry'])
    
    # Process each stock and collect results
    results = []
    with torch.no_grad():
        for i, symbol in enumerate(Stock_Symbols):
            tensor = datagenerator(
                stock_symbol=symbol,
                stock_name=Stock_Names[i],
                stock_industry=Stock_Industries[i],
                stock_sector=Stock_Sectors[i],
                bertopic_model=topic_model
                ,current_datetime= current_time
            )
            
            if tensor is not None:
                output = model(tensor)
                result = output.item()
                results.append((symbol, result))
    
    # Create directory structure and save results
    
    
    results_path = create_directory_structure(current_time,results_base_path)
    csv_filename = f"results_{current_string_time}.csv"
    csv_path = results_path / csv_filename
    
    with open(csv_path, 'w', newline='') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(['symbol', 'result', 'percent'])
        for symbol, result in results:
            percent = f"{result * 100:.2f}%"
            writer.writerow([symbol, result, percent])

if __name__ == "__main__":
    main()