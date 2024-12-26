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
import traceback
from pathlib import Path
import torch.nn as nn
import requests
import json
from collections import defaultdict

# Performance metrics for Market Cap categories
MARKET_CAP_METRICS = {
    'Q3': {'l1': 0.0481, 'rmse': 0.0616, 'error_std': 0.0384, 'precision': 0.6323},
    'Q4': {'l1': 0.0496, 'rmse': 0.0719, 'error_std': 0.0507, 'precision': 0.6189},
    'Q2': {'l1': 0.0499, 'rmse': 0.0638, 'error_std': 0.0397, 'precision': 0.5944},
    'Q1': {'l1': 0.0808, 'rmse': 0.3507, 'error_std': 0.3285, 'precision': 0.5803}
}

# Performance metrics for Volume categories
VOLUME_METRICS = {
    'Q1': {'l1': 0.0435, 'rmse': 0.0567, 'error_std': 0.0363, 'precision': 0.5859},
    'Q3': {'l1': 0.0548, 'rmse': 0.0740, 'error_std': 0.0493, 'precision': 0.5941},
    'Q4': {'l1': 0.0597, 'rmse': 0.0760, 'error_std': 0.0470, 'precision': 0.6355},
    'Q2': {'l1': 0.0703, 'rmse': 0.3423, 'error_std': 0.3251, 'precision': 0.6055}
}

# Performance metrics for Sectors
SECTOR_METRICS = {
    'Industrials': {'l1': 0.0444, 'rmse': 0.0582, 'error_std': 0.0376, 'precision': 0.5554},
    'Consumer Staples': {'l1': 0.0452, 'rmse': 0.0565, 'error_std': 0.0342, 'precision': 0.5290},
    'Real Estate': {'l1': 0.0468, 'rmse': 0.0561, 'error_std': 0.0313, 'precision': 0.6472},
    'Health Care': {'l1': 0.0472, 'rmse': 0.0597, 'error_std': 0.0367, 'precision': 0.4900},
    'Consumer Discretionary': {'l1': 0.0477, 'rmse': 0.0617, 'error_std': 0.0392, 'precision': 0.6003},
    'Miscellaneous': {'l1': 0.0505, 'rmse': 0.0605, 'error_std': 0.0463, 'precision': 0.2222},
    'Basic Materials': {'l1': 0.0570, 'rmse': 0.0656, 'error_std': 0.0333, 'precision': 0.7667},
    'Utilities': {'l1': 0.0577, 'rmse': 0.0898, 'error_std': 0.0683, 'precision': 0.5819},
    'Technology': {'l1': 0.0589, 'rmse': 0.0764, 'error_std': 0.0488, 'precision': 0.6787},
    'Energy': {'l1': 0.0589, 'rmse': 0.0722, 'error_std': 0.0415, 'precision': 0.6932},
    'Telecommunications': {'l1': 0.0694, 'rmse': 0.1238, 'error_std': 0.1051, 'precision': 0.6609},
    'Finance': {'l1': 0.0998, 'rmse': 0.4632, 'error_std': 0.4391, 'precision': 0.6380}
}

# Finnhub API functionality
def get_stock_news(symbol: str, date: str) -> List[str]:
    """Get news headlines from Finnhub API"""
    api_key = os.getenv('FINNHUB_API_KEY', '')  # Make sure to set your API key in environment variables
    if not api_key:
        print("Warning: Finnhub API key not found")
        return []
        
    end_date = datetime.strptime(date, '%Y-%m-%d')
    start_date = end_date - timedelta(days=1)
    
    url = f'https://finnhub.io/api/v1/company-news'
    params = {
        'symbol': symbol,
        'from': start_date.strftime('%Y-%m-%d'),
        'to': end_date.strftime('%Y-%m-%d'),
        'token': api_key
    }
    
    try:
        response = requests.get(url, params=params)
        if response.status_code == 200:
            news = response.json()
            return [item['headline'] for item in news if 'headline' in item]
    except Exception as e:
        print(f"Error fetching Finnhub news for {symbol}: {str(e)}")
    
    return []

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
        stock_data = x[:80]
        topic_vector = x[80:]
        
        stock_features = self.stock_path(stock_data)
        topic_features = self.topic_path(topic_vector)
        
        combined = torch.cat((stock_features, topic_features))
        return self.combined_path(combined.unsqueeze(0)).squeeze()

def get_market_cap_category(market_cap):
    """Determine market cap category based on value"""
    if market_cap <= 5e9:  # 5B
        return 'Q1'
    elif market_cap <= 20e9:  # 20B
        return 'Q2'
    elif market_cap <= 50e9:  # 50B
        return 'Q3'
    else:
        return 'Q4'

def get_volume_category(volume):
    """Determine volume category based on value"""
    if volume <= 1e6:  # 1M
        return 'Q1'
    elif volume <= 5e6:  # 5M
        return 'Q2'
    elif volume <= 20e6:  # 20M
        return 'Q3'
    else:
        return 'Q4'

def calculate_adjusted_metrics(row):
    """Calculate adjusted metrics based on stock categories"""
    market_cap_cat = get_market_cap_category(row['Market Cap'])
    volume_cat = get_volume_category(row['Volume'])
    sector = row['Sector']
    
    mc_metrics = MARKET_CAP_METRICS.get(market_cap_cat, {})
    vol_metrics = VOLUME_METRICS.get(volume_cat, {})
    sec_metrics = SECTOR_METRICS.get(sector, {})
    
    metrics = {}
    for metric in ['l1', 'rmse', 'error_std', 'precision']:
        values = [
            mc_metrics.get(metric, 0),
            vol_metrics.get(metric, 0),
            sec_metrics.get(metric, 0)
        ]
        metrics[metric] = np.median(values)
    
    return metrics

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
        time.sleep(0.1)
    
    if len(all_data) < min_points:
        print(f"Insufficient data points ({len(all_data)}/{min_points}) for {ticker_symbol}")
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
        return normalized_tensor, all_data[-1]
    except Exception as e:
        print(f"Error processing {ticker_symbol} data: {str(e)}")
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
        
        time.sleep(1)  # Rate limiting
    
    print(f"Scraping completed at {datetime.now()}")
    return list(set(all_titles))  # Remove duplicates

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

def datagenerator(
    stock_symbol: str,
    stock_name: str,
    stock_industry: str,
    stock_sector: str,
    bertopic_model: BERTopic,
    current_datetime: datetime
) -> Optional[torch.Tensor]:
    try:
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
        print(f"[    CHECKUP   ]  : {stock_symbol} {stock_name} {stock_industry}")
        
        # Get news from both sources
        second_headlines = new_scrape_articles(
            keywords=keywords,
            end_date=str_date,
            end_time=str_time,
            days_back=1,
            max_articles_per_keyword=10
        )
        
        headlines = get_stock_news(symbol=stock_symbol, date=str_date)
        headlines.extend(second_headlines)
        
        # Clean and deduplicate headlines
        clean_headlines = [x for x in headlines if x]
        clean_headlines = list(set(clean_headlines))
        
        if not clean_headlines:
            print(f"No headlines found for {stock_symbol}, skipping tensor creation")
            return None
            
        # Process headlines with BERTopic
        news_embedding = process_headlines_with_bertopic(clean_headlines, bertopic_model)
        if news_embedding is None:
            print(f"BERTopic processing failed for {stock_symbol}, skipping tensor creation")
            return None
            
        news_embedding = news_embedding[:30] if news_embedding is not None else None
        
        print(f"\nTensor Components for {stock_symbol}:")
        print(f"Stock Data (first 5): {stock_data[:5].tolist()}")
        print(f"Topic Vector (first 5): {news_embedding[:5].tolist()}\n")
        
        # Combine stock data and news embedding
        integrated_tensor = torch.cat([
            stock_data,
            news_embedding
        ])
        
        return integrated_tensor
        
    except Exception as e:
        print(f"Error in datagenerator for {stock_symbol}: {str(e)}")
        traceback.print_exc()
        return None

def process_predictions(df):
    """Process predictions and add expected metrics"""
    
    metric_columns = ['expected_l1', 'expected_rmse', 'expected_error_std', 'expected_precision']
    for col in metric_columns:
        df[col] = 0.0
    
    for idx, row in df.iterrows():
        metrics = calculate_adjusted_metrics(row)
        
        df.at[idx, 'expected_l1'] = metrics['l1']
        df.at[idx, 'expected_rmse'] = metrics['rmse']
        df.at[idx, 'expected_error_std'] = metrics['error_std']
        df.at[idx, 'expected_precision'] = metrics['precision']
    
    if 'percent' in df.columns:
        df = df.drop('percent', axis=1)
    
    return df

def create_directory_structure(current_date, base_path: str) -> Path:
    path = Path(base_path) / str(current_date.year) / f"{current_date.month:02d}" / f"{current_date.day:02d}"
    path.mkdir(parents=True, exist_ok=True)
    return path

def generate_analysis_report(df: pd.DataFrame, output_path: Path, timestamp: str):
    """Generate a detailed analysis report including the new metrics"""
    report_path = output_path / f"analysis_report_{timestamp}.txt"
    
    with open(report_path, 'w') as f:
        f.write("Stock Prediction Analysis Report\n")
        f.write("===============================\n\n")
        
        # Overall statistics
        f.write("Overall Statistics:\n")
        f.write("-----------------\n")
        f.write(f"Total stocks analyzed: {len(df)}\n")
        f.write(f"Average expected L1 distance: {df['expected_l1'].mean():.4f}\n")
        f.write(f"Average expected RMSE: {df['expected_rmse'].mean():.4f}\n")
        f.write(f"Average expected precision: {df['expected_precision'].mean():.4f}\n\n")
        
        # Analysis by sector
        f.write("Analysis by Sector:\n")
        f.write("-----------------\n")
        sector_stats = df.groupby('Sector').agg({
            'expected_l1': 'mean',
            'expected_rmse': 'mean',
            'expected_precision': 'mean',
            'symbol': 'count'
        }).round(4)
        
        for sector, stats in sector_stats.iterrows():
            f.write(f"\n{sector}:\n")
            f.write(f"  Number of stocks: {stats['symbol']}\n")
            f.write(f"  Expected L1 distance: {stats['expected_l1']:.4f}\n")
            f.write(f"  Expected RMSE: {stats['expected_rmse']:.4f}\n")
            f.write(f"  Expected precision: {stats['expected_precision']:.4f}\n")
        
        # Analysis by market cap and volume categories
        for category_type, func in [('Market Cap', get_market_cap_category), 
                                  ('Volume', get_volume_category)]:
            f.write(f"\nAnalysis by {category_type}:\n")
            f.write("-" * (len(category_type) + 12) + "\n")
            
            if category_type == 'Market Cap':
                df['Category'] = df['Market Cap'].apply(func)
            else:
                df['Category'] = df['Volume'].apply(func)
                
            category_stats = df.groupby('Category').agg({
                'expected_l1': 'mean',
                'expected_rmse': 'mean',
                'expected_precision': 'mean',
                'symbol': 'count'
            }).round(4)
            
            for category, stats in category_stats.iterrows():
                f.write(f"\n{category}:\n")
                f.write(f"  Number of stocks: {stats['symbol']}\n")
                f.write(f"  Expected L1 distance: {stats['expected_l1']:.4f}\n")
                f.write(f"  Expected RMSE: {stats['expected_rmse']:.4f}\n")
                f.write(f"  Expected precision: {stats['expected_precision']:.4f}\n")

def main():
    # Configuration paths
    csv_path = "/Users/daniellavin/Desktop/proj/MoneyTrainer/Hybrid_stockscreen.csv"
    bertopic_model_path = "/Users/daniellavin/Desktop/proj/MoneyTrainer/x_preprocess/2_BERTOPIC_MODEL_50.pt"
    checkpoint_path = "/Users/daniellavin/Desktop/proj/MoneyTrainer/checkpoints/checkpx1/checkpoint_epoch_43.pt"
    results_base_path = "/Users/daniellavin/Desktop/proj/MoneyTrainer/results"
    
    # Load models
    print("Loading models...")
    model = MultiModalMLP(device='cuda' if torch.cuda.is_available() else 'cpu')
    checkpoint = torch.load(checkpoint_path)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    
    topic_model = BERTopic.load(bertopic_model_path)
    
    # Read stock data
    print("Reading stock data...")
    Stock_Symbols = []
    Stock_Names = []
    Stock_Sectors = []
    Stock_Industries = []

    current_time = datetime.now()
    current_string_time = current_time.strftime("%Y%m%d_%H%M%S")

    with open(csv_path, 'r') as file:
        csv_reader = csv.DictReader(file)
        for row in csv_reader:
            Stock_Symbols.append(row['Symbol'])
            Stock_Names.append(row['Name'])
            Stock_Sectors.append(row['Sector'])
            Stock_Industries.append(row['Industry'])
    
    # Process stocks and collect results
    print("Processing stocks...")
    results = []
    with torch.no_grad():
        for i, symbol in enumerate(Stock_Symbols):
            print(f"\nProcessing {symbol} ({i+1}/{len(Stock_Symbols)})")
            tensor = datagenerator(
                stock_symbol=symbol,
                stock_name=Stock_Names[i],
                stock_industry=Stock_Industries[i],
                stock_sector=Stock_Sectors[i],
                bertopic_model=topic_model,
                current_datetime=current_time
            )
            
            if tensor is not None:
                output = model(tensor)
                result = output.item()
                results.append((symbol, result))
    
    # Create results DataFrame
    print("Creating results DataFrame...")
    results_df = pd.DataFrame(results, columns=['symbol', 'result'])
    
    # Add stock information
    info_df = pd.read_csv(csv_path)
    results_df = results_df.merge(info_df, left_on='symbol', right_on='Symbol', how='left')
    
    # Calculate adjusted metrics
    print("Calculating adjusted metrics...")
    results_df = process_predictions(results_df)
    
    # Save results
    print("Saving results...")
    results_path = create_directory_structure(current_time, results_base_path)
    csv_filename = f"results_{current_string_time}.csv"
    csv_path = results_path / csv_filename
    
    results_df.to_csv(csv_path, index=False)
    
    # Generate analysis report
    print("Generating analysis report...")
    generate_analysis_report(results_df, results_path, current_string_time)
    
    print(f"\nProcessing complete. Results saved to: {csv_path}")

if __name__ == "__main__":
    main()