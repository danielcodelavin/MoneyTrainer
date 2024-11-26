import torch
import pandas as pd
import yfinance as yf
from datetime import datetime, timedelta
from sklearn.metrics.pairwise import cosine_similarity
from typing import List, Tuple, Dict, Optional
import os
from typing import List, Optional
import torch
import numpy as np
from bertopic import BERTopic
import logging
import random
from tqdm import tqdm
from pygooglenews import GoogleNews
import time
from sentence_transformers import SentenceTransformer
from bertopic import BERTopic
import numpy as np
import csv
from finhub_scrape import get_stock_news
import traceback

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

def extract_symbols_from_csv(filepath: str) -> List[str]:
    with open(filepath, 'r') as file:
        reader = csv.reader(file)
        next(reader)
        symbols = [row[0].strip() for row in reader]
    return symbols

def calculate_percentage_change(latest_value: float, gt_value: float) -> float:
    return (gt_value - latest_value) / latest_value

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
        time.sleep(1)  # Rate limiting
    
    if len(all_data) < min_points:
        print(f"Insufficient data points ({len(all_data)}/{min_points}) for {ticker_symbol}")
        if len(all_data) <= (min_points*0.7):
            with open('problematic_tickers.txt', 'a') as f:
                f.write(f"{ticker_symbol}\n")
        return None
        
    try:
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
    




def returngroundtruthstock(stock_symbol: str, target_datetime: datetime, max_days: int = 4) -> torch.Tensor:
    """
    Get the first available hourly price for the next trading day.
    If target date is weekend/holiday, look ahead until we find the next trading day.
    """
    try:
        stock = yf.Ticker(stock_symbol)
        days_checked = 0
        
        while days_checked < max_days:
            check_date = target_datetime + timedelta(days=days_checked)
            if check_date.weekday() >= 5:  # Skip weekends
                days_checked += 1
                continue
                
            data = stock.history(
                start=check_date.strftime('%Y-%m-%d'),
                end=(check_date + timedelta(days=1)).strftime('%Y-%m-%d'),
                interval="60m",
                prepost=True
            )
            
            if not data.empty and 'Close' in data.columns:
                first_price = data['Close'].iloc[0]  # Get first available price
                if not np.isnan(first_price) and not np.isinf(first_price) and first_price > 0:
                    print(f"Found first price {first_price} for {stock_symbol} on {check_date.strftime('%Y-%m-%d')}")
                    return torch.tensor([first_price], dtype=torch.float32)
            
            days_checked += 1
            
        print(f"No valid price found for {stock_symbol} within {max_days} days")
        return torch.tensor([])
        
    except Exception as e:
        print(f"Error getting price for {stock_symbol}: {str(e)}")
        return torch.tensor([])



def process_headlines_with_bertopic(headlines: List[str], topic_model: BERTopic) -> Optional[torch.Tensor]:
    """
    Process headlines using the pretrained BERTopic model with 50 topics.
    
    Args:
        headlines (List[str]): List of headline strings to process
        topic_model (BERTopic): Pre-trained 50-topic BERTopic model
        
    Returns:
        Optional[torch.Tensor]: Mean-pooled probabilities tensor or None if error occurs
    """
    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger(__name__)
    
    try:
        # Input validation
        if not headlines or not isinstance(headlines, list):
            logger.error("Invalid headlines input: must be non-empty list of strings")
            return None
            
        if not all(isinstance(h, str) for h in headlines):
            logger.error("All headlines must be strings")
            return None
            
        # Clean headlines
        headlines = [h.strip() for h in headlines if h.strip()]
        
        if not headlines:
            logger.error("No valid headlines after cleaning")
            return None
            
        # Get total number of topics from model
        nr_topics = len(topic_model.get_topic_info())
        
        # Transform headlines using pretrained BERTopic model
        topics, probs = topic_model.transform(headlines)
        
        # Initialize full probability array
        full_probs = np.zeros((len(headlines), nr_topics))
        
        # Map probabilities to correct topics
        if isinstance(probs, (float, np.float64)):
            # Single probability case
            topic_idx = topics if isinstance(topics, (int, np.integer)) else topics[0]
            full_probs[0, topic_idx] = probs
        else:
            probs = np.array(probs)
            # Multiple probabilities case
            for i, (topic, prob) in enumerate(zip(topics, probs)):
                if isinstance(prob, np.ndarray):
                    for t, p in zip(topic, prob):
                        if 0 <= t < nr_topics:  # Ensure valid topic index
                            full_probs[i, t] = p
                else:
                    if 0 <= topic < nr_topics:  # Ensure valid topic index
                        full_probs[i, topic] = prob
                
        # Mean pool probabilities
        mean_probs = np.mean(full_probs, axis=0)
        
        # Verify we have the right number of topics
        if mean_probs.shape[0] != nr_topics:
            logger.error(f"Wrong number of topics in output: got {mean_probs.shape[0]}, expected {nr_topics}")
            return None
        
        # Convert to tensor
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
    
    # Create single instance of GoogleNews with shorter timeout
    gn = GoogleNews(lang='en', country='US')
    all_titles = []  # Changed from all_sentences
    
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
            for entry in search['entries'][:max_articles_per_keyword]:
                try:
                    pub_date = datetime.strptime(entry.published, '%a, %d %b %Y %H:%M:%S %Z')
                    if start_datetime <= pub_date <= end_datetime:
                        # Simply append the entire title after stripping whitespace
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
        
        # Shorter delay between keywords
        time.sleep(1)
    
    print(f"Scraping completed at {datetime.now()}")
    return all_titles  # Returns list of complete titles instead of individual sentences

def create_integrated_tensor(
    stock_symbol: str,
    symbol_index: int,
    current_datetime: datetime,
    bertopic_model_path: str,
    dataset_path: str,
    stock_name,
    stock_industry,
    stock_sector
) -> None:
    try:
        topic_model = BERTopic.load(bertopic_model_path)
        current_time = datetime.now()
        if current_datetime > current_time:
            current_datetime = current_time
        random_time = timedelta(
        hours=random.randint(9, 20),
        minutes=random.randint(0, 59),
        seconds=random.randint(0, 59))
        total_seconds = int(random_time.total_seconds())
        hours, remainder = divmod(total_seconds, 3600)
        minutes, seconds = divmod(remainder, 60)
        str_time = f"{hours:02}:{minutes:02}:{seconds:02}"

        earliest_date = datetime(2023, 12, 1)
        latest_date = min(datetime(2024, 11, 17), current_time - timedelta(days=1))
        
        while True:
                random_date = earliest_date + timedelta(days=random.randint(0, (latest_date - earliest_date).days))
                if random_date.weekday() < 5:
                    break
        str_date = random_date.strftime('%Y-%m-%d')
        stock_data_result = prepare_single_stock_data(
            ticker_symbol=stock_symbol,
            start_datetime=random_date,
            days=10,
            min_points=80
        )
        if stock_data_result is not None:
            stock_data, raw_latest_price = stock_data_result
                        
            
        gt_datetime = random_date + timedelta(days=1)
        gt_value = returngroundtruthstock(stock_symbol, gt_datetime)
        if gt_value.numel() == 0:
            print(f"No ground truth for {stock_symbol}")
            return
        gt_percentage = calculate_percentage_change(raw_latest_price, gt_value.item())
        keywords = [stock_name, stock_industry + ' Industry',]
        print("[    CHECKUP   ]  :" + stock_symbol +  stock_name +stock_industry)
        second_headlines = new_scrape_articles(
            keywords=keywords,end_date= str_date, end_time=str_time , days_back=1, max_articles_per_keyword=10)
        headlines = get_stock_news(symbol=stock_symbol, date=str_date)
        headlines.extend(second_headlines)
        clean_headlines = [x for x in headlines if x]
        clean_headlines = list(set(clean_headlines))
        if not clean_headlines:
            print(f"No headlines found for {stock_symbol}, skipping tensor creation")
            return
            
        news_embedding = process_headlines_with_bertopic(clean_headlines, topic_model)
        if news_embedding is None:
            print(f"BERTopic processing failed for {stock_symbol}, skipping tensor creation")
            return
        news_embedding = news_embedding[:20] if news_embedding is not None else None
        print(f"\nTensor Components for {stock_symbol}:")
        print(f"Ground Truth: {gt_percentage}")
        print(f"Stock Data (first 5): {stock_data[:10].tolist()}")
        print(f"Topic Vector (first 5): {news_embedding[:10].tolist()}\n")
        
        integrated_tensor = torch.cat([
            torch.tensor([gt_percentage], dtype=torch.float32),
            stock_data,
            news_embedding
        ])
        
        filename = f"{stock_symbol}_{current_datetime.strftime('%Y%m%d_%H%M%S')}.pt"
        filepath = os.path.join(dataset_path, filename)
        torch.save(integrated_tensor, filepath)
        
        for file in os.listdir(dataset_path):
            if file.startswith(stock_symbol) and file != filename:
                old_filepath = os.path.join(dataset_path, file)
                os.remove(old_filepath)
    except Exception as e:
        print(f"Error processing {stock_symbol}: {str(e)}")

def main(csv_path: str, bertopic_model_path: str, dataset_path: str,
         stock_names: List[str], stock_industries: List[str], stock_sectors: List[str], stock_symbols: List[str]):
    os.makedirs(dataset_path, exist_ok=True)
    failed_stocks = []
    successful_stocks = []
    Stock_Symbols = stock_symbols
    with tqdm(total=len(Stock_Symbols), desc="Processing stocks") as pbar:
        for i, stock_symbol in enumerate(Stock_Symbols):
            try:
                stock_symbol = stock_symbol.strip().replace("/", "_")
                print(f"\n[     SEARCH FOR   {stock_symbol}    ]")
                stock_name = stock_names[i]
                stock_industry = stock_industries[i]
                stock_sector = stock_sectors[i]
                create_integrated_tensor(
                    stock_symbol=stock_symbol,
                    symbol_index=i,
                    current_datetime=datetime.now(),
                    bertopic_model_path=bertopic_model_path,
                    dataset_path=dataset_path,
                    stock_name=stock_name,
                    stock_industry=stock_industry,
                    stock_sector=stock_sector
                )
                successful_stocks.append(stock_symbol)
            except Exception as e:
                print(f"Failed to process {stock_symbol}: {str(e)}")
                failed_stocks.append(stock_symbol)
            finally:
                pbar.update(1)
                time.sleep(1)
    print(f"\nProcessing complete!")
    print(f"Successful stocks: {len(successful_stocks)}")
    print(f"Failed stocks: {len(failed_stocks)}")
    if failed_stocks:
        print("Failed stock symbols:", failed_stocks)

if __name__ == "__main__":
    csv_path = "/Users/daniellavin/Desktop/proj/MoneyTrainer/Hybrid_stockscreen.csv"
    bertopic_model_path = "/Users/daniellavin/Desktop/proj/MoneyTrainer/x_preprocess/BERTOPIC_MODEL_50.pt"
    Stock_Symbols = []
    Stock_Names = []
    Stock_Sectors = []
    Stock_Industries = []
    attempts = 10
    
    # Read the CSV data
    with open(csv_path, 'r') as file:
        csv_reader = csv.DictReader(file)
        for row in csv_reader:
            Stock_Symbols.append(row['Symbol'])
            Stock_Names.append(row['Name'])
            Stock_Sectors.append(row['Sector'])
            Stock_Industries.append(row['Industry'])
    
    # Convert to numpy arrays for easier shuffling
    data = np.array([Stock_Symbols, Stock_Names, Stock_Sectors, Stock_Industries])
    
    for iterations in range(attempts):
        # Generate a random permutation index
        permutation = np.random.permutation(len(Stock_Symbols))
        
        # Apply the same permutation to all arrays
        shuffled_data = data[:, permutation]
        
        # Unpack the shuffled data
        shuffled_symbols = shuffled_data[0].tolist()
        shuffled_names = shuffled_data[1].tolist()
        shuffled_sectors = shuffled_data[2].tolist()
        shuffled_industries = shuffled_data[3].tolist()
        
        print(f"\nIteration {iterations + 1}/{attempts}")
        print(f"First few symbols after shuffling: {shuffled_symbols[:5]}")
        
        main(csv_path=csv_path, 
             bertopic_model_path=bertopic_model_path, 
             dataset_path="/Users/daniellavin/Desktop/proj/MoneyTrainer/X_findataset",
             stock_names=shuffled_names,
             stock_industries=shuffled_industries,
             stock_sectors=shuffled_sectors,
             stock_symbols=shuffled_symbols)