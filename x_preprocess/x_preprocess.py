import torch
import pandas as pd
import yfinance as yf
from datetime import datetime, timedelta
from typing import List, Tuple, Dict, Optional
import os
import random
from tqdm import tqdm
from gnews import GoogleNews
import time
from sentence_transformers import SentenceTransformer
from bertopic import BERTopic
import numpy as np
import csv

def validate_raw_data(data: pd.Series) -> bool:
    """
    Validate raw stock data for inf, nan, zero values.
    """
    if data is None or data.empty:
        return False
    
    # Check for inf values
    if np.isinf(data).any():
        return False
    
    # Check for nan values
    if np.isnan(data).any():
        return False
    
    # Check for zero values
    if (data <= 0).any():
        return False
    
    return True

def extract_symbols_from_csv(filepath: str) -> List[str]:
    """Extract stock symbols from CSV file."""
    with open(filepath, 'r') as file:
        reader = csv.reader(file)
        next(reader)  # Skip header
        symbols = [row[0].strip() for row in reader]
    return symbols

def calculate_percentage_change(latest_value: float, gt_value: float) -> float:
    """Calculate percentage change between latest value and ground truth."""
    return (gt_value - latest_value) / latest_value

def prepare_single_stock_data(ticker_symbol: str, start_datetime: datetime, days: int = 5, min_points: int = 80) -> Optional[torch.Tensor]:
    """Fetch and prepare stock data."""
    ticker = yf.Ticker(ticker_symbol)
    end_datetime = start_datetime
    
    all_data = []
    valid_days_collected = 0
    days_checked = 0

    while valid_days_collected < days and days_checked < 12:
        day_end = end_datetime - timedelta(days=days_checked)
        day_start = day_end - timedelta(days=1)
        
        try:
            data = ticker.history(start=day_start, end=day_end, interval="60m", prepost=True, auto_adjust=False)
            
            if not data.empty:
                # Validate the data before adding
                if validate_raw_data(data['Close']):
                    all_data.append(data['Close'])
                    valid_days_collected += 1
        except Exception as e:
            print(f"Error fetching data for {ticker_symbol}: {e}")
        
        days_checked += 1

    if not all_data:
        return None

    combined_data = pd.concat(all_data)
    
    if len(combined_data) >= min_points:
        tensor_data = torch.tensor(combined_data.values, dtype=torch.float32)
        
        # Normalize the data
        mean = tensor_data.mean()
        std = tensor_data.std()
        if std == 0:
            print(f"Zero standard deviation for {ticker_symbol}")
            return None
            
        normalized_tensor = (tensor_data - mean) / std
        return normalized_tensor
        
    return None

def returngroundtruthstock(stock_symbol: str, start_datetime: str, max_days: int = 4) -> torch.Tensor:
    try:
        stock = yf.Ticker(stock_symbol)
        current_datetime = pd.to_datetime(start_datetime)

        all_data = []
        valid_days_collected = 0
        days_checked = 0

        while valid_days_collected == 0 and days_checked < max_days:
            day_start = current_datetime + pd.Timedelta(days=days_checked)
            day_end = day_start + pd.Timedelta(days=1)

            print(f"Fetching data for {day_start.strftime('%Y-%m-%d')} to {day_end.strftime('%Y-%m-%d')}...")

            data = stock.history(start=day_start.strftime('%Y-%m-%d'),
                               end=day_end.strftime('%Y-%m-%d'),
                               interval="60m", prepost=True, auto_adjust=False)

            if not data.empty:
                all_data.append(data['Close'])
                valid_days_collected = 1
                print(f"Valid data found for {day_start} to {day_end}.")
            else:
                print(f"No data available for {day_start} to {day_end}. Checking next day.")
            
            days_checked += 1

        if all_data:
            combined_data = pd.concat(all_data)
            print(f"Returning first available closing price: {combined_data.iloc[0]}")
            return torch.tensor([combined_data.iloc[0]], dtype=torch.float32)
        else:
            print(f"No valid data found for {stock_symbol} within {max_days} days. Returning empty tensor.")
            return torch.tensor([])

    except Exception as e:
        print(f"Error fetching data for {stock_symbol}: {str(e)}")
        return torch.tensor([])

def scrape_articles(keywords: List[str], target_date: str, target_time: str, max_articles_per_keyword: int = 10) -> List[str]:
    """Scrape news articles and return headlines."""
    target_datetime = datetime.strptime(f"{target_date} {target_time}", '%Y-%m-%d %H:%M:%S')
    day_before_datetime = target_datetime - timedelta(days=1)
    
    gn = GoogleNews(lang='en', country='US')
    all_sentences = []
    
    for keyword in tqdm(keywords, desc="Processing keywords", leave=False):
        search = gn.search(keyword, from_=target_date, to_=target_date)
        article_count = 0
        articles_from_target_date = []
        
        for entry in search['entries']:
            pub_date = datetime.strptime(entry.published, '%a, %d %b %Y %H:%M:%S %Z')
            if pub_date.date() == target_datetime.date():
                sentences = [s.strip() for s in 
                           entry.title.replace('! ', '!SPLIT')
                           .replace('? ', '?SPLIT')
                           .replace('. ', '.SPLIT')
                           .split('SPLIT') 
                           if s.strip()]
                articles_from_target_date.extend(sentences)
                article_count += 1
                if article_count >= max_articles_per_keyword:
                    break
            time.sleep(0.1)
        
        if article_count < max_articles_per_keyword:
            search = gn.search(keyword,
                             from_=day_before_datetime.strftime('%Y-%m-%d'),
                             to_=day_before_datetime.strftime('%Y-%m-%d'))
            
            for entry in search['entries']:
                if article_count >= max_articles_per_keyword:
                    break
                    
                pub_date = datetime.strptime(entry.published, '%a, %d %b %Y %H:%M:%S %Z')
                if pub_date.date() == day_before_datetime.date():
                    sentences = [s.strip() for s in 
                               entry.title.replace('! ', '!SPLIT')
                               .replace('? ', '?SPLIT')
                               .replace('. ', '.SPLIT')
                               .split('SPLIT') 
                               if s.strip()]
                    articles_from_target_date.extend(sentences)
                    article_count += 1
                time.sleep(0.1)
        
        all_sentences.extend(articles_from_target_date)
    
    return all_sentences

def process_headlines_with_bertopic(headlines: List[str], topic_model: BERTopic) -> torch.Tensor:
    """Process headlines using pre-trained BERTopic model and return normalized mean-pooled embeddings."""
    if not headlines:
        return torch.zeros(50)
    
    topics, probs = topic_model.transform(headlines)
    mean_probs = torch.tensor(probs.mean(axis=0), dtype=torch.float32)
    
    # Normalize the probabilities
    mean = mean_probs.mean()
    std = mean_probs.std()
    if std == 0:
        return mean_probs  # Return unnormalized if std is 0 since it's already uniform
    
    normalized_probs = (mean_probs - mean) / std
    return normalized_probs






def create_integrated_tensor(
    stock_symbol: str,
    symbol_index: int,  # Added parameter for index
    current_datetime: datetime,
    bertopic_model_path: str,
    dataset_path: str,
    stock_names: List[str],
    stock_industries: List[str],
    stock_sectors: List[str]
) -> None:
    """Create and save integrated tensor combining stock, GT, and news data."""
    
    # Load BERTopic model
    topic_model = BERTopic.load(bertopic_model_path)
    
    # Generate random time for news scraping
    random_time = timedelta(
        hours=random.randint(9, 20),
        minutes=random.randint(0, 59),
        seconds=random.randint(0, 59)
    )
    
    str_date = current_datetime.strftime('%Y-%m-%d')
    total_seconds = int(random_time.total_seconds())
    hours, remainder = divmod(total_seconds, 3600)
    minutes, seconds = divmod(remainder, 60)
    str_time = f"{hours:02}:{minutes:02}:{seconds:02}"
    
    # Prepare stock data
    stock_data = prepare_single_stock_data(
        ticker_symbol=stock_symbol,
        start_datetime=current_datetime,
        days=6,
        min_points=45
    )
    
    if stock_data is None:
        print(f"Failed to get valid stock data for {stock_symbol}")
        return
    
    # Check stock data length and trim if necessary
    if len(stock_data) < 100:
        print(f"Insufficient stock data length for {stock_symbol}: {len(stock_data)}")
        return
    elif len(stock_data) > 100:
        stock_data = stock_data[-100:]
    
    # Get ground truth
    gt_value = returngroundtruthstock(stock_symbol, current_datetime.strftime('%Y-%m-%d'))
    if gt_value.numel() == 0:
        print(f"Failed to get ground truth for {stock_symbol}")
        return
        
    latest_value = stock_data[-1].item()
    gt_percentage = calculate_percentage_change(latest_value, gt_value.item())
    
    # Get news data using the index to access corresponding lists
    keywords = [
        stock_names[symbol_index],
        stock_names[symbol_index] + ' Stock',
        stock_industries[symbol_index] + ' Industry',
        stock_sectors[symbol_index] + ' Stocks'
    ]
    
    headlines = scrape_articles(
        keywords=keywords,
        target_date=str_date,
        target_time=str_time,
        max_articles_per_keyword=25
    )
    
    # Process headlines with BERTopic
    news_embedding = process_headlines_with_bertopic(headlines, topic_model)
    
    # Create integrated tensor
    integrated_tensor = torch.cat([
        torch.tensor([gt_percentage]),
        stock_data,
        news_embedding
    ])
    
    # Save tensor
    filename = f"{stock_symbol}_{current_datetime.strftime('%Y%m%d_%H%M%S')}.pt"
    filepath = os.path.join(dataset_path, filename)
    torch.save(integrated_tensor, filepath)
    
    # If duplicate exists, replace it
    for file in os.listdir(dataset_path):
        if file.startswith(stock_symbol) and file != filename:
            old_filepath = os.path.join(dataset_path, file)
            os.remove(old_filepath)

def main(
    csv_path: str,
    bertopic_model_path: str,
    dataset_path: str,
    stock_names: List[str],
    stock_industries: List[str],
    stock_sectors: List[str]
):
    """Main function to run the pipeline."""
    os.makedirs(dataset_path, exist_ok=True)
    
    # Process all stocks
    for i, stock_symbol in enumerate(tqdm(Stock_Symbols, desc="Processing stocks")):
        stock_symbol = stock_symbol.strip().replace("/", "_")
        create_integrated_tensor(
            stock_symbol=stock_symbol,
            symbol_index=i,  # Pass the index
            current_datetime=datetime.now(),
            bertopic_model_path=bertopic_model_path,
            dataset_path=dataset_path,
            stock_names=stock_names,
            stock_industries=stock_industries,
            stock_sectors=stock_sectors
        )
    
    # Delete the CSV file after processing
    try:
        os.remove(csv_path)
        print(f"Successfully deleted {csv_path}")
    except Exception as e:
        print(f"Error deleting CSV file: {e}")

if __name__ == "__main__":
    # Example usage
    csv_path= "/path/to/cleaned_stockscreen.csv"
    bertopic_model_path = "/path/to/bertopic_model"
    Stock_Symbols = []
    Stock_Names = []
    Stock_Sectors = []
    Stock_Industries = []

    with open(csv_path, 'r') as file:
        csv_reader = csv.DictReader(file)
        for row in csv_reader:
            Stock_Symbols.append(row['Symbol'])
            Stock_Names.append(row['Name'])
            Stock_Sectors.append(row['Sector'])
            Stock_Industries.append(row['Industry'])

    
    main(csv_path=csv_path, bertopic_model_path=bertopic_model_path, dataset_path="/path/to/dataset", stock_names=Stock_Names, stock_industries=Stock_Industries, stock_sectors=Stock_Sectors)

