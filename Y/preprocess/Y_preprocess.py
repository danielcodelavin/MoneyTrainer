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

def calculate_movement_score(prices: pd.Series, reference_price: float) -> float:
    """
    Calculate a composite score for stock movement quality based on:
    - Overall trend (50%): Open to close movement
    - Maximum positive deviation (30%): Highest point reached
    - Maximum negative deviation (20%): Lowest point reached
    
    All movements are capped at ±10% for score calculation
    Returns a score between -1 and 1
    """
    if prices.empty:
        return 0.0
        
    # Calculate percentage changes
    overall_change = (prices.iloc[-1] - prices.iloc[0]) / prices.iloc[0]
    max_positive = (prices.max() - prices.iloc[0]) / prices.iloc[0]
    max_negative = (prices.min() - prices.iloc[0]) / prices.iloc[0]
    
    # Cap all values at ±10%
    overall_change = np.clip(overall_change, -0.10, 0.10)
    max_positive = np.clip(max_positive, 0, 0.10)
    max_negative = np.clip(max_negative, -0.10, 0)
    
    # Scale to [-1, 1] range
    trend_score = overall_change / 0.10  # 50% weight
    positive_score = max_positive / 0.10  # 25% weight
    negative_score = max_negative / 0.10  # 25% weight
    
    # Calculate weighted score
    final_score = (0.5 * trend_score) + (0.25 * positive_score) + (0.25 * negative_score)
    
    return final_score

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
        time.sleep(0.3)  # Rate limiting
    
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





def returngroundtruthstock(stock_symbol: str, target_datetime: datetime) -> Optional[float]:
    """
    Get stock prices for the specific target date and calculate a movement quality score.
    Returns a score between -1 and 1 representing the quality of price movement,
    or None if no valid data is found for that day.
    """
    try:
        stock = yf.Ticker(stock_symbol)
        
        # Get prices for just the target day
        data = stock.history(
            start=target_datetime.strftime('%Y-%m-%d'),
            end=(target_datetime + timedelta(days=1)).strftime('%Y-%m-%d'),
            interval="60m",
            prepost=True
        )
        
        if not data.empty and 'Close' in data.columns:
            prices = data['Close'].dropna()
            if len(prices) > 0 and validate_raw_data(prices):
                movement_score = calculate_movement_score(prices, prices.iloc[0])
                print(f"Calculated movement score {movement_score:.3f} for {stock_symbol} on {target_datetime.strftime('%Y-%m-%d')}")
                return movement_score
        
        print(f"No valid prices found for {stock_symbol} on {target_datetime.strftime('%Y-%m-%d')}")
        return None
        
    except Exception as e:
        print(f"Error getting prices for {stock_symbol}: {str(e)}")
        return None
    


def process_headlines_with_bertopic(headlines: List[str], topic_model: BERTopic) -> Optional[torch.Tensor]:
    """
    Process headlines using BERTopic model and return mean-pooled probability distribution.
    Forces output to be exactly 100 topics.
    """
    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger(__name__)
    
    try:
        # Input validation
        if not headlines or not isinstance(headlines, list):
            logger.error("Invalid headlines input: must be non-empty list of strings")
            return None
            
        # Clean headlines
        headlines = [h.strip() for h in headlines if h.strip()]
        
        if not headlines:
            logger.error("No valid headlines after cleaning")
            return None
            
        # Get topics and probabilities
        topics, initial_probs = topic_model.transform(headlines)
        logger.info(f"Initial transform shape: {initial_probs.shape}")
        
        # Create full probability array
        full_probs = np.zeros((len(headlines), 100))
        
        # Map existing probabilities to correct indices
        topic_info = topic_model.get_topic_info()
        topic_mapping = {i: idx for idx, i in enumerate(sorted(topic_info['Topic'].unique()))}
        
        # Fill in probabilities for each topic
        for i, topic in enumerate(sorted(topic_info['Topic'].unique())):
            if i < initial_probs.shape[1]:  # Make sure we don't go out of bounds
                full_probs[:, i] = initial_probs[:, i]
        
        # Mean pool probability distributions
        mean_probs = np.mean(full_probs, axis=0)
        
        # Normalize probabilities
        prob_sum = np.sum(mean_probs)
        if not np.isclose(prob_sum, 1.0, rtol=1e-5):
            logger.warning(f"Normalizing probabilities (sum was {prob_sum})")
            mean_probs = mean_probs / (prob_sum if prob_sum > 0 else 1.0)
            
        # Convert to tensor
        mean_probs_tensor = torch.from_numpy(mean_probs).float()
        
        # Validation
        assert mean_probs_tensor.shape[0] == 100, f"Wrong number of topics: {mean_probs_tensor.shape[0]}"
        assert torch.isclose(mean_probs_tensor.sum(), torch.tensor(1.0), rtol=1e-5), "Probabilities don't sum to 1"
        assert not torch.any(mean_probs_tensor < 0), "Found negative probabilities"
        
        logger.info(f"Successfully processed {len(headlines)} headlines into {mean_probs_tensor.shape[0]} topic probabilities")
        return mean_probs_tensor
        
    except Exception as e:
        logger.error(f"Error processing headlines: {str(e)}")
        traceback.print_exc()
        return None



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
        #TIME GEN
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

        earliest_date = datetime(2024, 11, 2)
        latest_date = datetime(2025, 1, 17)
        
        while True:
            random_date = earliest_date + timedelta(days=random.randint(0, (latest_date - earliest_date).days))
            if random_date.weekday() < 4 or random_date.weekday() == 6:
                break
        str_date = random_date.strftime('%Y-%m-%d')
        
        #MONEY GEN
        stock_data_result , raw_boy = prepare_single_stock_data(
            ticker_symbol=stock_symbol,
            start_datetime=random_date,
            days=10,
            min_points=80
        )
        if stock_data_result is not None:
            stock_data = stock_data_result
            
            gt_datetime = random_date + timedelta(days=1)
            movement_score = returngroundtruthstock(stock_symbol, gt_datetime)
            if movement_score is None:
                print(f"No ground truth movement score for {stock_symbol}")
                return

            #HEADLINE GEN
            keywords = [stock_name, stock_industry + ' Industry']
            print("[    CHECKUP   ]  :" + stock_symbol + stock_name + stock_industry)
            
            headlines = get_stock_news(symbol=stock_symbol, date=str_date)
            
            clean_headlines = [x for x in headlines if x]
            clean_headlines = list(set(clean_headlines))
            if not clean_headlines:
                print(f"No headlines found for {stock_symbol}, skipping tensor creation")
                return
                
            news_embedding = process_headlines_with_bertopic(clean_headlines, topic_model)
            if news_embedding is None:
                print(f"BERTopic processing failed for {stock_symbol}, skipping tensor creation")
                return

            print(f"\nTensor Components for {stock_symbol}:")
            print(f"Movement Score: {movement_score}")
            print(f"Stock Data (first 5): {stock_data[:10].tolist()}")
            print(f"Topic Vector (first 5): {news_embedding[:10].tolist()}\n")
            
            integrated_tensor = torch.cat([
                torch.tensor([movement_score], dtype=torch.float32),
                stock_data,
                news_embedding
            ])
            
            filename = f"{stock_symbol}_{random_date.strftime('%Y%m%d')}_{str_time.replace(':', '')}.pt"
            filepath = os.path.join(dataset_path, filename)
            torch.save(integrated_tensor, filepath)
        
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
                time.sleep(2)
    print(f"\nProcessing complete!")
    print(f"Successful stocks: {len(successful_stocks)}")
    print(f"Failed stocks: {len(failed_stocks)}")
    if failed_stocks:
        print("Failed stock symbols:", failed_stocks)

if __name__ == "__main__":
    csv_path = "/Users/daniellavin/Desktop/proj/MoneyTrainer/Hybrid_stockscreen.csv"
    bertopic_model_path = "/Users/daniellavin/Desktop/proj/MoneyTrainer/Y/preprocess/Y100_BERTOPIC.pt"
    Stock_Symbols = []
    Stock_Names = []
    Stock_Sectors = []
    Stock_Industries = []
    attempts = 300
    
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
             dataset_path="/Users/daniellavin/Desktop/proj/MoneyTrainer/Y/y_100_val",
             stock_names=shuffled_names,
             stock_industries=shuffled_industries,
             stock_sectors=shuffled_sectors,
             stock_symbols=shuffled_symbols)
        
        # TRAINING TIME FRAME #
        # 2024-01-15 to 2024-11-1

        # Validation Time Frame #
        # 2024-11-3 to 2025-01-16