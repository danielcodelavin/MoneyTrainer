import torch
import numpy as np
import yfinance as yf
from datetime import datetime, timedelta
import csv
import logging
import os
import pandas as pd
from finhub_scrape import get_stock_news
import random
import time
from moneytensorgen import prepare_single_stock_data




import csv
import torch
import pandas as pd
import yfinance as yf
from typing import List, Tuple, Dict, Optional
import os
from typing import List, Optional
from typing import Dict, Any

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
    


def cleaner(old_csv_path: str, new_csv_path: str) -> None:
    """
    Cleans stock data by validating tensors and writing only valid entries to a new CSV file.
    Supports resuming from the last processed symbol.
    
    Parameters:
    - old_csv_path (str): Path to the input CSV file
    - new_csv_path (str): Path to the output CSV file where valid data will be written
    """
    # Get the last processed symbol from the new CSV if it exists
    last_processed_symbol = None
    resume_mode = os.path.exists(new_csv_path)
    
    if resume_mode:
        try:
            with open(new_csv_path, 'r') as file:
                csv_reader = csv.DictReader(file)
                # Get the last row's symbol
                for row in csv_reader:
                    last_processed_symbol = row['Symbol']
        except Exception as e:
            print(f"Error reading existing file: {str(e)}")
            return

    # Keep track of statistics
    total_processed = 0
    total_valid = 0
    
    # Generate random datetime logic (unchanged)
    earliest_date = datetime(2024, 1, 1)
    latest_date = datetime(2024, 11, 20)

    while True:
        random_date = earliest_date + timedelta(days=random.randint(0, (latest_date - earliest_date).days))
        if random_date.weekday() < 5:
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
    
    random_datetime = datetime.combine(random_date, datetime.min.time()) + random_time
    datetime_str = random_datetime.strftime('%Y-%m-%d %H:%M:%S')
    parsed_datetime = datetime.strptime(datetime_str, '%Y-%m-%d %H:%M:%S')

    # Read the old CSV
    with open(old_csv_path, 'r') as old_file:
        csv_reader = csv.DictReader(old_file)
        fieldnames = csv_reader.fieldnames

        # Open new file in append mode if resuming, write mode if starting fresh
        file_mode = 'a' if resume_mode else 'w'
        with open(new_csv_path, file_mode, newline='') as new_file:
            csv_writer = csv.DictWriter(new_file, fieldnames=fieldnames)
            
            # Write header only if starting fresh
            if not resume_mode:
                csv_writer.writeheader()

            # Flag to track when to start processing
            start_processing = not resume_mode

            # Process each row
            for row in csv_reader:
                symbol = row['Symbol']
                
                # Skip until we find the last processed symbol
                if not start_processing:
                    if symbol == last_processed_symbol:
                        start_processing = True
                    continue
                
                total_processed += 1
               # time.sleep(1)
                
                try:
                    news_data = get_stock_news(symbol=symbol, date=str_date)
                    stock_data, floaty = prepare_single_stock_data(
                        ticker_symbol=symbol, 
                        start_datetime=parsed_datetime, 
                        days=5, 
                        min_points=33
                    )
                    
                    if news_data is not None and len(news_data) > 0 and stock_data is not None and len(stock_data) > 0:
                        csv_writer.writerow(row)
                        total_valid += 1
                        print(f"// // // // // //Accepted {symbol}")
                    else:
                        print("NO TO     " + symbol)
                
                except Exception as e:
                    print(f"Error processing {symbol}: {str(e)}")

    # Print summary statistics
    print("\nProcessing Summary:")
    print(f"Total stocks processed: {total_processed}")
    print(f"Valid stocks: {total_valid}")
    print(f"Rejected stocks: {total_processed - total_valid}")

def main():
    old_csv_path = '/Users/daniellavin/Desktop/proj/MoneyTrainer/stockscreen.csv'
    new_csv_path = '/Users/daniellavin/Desktop/proj/MoneyTrainer/Hybrid_stockscreen.csv'
    cleaner(old_csv_path, new_csv_path)
    


if __name__ == "__main__":
    main()