import torch
import numpy as np
import yfinance as yf
from datetime import datetime, timedelta
import csv
import logging
import os
import pandas as pd
import hashlib


logging.basicConfig(level=logging.INFO)


def validate_and_clean_tensors(stock_dir, gt_dir):
    # Ensure the directories exist
    if not os.path.exists(stock_dir) or not os.path.exists(gt_dir):
        raise ValueError("One or both directories do not exist")

    # Get list of files in both directories
    stock_files = set(f for f in os.listdir(stock_dir) if f.endswith('.pt'))
    gt_files = set(f for f in os.listdir(gt_dir) if f.endswith('_GT.pt'))

    # Check for files in stock dir without GT counterparts
    for stock_file in stock_files:
        gt_file = stock_file.replace('.pt', '_GT.pt')
        if gt_file not in gt_files:
            print(f"No GT counterpart for {stock_file}. Deleting stock file.")
            os.remove(os.path.join(stock_dir, stock_file))
        else:
            # If GT counterpart exists, validate the tensor
            gt_path = os.path.join(gt_dir, gt_file)
            stock_path = os.path.join(stock_dir, stock_file)
            gt_tensor = torch.load(gt_path)

            if gt_tensor.shape[0] != 1:
                print(f"Invalid tensor found in {gt_file}. Deleting both GT and stock files.")
                os.remove(gt_path)
                os.remove(stock_path)
            else:
                print(f"Valid tensor pair: {stock_file} and {gt_file}")

    # Check for files in GT dir without stock counterparts
    for gt_file in gt_files:
        stock_file = gt_file.replace('_GT.pt', '.pt')
        if stock_file not in stock_files:
            print(f"No stock counterpart for {gt_file}. Deleting GT file.")
            os.remove(os.path.join(gt_dir, gt_file))

    print("Validation and cleaning complete.")


def encode_and_attach(label: str, vector: torch.Tensor) -> torch.Tensor:
    # Hash the label using SHA-256
    hash_object = hashlib.sha256(label.encode())
    hash_hex = hash_object.hexdigest()
    
    # Convert the first 8 characters of the hash to an integer
    hash_int = int(hash_hex[:8], 16)
    
    # Normalize the integer to a float between 0 and 1
    encoded_label = hash_int / (2**32 - 1)  # 2^32 - 1 is the maximum value for 8 hex digits
    
    # Create a tensor from the encoded label
    encoded_label_tensor = torch.tensor([encoded_label], dtype=torch.float)
    
    # Concatenate the encoded label with the input vector
    result = torch.cat([encoded_label_tensor, vector])
    
    return result


def prepare_single_stock_data(ticker_symbol: str, start_datetime: str, days: int = 7, min_points: int = 80):
    """
    Fetches stock data in batches (24 hourly data points per day) for a specified number of valid days
    (i.e., days with non-zero data) from the given starting date and time, with a 60-minute interval.
    The final data is concatenated into a tensor, and if the total number of points is less than the 
    'min_points' threshold, an empty tensor is returned.
    
    Parameters:
    - ticker_symbol (str): Stock ticker symbol to fetch data for.
    - start_datetime (str): Start date and time in 'YYYY-MM-DD HH:MM:SS' format.
    - days (int): The number of valid (non-zero) days to collect data for. Default is 5 days.
    - min_points (int): The minimum number of data points required in the final tensor.
    
    Returns:
    - torch.Tensor: A 1D tensor with the concatenated data, or an empty tensor if the
      total points are less than 'min_points'.
    """
    ticker = yf.Ticker(ticker_symbol)
    end_datetime = pd.to_datetime(start_datetime)
    print(f"[     SEARCH FOR   {ticker_symbol}    ]")
    all_data = []
    valid_days_collected = 0
    days_checked = 0  # Keep track of how many total days we’ve looked at

    # Keep fetching until we get the required number of valid days
    while valid_days_collected < days:
        day_end = end_datetime - pd.Timedelta(days=days_checked)
        day_start = day_end - pd.Timedelta(days=1)
        
        # try:
        #print(f"Fetching data for {day_start} to {day_end}...")
        # Fetch hourly data for the 24 hours before the given time
        data = ticker.history(start=day_start, end=day_end, interval="60m", prepost=True, auto_adjust=False)

        if not data.empty:  # Only append if we got valid data
            all_data.append(data['Close'])
            valid_days_collected += 1  # Increment the number of valid days
        #     else:
        #        print(f"No data available for {day_start} to {day_end}. Skipping day.")

        # except Exception as e:
        #     print(f"Error fetching data fosr stock {ticker_symbol} from {day_start} to {day_end}: {e}")

        # Move on to the next day (whether data was valid or not)
        days_checked += 1

        # Break out of the loop if we arent finding data within a max amount of time. reduce this number for faster performance
        if days_checked >= days + 4:
            break
    # Concatenate the collected data
    if not all_data:  # Check if all_data is empty
        print(f"No valid data collected. Returning an empty tensor.")
        return torch.tensor([])

    # Concatenate the collected data
    combined_data = pd.concat(all_data)

    # Ensure the final tensor has at least 'min_points' data points
    if len(combined_data) >= min_points:
        tensor_data = torch.tensor(combined_data.values, dtype=torch.float32)
        return tensor_data
    else:
        print(f"Insufficient data. Expected at least {min_points} points, got {len(combined_data)}.")
        return torch.tensor([])


def returngroundtruthstock(stock_symbol: str, start_datetime: str, max_days: int = 4) -> torch.Tensor:
    try:
        stock = yf.Ticker(stock_symbol)
        current_datetime = pd.to_datetime(start_datetime)

        all_data = []
        valid_days_collected = 0
        days_checked = 0

        # Keep fetching data until we get at least one valid day or exceed the max_days limit
        while valid_days_collected == 0 and days_checked < max_days:
            day_start = current_datetime + pd.Timedelta(days=days_checked)
            day_end = day_start + pd.Timedelta(days=1)

            # Use only the date part when fetching data
            print(f"Fetching data for {day_start.strftime('%Y-%m-%d')} to {day_end.strftime('%Y-%m-%d')}...")

            # Fetch hourly data for the 24 hours following the current day
            data = stock.history(start=day_start.strftime('%Y-%m-%d'),
                                 end=day_end.strftime('%Y-%m-%d'),
                                 interval="60m", prepost=True, auto_adjust=False)

            if not data.empty:  # If valid data is found
                all_data.append(data['Close'])
                valid_days_collected = 1  # Set valid day counter to 1 as soon as we get data
                print(f"Valid data found for {day_start} to {day_end}.")
            else:
                print(f"No data available for {day_start} to {day_end}. Checking next day.")
            
            days_checked += 1

        if all_data:  # Check if we found any valid data
            combined_data = pd.concat(all_data)
            print(f"Returning first available closing price: {combined_data.iloc[0]}")
            return torch.tensor([combined_data.iloc[0]], dtype=torch.float32)
        else:
            print(f"No valid data found for {stock_symbol} within {max_days} days. Returning empty tensor.")
            return torch.tensor([])

    except Exception as e:
        print(f"Error fetching data for {stock_symbol}: {str(e)}")
        return torch.tensor([])



def extract_symbols_from_csv(file_path):
    #extracts all stock symbols from the csv
    symbols = []
    with open(file_path, mode='r') as file:
        csv_reader = csv.reader(file)
        for row in csv_reader:
            if row:  # Check if the row is not empty
                symbol = row[0]  # The symbol is the first element in the row
                symbols.append(symbol)
    return symbols


def select_random_symbols(symbols: list[str], num_symbols: int) -> list[str]:
    # Selects 'num_symbols' random symbols from the 'symbols' list
    return np.random.choice(symbols, size=num_symbols, replace=False).tolist()








# Example usage
# symbols = ["AAPL", "GOOGL", "MSFT", "AMZN", "FB"]  # Example stock symbols
# end_date = "2024-07-24"  # Use the current date from the conversation

# labeled_stock_data = prepare_stock_data(symbols, end_date)

# print(f"Resulting tensor shape: {labeled_stock_data.shape}")
# print(f"First few rows:\n{labeled_stock_data[:3, :5]}")  # Show first 3 stocks, first 5 columns