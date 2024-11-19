import torch
import numpy as np
import yfinance as yf
from datetime import datetime, timedelta
import csv
import logging
import os
import pandas as pd
import random
from moneytensorgen import prepare_single_stock_data




import csv
import torch
import pandas as pd
import yfinance as yf
from typing import Dict, Any

def cleaner(old_csv_path: str, new_csv_path: str) -> None:
    """
    Cleans stock data by validating tensors and writing only valid entries to a new CSV file.
    
    Parameters:
    - old_csv_path (str): Path to the input CSV file
    - new_csv_path (str): Path to the output CSV file where valid data will be written
    """
    # Keep track of statistics
    total_processed = 0
    total_valid = 0
    rejection_reasons: Dict[str, int] = {
        "empty_tensor": 0,
        "contains_nan": 0,
        "contains_zero": 0,
        "contains_inf": 0,
        "too_short": 0
    }
    # generate a random datetime

    earliest_date = datetime(2022, 12, 1)
        
    latest_date = datetime(2024, 11, 15)

    while True:
        random_date = earliest_date + timedelta(days=random.randint(0, (latest_date - earliest_date).days))
        if random_date.weekday() < 5:  # 0 = Monday, 4 = Friday
            break

    # Generate a random time within a day
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
    
    
    # Combine random_date and random_time to get a full random datetime
    random_datetime = datetime.combine(random_date, datetime.min.time()) + random_time

    # Convert to string format (for example, '%Y-%m-%d %H:%M:%S')
    datetime_str = random_datetime.strftime('%Y-%m-%d %H:%M:%S')

    # Parse the datetime string back to a datetime object (if needed)
    parsed_datetime = datetime.strptime(datetime_str, '%Y-%m-%d %H:%M:%S')






    # Read the header from the old CSV to maintain the same structure
    with open(old_csv_path, 'r') as old_file:
        csv_reader = csv.DictReader(old_file)
        fieldnames = csv_reader.fieldnames

        # Create the new CSV file with the same headers
        with open(new_csv_path, 'w', newline='') as new_file:
            csv_writer = csv.DictWriter(new_file, fieldnames=fieldnames)
            csv_writer.writeheader()

            # Process each row
            for row in csv_reader:
                total_processed += 1
                symbol = row['Symbol']
                
                try:
                    # Get tensor data (you'll add date parameters later)
                    tensor_data = prepare_single_stock_data(symbol, parsed_datetime, 4, 10)
                    
                    # Validate the tensor
                    validation_result = validate_tensor(tensor_data)
                    
                    if validation_result['is_valid']:
                        # If tensor is valid, write the row to the new CSV
                        csv_writer.writerow(row)
                        total_valid += 1
                        print(f"// // // // // //Accepted {symbol}")
                    else:
                        # Update rejection statistics
                        for reason in validation_result['reasons']:
                            rejection_reasons[reason] += 1
                        
                        print(f"\\ \\ \\ \\ \\ \\Rejected {symbol}: {', '.join(validation_result['reasons'])}")
                
                except Exception as e:
                    print(f"Error processing {symbol}: {str(e)}")

    # Print summary statistics
    print("\nProcessing Summary:")
    print(f"Total stocks processed: {total_processed}")
    print(f"Valid stocks: {total_valid}")
    print(f"Rejected stocks: {total_processed - total_valid}")
    print("\nRejection Reasons:")
    for reason, count in rejection_reasons.items():
        print(f"- {reason}: {count}")

def validate_tensor(tensor: torch.Tensor) -> Dict[str, Any]:
    """
    Validates a tensor against multiple criteria.
    
    Parameters:
    - tensor (torch.Tensor): The tensor to validate
    
    Returns:
    - Dict containing validation result and reasons for rejection if any
    """
    result = {
        'is_valid': True,
        'reasons': []
    }
    
    # Check for empty tensor
    if tensor.nelement() == 0:
        result['is_valid'] = False
        result['reasons'].append('empty_tensor')
        return result
    
    # Check for NaN values
    if torch.isnan(tensor).any():
        result['is_valid'] = False
        result['reasons'].append('contains_nan')
    
    # Check for zero values
    if (tensor == 0).any():
        result['is_valid'] = False
        result['reasons'].append('contains_zero')

    #check for length
    if (tensor.numel() < 10):
        result['is_valid'] = False
        result['reasons'].append('too_short')
    
    # Check for infinite values
    if torch.isinf(tensor).any():
        result['is_valid'] = False
        result['reasons'].append('contains_inf')
    
    return result

def main():
    old_csv_path = '/Users/daniellavin/Desktop/proj/MoneyTrainer/stockscreen.csv'
    new_csv_path = '/Users/daniellavin/Desktop/proj/MoneyTrainer/X_cleaned_stockscreen.csv'
    cleaner(old_csv_path, new_csv_path)
    


if __name__ == "__main__":
    main()