import yfinance as yf
from datetime import datetime, timedelta
import os
import random
import csv
import torch
from moneytensorgen import prepare_single_stock_data , extract_symbols_from_csv , returngroundtruthstock,encode_and_attach, validate_and_clean_tensors
from postprocessing import process_GT_stock_torch_files, process_stock_torch_files

def prepare_dataset(passes:int, earliest_date: datetime, latest_date: datetime):
    for i in range(passes):
            # generate random date and time
            # generate random date and time, the day is imposed to be a weekday
            # the time is imposed between 9am to 9pm
            while True:
                random_date = earliest_date + timedelta(days=random.randint(0, (latest_date - earliest_date).days))
                if random_date.weekday() < 5:  # 0 = Monday, 4 = Friday
                    break

            # Generate a random time within a day
            random_time = timedelta(
                        hours=random.randint(9, 20),  # 9 AM to 9 PM (20 = 8:59 PM, end of the day)
                        minutes=random.randint(0, 59),
                        seconds=random.randint(0, 59))
            
            # Combine random_date and random_time to get a full random datetime
            random_datetime = datetime.combine(random_date, datetime.min.time()) + random_time

            # Convert to string format (for example, '%Y-%m-%d %H:%M:%S')
            datetime_str = random_datetime.strftime('%Y-%m-%d %H:%M:%S')

            # Parse the datetime string back to a datetime object (if needed)
            parsed_datetime = datetime.strptime(datetime_str, '%Y-%m-%d %H:%M:%S')
            
            #Point in time for ground truth is 1 day after the stock data
            gt_datetime = parsed_datetime + timedelta(days=1)
            # create a folder in dataset with date and time as name to start generating data in there
            #prepare stock symbols and save stock data for all symbols in datasetfolder
            stock_symbols = extract_symbols_from_csv('/Users/daniellavin/Desktop/proj/Moneytrain/cleaned_stockscreen.csv')
            #create string datetime to use for directory name
            safe_datetime = parsed_datetime.strftime("%Y-%m-%d_%H-%M-%S")
            safe_datetime = safe_datetime.replace(":", "_").replace(" ", "_")
            stock_dir = f'/Users/daniellavin/Desktop/proj/Moneytrain/stockdataset/{safe_datetime}'
            gt_dir = f'{stock_dir}/GT'
            os.makedirs(stock_dir, exist_ok=True)
            os.makedirs(gt_dir, exist_ok=True)

            for stock_symbol in stock_symbols:
                
                
                # Ensure that stock_symbol is clean and strip any trailing whitespace/newline
                stock_symbol = stock_symbol.strip()
                # Prepare stock data and ground truth data
                stock_data = prepare_single_stock_data(ticker_symbol=stock_symbol, start_datetime=parsed_datetime, days=6, min_points=45)
                gt_stock_data = returngroundtruthstock(stock_symbol, gt_datetime)
                #Ensure the stock dataset directory exists
                
                stock_symbol.strip(" ")
                stock_symbol = stock_symbol.replace("/", "_")
                if stock_data is not None and isinstance(stock_data, torch.Tensor):
                    stock_data = encode_and_attach(stock_symbol, stock_data)
                    # Construct proper file path with .pt extension for stock_data
                    stock_file_path = f'{stock_dir}/{stock_symbol}.pt'
                    # Save the stock data
                    torch.save(stock_data, stock_file_path)
                else:
                    print(f"Skipping {stock_symbol}: stock_data is invalid or None.")

                if gt_stock_data is not None and isinstance(gt_stock_data, torch.Tensor):
                    #gt_stock_data = encode_and_attach(stock_symbol, gt_stock_data)
                    # Construct proper file path with .pt extension for gt_stock_data
                    gt_file_path = f'{gt_dir}/{stock_symbol}_GT.pt'
                    # Save the ground truth data
                    torch.save(gt_stock_data, gt_file_path)
                else:
                    print(f"Skipping {stock_symbol}: gt_stock_data is invalid or None.")
                
            
            #delete all faulty torch vectors before we start the news pipeline

            process_stock_torch_files(f'{stock_dir}')

            #the same for GT but much simpler
            #process_GT_stock_torch_files(f'{gt_dir}')

            validate_and_clean_tensors(f'{stock_dir}', f'{gt_dir}')

            #prepare news data and save



    # Example usage
if __name__ == "__main__":
    #here we are passing the overall timeframe from which we want to collect datapoints, in general

    early = datetime(2024, 5, 5)
    late = datetime(2024, 9, 10)
    passes = 1
    prepare_dataset(passes, early, late)


        