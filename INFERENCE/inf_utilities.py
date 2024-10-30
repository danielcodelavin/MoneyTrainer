from pygooglenews import GoogleNews
import logging
from datetime import datetime, timedelta
from typing import List, Dict
import time
import re
import csv
import os
from crawl4ai import AsyncWebCrawler
import asyncio
import logging
import json
import torch


import yfinance as yf
import pandas as pd
import numpy as np
from transformers import BertTokenizer, BertModel
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.decomposition import PCA
import hashlib
from collections import Counter, deque
import math



logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

############### PLAY FUNCTION; TOUCH ################
def play_scrape_articles(keywords: List[str], end_date: str, end_time: str, days_back: int = 3, max_articles_per_keyword: int = 10) -> List[Dict[str, str]]:
    end_datetime = datetime.strptime(f"{end_date} {end_time}", '%Y-%m-%d %H:%M:%S')
    start_datetime = end_datetime - timedelta(days=days_back)
    
    gn = GoogleNews(lang='en', country='US')
    all_articles = []
    
    for keyword in keywords:
        search = gn.search(keyword, from_=start_datetime.strftime('%Y-%m-%d'), to_=end_datetime.strftime('%Y-%m-%d'))
        
        article_count = 0
        for entry in search['entries']:
            pub_date = datetime.strptime(entry.published, '%a, %d %b %Y %H:%M:%S %Z')
            if start_datetime <= pub_date <= end_datetime:
                article = {
                    'title': entry.title,
                    #'date': pub_date.strftime('%Y-%m-%d %H:%M:%S'),
                    #'source': entry.source.title,
                    #'content': _clean_text(entry.summary),
                    #'link': entry.link
                }
                all_articles.append(article)
                article_count += 1
                if article_count >= max_articles_per_keyword:
                    break
            time.sleep(1)  # Be polite, wait a second between requests
    
    return all_articles








################## FUNCTION WORKS NEVER TOUCH ####################
def scrape_articles(keywords: List[str], end_date: str, end_time: str, days_back: int = 3) -> List[Dict[str, str]]:
    end_datetime = datetime.strptime(f"{end_date} {end_time}", '%Y-%m-%d %H:%M:%S')
    start_datetime = end_datetime - timedelta(days=days_back)
    
    gn = GoogleNews(lang='en', country='US')
    all_articles = []
    
    for keyword in keywords:
        search = gn.search(keyword, from_=start_datetime.strftime('%Y-%m-%d'), to_=end_datetime.strftime('%Y-%m-%d'))
        
        for entry in search['entries']:
            pub_date = datetime.strptime(entry.published, '%a, %d %b %Y %H:%M:%S %Z')
            if start_datetime <= pub_date <= end_datetime:
                article = {
                    'title': entry.title,
                    'date': pub_date.strftime('%Y-%m-%d %H:%M:%S'),
                    'source': entry.source.title,
                    'content': _clean_text(entry.summary),
                    'link': entry.link
                }
                all_articles.append(article)
            time.sleep(1)  # Be polite, wait a second between requests
    
    return all_articles

def save_to_csv(articles: List[Dict[str, str]], filename: str):
    with open(filename, 'w', newline='', encoding='utf-8') as file:
        writer = csv.DictWriter(file, fieldnames=['title', 'date', 'source', 'content', 'link'])
        writer.writeheader()
        for article in articles:
            writer.writerow(article)

def play_save_to_csv(articles: List[Dict[str, str]], filename: str):
    with open(filename, 'w', newline='', encoding='utf-8') as file:
        writer = csv.DictWriter(file, fieldnames=['title'])
        writer.writeheader()
        for article in articles:
            writer.writerow(article)

def _clean_text(text: str) -> str:
    # Remove HTML tags
    cleaned_text = re.sub(r'<[^>]+>', '', text)
    # Remove all symbols except full stops and commas
    cleaned_text = re.sub(r'[^\w\s.,]', '', cleaned_text)
    # Remove extra whitespace
    cleaned_text = re.sub(r'\s+', ' ', cleaned_text).strip()
    return cleaned_text





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


def prepare_single_stock_data(ticker_symbol: str, start_datetime: str, days: int = 5, min_points: int = 80):
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

    all_data = []
    valid_days_collected = 0
    days_checked = 0  # Keep track of how many total days we’ve looked at

    # Keep fetching until we get the required number of valid days
    while valid_days_collected < days:
        day_end = end_datetime - pd.Timedelta(days=days_checked)
        day_start = day_end - pd.Timedelta(days=1)
        
        try:
            print(f"Fetching data for {day_start} to {day_end}...")
            # Fetch hourly data for the 24 hours before the given time
            data = ticker.history(start=day_start, end=day_end, interval="60m", prepost=True, auto_adjust=False)

            if not data.empty:  # Only append if we got valid data
                all_data.append(data['Close'])
                valid_days_collected += 1  # Increment the number of valid days
            else:
               print(f"No data available for {day_start} to {day_end}. Skipping day.")

        except Exception as e:
            print(f"Error fetching data fosr stock {ticker_symbol} from {day_start} to {day_end}: {e}")

        # Move on to the next day (whether data was valid or not)
        days_checked += 1

        # Break out of the loop if we arent finding data within a max amount of time. reduce this number for faster performance
        if days_checked >= 12:
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



def process_stock_torch_files(directory):
    # Initialize variables
    length_counter = Counter()
    delete_count = 0
    
    # Get all .pt files in the directory
    files = [f for f in os.listdir(directory) if f.endswith('.pt')]
    num_files = len(files)
    
    print(f"Number of files in the directory: {num_files}")

    # Step 1: Preliminary sweep to remove files with NaN, zero, or inf elements
    for file in files:
        file_path = os.path.join(directory, file)
        try:
            # Load the tensor
            tensor = torch.load(file_path)
            
            # Skip if it's not a tensor (e.g. dict or other objects)
            if not isinstance(tensor, torch.Tensor):
                continue
            
            # Check for NaN, zero, or inf values
            if torch.isnan(tensor).any() or torch.isinf(tensor).any() or (tensor == 0).any():
                os.remove(file_path)
                delete_count += 1
                continue

            # Get tensor length
            tensor_length = tensor.numel()
            
            # Only count lengths over 40
            if tensor_length > 40:
                length_counter[tensor_length] += 1
        
        except Exception as e:
            print(f"Error processing file {file}: {e}")
    
    # Find the most common length
    if length_counter:
        most_common_length = length_counter.most_common(1)[0][0]
    else:
        print("No valid tensors found.")
        return

    print(f"Most common tensor length: {most_common_length}")
    print(f"Number of files deleted in preliminary sweep: {delete_count}")

    # Step 2: Delete files with uncommon lengths
    for file in os.listdir(directory):
        if file.endswith('.pt'):
            file_path = os.path.join(directory, file)
            try:
                tensor = torch.load(file_path)
                if not isinstance(tensor, torch.Tensor) or tensor.numel() != most_common_length:
                    os.remove(file_path)
                    delete_count += 1
            except Exception as e:
                print(f"Error processing file {file}: {e}")

    print(f"Total number of files deleted: {delete_count}")

    # Step 2: Delete tensors not matching the most common length
    for file in files:
        file_path = os.path.join(directory, file)
        try:
            tensor = torch.load(file_path)

            # Skip if it's not a tensor or already deleted
            if not isinstance(tensor, torch.Tensor) or not os.path.exists(file_path):
                continue

            tensor_length = tensor.numel()

            # Delete tensors that don't match the most common length
            if tensor_length != most_common_length:
                os.remove(file_path)
                delete_count += 1

        except Exception as e:
            print(f"Error processing file {file}: {e}")
    
    # Output results
    print(f"Most common tensor length: {most_common_length}")
    print(f"Number of files deleted: {delete_count}")

def process_stock_torch_files_by_max(directory):
   # Initialize variables
    max_length = 0
    delete_count = 0
    
    # Get all .pt files in the directory
    files = [f for f in os.listdir(directory) if f.endswith('.pt')]
    num_files = len(files)
    
    print(f"Number of files in the directory: {num_files}")

    # Step 1: Find the highest tensor length
    for file in files:
        file_path = os.path.join(directory, file)
        try:
            # Load the tensor
            tensor = torch.load(file_path)
            
            # Skip if it's not a tensor (e.g. dict or other objects)
            if not isinstance(tensor, torch.Tensor):
                continue
            
            # Check for inf/-inf values and skip if found
            if torch.isinf(tensor).any():
                os.remove(file_path)
                delete_count += 1
                continue

            # Get tensor length
            tensor_length = tensor.numel()
            max_length = max(max_length, tensor_length)
        
        except Exception as e:
            print(f"Error processing file {file}: {e}")
    
    # Step 2: Delete tensors not matching the max length
    for file in files:
        file_path = os.path.join(directory, file)
        try:
            tensor = torch.load(file_path)

            # Skip if it's not a tensor or already deleted
            if not isinstance(tensor, torch.Tensor) or not os.path.exists(file_path):
                continue

            tensor_length = tensor.numel()

            # Delete tensors that don't match the max length
            if tensor_length != max_length:
                os.remove(file_path)
                delete_count += 1

        except Exception as e:
            print(f"Error processing file {file}: {e}")
    
    # Output results
    print(f"Highest tensor length: {max_length}")
    print(f"Number of files deleted: {delete_count}")

# Usage example: process_torch_files('/path/to/your/directory')


def process_GT_stock_torch_files(dataset_path):
    for root, dirs, files in os.walk(dataset_path):
        for file in files:
            if file.endswith('.pt'):
                tensor = torch.load(os.path.join(root, file))
                if (tensor.numel() == 0 or
                    torch.isnan(tensor).any() or 
                    torch.isinf(tensor).any() or 
                    torch.any(tensor == 0) or 
                    torch.any(tensor == -0) or 
                    torch.any(torch.isin(tensor, torch.tensor([float('inf'), float('-inf')])))):  # Fixed this line
                    os.remove(os.path.join(root, file))


def process_tensors_percentage_change(filepath: str):
    # Ensure the GT folder exists
    gt_folder = os.path.join(filepath, "GT")
    if not os.path.exists(gt_folder):
        print(f"Error: GT folder not found in {filepath}")
        return

    # Process all .pt files in the main folder
    for filename in os.listdir(filepath):
        if filename.endswith('.pt') and not filename.endswith('_GT.pt'):
            main_file_path = os.path.join(filepath, filename)
            gt_file_path = os.path.join(gt_folder, f"{filename[:-3]}_GT.pt")
            
            # Check if corresponding GT file exists
            if not os.path.exists(gt_file_path):
                print(f"Warning: No corresponding GT file found for {filename}")
                continue
            
            try:
                # Load the main tensor
                main_tensor = torch.load(main_file_path)
                
                # Ensure the main tensor is 1D and has at least 2 elements
                if not isinstance(main_tensor, torch.Tensor) or main_tensor.dim() != 1 or main_tensor.size(0) < 2:
                    print(f"Error: {filename} is not a valid 1D tensor with at least 2 elements")
                    continue
                
                # Get the reference value (second element)
                reference_value = main_tensor[1].item()
                
                # Load the GT tensor
                gt_tensor = torch.load(gt_file_path)
                
                # Ensure the GT tensor has only one element
                if not isinstance(gt_tensor, torch.Tensor) or gt_tensor.numel() != 1:
                    print(f"Error: {filename}_GT.pt is not a valid single-element tensor")
                    continue
                
                # Get the GT value
                gt_value = gt_tensor.item()
                
                # Calculate percentage change
                percentage_change = (gt_value - reference_value) / reference_value
                
                # Create new GT tensor with percentage change
                new_gt_tensor = torch.tensor([percentage_change])
                
                # Save the new GT tensor
                torch.save(new_gt_tensor, gt_file_path)
                
                print(f"Processed {filename}: Reference = {reference_value}, Original GT = {gt_value}, New GT = {percentage_change:.4f}")
                
            except Exception as e:
                print(f"Error processing {filename}: {str(e)}")

    print("Processing complete.")

# Usage
# process_tensors_percentage_change('/path/to/your/folder')


def normalize_tensors(folder_path: str):
    # List to store all tensors
    all_tensors = []
    
    # Walk through the directory
    for root, dirs, files in os.walk(folder_path):
        # Skip the GT subfolder
        if 'GT' in dirs:
            dirs.remove('GT')
        
        for file in files:
            if file.endswith('.pt'):
                file_path = os.path.join(root, file)
                try:
                    # Load the tensor
                    tensor = torch.load(file_path)
                    
                    # Check if it's a 1D tensor
                    if isinstance(tensor, torch.Tensor) and tensor.dim() == 1:
                        all_tensors.append(tensor)
                    else:
                        print(f"Skipping {file}: Not a 1D tensor")
                except Exception as e:
                    print(f"Error processing {file}: {e}")
    
    if not all_tensors:
        print("No valid tensors found.")
        return
    
    # Concatenate all tensors
    combined_tensor = torch.cat(all_tensors)
    
    # Separate labels and features
    labels = combined_tensor[::combined_tensor.size(0)//len(all_tensors)]
    features = combined_tensor.view(len(all_tensors), -1)[:, 1:]
    
    # Normalize features
    normalized_features = (features - features.mean()) / features.std()
    
    # Recombine labels and normalized features
    normalized_tensors = torch.cat([labels.unsqueeze(1), normalized_features], dim=1)
    
    # Save normalized tensors back to files
    for i, (root, _, files) in enumerate(os.walk(folder_path)):
        if 'GT' in root:
            continue
        for file in files:
            if file.endswith('.pt'):
                file_path = os.path.join(root, file)
                torch.save(normalized_tensors[i], file_path)
                print(f"Saved normalized tensor to {file_path}")

    print(f"Processed and normalized {len(all_tensors)} tensors.")

def add_positional_encodings(filepath):
    """
    This function loads a tensor from the specified file, applies positional encoding 
    to the time-dependent numerical data (ignoring the first element, which is the label),
    and overwrites the file with the updated tensor.

    Args:
    - filepath (str): Path to the tensor file.
    
    The function assumes the tensor is 1D, where the first element is a label and the 
    remaining elements are time-dependent numerical data.
    """

    # Load the tensor from the file
    tensor = torch.load(filepath)
    
    # Ensure the tensor is 1D and has more than 1 element
    assert len(tensor.shape) == 1, "Expected a 1D tensor."
    assert tensor.shape[0] > 1, "Tensor must have at least 2 elements (label + data)."
    
    # Split the tensor: first element is the label, rest is time-dependent numerical data
    label = tensor[0]  # First element is the label
    numerical_data = tensor[1:]  # Remaining elements are the time-dependent data

    # Get the length of the time-dependent numerical data
    sequence_length = numerical_data.shape[0]
    
    # Reshape numerical data to (sequence_length, 1) to prepare for positional encoding
    numerical_data = numerical_data.unsqueeze(1)  # Shape: (sequence_length, 1)
    
    # Generate positional encodings
    def positional_encoding(sequence_length, feature_dim):
        position = torch.arange(0, sequence_length, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, feature_dim, 2).float() * (-math.log(10000.0) / feature_dim))
        pe = torch.zeros(sequence_length, feature_dim)
        pe[:, 0::2] = torch.sin(position * div_term)  # Even indices
        pe[:, 1::2] = torch.cos(position * div_term)  # Odd indices
        return pe

    # Apply positional encoding to the numerical data
    pos_enc = positional_encoding(sequence_length, numerical_data.shape[1])
    numerical_data_with_pos = numerical_data + pos_enc  # Shape: (sequence_length, 1)

    # Recombine the label with the positionally encoded numerical data
    # We squeeze the data back to 1D for consistency
    updated_tensor = torch.cat((label.unsqueeze(0), numerical_data_with_pos.squeeze(1)), dim=0)

    # Overwrite the file with the updated tensor
    torch.save(updated_tensor, filepath)

    print(f"Positional encodings added and file '{filepath}' updated.")

def clean_text(text: str) -> str:
    # Remove all characters except alphanumeric and whitespace
    cleaned_text = re.sub(r'[^\w\s]', '', text)
    # Replace multiple whitespaces with a single space
    cleaned_text = re.sub(r'\s+', ' ', cleaned_text).strip()
    # Remove vertical line separator (not visible in the sample, but included as requested)
    cleaned_text = cleaned_text.replace('\u2758', '')
    return cleaned_text

def summarize_text(text: str, model_name='all-MiniLM-L6-v2', ratio=0.5) -> str:
    sentences = re.split(r'(?<=[.!?])\s+', text)
    model = SentenceTransformer(model_name)
    sentence_embeddings = model.encode(sentences)
    similarity_matrix = cosine_similarity(sentence_embeddings)
    scores = similarity_matrix.sum(axis=1)
    top_sentences_indices = scores.argsort()[-int(len(sentences) * ratio):]
    top_sentences_indices = sorted(top_sentences_indices)
    summarized_text = ' '.join([sentences[i] for i in top_sentences_indices])
    return summarized_text

def generate_embeddings(text: str, model_name='bert-base-uncased'):
    tokenizer = BertTokenizer.from_pretrained(model_name)
    model = BertModel.from_pretrained(model_name)
    inputs = tokenizer(text, return_tensors='pt', truncation=True, max_length=32, padding=False)
    with torch.no_grad():
        outputs = model(**inputs)
    embeddings = outputs.last_hidden_state[:, 0, :]
    return embeddings.squeeze(0)

def prepare_text_data(enddate: str, endtime: str, save_directory: str, keywords: List[str], days_back: int = 7, max_articles_per_keyword: int = 10) -> str:
    endtime = datetime.strptime(endtime, '%H:%M:%S').strftime('%H:%M:%S')
    endtime_str = endtime.replace(":", "").replace(" ", "")
    filename = f"scraped_articles{endtime_str}.csv"
    file_path = os.path.join(save_directory, filename)
    os.makedirs(save_directory, exist_ok=True)
    articles = play_scrape_articles(keywords=keywords, end_date=enddate, end_time=endtime, days_back=days_back, max_articles_per_keyword=max_articles_per_keyword)
    play_save_to_csv(articles, file_path)
    return file_path

def process_text_data(file_path: str) -> torch.Tensor:
    with open(file_path, 'r', encoding='utf-8') as file:
        reader = csv.DictReader(file)
        embeddings_list = []
        for row in reader:
            cleaned_text = clean_text(str(row))
            embedding = generate_embeddings(cleaned_text)
            embeddings_list.append(embedding)
    
    # Stack all embeddings
    all_embeddings = torch.stack(embeddings_list)
    
    # Determine the number of components for PCA
    n_components = 10   # number of components per article, for 50 articles, 10 components =500 elements
    # Use PCA to reduce dimensionality ############# ADJUST THE NUMBER OF COMPONENTS HERE FOR LENGTH OF FINAL TENSOR
    pca = PCA(n_components=n_components)
    reduced_embeddings = pca.fit_transform(all_embeddings.numpy())
    
    # Flatten the reduced embeddings into a 1D tensor
    final_embedding = torch.from_numpy(reduced_embeddings.flatten())
    print("[    NUM COMPONENTS   ]")
    print(n_components)
    os.remove(file_path)
    return final_embedding



