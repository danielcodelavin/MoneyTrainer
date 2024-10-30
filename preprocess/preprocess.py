import os
import torch
from datetime import datetime, timedelta
import random
from typing import List, Dict
import csv
#from textprep import 
from transformers import BertTokenizer, BertModel
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.decomposition import PCA
import numpy as np
import re
from genscraper import play_scrape_articles, _clean_text, play_save_to_csv
from moneytensorgen import prepare_single_stock_data , extract_symbols_from_csv , returngroundtruthstock, encode_and_attach, validate_and_clean_tensors
from postprocessing import process_GT_stock_torch_files, process_stock_torch_files
# Import statements for functions from other documents
import math



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


def prepare_dataset(passes:int, earliest_date: datetime, latest_date: datetime, config):
    ################### CONFIGS FOR TEXT GEN ##############
    
    
    
    
    for i in range(passes):
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
            
            #Point in time for ground truth is 1 day after the stock data
            gt_datetime = parsed_datetime + timedelta(days=1)
            # create a folder in dataset with date and time as name to start generating data in there
            #prepare stock symbols and save stock data for all symbols in datasetfolder
            stock_symbols = extract_symbols_from_csv('/Users/daniellavin/Desktop/proj/Moneytrain/cleaned_stockscreen.csv')
            ################# CHANGE HERE WHETHER TO USE stockscreen.csv or cleaned_stockscreen.csv
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
                # Ensure the stock dataset directory exists
                
                stock_symbol.strip("/")
                stock_symbol = stock_symbol.replace("/", "_")
                 # Check if stock_data and gt_stock_data are valid (e.g., not None or empty)
                
                
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
            process_GT_stock_torch_files(f'{gt_dir}')
            #the same for GT but much simpler
            validate_and_clean_tensors(f'{stock_dir}', f'{gt_dir}')
            # Here we want to convert all of the GTs to percentage changes 
            
            process_tensors_percentage_change(f'{stock_dir}')
            #normalize the main stock tensor
            normalize_tensors(f'{stock_dir}')

            # add positional encodings to the main stock tensor
            for files in os.listdir(f'{stock_dir}'):
                if files.endswith('.pt'):
                    add_positional_encodings(f'{stock_dir}/{files}')
            #prepare news data and save

        ###########  NEWS DATA IS STORED TOGETHER WITH STOCK DATA AND CONCATENATED AT DATALOAD TIME, SAVES 13x STORAGE ###########
        #prepare news data and save
        
            raw_news_data = prepare_text_data(enddate=str_date, endtime=str_time,save_directory=config['save_directory'], keywords=config['keywords'], days_back=7, max_articles_per_keyword=config['max_articles_per_keyword'])
            processed_news_data = process_text_data(raw_news_data)
            news_file_path = f'{stock_dir}/0news.pt'
            torch.save(processed_news_data, news_file_path)
            #print(processed_news_data)
            #print(processed_news_data.shape)
def load_data(dataset_path: str) -> Dict[str, torch.Tensor]:
    pass

# Example usage
if __name__ == "__main__":
   
    #here we are passing the overall timeframe from which we want to collect datapoints, in general
    config = {
        'keywords': ['financial', 'technology', 'stocks', 'funds','trading'],
        'save_directory': "/Users/daniellavin/Desktop/proj/Moneytrain/newscsv",
        'output_directory': "/Users/daniellavin/Desktop/proj/Moneytrain/stockdataset",
         'max_articles_per_keyword':  15,
    }
    
    early = datetime(2022, 12, 1)
    late = datetime(2024, 10, 10)
    
    passes = 100
    prepare_dataset(passes, early, late,config)
