import os
import torch
from datetime import datetime, timedelta
import random
from typing import List, Dict
import csv
import re
from transformers import BertTokenizer, BertModel
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.decomposition import PCA
import numpy as np
import numpy as np
from genscraper import scrape_articles, _clean_text, save_to_csv,play_scrape_articles , play_save_to_csv

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
    filename = f"scraped_articles_{enddate}_{endtime.replace(':', '-')}.csv"
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
    n_components = 15   # number of components of 10 corresponds to final tensor length 500
    # Use PCA to reduce dimensionality ############# ADJUST THE NUMBER OF COMPONENTS HERE FOR LENGTH OF FINAL TENSOR
    pca = PCA(n_components=n_components)
    reduced_embeddings = pca.fit_transform(all_embeddings.numpy())
    
    # Flatten the reduced embeddings into a 1D tensor
    final_embedding = torch.from_numpy(reduced_embeddings.flatten())
    print("[    NUM COMPONENTS   ]")
    print(n_components)
    os.remove(file_path)
    return final_embedding




def generate_random_datetime(earliest_date: datetime, latest_date: datetime) -> datetime:
    while True:
        random_date = earliest_date + timedelta(days=random.randint(0, (latest_date - earliest_date).days))
        if random_date.weekday() < 5:  # 0 = Monday, 4 = Friday
            break
    random_time = timedelta(
        hours=random.randint(9, 20),
        minutes=random.randint(0, 59),
        seconds=random.randint(0, 59))
    return datetime.combine(random_date, datetime.min.time()) + random_time

if __name__ == '__main__':
    config = {
        'earliest_date': datetime(2022, 10, 1),
        'latest_date': datetime(2024, 9, 10),
        'keywords': ['financial', 'technology', 'stocks', 'funds','trading'],
        'save_directory': "/Users/daniellavin/Desktop/proj/Moneytrain/newscsv",
        'output_directory': "/Users/daniellavin/Desktop/proj/Moneytrain/stockdataset",
        'max_articles_per_keyword':  15,
    }


    #################################TODO################################
    # finetune keyword amounr generation
    #concatenate titles before summarizaion to make it actually happen
    #

    random_datetime = generate_random_datetime(config['earliest_date'], config['latest_date'])
    random_date_str = random_datetime.strftime('%Y-%m-%d')
    random_time_str = random_datetime.strftime('%H:%M:%S')

    print("[     START     ]")
    raw_news_data = prepare_text_data(random_date_str, random_time_str, config['save_directory'], config['keywords'], max_articles_per_keyword=config['max_articles_per_keyword'])
    processed_news_data = process_text_data(raw_news_data)

    safe_datetime = random_datetime.strftime("%Y-%m-%d_%H-%M-%S")
    stock_dir = os.path.join(config['output_directory'], safe_datetime)
    os.makedirs(stock_dir, exist_ok=True)

    output_file = os.path.join(stock_dir, '01_news.pt')
    torch.save(processed_news_data, output_file)
    print(f"File saved to: {output_file}")
    print(f"Tensor shape: {processed_news_data.shape}")
    print(f"Number of elements: {processed_news_data.numel()}")