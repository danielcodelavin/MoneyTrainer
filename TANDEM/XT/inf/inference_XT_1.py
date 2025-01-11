import torch
import pandas as pd
import yfinance as yf
from datetime import datetime, timedelta
from typing import List, Tuple, Dict, Optional
import os
from typing import List, Optional
import numpy as np
from bertopic import BERTopic
import logging
import time
from pygooglenews import GoogleNews
import csv
from pathlib import Path
import torch.nn as nn
from finhub_scrape import get_stock_news
from collections import deque
import traceback
from torch.utils.data import Dataset, DataLoader

class StockBatchDataset(Dataset):
    def __init__(self, stocks_data: List[Tuple[torch.Tensor, torch.Tensor]]):
        self.stocks_data = stocks_data
    
    def __len__(self):
        return len(self.stocks_data)
    
    def __getitem__(self, idx):
        stock_data, topic_vector = self.stocks_data[idx]
        return stock_data, topic_vector

class ResidualBlock(nn.Module):
    def __init__(self, in_features):
        super(ResidualBlock, self).__init__()
        self.block = nn.Sequential(
            nn.Linear(in_features, in_features * 2),
            nn.LeakyReLU(),
            nn.Dropout(0.2),
            nn.Linear(in_features * 2, in_features)
        )
        self.leaky_relu = nn.LeakyReLU()
        self.norm = nn.LayerNorm(in_features)
        
    def forward(self, x):
        residual = x
        out = self.block(x)
        out += residual
        out = self.norm(out)
        return self.leaky_relu(out)

class PrecisionFocusedMLP(nn.Module):
    def __init__(self):
        super(PrecisionFocusedMLP, self).__init__()
        
        self.stock_path = nn.Sequential(
            nn.Linear(80, 256),
            nn.LeakyReLU(),
            nn.Dropout(0.2),
            ResidualBlock(256),
            nn.Linear(256, 384),
            nn.LeakyReLU(),
            nn.Dropout(0.2),
            ResidualBlock(384),
            nn.Linear(384, 512),
            nn.LeakyReLU(),
            nn.Dropout(0.2),
            ResidualBlock(512),
            nn.Linear(512, 384),
            nn.LeakyReLU(),
            nn.Dropout(0.2),
            ResidualBlock(384),
            nn.Linear(384, 256),
            nn.LeakyReLU(),
            nn.Dropout(0.2)
        )
        
        self.topic_path = nn.Sequential(
            nn.Linear(30, 128),
            nn.LeakyReLU(),
            nn.Dropout(0.2),
            nn.Linear(128, 256),
            nn.LeakyReLU(),
            nn.Dropout(0.2),
            ResidualBlock(256),
            nn.Linear(256, 192),
            nn.LeakyReLU(),
            nn.Dropout(0.2),
            ResidualBlock(192),
            nn.Linear(192, 128),
            nn.LeakyReLU(),
            nn.Dropout(0.2)
        )
        
        self.confidence_head = nn.Sequential(
            nn.Linear(384, 256),
            nn.LeakyReLU(),
            nn.Dropout(0.3),
            nn.Linear(256, 128),
            nn.LeakyReLU(),
            nn.Linear(128, 1),
            nn.Sigmoid()
        )
        
        self.prediction_head = nn.Sequential(
            nn.Linear(384, 512),
            nn.BatchNorm1d(512),
            nn.LeakyReLU(),
            nn.Dropout(0.2),
            ResidualBlock(512),
            nn.Linear(512, 384),
            nn.BatchNorm1d(384),
            nn.LeakyReLU(),
            nn.Dropout(0.2),
            ResidualBlock(384),
            nn.Linear(384, 256),
            nn.BatchNorm1d(256),
            nn.LeakyReLU(),
            nn.Dropout(0.2),
            ResidualBlock(256),
            nn.Linear(256, 128),
            nn.BatchNorm1d(128),
            nn.LeakyReLU(),
            nn.Dropout(0.2),
            nn.Linear(128, 1)
        )
    
    def forward(self, stock_data, topic_vector):
        stock_features = self.stock_path(stock_data)
        topic_features = self.topic_path(topic_vector)
        combined = torch.cat((stock_features, topic_features), dim=1)
        
        confidence = self.confidence_head(combined)
        prediction = self.prediction_head(combined)
        
        return prediction, confidence

class PredictionHistory:
    def __init__(self, window_size: int = 100):
        self.window_size = window_size
        self.predictions = deque(maxlen=window_size)
        self.ground_truths = deque(maxlen=window_size)
        self.confidences = deque(maxlen=window_size)
    
    def calculate_metrics(self) -> Tuple[float, float]:
        if len(self.predictions) < 10:
            return 0.5, 1.0
        
        preds = torch.tensor(list(self.predictions))
        truths = torch.tensor(list(self.ground_truths))
        
        pred_positive = preds > 0
        truth_positive = truths > 0
        predicted_positives = pred_positive.sum().item()
        
        if predicted_positives > 0:
            true_positives = (pred_positive & truth_positive).sum().item()
            precision = true_positives / predicted_positives
        else:
            precision = 0.5
        
        non_zero_preds = preds != 0
        if non_zero_preds.any():
            l1_score = torch.mean(torch.abs(
                preds[non_zero_preds] - truths[non_zero_preds]
            )).item()
        else:
            l1_score = 1.0
        
        return precision, l1_score

class WeightedEnsembleModel(nn.Module):
    def __init__(self, num_models: int = 7, precision_weight: float = 0.8,
                 window_size: int = 100):
        super(WeightedEnsembleModel, self).__init__()
        
        self.models = nn.ModuleList([
            PrecisionFocusedMLP() for _ in range(num_models)
        ])
        
        self.precision_weight = precision_weight
        self.l1_weight = 1 - precision_weight
        self.histories = [PredictionHistory(window_size) for _ in range(num_models)]
        self.model_weights = None
    
    def forward(self, stock_data, topic_vector):
        # Process entire batch at once for each model
        all_predictions = []
        all_confidences = []
        
        for model in self.models:
            pred, conf = model(stock_data, topic_vector)
            all_predictions.append(pred)
            all_confidences.append(conf)
        
        # Stack along model dimension
        predictions_stack = torch.stack(all_predictions, dim=1)  # [batch_size, num_models, 1]
        confidences_stack = torch.stack(all_confidences, dim=1)  # [batch_size, num_models, 1]
        
        # Use loaded weights from checkpoint for weighted prediction
        weighted_pred = torch.sum(
            predictions_stack * self.model_weights.view(1, -1, 1).to(stock_data.device),
            dim=1
        )
        ensemble_conf = confidences_stack.max(dim=1)[0]
        
        return weighted_pred, ensemble_conf

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
        time.sleep(0.1)
    
    if len(all_data) < min_points:
        print(f"Insufficient data points ({len(all_data)}/{min_points}) for {ticker_symbol}")
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
        return normalized_tensor, all_data[-1]
    except Exception as e:
        print(f"Error processing {ticker_symbol} data: {str(e)}")
        return None

def process_headlines_with_bertopic(headlines: List[str], topic_model: BERTopic) -> Optional[torch.Tensor]:
    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger(__name__)
    
    try:
        if not headlines or not isinstance(headlines, list):
            logger.error("Invalid headlines input: must be non-empty list of strings")
            return None
            
        if not all(isinstance(h, str) for h in headlines):
            logger.error("All headlines must be strings")
            return None
            
        headlines = [h.strip() for h in headlines if h.strip()]
        
        if not headlines:
            logger.error("No valid headlines after cleaning")
            return None
            
        nr_topics = len(topic_model.get_topic_info())
        
        topics, probs = topic_model.transform(headlines)
        
        full_probs = np.zeros((len(headlines), nr_topics))
        
        if isinstance(probs, (float, np.float64)):
            topic_idx = topics if isinstance(topics, (int, np.integer)) else topics[0]
            full_probs[0, topic_idx] = probs
        else:
            probs = np.array(probs)
            for i, (topic, prob) in enumerate(zip(topics, probs)):
                if isinstance(prob, np.ndarray):
                    for t, p in zip(topic, prob):
                        if 0 <= t < nr_topics:
                            full_probs[i, t] = p
                else:
                    if 0 <= topic < nr_topics:
                        full_probs[i, topic] = prob
                
        mean_probs = np.mean(full_probs, axis=0)
        
        if mean_probs.shape[0] != nr_topics:
            logger.error(f"Wrong number of topics in output: got {mean_probs.shape[0]}, expected {nr_topics}")
            return None
        
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
    
    gn = GoogleNews(lang='en', country='US')
    all_titles = []
    
    for i, keyword in enumerate(keywords):
        print(f"Processing keyword {i+1}/{len(keywords)}: {keyword}")
        start_time = time.time()
        
        try:
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
    
    print(f"Scraping completed at {datetime.now()}")
    return all_titles

def datagenerator(
    stock_symbol: str,
    stock_name: str,
    stock_industry: str,
    stock_sector: str,
    bertopic_model: BERTopic,
    current_datetime: datetime
) -> Optional[Tuple[torch.Tensor, torch.Tensor]]:
    try:
        str_date = current_datetime.strftime('%Y-%m-%d')
        str_time = current_datetime.strftime('%H:%M:%S')
        
        stock_data_result = prepare_single_stock_data(
            ticker_symbol=stock_symbol,
            start_datetime=current_datetime,
            days=10,
            min_points=80
        )
        
        if stock_data_result is None:
            return None
            
        stock_data, raw_latest_price = stock_data_result
        
        keywords = [stock_name, stock_industry + ' Industry']
        print("[    CHECKUP   ]  :" + stock_symbol + stock_name + stock_industry)
        second_headlines = new_scrape_articles(
            keywords=keywords,
            end_date=str_date,
            end_time=str_time,
            days_back=1,
            max_articles_per_keyword=10
        )
        
        headlines = get_stock_news(symbol=stock_symbol, date=str_date)
        headlines.extend(second_headlines)
        clean_headlines = [x for x in headlines if x]
        clean_headlines = list(set(clean_headlines))
        
        if not clean_headlines:
            print(f"No headlines found for {stock_symbol}, skipping tensor creation")
            return None
            
        news_embedding = process_headlines_with_bertopic(clean_headlines, bertopic_model)
        if news_embedding is None:
            print(f"BERTopic processing failed for {stock_symbol}, skipping tensor creation")
            return None
            
        news_embedding = news_embedding[:30] if news_embedding is not None else None
        
        print(f"\nTensor Components for {stock_symbol}:")
        print(f"Stock Data (first 5): {stock_data[:5].tolist()}")
        print(f"Topic Vector (first 5): {news_embedding[:5].tolist()}\n")
        
        return stock_data, news_embedding
        
    except Exception as e:
        print(f"Error in datagenerator for {stock_symbol}: {str(e)}")
        return None

def create_directory_structure(current_date: datetime, base_path: str) -> Path:
    path = Path(base_path) / str(current_date.year) / f"{current_date.month:02d}" / f"{current_date.day:02d}"
    path.mkdir(parents=True, exist_ok=True)
    return path

def process_in_batches(model: nn.Module, dataset: StockBatchDataset, 
                      batch_size: int, device: torch.device) -> List[Dict[str, float]]:
    """Process data in batches to maintain consistent BatchNorm behavior"""
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False)
    all_results = []
    
    with torch.no_grad():
        for batch_stock_data, batch_topic_vector in dataloader:
            batch_stock_data = batch_stock_data.to(device)
            batch_topic_vector = batch_topic_vector.to(device)
            
            predictions, confidences = model(batch_stock_data, batch_topic_vector)
            
            # Store batch results
            for pred, conf in zip(predictions, confidences):
                all_results.append({
                    'prediction': pred.item(),
                    'confidence': conf.item()
                })
    
    return all_results

def main(
    csv_path: str,
    bertopic_model_path: str,
    checkpoint_path: str,
    results_base_path: str,
    target_datetime: Optional[datetime] = None,
    batch_size: int = 128  # Same batch size as training
):
    if target_datetime is None:
        target_datetime = datetime.now()
    
    print(f"\nInitializing inference at {target_datetime}")
    print(f"Loading models and data...")
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Initialize model
    model = WeightedEnsembleModel(num_models=7, precision_weight=0.8, window_size=150)
    
    # Load checkpoint and weights
    checkpoint = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    
    if 'model_weights' in checkpoint:
        model.model_weights = torch.tensor(checkpoint['model_weights'])
    else:
        print("Warning: No model weights found in checkpoint, using equal weights")
        model.model_weights = torch.ones(7) / 7
    
    # Ensure model and all submodels are in eval mode
    model.eval()
    for sub_model in model.models:
        sub_model.eval()
    
    model = model.to(device)
    topic_model = BERTopic.load(bertopic_model_path)
    
    # Read stock data
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
    
    print(f"Processing {len(Stock_Symbols)} stocks...")
    
    # Collect all tensors first
    all_stock_data = []
    stock_indices = []  # To maintain stock symbol ordering
    
    for i, symbol in enumerate(Stock_Symbols):
        print(f"\nProcessing {symbol} ({i+1}/{len(Stock_Symbols)})")
        
        tensor_data = datagenerator(
            stock_symbol=symbol,
            stock_name=Stock_Names[i],
            stock_industry=Stock_Industries[i],
            stock_sector=Stock_Sectors[i],
            bertopic_model=topic_model,
            current_datetime=target_datetime
        )
        
        if tensor_data is not None:
            stock_data, topic_vector = tensor_data
            all_stock_data.append((stock_data, topic_vector))
            stock_indices.append(i)
    
    # Create dataset and process in batches
    dataset = StockBatchDataset(all_stock_data)
    batch_results = process_in_batches(model, dataset, batch_size, device)
    
    # Combine results with stock symbols
    results = []
    for idx, result in zip(stock_indices, batch_results):
        results.append({
            'name': Stock_Symbols[idx],
            'prediction': result['prediction']
        })
    
    # Save results
    results_path = create_directory_structure(target_datetime, results_base_path)
    date_string = target_datetime.strftime("%Y%m%d_%H%M%S")
    csv_filename = f"results_{date_string}.csv"
    csv_path = results_path / csv_filename
    
    with open(csv_path, 'w', newline='') as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=['name', 'prediction'])
        writer.writeheader()
        for result in results:
            writer.writerow(result)
    
    print(f"\nResults saved to: {csv_path}")
    print(f"Processed {len(results)} stocks successfully")

if __name__ == "__main__":
    dates = [datetime(2024,12,2,18,0,0),datetime(2024,12,3,18,0,0),datetime(2024,12,4,18,0,0),datetime(2024,12,5,18,0,0,),datetime(2024,12,9,18,0,0),
             datetime(2024,12,10,18,0,0),datetime(2024,12,11,18,0,0),datetime(2024,12,12,18,0,0),
             datetime(2024,12,16,18,0,0),datetime(2024,12,17,18,0,0),datetime(2024,12,18,18,0,0),datetime(2024,12,19,18,0,0)]
    
        
    for date in dates:    
        main(
        csv_path="/Users/daniellavin/Desktop/proj/MoneyTrainer/Hybrid_stockscreen.csv",
        bertopic_model_path="/Users/daniellavin/Desktop/proj/MoneyTrainer/x_preprocess/2_BERTOPIC_MODEL_50.pt",
        checkpoint_path="/Users/daniellavin/Desktop/proj/MoneyTrainer/TANDEM/XT/check/checkpoint_epoch_60.pt",
        results_base_path="/Users/daniellavin/Desktop/proj/MoneyTrainer/TANDEM/XT/results",
        target_datetime=date 
        )