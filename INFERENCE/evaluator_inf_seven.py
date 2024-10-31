import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List, Dict
import os
import math
import numpy as np
import csv
import random
from datetime import datetime, timedelta
from inf_utilities import prepare_single_stock_data, prepare_text_data, encode_and_attach ,process_text_data, extract_symbols_from_csv, returngroundtruthstock

class StableTransformerModel(nn.Module):
    def __init__(self, hidden_dim: int, num_layers: int, num_heads: int, dropout: float = 0.1):
        super(StableTransformerModel, self).__init__()
        self.hidden_dim = hidden_dim
        
        self.stock_embedding = nn.Sequential(
            nn.Linear(1, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.Tanh()
        )
        
        self.text_embedding = nn.Sequential(
            nn.Linear(1, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.Tanh()
        )
        
        self.type_embeddings = nn.Parameter(torch.zeros(2, hidden_dim))
        
        self.transformer_layers = nn.ModuleList([
            StableTransformerLayer(hidden_dim, num_heads, dropout)
            for _ in range(num_layers)
        ])
        
        self.output_head = nn.Sequential(
            nn.LayerNorm(hidden_dim),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.Tanh(),
            nn.Linear(hidden_dim // 2, 1),
            nn.Tanh()
        )
        
        self.dropout = nn.Dropout(dropout)
    
    def create_attention_mask(self, seq_len: int, stock_len: int, text_len: int):
        mask = torch.zeros(seq_len, seq_len, dtype=torch.bool)
        stock_end = 1 + stock_len
        text_start = stock_end
        
        # Text tokens can't attend to stock tokens
        mask[text_start:, :stock_end] = True
        return mask

    def forward(self, x):
        batch_size, seq_len = x.shape
        
        # Split input - maintaining same dimensions as training
        stock_len = seq_len - 750 - 1  # 750 is text length from training
        label = x[:, 0].unsqueeze(-1)
        stock_data = x[:, 1:stock_len+1].unsqueeze(-1)
        text_data = x[:, -750:].unsqueeze(-1)
        
        stock_embedded = self.stock_embedding(stock_data)
        text_embedded = self.text_embedding(text_data)
        label_embedded = self.stock_embedding(label).unsqueeze(1)
        
        stock_embedded = stock_embedded + self.type_embeddings[0]
        text_embedded = text_embedded + self.type_embeddings[1]
        
        x = torch.cat([label_embedded, stock_embedded, text_embedded], dim=1)
        x = self.dropout(x)
        
        attention_mask = self.create_attention_mask(x.shape[1], stock_len, 750).to(x.device)
        
        for layer in self.transformer_layers:
            x = layer(x, attention_mask)
        
        return self.output_head(x[:, 0])

class StableTransformerLayer(nn.Module):
    def __init__(self, hidden_dim: int, num_heads: int, dropout: float = 0.1):
        super().__init__()
        self.norm1 = nn.LayerNorm(hidden_dim)
        self.attention = nn.MultiheadAttention(hidden_dim, num_heads, dropout=dropout, batch_first=True)
        self.norm2 = nn.LayerNorm(hidden_dim)
        self.feedforward = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim * 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim * 2, hidden_dim)
        )
        self.dropout = nn.Dropout(dropout)
    
    def forward(self, x, attention_mask=None):
        normed = self.norm1(x)
        x = x + self.dropout(self.attention(normed, normed, normed, attn_mask=attention_mask)[0])
        
        normed = self.norm2(x)
        x = x + self.dropout(self.feedforward(normed))
        
        return x

def load_checkpoint(checkpoint_path: str, model: nn.Module) -> None:
    """Load model checkpoint."""
    checkpoint = torch.load(checkpoint_path, map_location='cpu')
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    print(f"Loaded checkpoint from {checkpoint_path}")

def micro_pos_encode(tensor):
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
    return updated_tensor



def generate_input_tensor(ticker,config):
    """Placeholder for generating an input tensor for inference."""
    earliest_date = datetime(2022, 12, 1)
    latest_date = datetime(2024, 10, 10)
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


    stock_symbol = ticker.strip()
    stock_data = prepare_single_stock_data(ticker_symbol=stock_symbol, start_datetime=parsed_datetime, days=6, min_points=45)
    
    reference_value = stock_data[0].item()
    features = stock_data

    features_mean = features.mean()
    features_std = features.std()
    normalized_features = (features - features_mean) / features_std

    gt_tensor = returngroundtruthstock(stock_symbol, gt_datetime)
    gt_value = gt_tensor.item()
    gt = (gt_value - reference_value) / reference_value
    # add positional encoding
    stock_data = micro_pos_encode(normalized_features)
    #validate
    if stock_data is not None and isinstance(stock_data, torch.Tensor):
                    stock_data = encode_and_attach(stock_symbol, stock_data)
    else:
        print("FAIL : STOCK DATA NOT PREPARED")
    raw_news_data = prepare_text_data(enddate=str_date, endtime=str_time,save_directory=config['save_directory'], keywords=config['keywords'], days_back=7, max_articles_per_keyword=config['max_articles_per_keyword'])
    processed_news_data = process_text_data(raw_news_data)

    output_tensor = torch.cat((stock_data, processed_news_data), dim=0)
    return output_tensor, gt



def inference(input_tensor: torch.Tensor, model: nn.Module, device: torch.device) -> float:
    """Run inference on the input tensor."""
    model.eval()
    with torch.no_grad():
        # Ensure input tensor is properly shaped and on correct device
        if len(input_tensor.shape) == 1:
            input_tensor = input_tensor.unsqueeze(0)
        input_tensor = input_tensor.to(device)
        
        # Run inference
        output = model(input_tensor)
        
        # Convert output to float prediction
        prediction = output.item()
        
    return prediction



def evaluate_inference(
    tickers: List[str],
    output_path: str = "evaluation_results.csv"
) -> None:
    """
    Evaluates AI inference against ground truth for multiple tickers.
    
    Args:
        tickers: List of ticker symbols to evaluate
        output_path: Path to save the CSV results
    """
    
    def calculate_metrics(predictions: List[float], ground_truth: List[float]) -> dict:
        """Calculate various metrics for binary classification and L1 distance."""
        # Convert to binary classifications (positive/negative)
        pred_binary = [1 if x >= 0 else 0 for x in predictions]
        gt_binary = [1 if x >= 0 else 0 for x in ground_truth]
        
        # Calculate L1 distances
        l1_distances = [abs(p - gt) for p, gt in zip(predictions, ground_truth)]
        
        # Calculate binary classification metrics
        tp = sum(1 for p, gt in zip(pred_binary, gt_binary) if p == 1 and gt == 1)
        fp = sum(1 for p, gt in zip(pred_binary, gt_binary) if p == 1 and gt == 0)
        tn = sum(1 for p, gt in zip(pred_binary, gt_binary) if p == 0 and gt == 0)
        fn = sum(1 for p, gt in zip(pred_binary, gt_binary) if p == 0 and gt == 1)
        
        # Avoid division by zero
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0
        accuracy = (tp + tn) / len(predictions) if len(predictions) > 0 else 0
        f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
        
        return {
            'average_l1': np.mean(l1_distances),
            'std_l1': np.std(l1_distances),
            'precision': precision,
            'recall': recall,
            'accuracy': accuracy,
            'f1_score': f1
        }
    
    def evaluate_single_ticker(ticker: str) -> tuple[str, dict]:
        """Evaluate a single ticker with multiple runs."""
        predictions = []
        ground_truth = []
        
        # Perform 50 evaluations
        for _ in range(50):

            pred, gt = returnpred(ticker)  # Replace with your tensor computation
                  
            predictions.append(pred)
            ground_truth.append(gt)
        
        metrics = calculate_metrics(predictions, ground_truth)
        return ticker, metrics
    
    # Prepare CSV headers
    headers = ['Ticker', 'Average_L1', 'Std_L1', 'Precision', 'Recall', 'Accuracy', 'F1_Score']
    
    # Evaluate all tickers and write results
    with open(output_path, 'w', newline='') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(headers)
        
        for ticker in tickers:
            ticker, metrics = evaluate_single_ticker(ticker)
            row = [
                ticker,
                metrics['average_l1'],
                metrics['std_l1'],
                metrics['precision'],
                metrics['recall'],
                metrics['accuracy'],
                metrics['f1_score']
            ]
            writer.writerow(row)



def returnpred(ticker):
    
    config = {
        'keywords': ['financial', 'technology', 'stocks', 'funds','trading'],
        'save_directory': "/Users/daniellavin/Desktop/proj/Moneytrainer/newscsv",
        'output_directory': "/Users/daniellavin/Desktop/proj/Moneytrainer/stockdataset",
        'max_articles_per_keyword': 15,
    }
    checkpoint_path = "/Users/daniellavin/Desktop/proj/MoneyTrainer/checkpoints/SEVEN_EPOCH_14.pt"
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = StableTransformerModel(hidden_dim=768, num_layers=8, num_heads=12, dropout=0.1)
    model.to(device)
    
    # Load checkpoint
    load_checkpoint(checkpoint_path, model)
    
    # Generate input tensor and run inference
    input_tensor ,gt = generate_input_tensor(ticker=ticker, config=config)
    output = inference(input_tensor, model, device)
    
    return output, gt




if __name__ == "__main__":
    
    stock_symbols = extract_symbols_from_csv('/Users/daniellavin/Desktop/proj/Moneytrainer/cleaned_stockscreen.csv')
    evaluate_inference(stock_symbols, '/Users/daniellavin/Desktop/proj/MoneyTrainer/EVALUATIONS/n50_seven_epoch_14.csv')