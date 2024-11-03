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


def generate_input_tensor(ticker, config):
    """Generate input tensor with error handling and validation."""
    max_retries = 3
    current_try = 0
    
    while current_try < max_retries:
        try:
            # Generate random datetime logic remains the same
            earliest_date = datetime(2022, 12, 1)
            latest_date = datetime(2024, 10, 10)
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
            gt_datetime = parsed_datetime + timedelta(days=1)

            stock_symbol = ticker.strip()
            stock_data = prepare_single_stock_data(ticker_symbol=stock_symbol, start_datetime=parsed_datetime, days=6, min_points=45)
            
            # Validate stock data
            if stock_data is None or torch.isnan(stock_data).any() or (stock_data == 0).all():
                raise ValueError("Invalid stock data received")
            
            reference_value = stock_data[0].item()
            features = stock_data

            # Validate features statistics
            features_mean = features.mean()
            features_std = features.std()
            if features_std == 0:
                raise ValueError("Zero standard deviation in features")
                
            normalized_features = (features - features_mean) / features_std

            gt_tensor = returngroundtruthstock(stock_symbol, gt_datetime)
            # Validate ground truth data
            if gt_tensor is None or torch.isnan(gt_tensor).any():
                raise ValueError("Invalid ground truth data received")
                
            gt_value = gt_tensor.item()
            gt = (gt_value - reference_value) / reference_value

            stock_data = micro_pos_encode(normalized_features)
            if stock_data is None or not isinstance(stock_data, torch.Tensor):
                raise ValueError("Failed to prepare stock data")
                
            stock_data = encode_and_attach(stock_symbol, stock_data)
            
            raw_news_data = prepare_text_data(
                enddate=str_date,
                endtime=str_time,
                save_directory=config['save_directory'],
                keywords=config['keywords'],
                days_back=7,
                max_articles_per_keyword=config['max_articles_per_keyword']
            )
            
            if raw_news_data is None:
                raise ValueError("Failed to prepare news data")
                
            processed_news_data = process_text_data(raw_news_data)
            if processed_news_data is None:
                raise ValueError("Failed to process news data")

            output_tensor = torch.cat((stock_data, processed_news_data), dim=0)
            
            # Final validation of output tensor
            if torch.isnan(output_tensor).any():
                raise ValueError("NaN values in final output tensor")
                
            return output_tensor, gt

        except Exception as e:
            current_try += 1
            print(f"Attempt {current_try} failed for ticker {ticker}: {str(e)}")
            if current_try >= max_retries:
                print(f"Failed to generate valid data for ticker {ticker} after {max_retries} attempts")
                return None, None
            
    return None, None




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
    Handles failures gracefully and continues processing.
    """
    def calculate_metrics(predictions: List[float], ground_truth: List[float]) -> dict:
        """Calculate various metrics for binary classification and L1 distance."""
        if not predictions or not ground_truth:
            return {
                'average_l1': 0.0,
                'std_l1': 0.0,
                'precision': 0.0,
                'recall': 0.0,
                'accuracy': 0.0,
                'f1_score': 0.0,
                'valid_samples': 0
            }
            
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
            'f1_score': f1,
            'valid_samples': len(predictions)
        }
    
    def evaluate_single_ticker(ticker: str) -> tuple[str, dict]:
        """Evaluate a single ticker with multiple runs."""
        predictions = []
        ground_truth = []
        failed_attempts = 0
        max_attempts = 75  # Higher than 50 to allow for some failures
        
        while len(predictions) < 50 and failed_attempts < max_attempts:
            try:
                pred, gt = returnpred(ticker)
                
                # Skip if we got None values
                if pred is None or gt is None:
                    failed_attempts += 1
                    continue
                    
                # Skip if we got invalid values
                if (math.isnan(pred) or math.isnan(gt) or 
                    math.isinf(pred) or math.isinf(gt)):
                    failed_attempts += 1
                    continue
                
                predictions.append(pred)
                ground_truth.append(gt)
                
            except Exception as e:
                failed_attempts += 1
                print(f"Error processing ticker {ticker}, attempt {failed_attempts}: {str(e)}")
                continue
        
        metrics = calculate_metrics(predictions, ground_truth)
        return ticker, metrics
    
    # Prepare CSV headers
    headers = ['Ticker', 'Average_L1', 'Std_L1', 'Precision', 'Recall', 
              'Accuracy', 'F1_Score', 'Valid_Samples', 'Status']
    
    # Evaluate all tickers and write results
    with open(output_path, 'w', newline='') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(headers)
        
        for ticker in tickers:
            try:
                print(f"Processing ticker: {ticker}")
                ticker, metrics = evaluate_single_ticker(ticker)
                
                # Determine status based on number of valid samples
                status = "Complete" if metrics['valid_samples'] >= 50 else \
                         "Partial" if metrics['valid_samples'] > 0 else "Failed"
                
                row = [
                    ticker,
                    metrics['average_l1'],
                    metrics['std_l1'],
                    metrics['precision'],
                    metrics['recall'],
                    metrics['accuracy'],
                    metrics['f1_score'],
                    metrics['valid_samples'],
                    status
                ]
                writer.writerow(row)
                csvfile.flush()  # Ensure data is written even if program crashes
                
            except Exception as e:
                print(f"Failed to process ticker {ticker}: {str(e)}")
                # Write error record
                row = [ticker, 0, 0, 0, 0, 0, 0, 0, f"Error: {str(e)}"]
                writer.writerow(row)
                csvfile.flush()
                continue  # Continue with next ticker

def returnpred(ticker):
    """Return prediction with error handling."""
    try:
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
        
        load_checkpoint(checkpoint_path, model)
        
        input_tensor, gt = generate_input_tensor(ticker=ticker, config=config)
        if input_tensor is None or gt is None:
            return None, None
            
        output = inference(input_tensor, model, device)
        if math.isnan(output) or math.isinf(output):
            return None, None
            
        return output, gt
        
    except Exception as e:
        print(f"Error in returnpred for ticker {ticker}: {str(e)}")
        return None, None


if __name__ == "__main__":
    
    stock_symbols = extract_symbols_from_csv('/Users/daniellavin/Desktop/proj/Moneytrainer/cleaned_stockscreen.csv')
    stock_symbols = stock_symbols[len(stock_symbols) // 4 : len(stock_symbols) // 2]
    evaluate_inference(stock_symbols, '/Users/daniellavin/Desktop/proj/MoneyTrainer/EVALUATIONS/second_n50_seven_epoch_14.csv')