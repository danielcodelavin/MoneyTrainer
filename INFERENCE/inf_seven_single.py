import torch
import torch.nn as nn
import os
from inf_utilities import scrape_articles, play_scrape_articles, play_save_to_csv, prepare_single_stock_data, extract_symbols_from_csv, returngroundtruthstock, encode_and_attach, validate_and_clean_tensors, process_GT_stock_torch_files, process_stock_torch_files, prepare_text_data, process_text_data
from datetime import datetime, timedelta
import math

class StableTransformerModel(nn.Module):
    # The model class should mirror the one used in training for compatibility.
    def __init__(self, hidden_dim: int, num_layers: int, num_heads: int, dropout: float = 0.1):
        super(StableTransformerModel, self).__init__()
        # Model definition based on the training file
        self.hidden_dim = hidden_dim
        self.layers = nn.ModuleList([
            nn.TransformerEncoderLayer(hidden_dim, num_heads, dim_feedforward=hidden_dim*2, dropout=dropout)
            for _ in range(num_layers)
        ])
        self.fc_out = nn.Linear(hidden_dim, 1)
        
    def forward(self, x):
        for layer in self.layers:
            x = layer(x)
        return self.fc_out(x[:, 0])
    
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

def load_checkpoint(filepath: str, model: nn.Module):
    """Loads the model checkpoint from the specified filepath."""
    if os.path.isfile(filepath):
        checkpoint = torch.load(filepath)
        model.load_state_dict(checkpoint['model_state_dict'])
        print(f"Loaded checkpoint from {filepath}")
    else:
        print(f"Checkpoint file not found at {filepath}")

def inference(input_tensor: torch.Tensor, model: nn.Module, device: torch.device):
    """Runs inference on the given input tensor."""
    model.eval()  # Set model to evaluation mode
    with torch.no_grad():
        input_tensor = input_tensor.to(device)
        output = model(input_tensor)
    return output

def generate_input_tensor(ticker,config):
    """Placeholder for generating an input tensor for inference."""
    now = datetime.now()
    str_date = now.strftime('%Y-%m-%d')

    total_seconds = int(now.time().total_seconds())
    hours, remainder = divmod(total_seconds, 3600)
    minutes, seconds = divmod(remainder, 60)
    str_time = f"{hours:02}:{minutes:02}:{seconds:02}"
    stock_symbol = ticker.strip()
    stock_data = prepare_single_stock_data(ticker_symbol=stock_symbol, start_datetime=now, days=6, min_points=45)
    features = stock_data.view(-1)[1:]  # Slice the tensor to remove the first element (label)

    features_mean = features.mean()
    features_std = features.std()
    normalized_features = (features - features_mean) / features_std

    # Recombine the label and normalized features
    label = stock_data[0]
    stock_data = torch.cat([label.unsqueeze(0), normalized_features])
    # add positional encoding
    stock_data = micro_pos_encode(stock_data)
    #validate
    if stock_data is not None and isinstance(stock_data, torch.Tensor):
                    stock_data = encode_and_attach(stock_symbol, stock_data)
    else:
        print("FAIL : STOCK DATA NOT PREPARED")
    raw_news_data = prepare_text_data(enddate=str_date, endtime=str_time,save_directory=config['save_directory'], keywords=config['keywords'], days_back=7, max_articles_per_keyword=config['max_articles_per_keyword'])
    processed_news_data = process_text_data(raw_news_data)
   
    

def main(checkpoint_path, ticker, config):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = StableTransformerModel(hidden_dim=768, num_layers=8, num_heads=12, dropout=0.1)
    model.to(device)
    
    # Load checkpoint
    load_checkpoint(checkpoint_path, model)
    
    # Generate input tensor and run inference
    input_tensor = generate_input_tensor(ticker=ticker,config=config)
    output = inference(input_tensor, model, device)
    
    print("Inference Output:", output)

if __name__ == "__main__":
    config = {
        'keywords': ['financial', 'technology', 'stocks', 'funds','trading'],
        'save_directory': "/Users/daniellavin/Desktop/proj/Moneytrain/newscsv",
        'output_directory': "/Users/daniellavin/Desktop/proj/Moneytrain/stockdataset",
         'max_articles_per_keyword':  15,
    }
    checkpoint_path="path/to/your/checkpoint.pt"
    ticker="AAPL"
    main(checkpoint_path,ticker, config)
