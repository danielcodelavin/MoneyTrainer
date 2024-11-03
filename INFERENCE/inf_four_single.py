import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List, Dict
import os
import math
from datetime import datetime
from inf_utilities import prepare_single_stock_data, prepare_text_data, encode_and_attach ,process_text_data

class ImprovedTransformerModel(nn.Module):
    def __init__(self, hidden_dim: int, num_layers: int, num_heads: int, dropout: float = 0.1):
        super(ImprovedTransformerModel, self).__init__()
        self.hidden_dim = hidden_dim
        
        # Input normalization
        self.input_norm = nn.LayerNorm(1)
        
        # Improved embedding
        self.embedding = nn.Sequential(
            nn.Linear(1, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU()
        )
        
        # Pre-norm transformer layers
        self.transformer_layers = nn.ModuleList([
            nn.TransformerEncoderLayer(
                d_model=hidden_dim,
                nhead=num_heads,
                dim_feedforward=hidden_dim*4,
                dropout=dropout,
                activation="gelu",  # Changed from nn.GELU() to "gelu"
                norm_first=True  # Pre-norm architecture
            )
            for _ in range(num_layers)
        ])
        
        # Sophisticated output head with multiple layers
        self.output_head = nn.Sequential(
            nn.LayerNorm(hidden_dim),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, hidden_dim // 4),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 4, 1),
            nn.Tanh()  # Added to constrain output to [-1, 1] range
        )
        
        self.dropout = nn.Dropout(dropout)
        self._init_weights()
    
    def _init_weights(self):
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
    
    def forward(self, x):
        x = x.unsqueeze(-1)
        x = self.input_norm(x)
        x = self.embedding(x)
        x = self.dropout(x)
        
        for layer in self.transformer_layers:
            x = layer(x)
        
        return self.output_head(x[:, -1, :]).squeeze(-1)

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
    now = datetime.now()
    
    str_date = now.strftime('%Y-%m-%d')
    str_time = now.strftime('%H:%M:%S')


    stock_symbol = ticker.strip()
    stock_data = prepare_single_stock_data(ticker_symbol=stock_symbol, start_datetime=now, days=6, min_points=45)
    features = stock_data

    features_mean = features.mean()
    features_std = features.std()
    normalized_features = (features - features_mean) / features_std

    
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
    return output_tensor



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

def main(checkpoint_path, ticker, config):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = ImprovedTransformerModel(hidden_dim=640, num_layers=8, num_heads=8, dropout=0.1)
    model.to(device)
    
    # Load checkpoint
    load_checkpoint(checkpoint_path, model)
    
    # Generate input tensor and run inference
    input_tensor = generate_input_tensor(ticker=ticker, config=config)
    output = inference(input_tensor, model, device)
    
    print("Inference Output:", output)

if __name__ == "__main__":
    config = {
        'keywords': ['financial', 'technology', 'stocks', 'funds','trading'],
        'save_directory': "/Users/daniellavin/Desktop/proj/Moneytrain/newscsv_four",
       # 'output_directory': "/Users/daniellavin/Desktop/proj/Moneytrain/stockdataset",
        'max_articles_per_keyword': 15,
    }
    checkpoint_path = "/Users/daniellavin/Desktop/proj/MoneyTrainer/checkpoints/FOUR_EPOCH_14.pt"
    ticker = "MAR"
    main(checkpoint_path, ticker, config)