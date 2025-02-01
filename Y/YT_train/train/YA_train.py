import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import numpy as np
from datetime import datetime
import os
import json
from pathlib import Path

def load_data_from_directory(data_dir):
    """
    Load all .pt files from a directory and organize them into stock data,
    probability vectors, and GT values.
    Expected format: Each sample is a 1D tensor with GT as first element,
    followed by 80 stock values and 100 probability values.
    """
    data_dir = Path(data_dir)
    all_data = []
    
    # Load all .pt files from directory
    for file_path in sorted(data_dir.glob('*.pt')):
        tensor = torch.load(file_path)
        all_data.append(tensor)
    
    if not all_data:
        raise ValueError(f"No .pt files found in {data_dir}")
    
    # Stack all tensors
    all_data = torch.stack(all_data)
    
    # Split into components
    gt_values = all_data[:, 0]  # First element is GT
    stock_data = all_data[:, 1:81]  # Next 80 elements are stock data
    prob_data = all_data[:, 81:]  # Last 100 elements are probability distribution
    
    # Verify shapes
    assert stock_data.shape[1] == 80, f"Stock data should have 80 values, got {stock_data.shape[1]}"
    assert prob_data.shape[1] == 100, f"Probability data should have 100 values, got {prob_data.shape[1]}"
    
    # Verify probability distribution properties
    assert torch.all(prob_data >= 0) and torch.all(prob_data <= 1), "Probability values must be between 0 and 1"
    assert torch.allclose(prob_data.sum(dim=1), torch.ones(prob_data.shape[0])), "Probability distributions must sum to 1"
    
    return stock_data, prob_data, gt_values

class MultimodalFusionModel(nn.Module):
    def __init__(self, stock_seq_length=80, prob_vector_length=100, hidden_dim=128):
        super().__init__()
        
        # Stock data processing (temporal) - LSTM branch
        self.lstm = nn.LSTM(
            input_size=1,
            hidden_size=hidden_dim,
            num_layers=2,
            batch_first=True,
            dropout=0.2
        )
        
        # Probability vector processing (distributional) - Dense branch
        self.prob_encoder = nn.Sequential(
            nn.Linear(prob_vector_length, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU()
        )
        
        # Attention mechanism for cross-modal interaction
        self.attention = nn.MultiheadAttention(hidden_dim, num_heads=4)
        
        # Final prediction layers
        self.fusion_layer = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(hidden_dim, 1)
        )
        
        # Kalman filter parameters (learnable)
        self.kf_transition = nn.Parameter(torch.ones(1))
        self.kf_measurement = nn.Parameter(torch.ones(1))
        
    def forward(self, stock_data, prob_data):
        # Process stock data through LSTM
        lstm_out, _ = self.lstm(stock_data.unsqueeze(-1))
        lstm_features = lstm_out[:, -1, :]  # Take last hidden state
        
        # Process probability vector
        prob_features = self.prob_encoder(prob_data)
        
        # Cross-modal attention
        attn_output, _ = self.attention(
            lstm_features.unsqueeze(0),
            prob_features.unsqueeze(0),
            prob_features.unsqueeze(0)
        )
        attn_output = attn_output.squeeze(0)
        
        # Concatenate features
        combined_features = torch.cat([attn_output, prob_features], dim=1)
        
        # Final prediction
        prediction = self.fusion_layer(combined_features)
        
        return prediction

class CustomDataset(Dataset):
    def __init__(self, stock_data, prob_data, targets):
        self.stock_data = stock_data
        self.prob_data = prob_data
        self.targets = targets
        
    def __len__(self):
        return len(self.targets)
        
    def __getitem__(self, idx):
        return self.stock_data[idx], self.prob_data[idx], self.targets[idx]

def compute_metrics(predictions, targets):
    predictions = predictions.detach().cpu().numpy()
    targets = targets.detach().cpu().numpy()
    
    # L1 loss
    l1_loss = np.mean(np.abs(predictions - targets))
    
    # Custom accuracy for regression (within a threshold)
    threshold = 0.5
    accuracy = np.mean(np.abs(predictions - targets) < threshold)
    
    # Custom precision (for positive predictions)
    positive_pred_mask = predictions > 0
    if np.sum(positive_pred_mask) > 0:
        precision = np.mean(targets[positive_pred_mask] > 0)
    else:
        precision = 0.0
        
    return {
        'l1_loss': l1_loss,
        'accuracy': accuracy,
        'precision': precision
    }

def train_model(config):
    # Load data
    print("Loading training data...")
    train_stock_data, train_prob_data, train_targets = load_data_from_directory(config['train_data_dir'])
    print("Loading validation data...")
    val_stock_data, val_prob_data, val_targets = load_data_from_directory(config['val_data_dir'])
    
    print(f"Loaded {len(train_targets)} training samples and {len(val_targets)} validation samples")
    
    # Set device
    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # Initialize model
    model = MultimodalFusionModel(
        stock_seq_length=80,
        prob_vector_length=100,
        hidden_dim=config['hidden_dim']
    ).to(device)
    
    # Initialize optimizer and loss
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=config['learning_rate'],
        weight_decay=config['weight_decay']
    )
    
    criterion = nn.HuberLoss(delta=1.0)
    
    # Create data loaders
    train_dataset = CustomDataset(train_stock_data, train_prob_data, train_targets)
    val_dataset = CustomDataset(val_stock_data, val_prob_data, val_targets)
    
    train_loader = DataLoader(
        train_dataset,
        batch_size=config['batch_size'],
        shuffle=True
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=config['batch_size'],
        shuffle=False
    )
    
    # Training loop
    best_val_loss = float('inf')
    
    # Create info file
    with open(config['info_path'], 'w') as f:
        f.write(f"Training started at {datetime.now()}\n")
        f.write(f"Configuration: {json.dumps(config, indent=2)}\n")
        f.write(f"Training samples: {len(train_targets)}\n")
        f.write(f"Validation samples: {len(val_targets)}\n\n")
    
    for epoch in range(config['num_epochs']):
        model.train()
        train_losses = []
        
        for batch_stock, batch_prob, batch_targets in train_loader:
            batch_stock = batch_stock.to(device)
            batch_prob = batch_prob.to(device)
            batch_targets = batch_targets.to(device)
            
            optimizer.zero_grad()
            outputs = model(batch_stock, batch_prob)
            loss = criterion(outputs.squeeze(), batch_targets)
            
            loss.backward()
            optimizer.step()
            
            train_losses.append(loss.item())
        
        # Validation phase
        model.eval()
        val_losses = []
        all_predictions = []
        all_targets = []
        
        with torch.no_grad():
            for batch_stock, batch_prob, batch_targets in val_loader:
                batch_stock = batch_stock.to(device)
                batch_prob = batch_prob.to(device)
                batch_targets = batch_targets.to(device)
                
                outputs = model(batch_stock, batch_prob)
                val_loss = criterion(outputs.squeeze(), batch_targets)
                
                val_losses.append(val_loss.item())
                all_predictions.append(outputs)
                all_targets.append(batch_targets)
        
        # Compute metrics
        val_predictions = torch.cat(all_predictions)
        val_targets = torch.cat(all_targets)
        metrics = compute_metrics(val_predictions, val_targets)
        
        # Log information
        avg_train_loss = np.mean(train_losses)
        avg_val_loss = np.mean(val_losses)
        
        with open(config['info_path'], 'a') as f:
            f.write(f"\nEpoch {epoch+1}/{config['num_epochs']}\n")
            f.write(f"Train Loss: {avg_train_loss:.4f}\n")
            f.write(f"Val Loss: {avg_val_loss:.4f}\n")
            f.write(f"Metrics: {json.dumps(metrics, indent=2)}\n")
        
        # Save checkpoint if needed
        if (epoch + 1) % config['checkpoint_frequency'] == 0:
            checkpoint_path = os.path.join(
                config['checkpoint_dir'],
                f'checkpoint_epoch_{epoch+1}.pt'
            )
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_loss': avg_val_loss,
            }, checkpoint_path)
        
        # Save best model
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            best_model_path = os.path.join(
                config['checkpoint_dir'],
                'best_model.pt'
            )
            torch.save(model.state_dict(), best_model_path)

if __name__ == "__main__":
    # Configuration
    config = {
        # Model parameters
        'hidden_dim': 512,
        
        # Training parameters
        'batch_size': 128,
        'learning_rate': 5e-5,
        'weight_decay': 1e-6,
        'num_epochs': 100,
        'checkpoint_frequency': 10,
        
        # Paths
        'train_data_dir': '/Users/daniellavin/Desktop/proj/MoneyTrainer/Y/y_100_train',  # Directory containing training .pt files
        'val_data_dir': '/Users/daniellavin/Desktop/proj/MoneyTrainer/Y/y_100_val',      # Directory containing validation .pt files
        'checkpoint_dir': '/Users/daniellavin/Desktop/proj/MoneyTrainer/Y/YT_train/check',
        'info_path': '/Users/daniellavin/Desktop/proj/MoneyTrainer/Y/YT_train/check/training_info.txt',
    }
    
    # Create checkpoint directory if it doesn't exist
    os.makedirs(config['checkpoint_dir'], exist_ok=True)
    
    # Start training
    train_model(config)