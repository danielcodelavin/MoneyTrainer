import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import numpy as np
from datetime import datetime
import os
import json
from pathlib import Path
from tqdm import tqdm

class PositionalEncoding(nn.Module):
    def __init__(self, d_model=32, max_len=80):
        super().__init__()
        position = torch.arange(max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) * (-np.log(10000.0) / d_model))
        pe = torch.zeros(max_len, d_model)
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe)

    def forward(self, x):
        x = x.unsqueeze(-1)
        x = x.repeat(1, 1, self.pe.shape[1])
        return x + self.pe

class MultimodalFusionModel(nn.Module):
    def __init__(self, hidden_dim=1024, dropout_rate=0.3):
        super().__init__()
        
        # Initial dimension processing
        self.initial_stock_proj = nn.Linear(1, hidden_dim)
        
        # Time series processing
        self.pos_encoder = PositionalEncoding(hidden_dim, 80)
        
        # Stock sequence processing
        self.stock_encoder = nn.Sequential(
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout_rate),
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim)
        )
        
        # LSTM for temporal processing
        self.lstm = nn.LSTM(
            input_size=hidden_dim,
            hidden_size=hidden_dim,
            num_layers=6,
            batch_first=True,
            dropout=dropout_rate,
            bidirectional=True
        )
        
        # Topic distribution processing
        self.topic_encoder = nn.Sequential(
            nn.Linear(100, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout_rate),
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim)
        )
        
        # Multi-layer cross attention
        self.cross_attention = nn.ModuleList([
            nn.MultiheadAttention(hidden_dim * 2, num_heads=8, dropout=0.1)
            for _ in range(3)
        ])
        
        # Layer normalization for attention
        self.layer_norms = nn.ModuleList([
            nn.LayerNorm(hidden_dim * 2)
            for _ in range(3)
        ])
        
        # Final prediction layers
        self.predictor = nn.Sequential(
            nn.Linear(hidden_dim * 6, hidden_dim),  # Input is now hidden_dim*6 due to concatenation
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout_rate),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.LayerNorm(hidden_dim // 2),
            nn.GELU(),
            nn.Dropout(dropout_rate),
            nn.Linear(hidden_dim // 2, 1)
        )
        
    def forward(self, stock_data, topic_data):
        batch_size = stock_data.shape[0]
        
        # Process stock data sequence
        stock_data = stock_data.unsqueeze(-1)  # [batch, 80, 1]
        stock_features = self.initial_stock_proj(stock_data)  # [batch, 80, hidden_dim]
        
        # Add positional encoding
        stock_features = stock_features + self.pos_encoder.pe
        
        # Process through stock encoder
        stock_features = self.stock_encoder(stock_features)  # [batch, 80, hidden_dim]
        
        # LSTM processing
        lstm_out, _ = self.lstm(stock_features)  # [batch, 80, hidden_dim*2] (bidirectional)
        
        # Process topic distribution
        topic_features = self.topic_encoder(topic_data)  # [batch, hidden_dim]
        topic_features = topic_features.unsqueeze(1)  # [batch, 1, hidden_dim]
        topic_features = topic_features.repeat(1, lstm_out.size(1), 1)  # [batch, 80, hidden_dim]
        topic_features = torch.cat([topic_features, topic_features], dim=-1)  # Match LSTM hidden dim
        
        # Multi-layer cross attention with residual connections
        x = lstm_out
        for attn, norm in zip(self.cross_attention, self.layer_norms):
            attn_out, _ = attn(
                x.transpose(0, 1),
                topic_features.transpose(0, 1),
                topic_features.transpose(0, 1)
            )
            x = norm(x + attn_out.transpose(0, 1))
        
        # Combine features - now properly accounting for dimensions
        # lstm_out is [batch, 80, hidden_dim*2] due to bidirectional
        lstm_final = x[:, -1, :]                    # [batch, hidden_dim*2]
        lstm_mean = x.mean(dim=1)                   # [batch, hidden_dim*2]
        topic_final = topic_features[:, 0, :]       # [batch, hidden_dim*2]
        
        # Concatenate all features
        global_features = torch.cat([
            lstm_final,      # hidden_dim*2
            lstm_mean,       # hidden_dim*2
            topic_final      # hidden_dim*2
        ], dim=1)           # Total: hidden_dim*6
        
        # Update predictor's first layer to match input size
        prediction = self.predictor(global_features)
        
        # Final prediction
        prediction = self.predictor(global_features)
        return prediction.squeeze(-1)

class CombinedLoss(nn.Module):
    def __init__(self, label_smoothing=0.1):
        super().__init__()
        self.mse = nn.MSELoss()
        self.huber = nn.HuberLoss(delta=0.1)
        self.label_smoothing = label_smoothing
        
    def forward(self, pred, target):
        # Smooth targets
        target_smooth = (1 - self.label_smoothing) * target + self.label_smoothing * target.mean()
        return self.mse(pred, target_smooth) * 0.7 + self.huber(pred, target) * 0.3

def load_data_from_directory(data_dir):
    data_dir = Path(data_dir)
    all_data = []
    
    for file_path in sorted(data_dir.glob('*.pt')):
        tensor = torch.load(file_path)
        all_data.append(tensor)
    
    if not all_data:
        raise ValueError(f"No .pt files found in {data_dir}")
    
    all_data = torch.stack(all_data)
    gt_values = all_data[:, 0]
    stock_data = all_data[:, 1:81]
    topic_data = all_data[:, 81:]
    
    return stock_data, topic_data, gt_values

class CustomDataset(Dataset):
    def __init__(self, stock_data, topic_data, targets):
        self.stock_data = stock_data
        self.topic_data = topic_data
        self.targets = targets
        
    def __len__(self):
        return len(self.targets)
        
    def __getitem__(self, idx):
        return self.stock_data[idx], self.topic_data[idx], self.targets[idx]

def compute_metrics(predictions, targets):
    predictions = predictions.squeeze().detach().cpu().numpy()
    targets = targets.squeeze().detach().cpu().numpy()
    
    l1_loss = np.mean(np.abs(predictions - targets))
    direction_accuracy = np.mean(np.sign(predictions) == np.sign(targets))
    
    ranges = [
        (-0.8, -0.4, "very negative"),
        (-0.4, -0.2, "negative"),
        (-0.2, 0.2, "neutral"),
        (0.2, 0.4, "positive"),
        (0.4, 0.8, "very positive")
    ]
    
    def get_range(value):
        for low, high, label in ranges:
            if low <= value < high:
                return label
        return "neutral"
    
    pred_ranges = np.array([get_range(x) for x in predictions])
    target_ranges = np.array([get_range(x) for x in targets])
    
    range_accuracy = np.mean(pred_ranges == target_ranges)
    
    extreme_mask = np.abs(targets) > 0.4
    if np.sum(extreme_mask) > 0:
        extreme_l1 = np.mean(np.abs(predictions[extreme_mask] - targets[extreme_mask]))
        extreme_direction_accuracy = np.mean(
            np.sign(predictions[extreme_mask]) == np.sign(targets[extreme_mask])
        )
    else:
        extreme_l1 = 0.0
        extreme_direction_accuracy = 0.0
    
    return {
        'l1_loss': float(l1_loss),
        'direction_accuracy': float(direction_accuracy),
        'range_accuracy': float(range_accuracy),
        'extreme_l1': float(extreme_l1),
        'extreme_direction_accuracy': float(extreme_direction_accuracy)
    }

def train_model(config):
    print("Loading training data...")
    train_stock_data, train_topic_data, train_targets = load_data_from_directory(config['train_data_dir'])
    print("Loading validation data...")
    val_stock_data, val_topic_data, val_targets = load_data_from_directory(config['val_data_dir'])
    
    print(f"Loaded {len(train_targets)} training samples and {len(val_targets)} validation samples")
    
    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    print(f"Using device: {device}")
    
    model = MultimodalFusionModel(
        hidden_dim=config['hidden_dim'],
        dropout_rate=config['dropout_rate']
    ).to(device)
    
    criterion = CombinedLoss(label_smoothing=config['label_smoothing'])
    
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=config['learning_rate'],
        weight_decay=config['weight_decay'],
        betas=(0.9, 0.999)
    )
    
    train_dataset = CustomDataset(train_stock_data, train_topic_data, train_targets)
    val_dataset = CustomDataset(val_stock_data, val_topic_data, val_targets)
    
    train_loader = DataLoader(
        train_dataset,
        batch_size=config['batch_size'],
        shuffle=True,
        num_workers=4,
        pin_memory=True
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=config['batch_size'],
        shuffle=False,
        num_workers=4,
        pin_memory=True
    )
    
    # Learning rate scheduler
    scheduler = torch.optim.lr_scheduler.OneCycleLR(
        optimizer,
        max_lr=config['learning_rate'],
        epochs=config['num_epochs'],
        steps_per_epoch=len(train_loader),
        pct_start=0.1,
        div_factor=25
    )
    
    best_val_loss = float('inf')
    accumulation_steps = 4
    
    with open(config['info_path'], 'w') as f:
        f.write(f"Training started at {datetime.now()}\n")
        f.write(f"Configuration: {json.dumps(config, indent=2)}\n")
        f.write(f"Training samples: {len(train_targets)}\n")
        f.write(f"Validation samples: {len(val_targets)}\n\n")
    
    for epoch in tqdm(range(config['num_epochs']), desc='Training epochs'):
        model.train()
        train_losses = []
        
        train_pbar = tqdm(train_loader, desc=f'Training batch (epoch {epoch+1}/{config["num_epochs"]})', 
                         leave=False)
        
        for i, (batch_stock, batch_topic, batch_targets) in enumerate(train_pbar):
            batch_stock = batch_stock.to(device)
            batch_topic = batch_topic.to(device)
            batch_targets = batch_targets.to(device)
            
            outputs = model(batch_stock, batch_topic)
            loss = criterion(outputs, batch_targets)
            
            # Scale loss for gradient accumulation
            loss = loss / accumulation_steps
            loss.backward()
            
            if (i + 1) % accumulation_steps == 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), config['grad_clip'])
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad()
            
            train_losses.append(loss.item() * accumulation_steps)
            train_pbar.set_postfix({'loss': f'{loss.item() * accumulation_steps:.4f}'})
        
        # Validation phase
        model.eval()
        val_losses = []
        all_predictions = []
        all_targets = []
        
        with torch.no_grad():
            val_pbar = tqdm(val_loader, desc=f'Validation batch (epoch {epoch+1}/{config["num_epochs"]})', 
                          leave=False)
            
            for batch_stock, batch_topic, batch_targets in val_pbar:
                batch_stock = batch_stock.to(device)
                batch_topic = batch_topic.to(device)
                batch_targets = batch_targets.to(device)
                
                outputs = model(batch_stock, batch_topic)
                val_loss = criterion(outputs, batch_targets)
                
                val_losses.append(val_loss.item())
                all_predictions.append(outputs)
                all_targets.append(batch_targets)
                val_pbar.set_postfix({'loss': f'{val_loss.item():.4f}'})
        
        val_predictions = torch.cat(all_predictions)
        val_targets = torch.cat(all_targets)
        metrics = compute_metrics(val_predictions, val_targets)
        
        avg_train_loss = np.mean(train_losses)
        avg_val_loss = np.mean(val_losses)
        
        with open(config['info_path'], 'a') as f:
            f.write(f"\nEpoch {epoch+1}/{config['num_epochs']}\n")
            f.write(f"Train Loss: {avg_train_loss:.4f}\n")
            f.write(f"Val Loss: {avg_val_loss:.4f}\n")
            f.write(f"Metrics: {json.dumps(metrics, indent=2)}\n")
            f.write(f"Learning Rate: {scheduler.get_last_lr()[0]:.6f}\n")
        
        if (epoch + 1) % config['checkpoint_frequency'] == 0:
            checkpoint_path = os.path.join(
                config['checkpoint_dir'],
                f'checkpoint_epoch_{epoch+1}.pt'
            )
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'scheduler_state_dict': scheduler.state_dict(),
                'val_loss': avg_val_loss,
            }, checkpoint_path)
        
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            best_model_path = os.path.join(
                config['checkpoint_dir'],
                'best_model.pt'
            )
            torch.save(model.state_dict(), best_model_path)

if __name__ == "__main__":
    config = {
        # Model parameters
        'hidden_dim': 768,
        'dropout_rate': 0.16,
        
        # Training parameters
        'batch_size': 64,
        'learning_rate': 2e-4,
        'weight_decay': 1e-4,
        'num_epochs': 100,
        'checkpoint_frequency': 10,
        'label_smoothing': 0.1,
        'grad_clip': 1.0,
        
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