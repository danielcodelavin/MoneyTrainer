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
        
        self.initial_stock_proj = nn.Linear(1, hidden_dim)
        self.pos_encoder = PositionalEncoding(hidden_dim, 80)
        
        self.stock_encoder = nn.Sequential(
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout_rate),
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim)
        )
        
        self.lstm = nn.LSTM(
            input_size=hidden_dim,
            hidden_size=hidden_dim,
            num_layers=6,
            batch_first=True,
            dropout=dropout_rate,
            bidirectional=True
        )
        
        self.topic_encoder = nn.Sequential(
            nn.Linear(100, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout_rate),
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim)
        )
        
        self.cross_attention = nn.ModuleList([
            nn.MultiheadAttention(hidden_dim * 2, num_heads=8, dropout=0.1)
            for _ in range(3)
        ])
        
        self.layer_norms = nn.ModuleList([
            nn.LayerNorm(hidden_dim * 2)
            for _ in range(3)
        ])
        
        self.predictor = nn.Sequential(
            nn.Linear(hidden_dim * 6, hidden_dim),
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
        
        stock_data = stock_data.unsqueeze(-1)
        stock_features = self.initial_stock_proj(stock_data)
        stock_features = stock_features + self.pos_encoder.pe
        stock_features = self.stock_encoder(stock_features)
        
        lstm_out, _ = self.lstm(stock_features)
        
        topic_features = self.topic_encoder(topic_data)
        topic_features = topic_features.unsqueeze(1)
        topic_features = topic_features.repeat(1, lstm_out.size(1), 1)
        topic_features = torch.cat([topic_features, topic_features], dim=-1)
        
        x = lstm_out
        for attn, norm in zip(self.cross_attention, self.layer_norms):
            attn_out, _ = attn(
                x.transpose(0, 1),
                topic_features.transpose(0, 1),
                topic_features.transpose(0, 1)
            )
            x = norm(x + attn_out.transpose(0, 1))
        
        lstm_final = x[:, -1, :]
        lstm_mean = x.mean(dim=1)
        topic_final = topic_features[:, 0, :]
        
        global_features = torch.cat([
            lstm_final,
            lstm_mean,
            topic_final
        ], dim=1)
        
        prediction = self.predictor(global_features)
        return prediction.squeeze(-1)

class PrecisionFocusedLoss(nn.Module):
    def __init__(self, high_threshold=0.3, fp_penalty=2.0, label_smoothing=0.1):
        super().__init__()
        self.mse = nn.MSELoss(reduction='none')
        self.huber = nn.HuberLoss(delta=0.1, reduction='none')
        self.label_smoothing = label_smoothing
        self.high_threshold = high_threshold
        self.fp_penalty = fp_penalty
        
    def forward(self, pred, target):
        # Smooth targets
        target_smooth = (1 - self.label_smoothing) * target + self.label_smoothing * target.mean()
        
        # Base losses with distance weighting
        mse_loss = self.mse(pred, target_smooth)
        huber_loss = self.huber(pred, target)
        
        # Distance-based weights (give more importance to larger movements)
        weights = 1.0 + torch.abs(target)
        
        # Additional weights for high values
        high_value_mask = target > self.high_threshold
        weights = torch.where(high_value_mask, weights * 4, weights)
        
        # Precision focus: penalize false positives for high values
        false_positive_mask = (pred > self.high_threshold) & (target <= self.high_threshold)
        weights = torch.where(false_positive_mask, weights * self.fp_penalty, weights)
        
        # Combine losses with weights
        total_loss = (0.7 * mse_loss + 0.3 * huber_loss) * weights
        
        return total_loss.mean()

class CustomDataset(Dataset):
    def __init__(self, stock_data, topic_data, targets, high_threshold=0.3):
        self.stock_data = stock_data
        self.topic_data = topic_data
        self.targets = targets
        
        # Calculate weights for sampling
        self.weights = torch.ones(len(targets))
        high_mask = targets > high_threshold
        self.weights[high_mask] = 10.0  # Moderate increase for high-value cases
        
    def __len__(self):
        return len(self.targets)
        
    def __getitem__(self, idx):
        return self.stock_data[idx], self.topic_data[idx], self.targets[idx]

def compute_metrics(predictions, targets, high_threshold=0.3):
    predictions = predictions.squeeze().detach().cpu().numpy()
    targets = targets.squeeze().detach().cpu().numpy()
    
    # Standard metrics
    l1_loss = np.mean(np.abs(predictions - targets))
    direction_accuracy = np.mean(np.sign(predictions) == np.sign(targets))
    
    # High value precision metrics
    pred_high = predictions > high_threshold
    true_high = targets > high_threshold
    
    tp = np.sum(pred_high & true_high)
    fp = np.sum(pred_high & ~true_high)
    fn = np.sum(~pred_high & true_high)
    
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0
    f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
    
    # Movement range metrics
    ranges = [
        (-0.8, -0.4, "very negative"),
        (-0.4, -0.2, "negative"),
        (-0.2, 0.2, "neutral"),
        (0.2, 0.3, "moderate positive"),
        (0.3, 0.8, "high positive")
    ]
    
    def get_range(value):
        for low, high, label in ranges:
            if low <= value < high:
                return label
        return "neutral"
    
    pred_ranges = np.array([get_range(x) for x in predictions])
    target_ranges = np.array([get_range(x) for x in targets])
    
    range_accuracy = np.mean(pred_ranges == target_ranges)
    
    # Distribution analysis
    value_distributions = {
        "high_positive_pred": float(np.mean(predictions > high_threshold)),
        "high_positive_true": float(np.mean(targets > high_threshold)),
        "avg_high_value": float(np.mean(predictions[pred_high])) if np.any(pred_high) else 0
    }
    
    return {
        'l1_loss': float(l1_loss),
        'direction_accuracy': float(direction_accuracy),
        'range_accuracy': float(range_accuracy),
        'high_value_precision': float(precision),
        'high_value_recall': float(recall),
        'high_value_f1': float(f1),
        'distributions': value_distributions
    }

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
    
    criterion = PrecisionFocusedLoss(
        high_threshold=config['high_threshold'],
        fp_penalty=config['fp_penalty'],
        label_smoothing=config['label_smoothing']
    )
    
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=config['learning_rate'],
        weight_decay=config['weight_decay'],
        betas=(0.9, 0.999)
    )
    
    train_dataset = CustomDataset(
        train_stock_data, 
        train_topic_data, 
        train_targets,
        high_threshold=config['high_threshold']
    )
    val_dataset = CustomDataset(
        val_stock_data, 
        val_topic_data, 
        val_targets,
        high_threshold=config['high_threshold']
    )
    
    # Use weighted sampling for training
    train_sampler = torch.utils.data.WeightedRandomSampler(
        weights=train_dataset.weights,
        num_samples=len(train_dataset),
        replacement=True
    )
    
    train_loader = DataLoader(
        train_dataset,
        batch_size=config['batch_size'],
        sampler=train_sampler,
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
    
    scheduler = torch.optim.lr_scheduler.OneCycleLR(
        optimizer,
        max_lr=config['learning_rate'],
        epochs=config['num_epochs'],
        steps_per_epoch=len(train_loader),
        pct_start=0.1,
        div_factor=25
    )
    
    best_val_precision = 0.0
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
            
            loss = loss / accumulation_steps
            loss.backward()
            
            if (i + 1) % accumulation_steps == 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), config['grad_clip'])
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad()
            
            train_losses.append(loss.item() * accumulation_steps)
            train_pbar.set_postfix({'loss': f'{loss.item() * accumulation_steps:.4f}'})
        
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
        metrics = compute_metrics(val_predictions, val_targets, high_threshold=config['high_threshold'])
        
        avg_train_loss = np.mean(train_losses)
        avg_val_loss = np.mean(val_losses)
        
        with open(config['info_path'], 'a') as f:
            f.write(f"\nEpoch {epoch+1}/{config['num_epochs']}\n")
            f.write(f"Train Loss: {avg_train_loss:.4f}\n")
            f.write(f"Val Loss: {avg_val_loss:.4f}\n")
            f.write(f"Metrics: {json.dumps(metrics, indent=2)}\n")
            f.write(f"Learning Rate: {scheduler.get_last_lr()[0]:.6f}\n")
        
        # Save checkpoint based on high-value precision instead of validation loss
        if metrics['high_value_precision'] > best_val_precision:
            best_val_precision = metrics['high_value_precision']
            best_model_path = os.path.join(
                config['checkpoint_dir'],
                'best_precision_model.pt'
            )
            torch.save(model.state_dict(), best_model_path)
        
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
                'metrics': metrics,
            }, checkpoint_path)

if __name__ == "__main__":
    config = {
        # Model parameters
        'hidden_dim': 768,
        'dropout_rate': 0.15,
        
        # Training parameters
        'batch_size': 64,
        'learning_rate': 2e-4,
        'weight_decay': 1e-4,
        'num_epochs': 100,
        'checkpoint_frequency': 10,
        'label_smoothing': 0.1,
        'grad_clip': 1.0,
        
        # Precision focus parameters
        'high_threshold': 0.35,
        'fp_penalty': 4.0,
        
        # Paths
        'train_data_dir': '/Users/daniellavin/Desktop/proj/MoneyTrainer/Y/y_100_train',
        'val_data_dir': '/Users/daniellavin/Desktop/proj/MoneyTrainer/Y/y_100_val',
        'checkpoint_dir': '/Users/daniellavin/Desktop/proj/MoneyTrainer/Y/YT_train/check2',
        'info_path': '/Users/daniellavin/Desktop/proj/MoneyTrainer/Y/YT_train/check2/training_info.txt',
    }
    
    os.makedirs(config['checkpoint_dir'], exist_ok=True)
    
    train_model(config)