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
from sklearn.utils.class_weight import compute_class_weight

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
    def __init__(self, hidden_dim=512, dropout_rate=0.5):  # Increased dropout
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
        
        # Reduced LSTM layers and added layer norm
        self.lstm = nn.LSTM(
            input_size=hidden_dim,
            hidden_size=hidden_dim,
            num_layers=4,  # Reduced from 6
            batch_first=True,
            dropout=dropout_rate,
            bidirectional=True
        )
        
        self.lstm_norm = nn.LayerNorm(hidden_dim * 2)
        
        self.topic_encoder = nn.Sequential(
            nn.Linear(100, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout_rate),
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim)
        )
        
        # Reduced attention layers
        self.cross_attention = nn.ModuleList([
            nn.MultiheadAttention(hidden_dim * 2, num_heads=8, dropout=dropout_rate)
            for _ in range(2)  # Reduced from 3
        ])
        
        self.layer_norms = nn.ModuleList([
            nn.LayerNorm(hidden_dim * 2)
            for _ in range(2)
        ])
        
        # Added L2 regularization via weight norm
        self.predictor = nn.Sequential(
            nn.Linear(hidden_dim * 6, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout_rate),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.LayerNorm(hidden_dim // 2),
            nn.GELU(),
            nn.Dropout(dropout_rate),
            nn.Linear(hidden_dim // 2, 1),
            nn.Sigmoid()
        )
        
    def forward(self, stock_data, topic_data):
        batch_size = stock_data.shape[0]
        
        stock_data = stock_data.unsqueeze(-1)
        stock_features = self.initial_stock_proj(stock_data)
        stock_features = stock_features + self.pos_encoder.pe
        stock_features = self.stock_encoder(stock_features)
        
        lstm_out, _ = self.lstm(stock_features)
        lstm_out = self.lstm_norm(lstm_out)
        
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
    def __init__(self, fp_weight=5.0, fn_weight=0.1):
        super().__init__()
        self.fp_weight = fp_weight
        self.fn_weight = fn_weight
        
    def forward(self, predictions, targets):
        # Base BCE loss
        bce_loss = F.binary_cross_entropy(predictions, targets, reduction='none')
        
        # Heavy penalty for false positives, very light penalty for false negatives
        false_positives = (1 - targets) * predictions 
        false_negatives = targets * (1 - predictions)
        
        weighted_loss = (
            bce_loss + 
            self.fp_weight * false_positives + 
            self.fn_weight * false_negatives
        )
        
        return weighted_loss.mean()

class CustomDataset(Dataset):
    def __init__(self, stock_data, topic_data, targets, threshold=0.35, augment_positives=True, target_ratio=0.15):
        # Use a more stringent threshold for positive samples
        binary_targets = (targets > threshold + 0.05).float()
        
        # Find positive samples
        positive_indices = torch.where(binary_targets == 1)[0]
        negative_indices = torch.where(binary_targets == 0)[0]
        
        num_positives = len(positive_indices)
        num_negatives = len(negative_indices)
        current_ratio = num_positives / (num_positives + num_negatives)
        
        print(f"Original class distribution - Positives: {current_ratio:.3f}, Samples: {len(targets)}")
        
        if augment_positives and current_ratio < target_ratio:
            # Calculate how many times to repeat positive samples
            target_positives = int((target_ratio * num_negatives) / (1 - target_ratio))
            repetitions_needed = int(np.ceil(target_positives / num_positives))
            
            # Repeat positive samples with small random variations
            augmented_positive_indices = []
            for _ in range(repetitions_needed):
                # Add small random noise to create variations
                noise_stock = torch.randn_like(stock_data[positive_indices]) * 0.01
                noise_topic = torch.randn_like(topic_data[positive_indices]) * 0.01
                
                augmented_stock = stock_data[positive_indices] + noise_stock
                augmented_topic = topic_data[positive_indices] + noise_topic
                
                if len(augmented_positive_indices) == 0:
                    self.stock_data = augmented_stock
                    self.topic_data = augmented_topic
                else:
                    self.stock_data = torch.cat([self.stock_data, augmented_stock])
                    self.topic_data = torch.cat([self.topic_data, augmented_topic])
                
                augmented_positive_indices.extend([1] * len(positive_indices))
            
            # Trim to desired length
            self.stock_data = self.stock_data[:target_positives]
            self.topic_data = self.topic_data[:target_positives]
            
            # Add negative samples
            self.stock_data = torch.cat([self.stock_data, stock_data[negative_indices]])
            self.topic_data = torch.cat([self.topic_data, topic_data[negative_indices]])
            self.targets = torch.cat([
                torch.ones(target_positives),
                torch.zeros(len(negative_indices))
            ])
            
            final_ratio = self.targets.mean().item()
            print(f"Augmented class distribution - Positives: {final_ratio:.3f}, Samples: {len(self.targets)}")
        else:
            self.stock_data = stock_data
            self.topic_data = topic_data
            self.targets = binary_targets
        
        # Calculate class weights for sampling
        class_weights = compute_class_weight(
            'balanced',
            classes=np.unique(self.targets.numpy()),
            y=self.targets.numpy()
        )
        self.weights = torch.tensor([class_weights[int(t)] for t in self.targets])
        
    def __len__(self):
        return len(self.targets)
        
    def __getitem__(self, idx):
        return self.stock_data[idx], self.topic_data[idx], self.targets[idx]



def compute_metrics(predictions, targets, threshold=0.7):  # Increased threshold for higher precision
    predictions = (predictions > threshold).float()
    predictions = predictions.cpu().numpy()
    targets = targets.cpu().numpy()
    
    tp = np.sum((predictions == 1) & (targets == 1))
    fp = np.sum((predictions == 1) & (targets == 0))
    tn = np.sum((predictions == 0) & (targets == 0))
    fn = np.sum((predictions == 0) & (targets == 1))
    
    accuracy = (tp + tn) / (tp + fp + tn + fn)
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0
    f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
    
    return {
        'accuracy': float(accuracy),
        'precision': float(precision),
        'recall': float(recall),
        'f1': float(f1),
        'true_positives': int(tp),
        'false_positives': int(fp),
        'true_negatives': int(tn),
        'false_negatives': int(fn)
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
        fp_weight=5.0,  # Very high penalty for false positives
        fn_weight=0.1   # Very low penalty for false negatives
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
        threshold=config['threshold'],
        augment_positives=config['augment_positives'],
        target_ratio=config['target_positive_ratio']
    )
    
    val_dataset = CustomDataset(
        val_stock_data, 
        val_topic_data, 
        val_targets,
        threshold=config['threshold'],
        augment_positives=False
    )
    
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
    
    # Changed to cosine annealing with warm restarts for better optimization
    scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
        optimizer,
        T_0=10,  # Reset every 10 epochs
        T_mult=2,  # Double the reset period after each restart
        eta_min=config['learning_rate'] * 1e-3  # Minimum learning rate
    )
    
    best_val_precision = 0.0

    accumulation_steps = 4  # Gradient accumulation for larger effective batch size
    
    with open(config['info_path'], 'w') as f:
        f.write(f"Training started at {datetime.now()}\n")
        f.write(f"Configuration: {json.dumps(config, indent=2)}\n")
        f.write(f"Training samples: {len(train_targets)}\n")
        f.write(f"Validation samples: {len(val_targets)}\n")
        f.write(f"Positive class ratio (train): {float((train_targets > config['threshold']).sum() / len(train_targets)):.3f}\n")
        f.write(f"Positive class ratio (val): {float((val_targets > config['threshold']).sum() / len(val_targets)):.3f}\n\n")
    
    for epoch in tqdm(range(config['num_epochs']), desc='Training epochs'):
        model.train()
        train_losses = []
        train_predictions = []
        train_true_labels = []
        
        train_pbar = tqdm(train_loader, desc=f'Training batch (epoch {epoch+1}/{config["num_epochs"]})', 
                         leave=False)
        
        optimizer.zero_grad()  # Reset gradients at the start of each epoch
        
        for i, (batch_stock, batch_topic, batch_targets) in enumerate(train_pbar):
            batch_stock = batch_stock.to(device)
            batch_topic = batch_topic.to(device)
            batch_targets = batch_targets.to(device)
            
            outputs = model(batch_stock, batch_topic)
            loss = criterion(outputs, batch_targets)
            
            # Normalize loss for gradient accumulation
            loss = loss / accumulation_steps
            loss.backward()
            
            if (i + 1) % accumulation_steps == 0:
                # Clip gradients to prevent explosion
                torch.nn.utils.clip_grad_norm_(model.parameters(), config['grad_clip'])
                optimizer.step()
                optimizer.zero_grad()
            
            train_losses.append(loss.item() * accumulation_steps)
            train_predictions.append(outputs.detach())
            train_true_labels.append(batch_targets)
            
            # Update progress bar
            train_pbar.set_postfix({
                'loss': f'{np.mean(train_losses[-10:]):.4f}',
                'lr': f'{scheduler.get_last_lr()[0]:.6f}'
            })
        
        # Compute training metrics
        train_predictions = torch.cat(train_predictions)
        train_true_labels = torch.cat(train_true_labels)
        train_metrics = compute_metrics(train_predictions, train_true_labels, threshold=config['prediction_threshold'])
        
        # Validation phase
        model.eval()
        val_losses = []
        val_predictions = []
        val_true_labels = []
        
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
                val_predictions.append(outputs)
                val_true_labels.append(batch_targets)
                
                val_pbar.set_postfix({'loss': f'{np.mean(val_losses[-10:]):.4f}'})
        
        # Compute validation metrics
        val_predictions = torch.cat(val_predictions)
        val_true_labels = torch.cat(val_true_labels)
        val_metrics = compute_metrics(val_predictions, val_true_labels, threshold=config['prediction_threshold'])
        
        # Update learning rate
        scheduler.step()
        
        # Calculate average losses
        avg_train_loss = np.mean(train_losses)
        avg_val_loss = np.mean(val_losses)
        
        # Log metrics
        with open(config['info_path'], 'a') as f:
            f.write(f"\nEpoch {epoch+1}/{config['num_epochs']}\n")
            f.write(f"Train Loss: {avg_train_loss:.4f}\n")
            f.write(f"Train Metrics: {json.dumps(train_metrics, indent=2)}\n")
            f.write(f"Val Loss: {avg_val_loss:.4f}\n")
            f.write(f"Val Metrics: {json.dumps(val_metrics, indent=2)}\n")
            f.write(f"Learning Rate: {scheduler.get_last_lr()[0]:.6f}\n")
        
        # Save best model based on precision
        if val_metrics['precision'] > best_val_precision:
            best_val_precision = val_metrics['precision']
            best_model_path = os.path.join(
                config['checkpoint_dir'],
                'best_precision_model.pt'
            )
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'scheduler_state_dict': scheduler.state_dict(),
                'val_metrics': val_metrics,
                'config': config,
            }, best_model_path)
        
        # Regular checkpointing
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
                'val_metrics': val_metrics,
                'train_metrics': train_metrics,
                'config': config,
            }, checkpoint_path)
        


if __name__ == "__main__":
    config = {
        # Model parameters
        'hidden_dim': 512,
        'dropout_rate': 0.7,  # Increased dropout
        
        # Class balance parameters
        'augment_positives': True,
        'target_positive_ratio': 0.15,
        
        # Training parameters
        'batch_size': 64,  # Reduced batch size
        'learning_rate': 1e-4,  # Reduced learning rate
        'weight_decay': 1e-3,  # Increased weight decay
        'num_epochs': 120,
        'checkpoint_frequency': 5,
        'grad_clip': 0.5,  # Reduced gradient clipping
        
        # Classification thresholds
        'threshold': 0.35,  # Threshold for converting regression values to binary classes
        'prediction_threshold': 0.7,  # Threshold for final predictions (higher for better precision)
        
        # Paths
        'train_data_dir': '/Users/daniellavin/Desktop/proj/MoneyTrainer/Y/y_100_train',
        'val_data_dir': '/Users/daniellavin/Desktop/proj/MoneyTrainer/Y/y_100_val',
        'checkpoint_dir': '/Users/daniellavin/Desktop/proj/MoneyTrainer/Y/YT_train/check3',
        'info_path': '/Users/daniellavin/Desktop/proj/MoneyTrainer/Y/YT_train/check3/training_info.txt',
    }
    
    os.makedirs(config['checkpoint_dir'], exist_ok=True)
    
    train_model(config)