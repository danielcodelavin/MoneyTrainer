import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import os
import random
from typing import List, Tuple
from tqdm import tqdm
from torch.utils.data import IterableDataset
import torch.nn.functional as F
from collections import deque
import math

class CustomDataLoader:
    def __init__(self, data_path, batch_size):
        self.data_path = data_path
        self.batch_size = batch_size
        self.folders = [f for f in os.listdir(data_path) if os.path.isdir(os.path.join(data_path, f))]
        self.folder_queue = deque(self.folders)
        self.current_epoch = 0
        self.current_folder_files = []
        self.current_news_data = None
        self.current_folder_path = None
        self.current_gt_folder_path = None

    def __iter__(self):
        return self

    def __next__(self):
        if not self.current_folder_files:
            if not self.folder_queue:
                self.folder_queue = deque(self.folders)
                random.shuffle(self.folder_queue)
                self.current_epoch += 1
                raise StopIteration(f"Epoch {self.current_epoch} completed")
            
            folder = self.folder_queue.popleft()
            self.current_folder_path = os.path.join(self.data_path, folder)
            self.current_gt_folder_path = os.path.join(self.current_folder_path, 'GT')
            
            self.current_news_data = torch.load(os.path.join(self.current_folder_path, '0news.pt'))
            
            self.current_folder_files = [f for f in os.listdir(self.current_folder_path) 
                                       if f.endswith('.pt') and f != '0news.pt']
            random.shuffle(self.current_folder_files)

        current_batch_size = min(self.batch_size, len(self.current_folder_files))
        selected_files = self.current_folder_files[:current_batch_size]
        self.current_folder_files = self.current_folder_files[current_batch_size:]
        
        batch_inputs = []
        batch_targets = []
        
        for file in selected_files:
            input_tensor = torch.load(os.path.join(self.current_folder_path, file))
            input_tensor = torch.cat([input_tensor, self.current_news_data], dim=0)
            batch_inputs.append(input_tensor)
            
            gt_file = file.replace('.pt', '_GT.pt')
            gt_tensor = torch.load(os.path.join(self.current_gt_folder_path, gt_file))
            batch_targets.append(gt_tensor.view(1))

        batch_inputs_padded = torch.stack([tensor for tensor in batch_inputs])
        
        return batch_inputs_padded, torch.cat(batch_targets)

    def __len__(self):
        return len(self.folders)


class ImprovedTransformerModel(nn.Module):
    def __init__(self, hidden_dim: int, num_layers: int, num_heads: int, dropout: float = 0.1):
        super(ImprovedTransformerModel, self).__init__()
        self.hidden_dim = hidden_dim
        
        # Separate normalization for different input types
        self.stock_norm = nn.LayerNorm(1)
        self.text_norm = nn.LayerNorm(1)
        
        # Separate embeddings for stock and text data
        self.stock_embedding = nn.Sequential(
            nn.Linear(1, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU()
        )
        
        self.text_embedding = nn.Sequential(
            nn.Linear(1, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU()
        )
        
        # Type embeddings to differentiate stock and text data
        self.type_embeddings = nn.Parameter(torch.randn(2, hidden_dim))
        
        # Transformer layers with skip connections
        self.transformer_layers = nn.ModuleList([
            TransformerLayerWithSkip(hidden_dim, num_heads, dropout)
            for _ in range(num_layers)
        ])
        
        # Sophisticated output head
        self.output_head = nn.Sequential(
            nn.LayerNorm(hidden_dim * 2),  # Double size to account for skip connections
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, 1)
        )
        
        self.dropout = nn.Dropout(dropout)
        self._init_weights()
    
    def _init_weights(self):
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
    
    def create_attention_mask(self, seq_len: int, stock_len: int, text_len: int):
        # Create causal mask within each segment
        mask = torch.ones(seq_len, seq_len, dtype=torch.bool)
        
        # Allow full attention within stock data segment
        stock_end = stock_len + 1  # +1 for label
        mask[:stock_end, :stock_end] = 0
        
        # Allow full attention within text data segment
        text_start = stock_end
        mask[text_start:, text_start:] = 0
        
        # Allow limited cross-attention between segments
        mask[text_start:, :stock_end] = 0  # Text can attend to stock
        mask[:stock_end, text_start:] = 0  # Stock can attend to text
        
        return mask
    
    def forward(self, x):
        batch_size, seq_len = x.shape
        
        # Split input into stock and text components
        stock_len = seq_len - 750 - 1  # -1 for label
        label = x[:, 0].unsqueeze(-1)  # Shape: [batch_size, 1]
        stock_data = x[:, 1:stock_len+1].unsqueeze(-1)  # Shape: [batch_size, stock_len, 1]
        text_data = x[:, -750:].unsqueeze(-1)  # Shape: [batch_size, 750, 1]
        
        # Process stock data
        stock_data = self.stock_norm(stock_data)
        stock_embedded = self.stock_embedding(stock_data)  # Shape: [batch_size, stock_len, hidden_dim]
        
        # Process text data
        text_data = self.text_norm(text_data)
        text_embedded = self.text_embedding(text_data)  # Shape: [batch_size, 750, hidden_dim]
        
        # Process label - need to embed it to match dimensions
        label_embedded = self.stock_embedding(label).unsqueeze(1)  # Shape: [batch_size, 1, hidden_dim]
        
        # Add type embeddings
        stock_embedded = stock_embedded + self.type_embeddings[0]
        text_embedded = text_embedded + self.type_embeddings[1]
        
        # Combine embeddings
        x = torch.cat([label_embedded, stock_embedded, text_embedded], dim=1)
        x = self.dropout(x)
        
        # Create attention mask
        attention_mask = self.create_attention_mask(seq_len, stock_len, 750).to(x.device)
        
        # Store skip connections
        skip_connections = []
        
        # Process through transformer layers
        for layer in self.transformer_layers:
            x, skip = layer(x, attention_mask)
            skip_connections.append(skip)
        
        # Combine final output with skip connections
        skip_tensor = torch.mean(torch.stack(skip_connections), dim=0)
        combined = torch.cat([x[:, 0], skip_tensor[:, 0]], dim=-1)
        
        return self.output_head(combined)

class TransformerLayerWithSkip(nn.Module):
    def __init__(self, hidden_dim: int, num_heads: int, dropout: float = 0.1):
        super().__init__()
        self.norm1 = nn.LayerNorm(hidden_dim)
        self.attention = nn.MultiheadAttention(hidden_dim, num_heads, dropout=dropout, batch_first=True)
        self.norm2 = nn.LayerNorm(hidden_dim)
        self.feedforward = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim * 4),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim * 4, hidden_dim)
        )
        self.dropout = nn.Dropout(dropout)
        
        # Skip connection processing
        self.skip_proj = nn.Linear(hidden_dim, hidden_dim)
        self.skip_norm = nn.LayerNorm(hidden_dim)
    
    def forward(self, x, attention_mask=None):
        # Self-attention
        residual = x
        x = self.norm1(x)
        x_att, _ = self.attention(x, x, x, attn_mask=attention_mask)
        x = residual + self.dropout(x_att)
        
        # Feedforward
        residual = x
        x = self.norm2(x)
        x = residual + self.dropout(self.feedforward(x))
        
        # Process skip connection
        skip = self.skip_norm(self.skip_proj(x))
        
        return x, skip

def log_cosh_loss(pred, target, weights=None):
    loss = torch.log(torch.cosh(pred - target))
    if weights is not None:
        loss = loss * weights
    return loss.mean()

def calculate_label_weights(targets, num_bins=100):
    # Move targets to CPU for histogram calculation
    targets_cpu = targets.cpu()
    
    # Create histogram of target values
    hist = torch.histogram(targets_cpu, bins=num_bins)
    bin_edges = hist.bin_edges
    bin_values = hist.hist
    
    # Calculate bin centers
    bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2
    
    # Calculate weights (inverse of frequency)
    epsilon = 1e-5  # Prevent division by zero
    weights = 1.0 / (bin_values + epsilon)
    weights = weights / weights.sum() * len(weights)  # Normalize weights
    
    # Create weight mapping function
    def get_weight(target):
        # Move inputs to CPU for bin calculation
        target_cpu = target.cpu()
        bin_idx = torch.searchsorted(bin_edges[:-1], target_cpu)
        bin_idx = torch.clamp(bin_idx, 0, len(weights) - 1)
        # Move weights to the same device as the target
        weights_device = weights.to(target.device)
        return weights_device[bin_idx]
    
    return get_weight


def train(model: nn.Module, train_loader: DataLoader, val_loader: DataLoader, 
          optimizer: optim.Optimizer, scheduler: torch.optim.lr_scheduler.OneCycleLR, 
          num_epochs: int, device: torch.device, 
          checkpoint_path: str, info_path: str, gradient_accumulation_steps: int):
    
    model.to(device)
    scaler = torch.cuda.amp.GradScaler()
    best_val_loss = float('inf')
    patience = 10
    patience_counter = 0
    
    # Collect initial batch of targets to calculate weights
    initial_targets = []
    for _, targets in train_loader:
        initial_targets.extend(targets.tolist())
        if len(initial_targets) > 10000:  # Sample size for distribution
            break
    get_weight = calculate_label_weights(torch.tensor(initial_targets))
    
    for epoch in range(num_epochs):
        model.train()
        train_loss = 0.0
        optimizer.zero_grad()

        total_folders = len(train_loader.folders)
        processed_folders = 0
        current_folder = None

        with tqdm(total=total_folders, desc=f"Epoch {epoch+1}/{num_epochs} - Training", unit="folder") as tbar:
            for i, (inputs, targets) in enumerate(train_loader):
                new_folder = train_loader.folder_queue[0] if train_loader.folder_queue else None
                if new_folder != current_folder:
                    current_folder = new_folder
                    processed_folders += 1
                    tbar.update(1)
                
                inputs, targets = inputs.to(device), targets.to(device)
                weights = get_weight(targets)
                
                with torch.cuda.amp.autocast():
                    outputs = model(inputs)
                    targets = targets.view(-1)
                    loss = log_cosh_loss(outputs, targets, weights)
                    scaled_loss = loss / gradient_accumulation_steps
                
                scaler.scale(scaled_loss).backward()

                if (i + 1) % gradient_accumulation_steps == 0:
                    scaler.unscale_(optimizer)
                    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                    
                    scaler.step(optimizer)
                    scaler.update()
                    scheduler.step()
                    optimizer.zero_grad()

                train_loss += loss.item()
                tbar.set_postfix(loss=loss.item(),
                               folders=f"{processed_folders}/{total_folders}",
                               lr=f"{scheduler.get_last_lr()[0]:.2e}")

        train_loss /= len(train_loader)

        model.eval()
        val_loss = 0.0
        total_val_folders = len(val_loader.folders)
        processed_val_folders = 0
        current_val_folder = None

        with tqdm(total=total_val_folders, desc=f"Epoch {epoch+1}/{num_epochs} - Validation", unit="folder") as vbar:
            with torch.no_grad():
                for inputs, targets in val_loader:
                    new_folder = val_loader.folder_queue[0] if val_loader.folder_queue else None
                    if new_folder != current_val_folder:
                        current_val_folder = new_folder
                        processed_val_folders += 1
                        vbar.update(1)
                    
                    inputs, targets = inputs.to(device), targets.to(device)
                    outputs = model(inputs)
                    targets = targets.view(-1)
                    loss = log_cosh_loss(outputs, targets)
                    val_loss += loss.item()
                    vbar.set_postfix(loss=loss.item(), 
                                   folders=f"{processed_val_folders}/{total_val_folders}")

        val_loss /= len(val_loader)

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience_counter = 0
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'scheduler_state_dict': scheduler.state_dict(),
                'train_loss': train_loss,
                'val_loss': val_loss,
            }, f'{checkpoint_path}_best.pt')
        else:
            patience_counter += 1
            if patience_counter >= patience:
                print(f"Early stopping triggered after {epoch + 1} epochs")
                break

        if (epoch + 1) % 2 == 0:
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'scheduler_state_dict': scheduler.state_dict(),
                'train_loss': train_loss,
                'val_loss': val_loss,
            }, f'{checkpoint_path}_{epoch+1}.pt')

        print(f'Epoch {epoch+1}/{num_epochs}, Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}')
        with open(info_path, 'a') as f:
            f.write(f'Epoch {epoch+1}, Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}\n')

def main():
    # Hyperparameters and file paths
    hidden_dim = 768  # Increased from 640
    num_layers = 8
    num_heads = 12    # Increased from 8
    batch_size = 16
    num_epochs = 500
    learning_rate = 5e-5  # Reduced for more stable training
    gradient_accumulation_steps = 8
    
    train_data_path = '/usr/prakt/s0097/stockdataset'
    val_data_path = '/usr/prakt/s0097/valstockdataset'
    checkpoint_path = '/usr/prakt/s0097/newcheckpoints/FIVE_EPOCH'
    info_path = '/usr/prakt/s0097/epochinformations/fiveepochinfo.txt'
    load_checkpoint = None
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    if torch.cuda.is_available():
        print(f"GPU: {torch.cuda.get_device_name(0)}")
        # Enable cudnn autotuner
        torch.backends.cudnn.benchmark = True

    # Create datasets and dataloaders
    train_dataset = CustomDataLoader(train_data_path, batch_size)
    val_dataset = CustomDataLoader(val_data_path, batch_size)
    train_loader = train_dataset
    val_loader = val_dataset
    
    # Create model and optimizer
    model = ImprovedTransformerModel(hidden_dim, num_layers, num_heads, dropout=0.1)
    optimizer = optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=0.01,
                           betas=(0.9, 0.999), eps=1e-8)
    
    scheduler = torch.optim.lr_scheduler.OneCycleLR(
        optimizer,
        max_lr=learning_rate,
        epochs=num_epochs,
        steps_per_epoch=len(train_dataset),
        pct_start=0.3,
        div_factor=10,
        final_div_factor=1e4
    )
    
    if load_checkpoint:
        checkpoint = torch.load(load_checkpoint)
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        print(f"Loaded checkpoint from {load_checkpoint}")

    # Train the model
    train(model, train_loader, val_loader, optimizer, scheduler, 
          num_epochs, device, checkpoint_path, info_path, 
          gradient_accumulation_steps)

if __name__ == "__main__":
    main()