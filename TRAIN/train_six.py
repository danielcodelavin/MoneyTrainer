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


class StableTransformerModel(nn.Module):
    def __init__(self, hidden_dim: int, num_layers: int, num_heads: int, dropout: float = 0.1):
        super(StableTransformerModel, self).__init__()
        self.hidden_dim = hidden_dim
        
        # Remove normalization since data is already well-scaled
        self.stock_embedding = nn.Sequential(
            nn.Linear(1, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.Tanh()  # Changed to Tanh since our data is in [-1, 1]
        )
        
        self.text_embedding = nn.Sequential(
            nn.Linear(1, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.Tanh()  # Changed to Tanh since our data is in [-1, 1]
        )
        
        self.type_embeddings = nn.Parameter(torch.zeros(2, hidden_dim))
        
        self.transformer_layers = nn.ModuleList([
            StableTransformerLayer(hidden_dim, num_heads, dropout)
            for _ in range(num_layers)
        ])
        
        # Simplified output head
        self.output_head = nn.Sequential(
            nn.LayerNorm(hidden_dim),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.Tanh(),
            nn.Linear(hidden_dim // 2, 1),
            nn.Tanh()  # Final Tanh to ensure predictions stay in [-1, 1]
        )
        
        self.dropout = nn.Dropout(dropout)
        self._init_weights()
    
    def _init_weights(self):
        for module in self.modules():
            if isinstance(module, nn.Linear):
                # Using Kaiming initialization for ReLU activation
                nn.init.kaiming_normal_(module.weight, mode='fan_out', nonlinearity='relu')
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
    
    def create_attention_mask(self, seq_len: int, stock_len: int, text_len: int):
        mask = torch.zeros(seq_len, seq_len, dtype=torch.bool)
        stock_end = 1 + stock_len
        text_start = stock_end
        
        # Simplified masking strategy
        mask[text_start:, :stock_end] = True
        return mask

    def forward(self, x):
        batch_size, seq_len = x.shape
        
        # Split input
        stock_len = seq_len - 750 - 1
        label = x[:, 0].unsqueeze(-1)
        stock_data = x[:, 1:stock_len+1].unsqueeze(-1)
        text_data = x[:, -750:].unsqueeze(-1)
        
        # Embeddings (removed normalization)
        stock_embedded = self.stock_embedding(stock_data)
        text_embedded = self.text_embedding(text_data)
        label_embedded = self.stock_embedding(label).unsqueeze(1)
        
        # Add type embeddings
        stock_embedded = stock_embedded + self.type_embeddings[0]
        text_embedded = text_embedded + self.type_embeddings[1]
        
        # Combine sequences
        x = torch.cat([label_embedded, stock_embedded, text_embedded], dim=1)
        x = self.dropout(x)
        
        # Create attention mask
        attention_mask = self.create_attention_mask(x.shape[1], stock_len, 750).to(x.device)
        
        # Process through transformer layers
        for layer in self.transformer_layers:
            x = layer(x, attention_mask)
        
        # Extract first token for prediction
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
        # Pre-norm architecture for better training stability
        normed = self.norm1(x)
        x = x + self.dropout(self.attention(normed, normed, normed, attn_mask=attention_mask)[0])
        
        normed = self.norm2(x)
        x = x + self.dropout(self.feedforward(normed))
        
        return x

def train(model: nn.Module, train_loader: DataLoader, val_loader: DataLoader, 
          optimizer: optim.Optimizer, scheduler: torch.optim.lr_scheduler.OneCycleLR, 
          num_epochs: int, device: torch.device, 
          checkpoint_path: str, info_path: str, gradient_accumulation_steps: int):
    
    model.to(device)
    scaler = torch.cuda.amp.GradScaler()
    best_val_loss = float('inf')
    patience = 10
    patience_counter = 0
    
    # Use MSE loss with gradient clipping
    criterion = nn.MSELoss()
    
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
                
                with torch.cuda.amp.autocast():
                    outputs = model(inputs)
                    targets = targets.view(-1, 1)
                    loss = criterion(outputs, targets)
                    scaled_loss = loss / gradient_accumulation_steps
                
                scaler.scale(scaled_loss).backward()

                if (i + 1) % gradient_accumulation_steps == 0:
                    scaler.unscale_(optimizer)
                    # Clip gradients to prevent explosion
                    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=0.5)
                    
                    scaler.step(optimizer)
                    scaler.update()
                    scheduler.step()
                    optimizer.zero_grad()

                train_loss += loss.item()
                tbar.set_postfix(loss=f"{loss.item():.6f}",
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
                    targets = targets.view(-1, 1)
                    loss = criterion(outputs, targets)
                    val_loss += loss.item()
                    vbar.set_postfix(loss=f"{loss.item():.6f}", 
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

        print(f'Epoch {epoch+1}/{num_epochs}, Train Loss: {train_loss:.6f}, Val Loss: {val_loss:.6f}')
        with open(info_path, 'a') as f:
            f.write(f'Epoch {epoch+1}, Train Loss: {train_loss:.6f}, Val Loss: {val_loss:.6f}\n')

def main():
    # Modified hyperparameters
    hidden_dim = 512  # Reduced further since we don't need as much capacity
    num_layers = 6    # Reduced layers
    num_heads = 8
    batch_size = 8
    num_epochs = 100
    learning_rate = 5e-7  
    gradient_accumulation_steps = 16
    
    train_data_path = '/usr/prakt/s0097/stockdataset'
    val_data_path = '/usr/prakt/s0097/valstockdataset'
    checkpoint_path = '/usr/prakt/s0097/newcheckpoints/SIX_EPOCH_low'
    info_path = '/usr/prakt/s0097/epochinformations/sixepochinfolower.txt'
    load_checkpoint = None
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    if torch.cuda.is_available():
        print(f"GPU: {torch.cuda.get_device_name(0)}")
        torch.backends.cudnn.benchmark = True

    # Create datasets and dataloaders
    train_dataset = CustomDataLoader(train_data_path, batch_size)
    val_dataset = CustomDataLoader(val_data_path, batch_size)
    train_loader = train_dataset
    val_loader = val_dataset
    
    model = StableTransformerModel(hidden_dim, num_layers, num_heads, dropout=0.1)
    
    # Use AdamW with smaller weight decay
    optimizer = optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=0.001,
                           betas=(0.9, 0.999), eps=1e-8)
    
    scheduler = torch.optim.lr_scheduler.OneCycleLR(
        optimizer,
        max_lr=learning_rate,
        epochs=num_epochs,
        steps_per_epoch=len(train_dataset),
        pct_start=0.1,
        div_factor=10,  # Smaller div factor for more stable learning
        final_div_factor=100
    )
    
    if load_checkpoint:
        checkpoint = torch.load(load_checkpoint)
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        print(f"Loaded checkpoint from {load_checkpoint}")

    train(model, train_loader, val_loader, optimizer, scheduler, 
          num_epochs, device, checkpoint_path, info_path, 
          gradient_accumulation_steps)

if __name__ == "__main__":
    main()