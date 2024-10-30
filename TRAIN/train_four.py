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
                activation=nn.GELU(),
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

def custom_criterion(pred, target):
    mse_loss = F.mse_loss(pred, target)
    huber_loss = F.smooth_l1_loss(pred, target)
    return mse_loss + 0.5 * huber_loss

def train(model: nn.Module, train_loader: DataLoader, val_loader: DataLoader, 
          optimizer: optim.Optimizer, scheduler: torch.optim.lr_scheduler.OneCycleLR, 
          criterion: nn.Module, num_epochs: int, device: torch.device, 
          checkpoint_path: str, info_path: str, gradient_accumulation_steps: int):
    
    model.to(device)
    scaler = torch.cuda.amp.GradScaler()
    best_val_loss = float('inf')
    patience = 10
    patience_counter = 0
    
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
                    targets = targets.view(-1)
                    loss = criterion(outputs, targets)
                    scaled_loss = loss / gradient_accumulation_steps
                
                scaler.scale(scaled_loss).backward()

                if (i + 1) % gradient_accumulation_steps == 0:
                    scaler.unscale_(optimizer)
                    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                    
                    scaler.step(optimizer)
                    scaler.update()
                    scheduler.step()
                    optimizer.zero_grad()

                train_loss += loss.item()  # Store the true loss, not the scaled loss
                tbar.set_postfix(loss=loss.item(),  # Show true loss in progress bar
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
                    loss = criterion(outputs, targets)
                    val_loss += loss.item()
                    vbar.set_postfix(loss=loss.item(), 
                                   folders=f"{processed_val_folders}/{total_val_folders}")

        val_loss /= len(val_loader)

        # Early stopping check
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
    hidden_dim = 640
    num_layers = 8
    num_heads = 8
    batch_size = 32
    num_epochs = 500
    learning_rate = 2e-4  # Reduced from original
    gradient_accumulation_steps = 8
    
    train_data_path = '/usr/prakt/s0097/stockdataset'
    val_data_path = '/usr/prakt/s0097/valstockdataset'
    checkpoint_path = '/usr/prakt/s0097/newcheckpoints/FOUR_EPOCH'  # Changed path
    info_path = '/usr/prakt/s0097/epochinformations/fourepochinfo.txt'  # Changed path
    load_checkpoint = None
    
    # Device selection
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    if torch.cuda.is_available():
        print(f"GPU: {torch.cuda.get_device_name(0)}")

    # Create datasets and dataloaders
    train_dataset = CustomDataLoader(train_data_path, batch_size)
    val_dataset = CustomDataLoader(val_data_path, batch_size)
    train_loader = train_dataset
    val_loader = val_dataset
    
    # Create model and optimizer
    model = ImprovedTransformerModel(hidden_dim, num_layers, num_heads, dropout=0.1)
    optimizer = optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=0.01)
    
    scheduler = torch.optim.lr_scheduler.OneCycleLR(
        optimizer,
        max_lr=learning_rate,
        epochs=num_epochs,
        steps_per_epoch=len(train_dataset),
        pct_start=0.3,  # 30% warmup
        div_factor=10,
        final_div_factor=1e4
    )
    
    criterion = custom_criterion

    if load_checkpoint:
        checkpoint = torch.load(load_checkpoint, map_location=device)
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        if 'scheduler_state_dict' in checkpoint:
            scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        start_epoch = checkpoint['epoch'] + 1
        print(f'Loaded checkpoint from epoch {start_epoch}')
    
    model.to(device)

    train(model, train_loader, val_loader, optimizer, scheduler, criterion, 
          num_epochs, device, checkpoint_path, info_path, gradient_accumulation_steps)

if __name__ == '__main__':
    main()