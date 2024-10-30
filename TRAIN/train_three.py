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
import torch
from torch.utils.data import Dataset
import random
from torch.optim.lr_scheduler import OneCycleLR

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
        # If we don't have any files left in current folder, move to next folder
        if not self.current_folder_files:
            if not self.folder_queue:
                self.folder_queue = deque(self.folders)
                random.shuffle(self.folder_queue)
                self.current_epoch += 1
                raise StopIteration(f"Epoch {self.current_epoch} completed")
            
            # Set up new folder
            folder = self.folder_queue.popleft()
            self.current_folder_path = os.path.join(self.data_path, folder)
            self.current_gt_folder_path = os.path.join(self.current_folder_path, 'GT')
            
            # Load news data for new folder
            self.current_news_data = torch.load(os.path.join(self.current_folder_path, '0news.pt'))
            
            # Get all PT files except news.pt and shuffle them
            self.current_folder_files = [f for f in os.listdir(self.current_folder_path) 
                                       if f.endswith('.pt') and f != '0news.pt']
            random.shuffle(self.current_folder_files)

        # Calculate batch size for this iteration
        current_batch_size = min(self.batch_size, len(self.current_folder_files))
        
        # Take the files for this batch
        selected_files = self.current_folder_files[:current_batch_size]
        # Remove used files from the list
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

class TransformerModel(nn.Module):
    def __init__(self, hidden_dim: int, num_layers: int, num_heads: int, dropout: float = 0.1):
        super(TransformerModel, self).__init__()
        self.hidden_dim = hidden_dim
        self.embedding = nn.Linear(1, hidden_dim)
        self.transformer_layers = nn.ModuleList([
            nn.TransformerEncoderLayer(
                d_model=hidden_dim, 
                nhead=num_heads, 
                dim_feedforward=hidden_dim*4, 
                activation=nn.SiLU(),
                dropout=dropout  # Add dropout here
            )
            for _ in range(num_layers)
        ])
        self.output_layer = nn.Linear(hidden_dim, 1)
        self.dropout = nn.Dropout(dropout)  # Add dropout layer

    def forward(self, x):
        x = x.unsqueeze(-1)  # Add feature dimension
        x = self.embedding(x)
        x = self.dropout(x)  # Apply dropout after embedding
        for layer in self.transformer_layers:
            x = layer(x)
        return self.output_layer(x[:, -1, :]).squeeze(-1)  # Ensure output is [batch_size]
def train(model: nn.Module, train_loader: DataLoader, val_loader: DataLoader, optimizer: optim.Optimizer, 
          scheduler: torch.optim.lr_scheduler.OneCycleLR, criterion: nn.Module, num_epochs: int, 
          device: torch.device, checkpoint_path: str, info_path: str, gradient_accumulation_steps: int):
    model.to(device)
    
    for epoch in range(num_epochs):
        print("[  SAVING EPOCH  ]")
        if (epoch + 1) % 2 == 0:
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'scheduler_state_dict': scheduler.state_dict(),  # Save scheduler state
                'train_loss': train_loss,
                'val_loss': val_loss,}, f'{checkpoint_path}_{epoch+1}.pt')
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
                outputs = model(inputs)
                
                targets = targets.view(-1)
                
                loss = criterion(outputs, targets)
                loss = loss / gradient_accumulation_steps
                loss.backward()

                if (i + 1) % gradient_accumulation_steps == 0:
                    optimizer.step()
                    scheduler.step()  # Step the scheduler
                    optimizer.zero_grad()

                train_loss += loss.item() * gradient_accumulation_steps
                tbar.set_postfix(loss=loss.item() * gradient_accumulation_steps, 
                               folders=f"{processed_folders}/{total_folders}",
                               lr=f"{scheduler.get_last_lr()[0]:.2e}") 

        train_loss /= len(train_loader)

        model.eval()
        val_loss = 0.0
        # Similar modification for validation loader
        total_val_folders = len(val_loader.folders)
        processed_val_folders = 0
        current_val_folder = None

        with tqdm(total=total_val_folders, desc=f"Epoch {epoch+1}/{num_epochs} - Validation", unit="folder") as vbar:
            with torch.no_grad():
                for inputs, targets in val_loader:
                    # Check if we've moved to a new folder
                    new_folder = val_loader.folder_queue[0] if val_loader.folder_queue else None
                    if new_folder != current_val_folder:
                        current_val_folder = new_folder
                        processed_val_folders += 1
                        vbar.update(1)
                    
                    inputs, targets = inputs.to(device), targets.to(device)
                    outputs = model(inputs)
                    
                    # Ensure targets are the same shape as outputs
                    targets = targets.view(-1)
                    
                    loss = criterion(outputs, targets)
                    val_loss += loss.item()
                    # Update tqdm bar with current validation loss and folders processed
                    vbar.set_postfix(loss=loss.item(), 
                                   folders=f"{processed_val_folders}/{total_val_folders}")

        val_loss /= len(val_loader)

        print(f'Epoch {epoch+1}/{num_epochs}, Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}')

        # Write to info_path file
        with open(info_path, 'a') as f:
            f.write(f'Epoch {epoch+1}, Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}\n')


def main():
    # Hyperparameters and file paths
    hidden_dim = 640  # HIDDEN SIZE
    num_layers = 8
    num_heads = 8
    batch_size = 32
    num_epochs = 500
    learning_rate = 3e-5
    gradient_accumulation_steps = 16
    #dropout = 0.1
    train_data_path = '/usr/prakt/s0097/stockdataset'
    val_data_path = '/usr/prakt/s0097/valstockdataset'
    checkpoint_path = '/usr/prakt/s0097/newcheckpoints/EPOCH'
    info_path = '/usr/prakt/s0097/epochinformations/epochinfo.txt'
    #load_checkpoint = '/Users/daniellavin/Desktop/proj/Moneytrain/minitrash/miniepochs_epoch_100.pt'
    load_checkpoint = None
    use_cuda = True  # Set to False to force CPU usage

    # Device selection
    if use_cuda and torch.cuda.is_available():
        device = torch.device('cuda')
        print(f"Using CUDA. Device: {torch.cuda.get_device_name(0)}")
    else:
        device = torch.device('cpu')
        print("Using CPU.")

    # Create datasets and dataloaders
    train_dataset = CustomDataLoader(train_data_path,batch_size)
    val_dataset = CustomDataLoader(val_data_path,batch_size)
    # train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=False)
    # val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    train_loader = train_dataset
    val_loader = val_dataset    
    # Create model and optimizer
    model = TransformerModel(hidden_dim, num_layers, num_heads, dropout=0.1)  
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    scheduler = torch.optim.lr_scheduler.OneCycleLR(
    optimizer,
    max_lr=learning_rate,
    epochs=num_epochs,
    steps_per_epoch=len(train_dataset),  # Use dataset length
    pct_start=0.1,  # 10% warmup
    div_factor=25,  # Initial learning rate = max_lr/25
    final_div_factor=1e4  # Final learning rate = max_lr/10000
)
    criterion = nn.MSELoss()

    # Load from checkpoint if specified
    start_epoch = 0
    if load_checkpoint:
        checkpoint = torch.load(load_checkpoint, map_location=device)
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        if 'scheduler_state_dict' in checkpoint:
            scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        start_epoch = checkpoint['epoch'] + 1
        print(f'Loaded checkpoint from epoch {start_epoch}')

    # Move model to the selected device
    model.to(device)

    # Train the model
    train(model, train_loader, val_loader, optimizer, scheduler, criterion, 
      num_epochs - start_epoch, device, checkpoint_path, info_path, 
      gradient_accumulation_steps)

if __name__ == '__main__':
    main()