import torch
import torch.nn as nn
import torch.optim as optim
import os
import random
from collections import deque
from tqdm import tqdm
import matplotlib.pyplot as plt


def test_model(model, test_loader, criterion):
    model.eval()
    test_loss = 0.0
    all_predictions = []
    all_targets = []
    
    test_progress_bar = tqdm(total=len(test_loader), desc="Testing")
    
    with torch.no_grad():
        while True:
            try:
                inputs, targets = next(test_loader)
                outputs = model(inputs)
                loss = criterion(outputs, targets)
                test_loss += loss.item()
                
                all_predictions.extend(outputs.cpu().numpy())
                all_targets.extend(targets.cpu().numpy())
                
                test_progress_bar.update(1)
                test_progress_bar.set_postfix({'loss': f"{loss.item():.4f}"})
            except StopIteration:
                break  # End of test set
    
    test_progress_bar.close()
    avg_test_loss = test_loss / len(test_loader)
    
    # Calculate additional metrics if needed
    # For example, you could calculate Mean Absolute Error (MAE)
    mae = torch.mean(torch.abs(torch.tensor(all_predictions) - torch.tensor(all_targets))).item()
    
    print(f"Test Loss: {avg_test_loss:.4f}, MAE: {mae:.4f}")
    
    return avg_test_loss, mae, all_predictions, all_targets

class SwiGLU(nn.Module):
    def __init__(self, in_features):
        super().__init__()
        self.linear1 = nn.Linear(in_features, in_features)
        self.linear2 = nn.Linear(in_features, in_features)
        self.beta = nn.Parameter(torch.ones(1))

    def forward(self, x):
        return self.linear1(x) * torch.sigmoid(self.beta * self.linear2(x))

class TransformerBlock(nn.Module):
    def __init__(self, d_model, nhead):
        super().__init__()
        self.attention = nn.MultiheadAttention(d_model, nhead)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.swiglu = SwiGLU(d_model)

    def forward(self, x):
        attn_output, _ = self.attention(x, x, x)
        x = self.norm1(x + attn_output)
        swiglu_output = self.swiglu(x)
        x = self.norm2(x + swiglu_output)
        return x

class TransformerRegression(nn.Module):
    def __init__(self, d_model, nhead, num_layers):
        super().__init__()
        self.d_model = d_model
        self.input_projection = nn.Linear(1, d_model)  # Project each scalar to d_model dimensions
        self.transformer_blocks = nn.ModuleList([TransformerBlock(d_model, nhead) for _ in range(num_layers)])
        self.regression_head = nn.Linear(d_model, 1)

    def forward(self, x):
        # x shape: (batch_size, sequence_length)
        x = x.unsqueeze(-1)  # Add feature dimension: (batch_size, sequence_length, 1)
        x = self.input_projection(x)  # Project to d_model: (batch_size, sequence_length, d_model)
        x = x.transpose(0, 1)  # Transpose to (sequence_length, batch_size, d_model) for transformer input
        
        for block in self.transformer_blocks:
            x = block(x)
        
        # Use the output of the last token for regression
        x = x[-1]
        return self.regression_head(x).squeeze(-1)

class CustomDataLoader:
    def __init__(self, data_path, batch_size):
        self.data_path = data_path
        self.batch_size = batch_size
        self.folders = [f for f in os.listdir(data_path) if os.path.isdir(os.path.join(data_path, f))]
        self.folder_queue = deque(self.folders)
        self.current_epoch = 0

    def __iter__(self):
        return self

    def __next__(self):
        if not self.folder_queue:
            self.folder_queue = deque(self.folders)
            random.shuffle(self.folder_queue)
            self.current_epoch += 1
            raise StopIteration(f"Epoch {self.current_epoch} completed")

        folder = self.folder_queue.popleft()
        folder_path = os.path.join(self.data_path, folder)
        gt_folder_path = os.path.join(folder_path, '/GT')
        
        # Load news data
        news_data = torch.load(os.path.join(folder_path, '0news.pt'))
        
        # Get list of all .pt files in the folder (excluding 0news.pt)
        pt_files = [f for f in os.listdir(folder_path) if f.endswith('.pt') and f != '0news.pt']
        
        # Randomly select batch_size files
        selected_files = random.sample(pt_files, min(self.batch_size, len(pt_files)))
        
        batch_inputs = []
        batch_targets = []
        
        for file in selected_files:
            # Load input tensor
            input_tensor = torch.load(os.path.join(folder_path, file))
            input_tensor = torch.cat([input_tensor, news_data], dim=0)  # Append news data
            batch_inputs.append(input_tensor)
            
            # Load ground truth
            gt_file = file.replace('.pt', '_GT.pt')
            gt_tensor = torch.load(os.path.join(gt_folder_path, gt_file))
            batch_targets.append(gt_tensor)
        
        # Pad sequences to the same length
        max_len = max(tensor.size(0) for tensor in batch_inputs)
        batch_inputs_padded = torch.stack([torch.nn.functional.pad(tensor, (0, max_len - tensor.size(0))) for tensor in batch_inputs])
        
        return batch_inputs_padded, torch.stack(batch_targets)

    def __len__(self):
        return len(self.folders)
    

def train_model(model, train_loader, val_loader, num_epochs, learning_rate, weight_decay, checkpoint_dir, accumulation_steps):
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
    
    train_losses = []
    val_losses = []
    steps = 0
    
    for epoch in range(num_epochs):
        model.train()
        train_loss = 0.0
        train_progress_bar = tqdm(total=len(train_loader), desc=f"Epoch {epoch+1}/{num_epochs} [Train]")
        optimizer.zero_grad()  # Zero gradients at the beginning of each epoch
        
        batch_count = 0
        while True:
            try:
                inputs, targets = next(train_loader)
                outputs = model(inputs)
                loss = criterion(outputs, targets)
                loss = loss / accumulation_steps  # Normalize the loss
                loss.backward()
                
                batch_count += 1
                if batch_count % accumulation_steps == 0 or batch_count == len(train_loader):
                    optimizer.step()
                    optimizer.zero_grad()
                    steps += 1
                    
                    if steps % 20000 == 0:
                        checkpoint_path = os.path.join(checkpoint_dir, f"checkpoint_step_{steps}.pt")
                        torch.save({
                            'step': steps,
                            'model_state_dict': model.state_dict(),
                            'optimizer_state_dict': optimizer.state_dict(),
                            'loss': loss.item() * accumulation_steps,  # Denormalize the loss for logging
                        }, checkpoint_path)
                
                train_loss += loss.item() * accumulation_steps  # Denormalize the loss for logging
                train_progress_bar.update(1)
                train_progress_bar.set_postfix({'loss': f"{loss.item() * accumulation_steps:.4f}"})
                
            except StopIteration:
                break  # End of epoch
        
        train_progress_bar.close()
        avg_train_loss = train_loss / len(train_loader)
        train_losses.append(avg_train_loss)
        
        model.eval()
        val_loss = 0.0
        val_progress_bar = tqdm(total=len(val_loader), desc=f"Epoch {epoch+1}/{num_epochs} [Val]")
        with torch.no_grad():
            while True:
                try:
                    inputs, targets = next(val_loader)
                    outputs = model(inputs)
                    loss = criterion(outputs, targets)
                    val_loss += loss.item()
                    val_progress_bar.update(1)
                    val_progress_bar.set_postfix({'loss': f"{loss.item():.4f}"})
                except StopIteration:
                    break  # End of epoch
        
        val_progress_bar.close()
        avg_val_loss = val_loss / len(val_loader)
        val_losses.append(avg_val_loss)
        
        print(f"Epoch {epoch+1}/{num_epochs}, Train Loss: {avg_train_loss:.4f}, Val Loss: {avg_val_loss:.4f}")
    
    # Plot training and validation loss
    plt.figure(figsize=(10, 5))
    plt.plot(train_losses, label='Training Loss')
    plt.plot(val_losses, label='Validation Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Training and Validation Loss')
    plt.legend()
    plt.savefig(os.path.join(checkpoint_dir, 'loss_plot.png'))
    plt.close()

# Hyperparameters
d_model = 768
nhead = 8
num_layers = 12
num_epochs = 100
learning_rate = 0.001
weight_decay = 0.01
batch_size = 8
accumulation_steps = 8  # New hyperparameter for gradient accumulation

# Create model
model = TransformerRegression(d_model, nhead, num_layers)

# Create dataloaders
train_data_path = 'path/to/train/data'
val_data_path = 'path/to/val/data'
train_loader = CustomDataLoader(train_data_path, batch_size)
val_loader = CustomDataLoader(val_data_path, batch_size)

# Set up checkpoint directory
checkpoint_dir = 'path/to/checkpoints'
os.makedirs(checkpoint_dir, exist_ok=True)

# Train the model
train_model(model, train_loader, val_loader, num_epochs, learning_rate, weight_decay, checkpoint_dir, accumulation_steps)

# Example usage for TESTING:
# test_data_path = 'path/to/test/data'
# test_loader = CustomDataLoader(test_data_path, batch_size)
# criterion = nn.MSELoss()
# test_loss, mae, predictions, targets = test_model(model, test_loader, criterion)