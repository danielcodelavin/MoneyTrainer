import torch
import torch.nn as nn
import torch.optim as optim
import os
import random
from collections import deque
from tqdm import tqdm
import matplotlib.pyplot as plt
import numpy as np

def test_model(model, test_loader, criterion, device):
    model.eval()
    test_loss = 0.0
    all_predictions = []
    all_targets = []

    test_progress_bar = tqdm(total=len(test_loader), desc="Testing")

    with torch.no_grad():
        while True:
            try:
                inputs, targets = next(test_loader)
                inputs, targets = inputs.to(device), targets.to(device)
                outputs = model(inputs)
                loss = criterion(outputs, targets)
                test_loss += loss.item()

                all_predictions.extend(outputs.cpu().numpy())
                all_targets.extend(targets.cpu().numpy())

                test_progress_bar.update(1)
                test_progress_bar.set_postfix({'loss': f"{loss.item():.4f}"})
            except StopIteration:
                break

    test_progress_bar.close()
    avg_test_loss = test_loss / len(test_loader)

    mae = torch.mean(torch.abs(torch.tensor(all_predictions) - torch.tensor(all_targets))).item()
    mse = torch.mean((torch.tensor(all_predictions) - torch.tensor(all_targets))**2).item()
    r2 = 1 - (torch.sum((torch.tensor(all_targets) - torch.tensor(all_predictions))**2) / 
              torch.sum((torch.tensor(all_targets) - torch.mean(torch.tensor(all_targets)))**2)).item()

    print(f"Test Loss: {avg_test_loss:.4f}, MAE: {mae:.4f}, MSE: {mse:.4f}, R2: {r2:.4f}")

    return avg_test_loss, mae, mse, r2, all_predictions, all_targets


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
        self.batch_norm = nn.BatchNorm1d(d_model)  # Added BatchNorm

    def forward(self, x):
        attn_output, _ = self.attention(x, x, x)
        x = self.norm1(x + attn_output)
        swiglu_output = self.swiglu(x)
        x = self.norm2(x + swiglu_output)
        x = x.transpose(0, 2)  # Transpose for BatchNorm
        x = self.batch_norm(x)
        x = x.transpose(0, 2)  # Transpose back
        return x


class TransformerRegression(nn.Module):
    def __init__(self, d_model, nhead, num_layers):
        super().__init__()
        self.d_model = d_model
        self.input_projection = nn.Linear(1, d_model)
        self.transformer_blocks = nn.ModuleList([TransformerBlock(d_model, nhead) for _ in range(num_layers)])
        self.regression_head = nn.Linear(d_model, 1)

        # Xavier/Glorot initialization
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    def forward(self, x):
        x = x.unsqueeze(-1)
        x = self.input_projection(x)
        x = x.transpose(0, 1)

        for block in self.transformer_blocks:
            x = block(x)

        x = x[-1]
        return self.regression_head(x).view(-1)


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
        gt_folder_path = os.path.join(folder_path, 'GT')
        
        news_data = torch.load(os.path.join(folder_path, '0news.pt'))
        
        pt_files = [f for f in os.listdir(folder_path) if f.endswith('.pt') and f != '0news.pt']
        
        selected_files = random.sample(pt_files, min(self.batch_size, len(pt_files)))
        
        batch_inputs = []
        batch_targets = []
        
        for file in selected_files:
            input_tensor = torch.load(os.path.join(folder_path, file))
            input_tensor = torch.cat([input_tensor, news_data], dim=0)
            batch_inputs.append(input_tensor)
            
            gt_file = file.replace('.pt', '_GT.pt')
            gt_tensor = torch.load(os.path.join(gt_folder_path, gt_file))
            batch_targets.append(gt_tensor.view(1))

        max_len = max(tensor.size(0) for tensor in batch_inputs)
        batch_inputs_padded = torch.stack([torch.nn.functional.pad(tensor, (0, max_len - tensor.size(0))) for tensor in batch_inputs])
        
        return batch_inputs_padded, torch.cat(batch_targets)

    def __len__(self):
        return len(self.folders)
    

def train_model(model, train_loader, val_loader, num_epochs, learning_rate, weight_decay, checkpoint_path, accumulation_steps, epoch_info_path, device, start_epoch=0):
    criterion = nn.HuberLoss(delta=100)  # Adjusted delta to 100
    optimizer = optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=20, verbose=True)  # Increased patience to 20

    train_losses = []
    val_losses = []

    epoch_info_file = os.path.join(epoch_info_path, "epoch_info.txt")

    for epoch in range(start_epoch, num_epochs):
        model.train()
        train_loss = 0.0
        train_progress_bar = tqdm(total=len(train_loader), desc=f"Epoch {epoch+1}/{num_epochs} [Train]")
        optimizer.zero_grad()

        batch_count = 0
        while True:
            try:
                inputs, targets = next(train_loader)
                inputs, targets = inputs.to(device), targets.to(device)
                outputs = model(inputs)
                loss = criterion(outputs, targets)
                loss = loss / accumulation_steps
                loss.backward()

                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=0.1)  # Reduced max_norm to 0.1

                batch_count += 1
                if batch_count % accumulation_steps == 0 or batch_count == len(train_loader):
                    optimizer.step()
                    optimizer.zero_grad()

                train_loss += loss.item() * accumulation_steps
                train_progress_bar.update(1)
                train_progress_bar.set_postfix({'loss': f"{loss.item() * accumulation_steps:.4f}"})

            except StopIteration:
                break

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
                    inputs, targets = inputs.to(device), targets.to(device)
                    outputs = model(inputs)
                    loss = criterion(outputs, targets)
                    val_loss += loss.item()
                    val_progress_bar.update(1)
                    val_progress_bar.set_postfix({'loss': f"{loss.item():.4f}"})
                except StopIteration:
                    break

        val_progress_bar.close()
        avg_val_loss = val_loss / len(val_loader)
        val_losses.append(avg_val_loss)

        print(f"Epoch {epoch+1}/{num_epochs}, Train Loss: {avg_train_loss:.4f}, Val Loss: {avg_val_loss:.4f}")

        scheduler.step(avg_val_loss)

        with open(epoch_info_file, "a") as f:
            f.write(f"Epoch: {epoch+1}\n")
            f.write(f"Train Loss: {avg_train_loss:.4f}\n")
            f.write(f"Val Loss: {avg_val_loss:.4f}\n")
            f.write("\n")

        if(epoch%5 == 0):
            torch.save({
                'epoch': epoch + 1,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'train_loss': avg_train_loss,
                'val_loss': avg_val_loss,
            }, os.path.join(checkpoint_path, f"checkpoint_epoch_{epoch+1}.pt"))

    plt.figure(figsize=(10, 5))
    plt.plot(train_losses, label='Train Loss')
    plt.plot(val_losses, label='Validation Loss')
    plt.legend()
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Training and Validation Loss')
    plt.savefig(os.path.join(epoch_info_path, 'loss_plot.png'))
    plt.close()

    return model


if __name__ == "__main__":
    train_data_path = '/Users/daniellavin/Desktop/proj/Moneytrain/minitrash/ministockdataset'
    val_data_path = '/Users/daniellavin/Desktop/proj/Moneytrain/minitrash/minivalset'
    checkpoint_path = '/Users/daniellavin/Desktop/proj/Moneytrain/minitrash/miniepochs'
    info_path = '/Users/daniellavin/Desktop/proj/Moneytrain/minitrash/minireport.txt'

    batch_size = 32
    learning_rate = 0.0003  # Reduced to 0.0003
    weight_decay = 1e-6
    num_epochs = 300
    accumulation_steps = 8  # Increased back to 8

    train_loader = CustomDataLoader(train_data_path, batch_size)
    val_loader = CustomDataLoader(val_data_path, batch_size)

    d_model = 640  # HIDDEN SIZE
    nhead = 8
    num_layers = 8
    model = TransformerRegression(d_model=d_model, nhead=nhead, num_layers=num_layers)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)

    # Add option to continue training from a checkpoint
    start_epoch = 0
    checkpoint_file = None  # Set this to the checkpoint file path if you want to continue training

    if checkpoint_file:
        checkpoint = torch.load(checkpoint_file)
        model.load_state_dict(checkpoint['model_state_dict'])
        start_epoch = checkpoint['epoch']
        print(f"Resuming training from epoch {start_epoch}")

    trained_model = train_model(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        num_epochs=num_epochs,
        learning_rate=learning_rate,
        weight_decay=weight_decay,
        checkpoint_path=checkpoint_path,
        accumulation_steps=accumulation_steps,
        epoch_info_path=info_path,
        device=device,
        start_epoch=start_epoch
    )