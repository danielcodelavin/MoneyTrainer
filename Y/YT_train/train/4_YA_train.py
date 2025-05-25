import os
import json
import numpy as np
from datetime import datetime
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm

# Configuration and Hyperparameters
config = {
    "model_size": 'xlarge',
    "dropout_rate": 0.5,
    "batch_size": 128,
    "learning_rate": 0.0001,
    "weight_decay": 0.0001,
    "num_epochs": 100,
    "checkpoint_frequency": 10,
    "grad_clip": 1.0,
    "train_data_dir": "/Users/daniellavin/Desktop/proj/MoneyTrainer/Y/y_100_train",
    "val_data_dir": "/Users/daniellavin/Desktop/proj/MoneyTrainer/Y/y_100_val",
    "checkpoint_dir": "/Users/daniellavin/Desktop/proj/MoneyTrainer/Y/YT_train/check4",
}

size_dict = {'small': 128, 'medium': 256, 'large': 512 , 'xlarge': 768 , 'xxlarge': 1024}
config["hidden_dim"] = size_dict.get(config["model_size"], config["model_size"])
config["info_path"] = os.path.join(config["checkpoint_dir"], "4_training_info.txt")

class StockDataset(Dataset):
    def __init__(self, data_dir):
        self.data_files = sorted([os.path.join(data_dir, f)
                                  for f in os.listdir(data_dir) if f.endswith('.pt')])

    def __len__(self):
        return len(self.data_files)

    def __getitem__(self, idx):
        data = torch.load(self.data_files[idx])
        target = data[0]
        stock_seq = data[1:81]
        topic_vec = data[81:]

        stock_mean = stock_seq.mean()
        stock_std = stock_seq.std() if stock_seq.std() > 0 else 1.0
        stock_seq = (stock_seq - stock_mean) / stock_std

        return stock_seq, topic_vec, target

class StockMovementPredictor(nn.Module):
    def __init__(self, seq_len=80, topic_dim=100, hidden_dim=256, dropout_rate=0.5):
        super().__init__()

        self.cnn = nn.Sequential(
            nn.Conv1d(1, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Dropout(dropout_rate)
        )

        self.lstm = nn.LSTM(32, hidden_dim, batch_first=True)
        self.attn_fc = nn.Linear(hidden_dim, 1)

        self.topic_mlp = nn.Sequential(
            nn.Linear(topic_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(hidden_dim // 2, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout_rate)
        )

        fusion_dim = hidden_dim + (hidden_dim // 2)
        self.fc_out = nn.Sequential(
            nn.Linear(fusion_dim, fusion_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(fusion_dim // 2, 1)
        )

    def forward(self, stock_seq, topic_vec):
        x = stock_seq.unsqueeze(1)
        x = self.cnn(x)
        x = x.permute(0, 2, 1)

        lstm_out, _ = self.lstm(x)
        attn_scores = self.attn_fc(lstm_out).squeeze(-1)
        attn_weights = torch.softmax(attn_scores, dim=1).unsqueeze(-1)
        seq_repr = (lstm_out * attn_weights).sum(dim=1)

        topic_repr = self.topic_mlp(topic_vec)
        fused = torch.cat([seq_repr, topic_repr], dim=1)
        output = self.fc_out(fused)
        return output

# Device setup
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Data loading
train_loader = DataLoader(StockDataset(config["train_data_dir"]), batch_size=config["batch_size"], shuffle=True)
val_loader = DataLoader(StockDataset(config["val_data_dir"]), batch_size=config["batch_size"], shuffle=False)

# Model, optimizer, loss
model = StockMovementPredictor(hidden_dim=config["hidden_dim"], dropout_rate=config["dropout_rate"]).to(device)
criterion = nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=config["learning_rate"], weight_decay=config["weight_decay"])
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=5)

os.makedirs(config["checkpoint_dir"], exist_ok=True)
info_file = open(config["info_path"], "w")
info_file.write(f"Training started at {datetime.now()}\nConfig: {json.dumps(config)}\n")

# Training loop
for epoch in range(1, config["num_epochs"] + 1):
    model.train()
    train_loss, train_mae, train_correct = 0, 0, 0
    for stock_seq, topic_vec, target in tqdm(train_loader, desc=f"Epoch {epoch} Training"):
        stock_seq, topic_vec, target = stock_seq.to(device), topic_vec.to(device), target.to(device)
        optimizer.zero_grad()
        output = model(stock_seq, topic_vec).squeeze(1)
        loss = criterion(output, target)
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), config["grad_clip"])
        optimizer.step()
        train_loss += loss.item() * stock_seq.size(0)
        train_mae += torch.sum(torch.abs(output - target)).item()
        train_correct += torch.sum(torch.sign(output) == torch.sign(target)).item()

    train_loss /= len(train_loader.dataset)
    train_mae /= len(train_loader.dataset)
    train_acc = train_correct / len(train_loader.dataset)

    model.eval()
    val_loss, val_mae, val_correct = 0, 0, 0
    with torch.no_grad():
        for stock_seq, topic_vec, target in tqdm(val_loader, desc=f"Epoch {epoch} Validation"):
            stock_seq, topic_vec, target = stock_seq.to(device), topic_vec.to(device), target.to(device)
            output = model(stock_seq, topic_vec).squeeze(1)
            loss = criterion(output, target)
            val_loss += loss.item() * stock_seq.size(0)
            val_mae += torch.sum(torch.abs(output - target)).item()
            val_correct += torch.sum(torch.sign(output) == torch.sign(target)).item()

    val_loss /= len(val_loader.dataset)
    val_mae /= len(val_loader.dataset)
    val_acc = val_correct / len(val_loader.dataset)

    scheduler.step(val_loss)

    epoch_summary = (f"Epoch {epoch}/{config['num_epochs']} - Train Loss: {train_loss:.4f}, MAE: {train_mae:.4f}, Acc: {train_acc:.4f} | "
                     f"Val Loss: {val_loss:.4f}, MAE: {val_mae:.4f}, Acc: {val_acc:.4f}")

    print(epoch_summary)
    info_file.write(epoch_summary + '\n')

    if epoch % config["checkpoint_frequency"] == 0:
        torch.save(model.state_dict(), os.path.join(config["checkpoint_dir"], f"model_epoch_{epoch}.pt"))

info_file.close()
