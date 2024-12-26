import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import os
from datetime import datetime
import glob

class FinancialDataset(Dataset):
    def __init__(self, data_path):
        """
        Dataset class for loading financial tensors
        Each tensor is expected to have shape (111,) where:
        - First element is ground truth
        - Next 80 elements are stock data
        - Last 30 elements are topic vector
        """
        self.tensor_files = glob.glob(os.path.join(data_path, "*.pt"))
        
    def __len__(self):
        return len(self.tensor_files)
    
    def __getitem__(self, idx):
        tensor = torch.load(self.tensor_files[idx])
        ground_truth = tensor[0]
        stock_data = tensor[1:81]
        topic_vector = tensor[81:]
        
        return {
            'stock_data': stock_data,
            'topic_vector': topic_vector,
            'ground_truth': ground_truth
        }

class ResidualBlock(nn.Module):
    def __init__(self, in_features):
        super(ResidualBlock, self).__init__()
        self.block = nn.Sequential(
            nn.Linear(in_features, in_features * 2),
            nn.LeakyReLU(),
            nn.Dropout(0.2),
            nn.Linear(in_features * 2, in_features)
        )
        self.leaky_relu = nn.LeakyReLU()
        self.norm = nn.LayerNorm(in_features)
        
    def forward(self, x):
        residual = x
        out = self.block(x)
        out += residual
        out = self.norm(out)
        return self.leaky_relu(out)

class MultiModalMLP(nn.Module):
    def __init__(self, device='cuda'):
        super(MultiModalMLP, self).__init__()
        
        # Stock data path with increased complexity
        self.stock_path = nn.Sequential(
            nn.Linear(80, 256),
            nn.LeakyReLU(),
            nn.Dropout(0.2),
            ResidualBlock(256),
            nn.Linear(256, 384),
            nn.LeakyReLU(),
            nn.Dropout(0.2),
            ResidualBlock(384),
            nn.Linear(384, 512),
            nn.LeakyReLU(),
            nn.Dropout(0.2),
            ResidualBlock(512),
            nn.Linear(512, 384),
            nn.LeakyReLU(),
            nn.Dropout(0.2),
            ResidualBlock(384),
            nn.Linear(384, 256),
            nn.LeakyReLU(),
            nn.Dropout(0.2)
        )
        
        # Topic vector path with increased complexity
        self.topic_path = nn.Sequential(
            nn.Linear(30, 128),
            nn.LeakyReLU(),
            nn.Dropout(0.2),
            nn.Linear(128, 256),
            nn.LeakyReLU(),
            nn.Dropout(0.2),
            ResidualBlock(256),
            nn.Linear(256, 192),
            nn.LeakyReLU(),
            nn.Dropout(0.2),
            ResidualBlock(192),
            nn.Linear(192, 128),
            nn.LeakyReLU(),
            nn.Dropout(0.2)
        )
        
        # Combined path with increased complexity
        self.combined_path = nn.Sequential(
            nn.Linear(384, 512),
            nn.BatchNorm1d(512),
            nn.LeakyReLU(),
            nn.Dropout(0.2),
            ResidualBlock(512),
            nn.Linear(512, 384),
            nn.BatchNorm1d(384),
            nn.LeakyReLU(),
            nn.Dropout(0.2),
            ResidualBlock(384),
            nn.Linear(384, 256),
            nn.BatchNorm1d(256),
            nn.LeakyReLU(),
            nn.Dropout(0.2),
            ResidualBlock(256),
            nn.Linear(256, 128),
            nn.BatchNorm1d(128),
            nn.LeakyReLU(),
            nn.Dropout(0.2),
            nn.Linear(128, 1)
        )
        
        self.to(device)
        
    def forward(self, stock_data, topic_vector):
        stock_features = self.stock_path(stock_data)
        topic_features = self.topic_path(topic_vector)
        combined = torch.cat((stock_features, topic_features), dim=1)
        return self.combined_path(combined)

class CustomLoss(nn.Module):
    def __init__(self, direction_weight=0.3):
        super(CustomLoss, self).__init__()
        self.mse = nn.MSELoss()
        self.direction_weight = direction_weight
        
    def forward(self, predictions, targets):
        mse_loss = self.mse(predictions, targets)
        pred_signs = torch.sign(predictions)
        target_signs = torch.sign(targets)
        direction_loss = torch.mean((pred_signs - target_signs).pow(2))
        total_loss = mse_loss + self.direction_weight * direction_loss
        return total_loss

def calculate_metrics(predictions, ground_truth):
    """Calculate various performance metrics"""
    pred_positive = predictions >= 0
    truth_positive = ground_truth >= 0
    
    # Precision
    true_positives = (pred_positive & truth_positive).sum().item()
    predicted_positives = pred_positive.sum().item()
    precision = true_positives / predicted_positives if predicted_positives > 0 else 0
    
    # Recall
    actual_positives = truth_positive.sum().item()
    recall = true_positives / actual_positives if actual_positives > 0 else 0
    
    # Accuracy
    correct_predictions = (pred_positive == truth_positive).sum().item()
    total_predictions = len(predictions)
    accuracy = correct_predictions / total_predictions
    
    # L1 Distance (MAE)
    l1_distance = torch.mean(torch.abs(predictions - ground_truth)).item()
    
    return {
        'precision': precision,
        'recall': recall,
        'accuracy': accuracy,
        'l1_distance': l1_distance
    }

def train_model(
    train_data_path,
    val_data_path,
    checkpoint_dir,
    report_path,
    params
):
    train_dataset = FinancialDataset(train_data_path)
    val_dataset = FinancialDataset(val_data_path)
    
    train_loader = DataLoader(train_dataset, batch_size=params['batch_size'], shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=params['batch_size'], shuffle=False)
    
    model = MultiModalMLP(device=params['device'])
    
    if params['resume_checkpoint']:
        checkpoint = torch.load(params['resume_checkpoint'])
        model.load_state_dict(checkpoint['model_state_dict'])
        start_epoch = checkpoint['epoch']
    else:
        start_epoch = 0
    
    optimizer = optim.Adam(
        model.parameters(), 
        lr=params['learning_rate'], 
        weight_decay=params['weight_decay']
    )
    
    if params['scheduler_enabled']:
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, 
            mode='min', 
            factor=0.5, 
            patience=10, 
            verbose=True
        )
    
    criterion = CustomLoss(direction_weight=params['direction_weight'])
    
    os.makedirs(checkpoint_dir, exist_ok=True)
    
    with open(report_path, 'a') as report_file:
        report_file.write(f"Training started at {datetime.now()}\n")
        report_file.write("Epoch,Train_Loss,Val_Loss,Precision,Recall,Accuracy,L1_Distance\n")
    
    for epoch in range(start_epoch, params['num_epochs']):
        model.train()
        train_loss = 0.0
        
        for batch in train_loader:
            stock_data = batch['stock_data'].to(params['device'])
            topic_vector = batch['topic_vector'].to(params['device'])
            ground_truth = batch['ground_truth'].to(params['device'])
            
            optimizer.zero_grad()
            predictions = model(stock_data, topic_vector).squeeze()
            loss = criterion(predictions, ground_truth)
            loss.backward()
            
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=params['grad_clip'])
            
            optimizer.step()
            train_loss += loss.item()
        
        train_loss /= len(train_loader)
        
        # Validation step
        model.eval()
        val_loss = 0.0
        all_predictions = []
        all_ground_truths = []
        
        with torch.no_grad():
            for batch in val_loader:
                stock_data = batch['stock_data'].to(params['device'])
                topic_vector = batch['topic_vector'].to(params['device'])
                ground_truth = batch['ground_truth'].to(params['device'])
                
                predictions = model(stock_data, topic_vector).squeeze()
                loss = criterion(predictions, ground_truth)
                val_loss += loss.item()
                
                all_predictions.extend(predictions.cpu())
                all_ground_truths.extend(ground_truth.cpu())
        
        val_loss /= len(val_loader)
        
        if params['scheduler_enabled']:
            scheduler.step(val_loss)
        
        metrics = calculate_metrics(
            torch.tensor(all_predictions),
            torch.tensor(all_ground_truths)
        )
        
        with open(report_path, 'a') as report_file:
            report_file.write(
                f"{epoch},{train_loss:.6f},{val_loss:.6f},"
                f"{metrics['precision']:.6f},{metrics['recall']:.6f},"
                f"{metrics['accuracy']:.6f},{metrics['l1_distance']:.6f}\n"
            )
        
        if (epoch + 1) % params['checkpoint_frequency'] == 0:
            checkpoint_path = os.path.join(checkpoint_dir, f"checkpoint_epoch_{epoch+1}.pt")
            torch.save({
                'epoch': epoch + 1,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'train_loss': train_loss,
                'val_loss': val_loss,
                'metrics': metrics
            }, checkpoint_path)
            
        print(f"Epoch {epoch+1}/{params['num_epochs']}")
        print(f"Train Loss: {train_loss:.6f}")
        print(f"Val Loss: {val_loss:.6f}")
        print(f"Precision: {metrics['precision']:.6f}")
        print(f"Recall: {metrics['recall']:.6f}")
        print(f"Accuracy: {metrics['accuracy']:.6f}")
        print(f"L1 Distance: {metrics['l1_distance']:.6f}")
        print("----------------------------------------")

if __name__ == "__main__":
    # All hyperparameters and configuration in one place
    params = {
        'num_epochs': 500,
        'batch_size': 64,
        'learning_rate': 3e-5,
        'weight_decay': 1e-5,
        'direction_weight': 0.3,
        'grad_clip': 1.0,
        'checkpoint_frequency': 1,
        'device': 'cuda' if torch.cuda.is_available() else 'cpu',
        'resume_checkpoint': None,
        'scheduler_enabled': True
    }
    
    train_model(
        train_data_path="/Users/daniellavin/Desktop/proj/MoneyTrainer/X_findataset",
        val_data_path="/Users/daniellavin/Desktop/proj/MoneyTrainer/minivals",
        checkpoint_dir="/Users/daniellavin/Desktop/proj/MoneyTrainer/checkpoints/checkpx1",
        report_path="/Users/daniellavin/Desktop/proj/MoneyTrainer/reports/1X_report.txt",
        params=params
    )


