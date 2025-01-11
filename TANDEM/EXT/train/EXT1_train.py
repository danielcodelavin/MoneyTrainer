import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import os
from datetime import datetime
import glob
import numpy as np
from typing import List, Dict, Tuple
from collections import deque
import math
from tqdm import tqdm

class PredictionHistory:
    def __init__(self, window_size: int = 100):
        self.window_size = window_size
        self.predictions = deque(maxlen=window_size)
        self.ground_truths = deque(maxlen=window_size)
        self.confidences = deque(maxlen=window_size)
    
    def add_prediction(self, prediction: float, ground_truth: float, confidence: float):
        self.predictions.append(prediction)
        self.ground_truths.append(ground_truth)
        self.confidences.append(confidence)
    
    def calculate_metrics(self) -> Tuple[float, float]:
        if len(self.predictions) < 10:
            return 0.5, 1.0
        
        preds = torch.tensor(list(self.predictions))
        truths = torch.tensor(list(self.ground_truths))
        
        pred_positive = preds > 0
        truth_positive = truths > 0
        predicted_positives = pred_positive.sum().item()
        
        if predicted_positives > 0:
            true_positives = (pred_positive & truth_positive).sum().item()
            precision = true_positives / predicted_positives
        else:
            precision = 0.5
        
        non_zero_preds = preds != 0
        if non_zero_preds.any():
            l1_score = torch.mean(torch.abs(
                preds[non_zero_preds] - truths[non_zero_preds]
            )).item()
        else:
            l1_score = 1.0
        
        return precision, l1_score

class FinancialDataset(Dataset):
    def __init__(self, data_path):
        """
        Dataset class for loading financial tensors with 100-topic vectors
        Each tensor has shape (181,) where:
        - First element is ground truth
        - Next 80 elements are stock data
        - Last 100 elements are topic vector
        """
        self.tensor_files = glob.glob(os.path.join(data_path, "*.pt"))
    
    def __len__(self):
        return len(self.tensor_files)
    
    def __getitem__(self, idx):
        tensor = torch.load(self.tensor_files[idx], map_location='cpu')
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

class PrecisionFocusedMLP(nn.Module):
    def __init__(self):
        super(PrecisionFocusedMLP, self).__init__()
        
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
        
        self.topic_path = nn.Sequential(
            nn.Linear(100, 128),  # Changed from 30 to 100 input features
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
        
        self.confidence_head = nn.Sequential(
            nn.Linear(384, 256),
            nn.LeakyReLU(),
            nn.Dropout(0.3),
            nn.Linear(256, 128),
            nn.LeakyReLU(),
            nn.Linear(128, 1),
            nn.Sigmoid()
        )
        
        self.prediction_head = nn.Sequential(
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
    
    def forward(self, stock_data, topic_vector):
        stock_features = self.stock_path(stock_data)
        topic_features = self.topic_path(topic_vector)
        combined = torch.cat((stock_features, topic_features), dim=1)
        
        confidence = self.confidence_head(combined)
        prediction = self.prediction_head(combined)
        
        return prediction, confidence

class WeightedEnsembleModel(nn.Module):
    def __init__(self, num_models: int = 7, precision_weight: float = 0.8,
                 window_size: int = 100):
        super(WeightedEnsembleModel, self).__init__()
        
        self.models = nn.ModuleList([
            PrecisionFocusedMLP() for _ in range(num_models)
        ])
        
        self.precision_weight = precision_weight
        self.l1_weight = 1 - precision_weight
        self.histories = [PredictionHistory(window_size) for _ in range(num_models)]
    
    def calculate_model_weights(self) -> torch.Tensor:
        metrics = [history.calculate_metrics() for history in self.histories]
        precisions, l1_scores = zip(*metrics)
        
        precisions = torch.tensor(precisions)
        l1_scores = torch.tensor(l1_scores)
        
        l1_scores = 1 / (l1_scores + 1e-6)
        l1_scores = l1_scores / l1_scores.sum()
        
        precisions = precisions / precisions.sum()
        
        weights = (self.precision_weight * precisions + 
                  self.l1_weight * l1_scores)
        
        return weights / weights.sum()
    
    def forward(self, stock_data, topic_vector, ground_truth=None):
        all_predictions = []
        all_confidences = []
        
        for model in self.models:
            pred, conf = model(stock_data, topic_vector)
            all_predictions.append(pred.view(stock_data.size(0), -1))
            all_confidences.append(conf.view(stock_data.size(0), -1))
        
        predictions_stack = torch.stack(all_predictions, dim=1)  # [batch_size, num_models, 1]
        confidences_stack = torch.stack(all_confidences, dim=1)  # [batch_size, num_models, 1]
        
        if self.training:
            return predictions_stack.transpose(0, 1), confidences_stack.transpose(0, 1)
        else:
            model_weights = self.calculate_model_weights()
            weighted_pred = torch.sum(
                predictions_stack * model_weights.view(1, -1, 1),
                dim=1
            )
            ensemble_conf = confidences_stack.max(dim=1)[0]
            
            if ground_truth is not None:
                for i, (pred, conf) in enumerate(zip(all_predictions, all_confidences)):
                    for p, c, gt in zip(pred.cpu().numpy(), 
                                    conf.cpu().numpy(), 
                                    ground_truth.cpu().numpy()):
                        self.histories[i].add_prediction(p.item(), gt, c.item())
            
            return weighted_pred, ensemble_conf

class PrecisionFocusedLoss(nn.Module):
    def __init__(self, confidence_weight=0.3, fp_penalty=2.0):
        super(PrecisionFocusedLoss, self).__init__()
        self.confidence_weight = confidence_weight
        self.fp_penalty = fp_penalty
        self.mse = nn.MSELoss(reduction='none')
    
    def forward(self, predictions, confidences, targets):
        if len(predictions.shape) == 3:  # [num_models, batch_size, 1]
            batch_losses = []
            for i in range(predictions.shape[0]):
                model_loss = self._single_model_loss(
                    predictions[i],
                    confidences[i],
                    targets
                )
                batch_losses.append(model_loss)
            return torch.stack(batch_losses).mean()
        else:
            return self._single_model_loss(predictions, confidences, targets)
    
    def _single_model_loss(self, predictions, confidences, targets):
        predictions = predictions.view(-1)
        confidences = confidences.view(-1)
        targets = targets.view(-1)
        
        assert predictions.shape == targets.shape, f"Shape mismatch: pred {predictions.shape} vs target {targets.shape}"
        
        percentage_loss = self.mse(predictions, targets)
        
        pred_direction = torch.sign(predictions)
        target_direction = torch.sign(targets)
        
        false_positives = (pred_direction > 0) & (target_direction <= 0)
        weighted_loss = torch.where(
            false_positives,
            percentage_loss * self.fp_penalty,
            percentage_loss
        )
        
        direction_correct = pred_direction == target_direction
        confidence_targets = direction_correct.float()
        
        confidence_loss = nn.BCELoss()(
            confidences,
            confidence_targets
        )
        
        confidence_penalty = torch.mean(
            confidences * (~direction_correct).float()
        )
        
        return (weighted_loss.mean() + 
                confidence_loss * self.confidence_weight +
                confidence_penalty * self.confidence_weight)

def calculate_metrics(predictions: torch.Tensor, ground_truth: torch.Tensor) -> Dict[str, float]:
    pred_positive = predictions > 0
    truth_positive = ground_truth > 0
    
    true_positives = (pred_positive & truth_positive).sum().item()
    predicted_positives = pred_positive.sum().item()
    precision = true_positives / predicted_positives if predicted_positives > 0 else 0
    
    actual_positives = truth_positive.sum().item()
    recall = true_positives / actual_positives if actual_positives > 0 else 0
    
    correct_predictions = (pred_positive == truth_positive).sum().item()
    total_predictions = len(predictions)
    accuracy = correct_predictions / total_predictions
    
    non_zero_preds = predictions != 0
    l1_distance = torch.mean(torch.abs(
        predictions[non_zero_preds] - ground_truth[non_zero_preds]
    )).item() if non_zero_preds.any() else 0
    
    prediction_rate = non_zero_preds.float().mean().item()
    
    return {
        'precision': precision,
        'recall': recall,
        'accuracy': accuracy,
        'l1_distance': l1_distance,
        'prediction_rate': prediction_rate
    }

def train_ensemble_model(
    train_data_path: str,
    val_data_path: str,
    checkpoint_dir: str,
    report_path: str,
    params: Dict
) -> None:
    print("\nInitializing training with configuration:")
    print(f"Number of models: {params['num_models']}")
    print(f"Batch size: {params['batch_size']}")
    print(f"Training data path: {train_data_path}")
    print(f"Validation data path: {val_data_path}")
    
    train_dataset = FinancialDataset(train_data_path)
    val_dataset = FinancialDataset(val_data_path)
    
    train_loader = DataLoader(train_dataset, batch_size=params['batch_size'], shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=params['batch_size'], shuffle=False)
    
    model = WeightedEnsembleModel(
        num_models=params['num_models'],
        precision_weight=params['precision_weight'],
        window_size=params['window_size']
    )
    
    start_epoch = 0
    if params['resume_checkpoint']:
        checkpoint = torch.load(params['resume_checkpoint'], map_location='cpu')
        model.load_state_dict(checkpoint['model_state_dict'])
        start_epoch = checkpoint['epoch']
        if 'histories' in checkpoint:
            model.histories = checkpoint['histories']
    
    optimizer = optim.Adam(
        model.parameters(),
        lr=params['learning_rate'],
        weight_decay=params['weight_decay']
    )
    
    criterion = PrecisionFocusedLoss(
        confidence_weight=params['confidence_weight'],
        fp_penalty=params['fp_penalty']
    )
    
    if params['scheduler_enabled']:
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            optimizer,
            mode='max',
            factor=0.5,
            patience=10,
            verbose=True
        )
    
    os.makedirs(checkpoint_dir, exist_ok=True)
    
    with open(report_path, 'a') as report_file:
        report_file.write(f"Training started at {datetime.now()}\n")
        report_file.write("Epoch,Train_Loss,Val_Loss,Precision,Recall,Accuracy,L1_Distance,Prediction_Rate,Model_Weights\n")
    
    for epoch in tqdm(range(start_epoch, params['num_epochs']), desc='Training Progress'):
        model.train()
        train_loss = 0.0
        
        for batch in train_loader:
            stock_data = batch['stock_data']
            topic_vector = batch['topic_vector']
            ground_truth = batch['ground_truth']
            
            optimizer.zero_grad()
            predictions, confidences = model(stock_data, topic_vector)
            loss = criterion(predictions, confidences, ground_truth)
            loss.backward()
            
            if params['grad_clip'] > 0:
                torch.nn.utils.clip_grad_norm_(
                    model.parameters(), 
                    max_norm=params['grad_clip']
                )
            
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
                stock_data = batch['stock_data']
                topic_vector = batch['topic_vector']
                ground_truth = batch['ground_truth']
                
                predictions, confidences = model(stock_data, topic_vector, ground_truth)
                loss = criterion(predictions, confidences, ground_truth)
                val_loss += loss.item()
                
                all_predictions.extend(predictions.cpu())
                all_ground_truths.extend(ground_truth.cpu())
        
        val_loss /= len(val_loader)
        
        metrics = calculate_metrics(
            torch.tensor(all_predictions),
            torch.tensor(all_ground_truths)
        )
        
        model_weights = model.calculate_model_weights().numpy()
        weights_str = ','.join([f"{w:.4f}" for w in model_weights])
        
        if params['scheduler_enabled']:
            scheduler.step(metrics['precision'])
        
        with open(report_path, 'a') as report_file:
            report_file.write(
                f"{epoch},{train_loss:.6f},{val_loss:.6f},"
                f"{metrics['precision']:.6f},{metrics['recall']:.6f},"
                f"{metrics['accuracy']:.6f},{metrics['l1_distance']:.6f},"
                f"{metrics['prediction_rate']:.6f},{weights_str}\n"
            )
        
        if (epoch + 1) % params['checkpoint_frequency'] == 0:
            checkpoint_path = os.path.join(
                checkpoint_dir, 
                f"checkpoint_epoch_{epoch+1}.pt"
            )
            torch.save({
                'epoch': epoch + 1,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'train_loss': train_loss,
                'val_loss': val_loss,
                'metrics': metrics,
                'histories': model.histories,
                'model_weights': model_weights,
                'model_config': {
                    'num_models': params['num_models'],
                    'precision_weight': params['precision_weight'],
                    'window_size': params['window_size']
                }
            }, checkpoint_path)
        
        print(f"\nEpoch {epoch+1}/{params['num_epochs']}")
        print(f"Train Loss: {train_loss:.6f}")
        print(f"Val Loss: {val_loss:.6f}")
        print(f"Precision: {metrics['precision']:.6f}")
        print(f"Recall: {metrics['recall']:.6f}")
        print(f"Accuracy: {metrics['accuracy']:.6f}")
        print(f"L1 Distance: {metrics['l1_distance']:.6f}")
        print(f"Prediction Rate: {metrics['prediction_rate']:.6f}")
        print(f"Model Weights: {weights_str}")
        print("----------------------------------------")

if __name__ == "__main__":
    params = {
        'num_epochs': 1500,
        'batch_size':128,
        'num_models': 7,
        'precision_weight': 0.8,
        'window_size': 150,
        'learning_rate': 2e-5,
        'weight_decay': 1e-5,
        'confidence_weight': 0.3,
        'fp_penalty': 16.0,
        'grad_clip': 1.0,
        'checkpoint_frequency': 20,
        'resume_checkpoint': None,
        'scheduler_enabled': True
    }
    
    train_ensemble_model(
        train_data_path="/Users/daniellavin/Desktop/proj/MoneyTrainer/train_energydataset100",
        val_data_path="/Users/daniellavin/Desktop/proj/MoneyTrainer/val_energydataset100",
        checkpoint_dir="/Users/daniellavin/Desktop/proj/MoneyTrainer/TANDEM/EXT/check",
        report_path="/Users/daniellavin/Desktop/proj/MoneyTrainer/TANDEM/EXT/check/TEX1_report.txt",
        params=params
    )