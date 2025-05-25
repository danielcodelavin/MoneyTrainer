import torch
from pathlib import Path
import numpy as np
from collections import defaultdict

def analyze_gt_distribution(data_dir):
    data_dir = Path(data_dir)
    gt_values = []
    
    # Load all GT values
    for file_path in sorted(data_dir.glob('*.pt')):
        tensor = torch.load(file_path)
        gt_values.append(tensor[0].item())  # First element is GT
    
    gt_array = np.array(gt_values)
    
    # Basic statistics
    print("\n=== Basic Statistics ===")
    print(f"Total samples: {len(gt_array)}")
    print(f"Mean: {gt_array.mean():.4f}")
    print(f"Median: {np.median(gt_array):.4f}")
    print(f"Std: {gt_array.std():.4f}")
    print(f"Min: {gt_array.min():.4f}")
    print(f"Max: {gt_array.max():.4f}")
    
    # Distribution analysis using numpy boolean indexing
    print("\n=== Value Distribution ===")
    
    # Very negative
    count = np.sum(gt_array < -0.4)
    print(f"Very negative (<-0.4): {count} samples ({(count/len(gt_array))*100:.2f}%)")
    
    # Negative
    count = np.sum((gt_array >= -0.4) & (gt_array < -0.2))
    print(f"Negative (-0.4 to -0.2): {count} samples ({(count/len(gt_array))*100:.2f}%)")
    
    # Slight negative
    count = np.sum((gt_array >= -0.2) & (gt_array < 0))
    print(f"Slight negative (-0.2 to 0): {count} samples ({(count/len(gt_array))*100:.2f}%)")
    
    # Slight positive
    count = np.sum((gt_array >= 0) & (gt_array < 0.2))
    print(f"Slight positive (0 to 0.2): {count} samples ({(count/len(gt_array))*100:.2f}%)")
    
    # Moderate positive
    count = np.sum((gt_array >= 0.2) & (gt_array < 0.35))
    print(f"Moderate positive (0.2 to 0.35): {count} samples ({(count/len(gt_array))*100:.2f}%)")
    
    # High positive
    count = np.sum(gt_array >= 0.35)
    print(f"High positive (>=0.35): {count} samples ({(count/len(gt_array))*100:.2f}%)")
    
    # Detailed high value analysis
    high_values = gt_array[gt_array >= 0.25]
    if len(high_values) > 0:
        print("\n=== High Value Analysis ===")
        print(f"Number of high values (>=0.35): {len(high_values)}")
        print(f"Percentage of high values: {(len(high_values)/len(gt_array))*100:.2f}%")
        print(f"Mean of high values: {high_values.mean():.4f}")
        print(f"Max high value: {high_values.max():.4f}")
        
        # Distribution of high values
        print("\nHigh value distribution:")
        thresholds = [0.25, 0.3, 0.35, 0.4, 0.45, 0.6, 0.65, 0.7, 0.75]
        for i in range(len(thresholds)-1):
            count = np.sum((gt_array >= thresholds[i]) & (gt_array < thresholds[i+1]))
            print(f"{thresholds[i]:.2f} to {thresholds[i+1]:.2f}: {count} samples ({(count/len(gt_array))*100:.2f}%)")
    
    # Direction distribution
    positive_samples = np.sum(gt_array > 0)
    negative_samples = np.sum(gt_array < 0)
    zero_samples = np.sum(gt_array == 0)
    
    print("\n=== Direction Distribution ===")
    print(f"Positive movements: {positive_samples} ({(positive_samples/len(gt_array))*100:.2f}%)")
    print(f"Negative movements: {negative_samples} ({(negative_samples/len(gt_array))*100:.2f}%)")
    print(f"Zero movements: {zero_samples} ({(zero_samples/len(gt_array))*100:.2f}%)")

if __name__ == "__main__":
    val_data_dir = '/Users/daniellavin/Desktop/proj/MoneyTrainer/Y/y_100_val'
    analyze_gt_distribution(val_data_dir)