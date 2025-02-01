import torch
import os
from typing import List, Tuple
import numpy as np
from pathlib import Path

def analyze_element_across_tensors(tensors: List[torch.Tensor], element_idx: int) -> Tuple[float, float, float]:
    """
    Analyze a specific element across multiple tensors.
    Returns (mean, max_divergence, min_value)
    """
    values = [tensor[element_idx].item() for tensor in tensors]
    mean_val = np.mean(values)
    max_div = max(abs(val - mean_val) for val in values)
    min_val = min(values)
    
    return mean_val, max_div, min_val

def analyze_pt_files(folder_path: str):
    """
    Analyze PT files in the specified folder.
    For each relevant element, compute mean, max divergence, and lowest value.
    """
    # Load all PT files
    pt_files = []
    for file in Path(folder_path).glob('*.pt'):
        tensor = torch.load(file)
        if len(tensor.shape) != 1 or tensor.shape[0] != 181:
            print(f"Warning: Skipping {file} - incorrect shape")
            continue
        pt_files.append(tensor)
    
    if not pt_files:
        print("No valid PT files found!")
        return

    # Elements to analyze
    gt_idx = 0  # Ground truth
    stock_indices = list(range(1, 21))  # Stock tensors (20 elements after GT)
    additional_indices = list(range(81, 111))  # Elements 81-111
    
    # Analyze ground truth element
    print("\n=== Ground Truth Element (index 0) ===")
    mean_val, max_div, min_val = analyze_element_across_tensors(pt_files, gt_idx)
    print(f"Mean: {mean_val:.4f}")
    print(f"Max Divergence: {max_div:.4f}")
    print(f"Lowest Value: {min_val:.4f}")
    
    # Analyze stock tensor elements
    print("\n=== Stock Tensor Elements (indices 1-20) ===")
    for idx in stock_indices:
        mean_val, max_div, min_val = analyze_element_across_tensors(pt_files, idx)
        print(f"\nElement {idx}:")
        print(f"Mean: {mean_val:.4f}")
        print(f"Max Divergence: {max_div:.4f}")
        print(f"Lowest Value: {min_val:.4f}")
    
    # Analyze additional elements
    print("\n=== Additional Elements (indices 81-110) ===")
    for idx in additional_indices:
        mean_val, max_div, min_val = analyze_element_across_tensors(pt_files, idx)
        print(f"\nElement {idx}:")
        print(f"Mean: {mean_val:.4f}")
        print(f"Max Divergence: {max_div:.4f}")
        print(f"Lowest Value: {min_val:.4f}")

if __name__ == "__main__":
    folder_path = "/Users/daniellavin/Desktop/proj/MoneyTrainer/Y/y_100_train"
    analyze_pt_files(folder_path)