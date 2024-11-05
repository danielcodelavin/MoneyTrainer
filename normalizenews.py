import os
import torch
from pathlib import Path

def normalize_tensors(root_path):
    """
    Walk through all subfolders in root_path, find '0news.pt' files,
    normalize the tensors, and save them back.
    
    Args:
        root_path (str): Path to the root directory to start searching from
    """
    # Convert string path to Path object for better handling
    root = Path(root_path)
    
    # Counter for processed files
    processed = 0
    
    # Walk through all subdirectories
    for folder_path in root.rglob("*"):
        if folder_path.is_dir():
            tensor_path = folder_path / "0news.pt"
            
            # Check if the tensor file exists in this folder
            if tensor_path.exists():
                try:
                    # Load the tensor
                    tensor = torch.load(tensor_path)
                    
                    # Calculate mean and std across all dimensions
                    mean = torch.mean(tensor)
                    std = torch.std(tensor)
                    
                    # Normalize the tensor
                    normalized_tensor = (tensor - mean) / (std + 1e-8)  # Adding small epsilon to prevent division by zero
                    
                    # Save the normalized tensor back to the same file
                    torch.save(normalized_tensor, tensor_path)
                    
                    processed += 1
                    print(f"Processed: {tensor_path}")
                    
                except Exception as e:
                    print(f"Error processing {tensor_path}: {str(e)}")
    
    print(f"\nCompleted! Processed {processed} tensor files.")

if __name__ == "__main__":
    import sys
    
    root_folder = '/Users/daniellavin/Desktop/proj/MoneyTrainer/valstockdataset'
        
    normalize_tensors(root_folder)