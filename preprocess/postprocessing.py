import torch
import os
from collections import Counter

def process_stock_torch_files(directory):
    # Initialize variables
    length_counter = Counter()
    delete_count = 0
    
    # Get all .pt files in the directory
    files = [f for f in os.listdir(directory) if f.endswith('.pt')]
    num_files = len(files)
    
    print(f"Number of files in the directory: {num_files}")

    # Step 1: Preliminary sweep to remove files with NaN, zero, or inf elements
    for file in files:
        file_path = os.path.join(directory, file)
        try:
            # Load the tensor
            tensor = torch.load(file_path)
            
            # Skip if it's not a tensor (e.g. dict or other objects)
            if not isinstance(tensor, torch.Tensor):
                continue
            
            # Check for NaN, zero, or inf values
            if torch.isnan(tensor).any() or torch.isinf(tensor).any() or (tensor == 0).any():
                os.remove(file_path)
                delete_count += 1
                continue

            # Get tensor length
            tensor_length = tensor.numel()
            
            # Only count lengths over 40
            if tensor_length > 40:
                length_counter[tensor_length] += 1
        
        except Exception as e:
            print(f"Error processing file {file}: {e}")
    
    # Find the most common length
    if length_counter:
        most_common_length = length_counter.most_common(1)[0][0]
    else:
        print("No valid tensors found.")
        return

    print(f"Most common tensor length: {most_common_length}")
    print(f"Number of files deleted in preliminary sweep: {delete_count}")

    # Step 2: Delete files with uncommon lengths
    for file in os.listdir(directory):
        if file.endswith('.pt'):
            file_path = os.path.join(directory, file)
            try:
                tensor = torch.load(file_path)
                if not isinstance(tensor, torch.Tensor) or tensor.numel() != most_common_length:
                    os.remove(file_path)
                    delete_count += 1
            except Exception as e:
                print(f"Error processing file {file}: {e}")

    print(f"Total number of files deleted: {delete_count}")

    # Step 2: Delete tensors not matching the most common length
    for file in files:
        file_path = os.path.join(directory, file)
        try:
            tensor = torch.load(file_path)

            # Skip if it's not a tensor or already deleted
            if not isinstance(tensor, torch.Tensor) or not os.path.exists(file_path):
                continue

            tensor_length = tensor.numel()

            # Delete tensors that don't match the most common length
            if tensor_length != most_common_length:
                os.remove(file_path)
                delete_count += 1

        except Exception as e:
            print(f"Error processing file {file}: {e}")
    
    # Output results
    print(f"Most common tensor length: {most_common_length}")
    print(f"Number of files deleted: {delete_count}")

def process_stock_torch_files_by_max(directory):
   # Initialize variables
    max_length = 0
    delete_count = 0
    
    # Get all .pt files in the directory
    files = [f for f in os.listdir(directory) if f.endswith('.pt')]
    num_files = len(files)
    
    print(f"Number of files in the directory: {num_files}")

    # Step 1: Find the highest tensor length
    for file in files:
        file_path = os.path.join(directory, file)
        try:
            # Load the tensor
            tensor = torch.load(file_path)
            
            # Skip if it's not a tensor (e.g. dict or other objects)
            if not isinstance(tensor, torch.Tensor):
                continue
            
            # Check for inf/-inf values and skip if found
            if torch.isinf(tensor).any():
                os.remove(file_path)
                delete_count += 1
                continue

            # Get tensor length
            tensor_length = tensor.numel()
            max_length = max(max_length, tensor_length)
        
        except Exception as e:
            print(f"Error processing file {file}: {e}")
    
    # Step 2: Delete tensors not matching the max length
    for file in files:
        file_path = os.path.join(directory, file)
        try:
            tensor = torch.load(file_path)

            # Skip if it's not a tensor or already deleted
            if not isinstance(tensor, torch.Tensor) or not os.path.exists(file_path):
                continue

            tensor_length = tensor.numel()

            # Delete tensors that don't match the max length
            if tensor_length != max_length:
                os.remove(file_path)
                delete_count += 1

        except Exception as e:
            print(f"Error processing file {file}: {e}")
    
    # Output results
    print(f"Highest tensor length: {max_length}")
    print(f"Number of files deleted: {delete_count}")

# Usage example: process_torch_files('/path/to/your/directory')


def process_GT_stock_torch_files(dataset_path):
    for root, dirs, files in os.walk(dataset_path):
        for file in files:
            if file.endswith('.pt'):
                tensor = torch.load(os.path.join(root, file))
                if (tensor.numel() == 0 or
                    torch.isnan(tensor).any() or 
                    torch.isinf(tensor).any() or 
                    torch.any(tensor == 0) or 
                    torch.any(tensor == -0) or 
                    torch.any(torch.isin(tensor, torch.tensor([float('inf'), float('-inf')])))):  # Fixed this line
                    os.remove(os.path.join(root, file))


def thorough_process_GT_stock_torch_files(dataset_path):
    pass