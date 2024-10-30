import os
import torch

def print_tensors_in_folder(folder_path: str):
    print(f"Printing tensors in folder: {folder_path}")
    print("=" * 50)

    for root, dirs, files in os.walk(folder_path):
        for file in files:
            if file.endswith('.pt'):
                file_path = os.path.join(root, file)
                try:
                    # Load the tensor
                    tensor = torch.load(file_path)
                    
                    # Check if it's a tensor
                    if isinstance(tensor, torch.Tensor):
                        print(f"File: {file_path}")
                        print("Values:", tensor.tolist())
                        print("-" * 50)
                    else:
                        print(f"File: {file_path}")
                        print("Not a tensor. Type:", type(tensor))
                        print("-" * 50)
                except Exception as e:
                    print(f"Error loading {file_path}: {str(e)}")
                    print("-" * 50)

    print("Finished printing all tensors.")

# Usage
print_tensors_in_folder('/Users/daniellavin/Desktop/proj/Moneytrain/stockdataset/2023-03-20_12-36-56')


