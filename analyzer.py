import os
import random
import torch
import argparse
from collections import Counter

def count_lengths_in_subfolder(folder_path):
    lengths = []
    
    # Iterate over all files in the folder
    for file_name in os.listdir(folder_path):
        if file_name.endswith('.pt') or file_name.endswith('.pth'):
            file_path = os.path.join(folder_path, file_name)
            try:
                # Load the torch file
                tensor = torch.load(file_path)
                
                # If the loaded file is a tensor, get its length
                if isinstance(tensor, torch.Tensor):
                    lengths.append(tensor.size(0))
                # If it's a dictionary (like a state_dict), calculate based on the tensor lengths in it
                elif isinstance(tensor, dict):
                    for key, val in tensor.items():
                        if isinstance(val, torch.Tensor):
                            lengths.append(val.size(0))
            except Exception as e:
                print(f"Error loading {file_name}: {e}")

    # Count how often each length appears
    length_counts = Counter(lengths)
    return length_counts

def process_random_subfolders(base_folder, n=5):
    # Get the list of all subfolders
    subfolders = [f.path for f in os.scandir(base_folder) if f.is_dir()]

    if len(subfolders) < n:
        print(f"Not enough subfolders, found {len(subfolders)}.")
        return
    
    # Randomly select n subfolders
    selected_subfolders = random.sample(subfolders, n)

    for subfolder in selected_subfolders:
        print(f"\nProcessing folder: {subfolder}")
        length_counts = count_lengths_in_subfolder(subfolder)
        
        if length_counts:
            for length, count in length_counts.items():
                print(f"Length: {length}, Count: {count}")
        else:
            print(f"No torch files found or no valid lengths in folder: {subfolder}")

if __name__ == "__main__":
   
    process_random_subfolders('stockdataset')
