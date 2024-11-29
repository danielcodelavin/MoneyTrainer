import os
import torch

def print_tensors_in_folder(folder_path: str):
   count = 0
   valid_count = 0
   lens = []
   max_gt = float('-inf')
   deleted = 0
   
   for root, dirs, files in os.walk(folder_path):
       for file in files:
           if file.endswith('.pt'):
               file_path = os.path.join(root, file)
               try:
                   tensor = torch.load(file_path)
                   count += 1
                   
                   # Check for invalid conditions
                   if (abs(float(tensor[0])) > 1.7 or 
                       torch.isnan(tensor).any() or 
                       torch.isinf(tensor).any()):
                       os.remove(file_path)
                       deleted += 1
                       continue
                       
                   valid_count += 1
                   gt = float(tensor[0])
                   max_gt = max(max_gt, gt)
                   lens.append(tensor.numel())
                   
               except Exception as e:
                   print(f"Error loading {file_path}: {str(e)}")
   
   print(f"\nDATASET STATISTICS:")
   print(f"Total tensors processed: {count}")
   print(f"Valid tensors remaining: {valid_count}")
   print(f"Tensors deleted: {deleted}")
   print(f"Average tensor length: {sum(lens) / valid_count if lens else 0:.2f}")
   print(f"Maximum GT value: {max_gt:.4f}")
   if lens:
       print(f"Length distribution: min={min(lens)}, max={max(lens)}")

print_tensors_in_folder('/Users/daniellavin/Desktop/proj/Moneytrainer/X_findataset')


