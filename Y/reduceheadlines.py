import os
import torch
from pathlib import Path

def analyze_pt_file(file_path):
    """
    Analyze a PyTorch file and return its content information.
    """
    try:
        # Load the PyTorch file
        data = torch.load(file_path, map_location=torch.device('cpu'))
        
        # Function to recursively analyze tensors
        def analyze_tensor(obj, path=""):
            results = []
            
            if isinstance(obj, torch.Tensor):
                results.append({
                    'path': path,
                    'shape': list(obj.shape),
                    'numel': obj.numel(),
                    'dtype': str(obj.dtype)
                })
            elif isinstance(obj, dict):
                for key, value in obj.items():
                    new_path = f"{path}.{key}" if path else key
                    results.extend(analyze_tensor(value, new_path))
            elif isinstance(obj, (list, tuple)):
                for idx, value in enumerate(obj):
                    new_path = f"{path}[{idx}]"
                    results.extend(analyze_tensor(value, new_path))
            
            return results
        
        return analyze_tensor(data)
    
    except Exception as e:
        return [{'error': str(e)}]

def analyze_folder(folder_path):
    """
    Print analysis of all PyTorch files in the specified folder.
    """
    folder_path = Path(folder_path)
    pt_files = list(folder_path.rglob("*.pt"))
    
    print(f"\nAnalyzing PyTorch files in: {folder_path.absolute()}\n")
    
    if not pt_files:
        print("No PyTorch files found in the specified directory.")
        return
    
    for file_path in pt_files:
        rel_path = file_path.relative_to(folder_path)
        print(f"File: {rel_path}")
        print("-" * 40)
        
        results = analyze_pt_file(file_path)
        
        if results and 'error' in results[0]:
            print(f"Error analyzing file: {results[0]['error']}")
        else:
            total_params = 0
            for item in results:
                path = item['path']
                shape = item['shape']
                numel = item['numel']
                total_params += numel
                
                print(f"Tensor: {path}")
                print(f"  Shape: {shape}")
                print(f"  Elements: {numel:,}\n")
            
            print(f"Total parameters in file: {total_params:,}")
        
        print("\n" + "=" * 40 + "\n")
    
    print(f"Total files analyzed: {len(pt_files)}")

if __name__ == "__main__":
    # Define your folder path here
    folder_path = "/Users/daniellavin/Desktop/proj/MoneyTrainer/Y/y_100_train"  # <-- Change this to your folder path
    analyze_folder(folder_path)