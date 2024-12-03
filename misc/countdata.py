import os

def count_folders_and_files(folder_path):
    subfolder_count = 0
    file_count = 0
    
    # Walk through the directory tree
    for root, dirs, files in os.walk(folder_path):
        subfolder_count += len(dirs)   # Add the number of subfolders at this level
        file_count += len(files)       # Add the number of files at this level

    return subfolder_count, file_count

# Example usage:
folder_path = '/Users/daniellavin/Desktop/proj/Moneytrain/stockdataset'  # Replace with your folder path
subfolders, files = count_folders_and_files(folder_path)

print(f"Subfolders: {int(subfolders/2)}")
print(f"Total files: {int(files/2)}")
