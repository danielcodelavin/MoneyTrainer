import os
import shutil

"""
This script is designed to clean up subfolders within a main folder based on specific criteria:

1. It iterates through all subfolders in the specified main folder.
2. For each subfolder, it checks two conditions:
   a. The subfolder must contain more than 5 files.
   b. The subfolder must contain a file named "0news.pt".
3. If both conditions are met, the subfolder is kept.
4. If either condition is not met, the subfolder is deleted.

The script uses the os module for file and directory operations and the shutil module
for removing directories. It provides feedback by printing which subfolders are kept
and which are deleted.

Usage:
Call the cleanup_subfolders() function with the path to the main folder as an argument.
"""


def cleanup_subfolders(main_folder_path):
    for subfolder_name in os.listdir(main_folder_path):
        subfolder_path = os.path.join(main_folder_path, subfolder_name)
        
        # Check if it's a directory
        if not os.path.isdir(subfolder_path):
            continue
        
        # Count files in the subfolder
        file_count = len([f for f in os.listdir(subfolder_path) if os.path.isfile(os.path.join(subfolder_path, f))])
        
        # Check for the presence of "0news.pt"
        has_news_file = "0news.pt" in os.listdir(subfolder_path)
        
        # If both conditions are not met, delete the subfolder
        if not (file_count > 5 and has_news_file):
            print(f"Deleting subfolder: {subfolder_path}")
            shutil.rmtree(subfolder_path)
        else:
            print(f"Keeping subfolder: {subfolder_path}")

# Example usage
cleanup_subfolders("/Users/daniellavin/Desktop/proj/Moneytrain/stockdataset")