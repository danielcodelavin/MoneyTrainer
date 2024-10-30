import os
import csv
import torch
from moneytensorgen import returngroundtruthstock

def process_torch_and_csv(torch_dir, csv_path, chunk_size=1000):
    # Get all .pt files and process their names
    torch_names = set()
    for filename in os.listdir(torch_dir):
        if filename.endswith('.pt'):
            name = filename[:-3].replace('_', '/')
            torch_names.add(name)

    # Process the CSV file in chunks
    temp_csv_path = csv_path + '.temp'
    with open(csv_path, 'r') as input_file, open(temp_csv_path, 'w', newline='') as output_file:
        csv_reader = csv.reader(input_file)
        csv_writer = csv.writer(output_file)

        chunk = []
        for row in csv_reader:
            chunk.append(row)
            if len(chunk) >= chunk_size:
                process_chunk(chunk, torch_names, csv_writer)
                chunk = []
        
        # Process any remaining rows
        if chunk:
            process_chunk(chunk, torch_names, csv_writer)

    # Replace the original CSV with the filtered version
    os.replace(temp_csv_path, csv_path)

    print(f"Processed {len(torch_names)} torch files.")
    print(f"CSV file has been updated at {csv_path}")

def process_chunk(chunk, torch_names, csv_writer):
    for row in chunk:
        if row and any(row[0] in name for name in torch_names):
            csv_writer.writerow(row)





def lookatfolder(filepath):
       
        for file in filepath:
            if file.endswith('.pt') or file.endswith('.pth'):
                file_path = os.path.join(root, file)
                print(f"File name: {file}")
                try:
                    content = torch.load(file_path)
                    print(f"Content: {content}")
                    print(f"Tensor length: {content.numel()}")
                except Exception as e:
                    print(f"Error loading file: {e}")
                print()  # Add a blank line between files





if __name__ == '__main__':
    #  print("GT:")
    #  path = '/Users/daniellavin/Desktop/proj/Moneytrain/stockdataset/2024-05-21_20-09-03/GT'
    #  for root, dirs, files in os.walk(path):
    #      lookatfolder(files)
    #process_torch_and_csv('/Users/daniellavin/Desktop/proj/Moneytrain/stockdataset/2024-05-24_15-45-59', '/Users/daniellavin/Desktop/proj/Moneytrain/cleaned_stockscreen.csv')
    torchvector = returngroundtruthstock('AAPL', '2024-05-21 20:09:03')
    print(torchvector)
    print(torchvector.numel())
