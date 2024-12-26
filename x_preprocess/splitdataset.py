import os
import glob
from datetime import datetime
import sys

def parse_filename(filename):
    """
    Parse the filename to extract symbol, date, and time
    Example: ABNB_20240226_115340.pt -> (ABNB, 2024-02-26 11:53:40)
    """
    try:
        # Split by underscore and remove .pt extension
        parts = filename.replace('.pt', '').split('_')
        
        # The symbol is the first part
        symbol = parts[0]
        
        # Parse date and time
        date_str = parts[1]
        time_str = parts[2]
        
        # Combine date and time strings and convert to datetime
        datetime_str = f"{date_str[:4]}-{date_str[4:6]}-{date_str[6:]} {time_str[:2]}:{time_str[2:4]}:{time_str[4:]}"
        file_datetime = datetime.strptime(datetime_str, '%Y-%m-%d %H:%M:%S')
        
        return symbol, file_datetime
    except (IndexError, ValueError) as e:
        print(f"Error parsing filename {filename}: {str(e)}")
        return None, None

def delete_old_pt_files(data_dir, cutoff_date, dry_run=True):
    """
    Delete .pt files older than the cutoff date
    dry_run: If True, only prints what would be deleted without actually deleting
    """
    # Counter for deleted files
    delete_count = 0
    error_count = 0
    
    # Get all .pt files in directory
    pt_files = glob.glob(os.path.join(data_dir, "*.pt"))
    
    print(f"Found {len(pt_files)} .pt files in directory")
    print(f"{'Would delete' if dry_run else 'Deleting'} files older than {cutoff_date}")
    
    for file_path in pt_files:
        filename = os.path.basename(file_path)
        symbol, file_datetime = parse_filename(filename)
        
        if symbol is None or file_datetime is None:
            error_count += 1
            continue
            
        if file_datetime < cutoff_date:
            try:
                if dry_run:
                    print(f"Would delete: {filename} (date: {file_datetime})")
                else:
                    os.remove(file_path)
                    print(f"Deleted: {filename} (date: {file_datetime})")
                delete_count += 1
            except Exception as e:
                print(f"Error processing {filename}: {str(e)}")
                error_count += 1
    
    return delete_count, error_count

def main():
    # Define your data directory here
    data_dir = "/Users/daniellavin/Desktop/proj/MoneyTrainer/minivals"  # Replace with your path
    
    # Define cutoff date - files BEFORE this date will be deleted
    # Example: January 1, 2024 00:00:00
    cutoff_date = datetime(2024, 11, 5, 0, 0, 0)
    
    # Safety: First run in dry-run mode
    dry_run = False  # Set to False to actually delete files
    
    print(f"Processing files in {data_dir}")
    print(f"Cutoff date: {cutoff_date}")
    print(f"DRY RUN: {'Yes' if dry_run else 'No'}")
    
    # Check if directory exists
    if not os.path.exists(data_dir):
        print(f"Error: Directory '{data_dir}' does not exist!")
        sys.exit(1)
    
    # Confirm if not in dry-run mode
    if not dry_run:
        confirm = input("This will permanently delete files. Are you sure? (yes/no): ")
        if confirm.lower() != 'yes':
            print("Operation cancelled.")
            sys.exit(0)
    
    # Delete files
    delete_count, error_count = delete_old_pt_files(data_dir, cutoff_date, dry_run)
    
    # Print summary
    print("\nOperation completed!")
    if dry_run:
        print(f"Would have deleted: {delete_count} files")
    else:
        print(f"Files deleted: {delete_count}")
    print(f"Errors encountered: {error_count}")

if __name__ == "__main__":
    main()