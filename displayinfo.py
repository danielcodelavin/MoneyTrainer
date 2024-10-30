import re

def correct_loss_values(file_path):
    # Read the file
    with open(file_path, 'r') as f:
        lines = f.readlines()
    
    print("\nOriginal Values -> Corrected Values (divided by 16)\n")
    print("=" * 50)
    
    # Process each line
    for line in lines:
        # Extract epoch number and loss values using regex
        match = re.match(r'Epoch (\d+), Train Loss: (\d+\.\d+), Val Loss: (\d+\.\d+)', line.strip())
        
        if match:
            epoch = int(match.group(1))
            train_loss = float(match.group(2))
            val_loss = float(match.group(3))
            
            # Calculate corrected values
            corrected_train = train_loss / (1660.0 * 16.0)
            corrected_val = val_loss / (1660.0 * 16.0)
            
            # Print original and corrected values
            print(f"Epoch {epoch:3d}:")
            print(f"  Train Loss: {corrected_train:.8f}")
            print(f"  Val Loss:  {corrected_val:.8f}")
            #print("-" * 50)

if __name__ == "__main__":
    file_path = "/usr/prakt/s0097/epochinformations/sixepochinfo.txt"  # Update this path to your txt file location
    try:
        correct_loss_values(file_path)
    except FileNotFoundError:
        print(f"Error: Could not find file at {file_path}")
        print("Please update the file_path variable to point to your epochinfo.txt file.")