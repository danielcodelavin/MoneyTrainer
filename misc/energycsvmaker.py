import pandas as pd

def filter_energy_sector(input_path: str, output_path: str) -> None:
    """
    Filter a CSV file to keep only companies in the Energy sector.
    
    Args:
        input_path (str): Path to the input CSV file
        output_path (str): Path where the filtered CSV will be saved
    """
    try:
        # Read the CSV file
        df = pd.read_csv(input_path)
        
        # Filter for Energy sector
        energy_df = df[df['Sector'] == 'Energy']
        
        # Save to new CSV file
        energy_df.to_csv(output_path, index=False)
        
        print(f"Successfully created filtered CSV at {output_path}")
        print(f"Found {len(energy_df)} companies in the Energy sector")
        
    except FileNotFoundError:
        print(f"Error: Could not find input file at {input_path}")
    except Exception as e:
        print(f"An error occurred: {str(e)}")

if __name__ == "__main__":
    # Set your input and output paths here
    input_file = "/Users/daniellavin/Desktop/proj/MoneyTrainer/Hybrid_stockscreen.csv"
    output_file = "/Users/daniellavin/Desktop/proj/MoneyTrainer/ENERGY_stockscreen.csv"
    
    filter_energy_sector(input_file, output_file)