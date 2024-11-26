import pandas as pd

def delete_symbols_from_csv(txt_path: str, csv_path: str) -> None:
    """
    Reads symbols from a text file and removes corresponding rows from a CSV file.
    
    Args:
        txt_path (str): Path to the text file containing symbols (one per line)
        csv_path (str): Path to the CSV file containing market data
        
    The CSV file should have the structure:
    Symbol,Name,Last Sale,Net Change,% Change,Market Cap,Country,IPO Year,Volume,Sector,Industry
    """
    # Read symbols from text file, creating a set for efficient lookup
    with open(txt_path, 'r') as f:
        symbols_to_delete = {line.strip() for line in f if line.strip()}
    
    # Read the CSV and filter out rows with matching symbols
    df = pd.read_csv(csv_path)
    df_filtered = df[~df['Symbol'].isin(symbols_to_delete)]
    
    # Save the filtered data to a new CSV
    output_path = csv_path.replace('.csv', '_filtered.csv')
    df_filtered.to_csv(output_path, index=False)
    
    print(f"Removed {len(df) - len(df_filtered)} rows")
    print(f"Saved to {output_path}")

if __name__ == "__main__":
    # Define your file paths here
    txt_path = 'symbols.txt'
    csv_path = 'market_data.csv'
    
    delete_symbols_from_csv(txt_path, csv_path)