import pandas as pd
import datetime
from datetime import timedelta
from pathlib import Path
import numpy as np
import torch
import time
from typing import List, Tuple, Dict, Optional
import os
from typing import List, Optional
import yfinance as yf
import json
from collections import defaultdict

def calculate_percentage_change(latest_value: float, gt_value: float) -> float:
    return (gt_value - latest_value) / latest_value

def prepare_single_stock_data(ticker_symbol: str, start_datetime: datetime, days: int = 7, min_points: int = 80) -> Optional[Tuple[torch.Tensor, float]]:
    ticker = yf.Ticker(ticker_symbol)
    all_data = []
    days_checked = 0
    print(f"Fetching data for {ticker_symbol}")
    
    while len(all_data) < min_points and days_checked < days:
        try:
            day_end = start_datetime - timedelta(days=days_checked)
            day_start = day_end - timedelta(days=1)
            data = ticker.history(start=day_start, end=day_end, interval="60m", prepost=True, auto_adjust=False)
            
            if not data.empty and 'Close' in data.columns:
                valid_data = data['Close'].dropna()
                if validate_raw_data(valid_data):
                    all_data.extend(valid_data.values)
                    print(f"Added {len(valid_data)} points for {ticker_symbol} - Total: {len(all_data)}")
        except Exception as e:
            print(f"Error on {ticker_symbol} for {day_start}: {str(e)}")
        
        days_checked += 1
        time.sleep(0.3)  # Rate limiting
    
    if len(all_data) < min_points:
        print(f"Insufficient data points ({len(all_data)}/{min_points}) for {ticker_symbol}")
        if len(all_data) <= (min_points*0.7):
            with open('problematic_tickers.txt', 'a') as f:
                f.write(f"{ticker_symbol}\n")
        return None
        
    try:
        if len(all_data) > min_points:
            all_data = all_data[:min_points]
            
        tensor_data = torch.tensor(all_data, dtype=torch.float32)
        mean = tensor_data.mean()
        std = tensor_data.std()
        if std == 0:
            return None
        normalized_tensor = (tensor_data - mean) / std
        return normalized_tensor, all_data[-1]  # Return normalized data and last raw price
    except Exception as e:
        print(f"Error processing {ticker_symbol} data: {str(e)}")
        return None

def returngroundtruthstock(stock_symbol: str, target_datetime: datetime, max_days: int = 4) -> torch.Tensor:
    try:
        stock = yf.Ticker(stock_symbol)
        days_checked = 0
        
        while days_checked < max_days:
            check_date = target_datetime + timedelta(days=days_checked)
            if check_date.weekday() >= 5:  # Skip weekends
                days_checked += 1
                continue
                
            data = stock.history(
                start=check_date.strftime('%Y-%m-%d'),
                end=(check_date + timedelta(days=1)).strftime('%Y-%m-%d'),
                interval="60m",
                prepost=True
            )
            
            if not data.empty and 'Close' in data.columns:
                first_price = data['Close'].iloc[0]
                if not np.isnan(first_price) and not np.isinf(first_price) and first_price > 0:
                    print(f"Found first price {first_price} for {stock_symbol} on {check_date.strftime('%Y-%m-%d')}")
                    return torch.tensor([first_price], dtype=torch.float32)
            
            days_checked += 1
            
        print(f"No valid price found for {stock_symbol} within {max_days} days")
        return torch.tensor([])
        
    except Exception as e:
        print(f"Error getting price for {stock_symbol}: {str(e)}")
        return torch.tensor([])

def validate_raw_data(data: pd.Series) -> bool:
    if data is None or data.empty:
        return False
    if np.isinf(data).any():
        return False
    if np.isnan(data).any():
        return False
    if (data <= 0).any():
        return False
    return True

def extract_date_from_filename(filepath: str) -> datetime.datetime:
    filename = Path(filepath).name
    date_str = filename.split('_')[1]
    
    year = int(date_str[:4])
    month = int(date_str[4:6])
    day = int(date_str[6:8])
    
    return datetime.datetime(year, month, day)

def calculate_metrics(true_values: np.ndarray, predicted_values: np.ndarray) -> dict:
    l1_distance = np.abs(true_values - predicted_values)
    
    true_signs = np.sign(true_values)
    pred_signs = np.sign(predicted_values)
    
    tp = np.sum((true_signs == 1) & (pred_signs == 1))
    fp = np.sum((true_signs == -1) & (pred_signs == 1))
    tn = np.sum((true_signs == -1) & (pred_signs == -1))
    fn = np.sum((true_signs == 1) & (pred_signs == -1))
    
    accuracy = (tp + tn) / (tp + tn + fp + fn) if (tp + tn + fp + fn) > 0 else 0
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0
    f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
    
    # Add RMSE
    rmse = np.sqrt(np.mean((true_values - predicted_values) ** 2))
    
    # Add directional accuracy
    directional_accuracy = np.mean(np.sign(true_values) == np.sign(predicted_values))
    
    return {
        'l1_distance': l1_distance,
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'f1_score': f1,
        'rmse': rmse,
        'directional_accuracy': directional_accuracy
    }

def analyze_by_factor(df: pd.DataFrame, info_df: pd.DataFrame, factor: str) -> Tuple[dict, dict]:
    results = defaultdict(lambda: defaultdict(list))
    stock_listings = defaultdict(list)
    
    # Merge dataframes
    merged_df = df.merge(info_df, left_on='name', right_on='Symbol', how='left')
    
    if factor == 'Market Cap':
        try:
            merged_df['Market Cap'] = pd.to_numeric(merged_df['Market Cap'], errors='coerce')
            merged_df = merged_df[merged_df['Market Cap'].notna() & (merged_df['Market Cap'] > 0)]
            
            if len(merged_df) > 0:
                quartile_values = merged_df['Market Cap'].quantile([0.25, 0.5, 0.75])
                
                def assign_market_cap_quartile(x):
                    if x <= quartile_values[0.25]:
                        return f'Q1 (Small Cap: <${round(quartile_values[0.25]/1e9, 1)}B)'
                    elif x <= quartile_values[0.5]:
                        return f'Q2 (Mid Cap: ${round(quartile_values[0.25]/1e9, 1)}-{round(quartile_values[0.5]/1e9, 1)}B)'
                    elif x <= quartile_values[0.75]:
                        return f'Q3 (Large Cap: ${round(quartile_values[0.5]/1e9, 1)}-{round(quartile_values[0.75]/1e9, 1)}B)'
                    else:
                        return f'Q4 (Mega Cap: >${round(quartile_values[0.75]/1e9, 1)}B)'
                
                merged_df['Group'] = merged_df['Market Cap'].apply(assign_market_cap_quartile)
            else:
                return {}, {}
                
        except Exception as e:
            print(f"Error processing Market Cap data: {e}")
            return {}, {}
            
    elif factor == 'Volume':
        try:
            merged_df['Volume'] = pd.to_numeric(merged_df['Volume'], errors='coerce')
            merged_df = merged_df[merged_df['Volume'].notna() & (merged_df['Volume'] > 0)]
            
            if len(merged_df) > 0:
                quartile_values = merged_df['Volume'].quantile([0.25, 0.5, 0.75])
                
                def assign_volume_quartile(x):
                    if x <= quartile_values[0.25]:
                        return f'Q1 (Low Volume: <{int(quartile_values[0.25]):,})'
                    elif x <= quartile_values[0.5]:
                        return f'Q2 (Medium-Low: {int(quartile_values[0.25]):,}-{int(quartile_values[0.5]):,})'
                    elif x <= quartile_values[0.75]:
                        return f'Q3 (Medium-High: {int(quartile_values[0.5]):,}-{int(quartile_values[0.75]):,})'
                    else:
                        return f'Q4 (High Volume: >{int(quartile_values[0.75]):,})'
                
                merged_df['Group'] = merged_df['Volume'].apply(assign_volume_quartile)
            else:
                return {}, {}
                
        except Exception as e:
            print(f"Error processing Volume data: {e}")
            return {}, {}
    
    elif factor == 'Sector':
        if 'Sector' not in merged_df.columns:
            print(f"Sector column not found in data")
            return {}, {}
        merged_df = merged_df[merged_df['Sector'].notna()]
        merged_df['Group'] = merged_df['Sector']
    else:
        print(f"Unsupported factor: {factor}")
        return {}, {}
    
    # Calculate metrics for each group
    for name, group in merged_df.groupby('Group'):
        if name == 'SUMMARY':
            continue
            
        try:
            # Calculate metrics
            true_positives = len(group[
                (group['actual_percentage_change'] > 0) & 
                (group['prediction'] > 0)
            ])
            predicted_positives = len(group[group['prediction'] > 0])
            precision = true_positives / predicted_positives if predicted_positives > 0 else 0
            
            # Store stock information
            stock_listings[str(name)] = [
                {
                    'symbol': str(row['name']),
                    'name': str(row.get('Name', '')),
                    'market_cap': float(row.get('Market Cap', 0)),
                    'volume': float(row.get('Volume', 0)),
                    'sector': str(row.get('Sector', ''))
                }
                for _, row in group.iterrows()
                if not pd.isna(row['name'])
            ]
            
            # Calculate aggregate metrics
            metrics = {
                'count': len(group),
                'avg_l1_distance': group['l1_distance'].mean(),
                'avg_rmse': np.sqrt(np.mean(group['prediction_error'] ** 2)) if 'prediction_error' in group.columns else None,
                'prediction_error_std': group['prediction_error'].std() if 'prediction_error' in group.columns else None,
                'precision': precision
            }
            
            # Filter out None values and NaN
            metrics = {k: float(v) for k, v in metrics.items() if v is not None and not pd.isna(v)}
            
            if metrics:
                results[factor][str(name)] = metrics
            
        except Exception as e:
            print(f"Error calculating metrics for group {name}: {e}")
            continue
    
    return dict(results), dict(stock_listings)

def generate_analysis_report(results: dict, stock_listings: dict, output_path: str):
    with open(output_path, 'w') as f:
        f.write("Stock Prediction Analysis Report\n")
        f.write("===============================\n\n")
        
        for factor, factor_results in results.items():
            if not factor_results:
                continue
                
            f.write(f"\n{factor} Analysis\n")
            f.write("-" * (len(factor) + 9) + "\n\n")
            
            sorted_results = sorted(
                factor_results.items(),
                key=lambda x: x[1].get('avg_l1_distance', float('inf'))
            )
            
            for category, metrics in sorted_results:
                if not metrics:
                    continue
                    
                f.write(f"\n{category}:\n")
                f.write(f"  Number of stocks: {metrics['count']}\n")
                
                metric_display = {
                    'avg_l1_distance': 'Average L1 Distance',
                    'avg_rmse': 'Average RMSE',
                    'prediction_error_std': 'Prediction Error Std',
                    'precision': 'Precision'
                }
                
                for metric_key, display_name in metric_display.items():
                    if metric_key in metrics:
                        f.write(f"  {display_name}: {metrics[metric_key]:.4f}\n")
            
            if sorted_results:
                f.write("\nKey Insights:\n")
                best_category = sorted_results[0][0]
                worst_category = sorted_results[-1][0]
                f.write(f"- Best performing category: {best_category}\n")
                f.write(f"- Worst performing category: {worst_category}\n")
                
                if ('avg_l1_distance' in sorted_results[0][1] and 
                    'avg_l1_distance' in sorted_results[-1][1]):
                    perf_diff = abs(sorted_results[0][1]['avg_l1_distance'] - 
                                  sorted_results[-1][1]['avg_l1_distance']) * 100
                    f.write(f"- Performance difference: {perf_diff:.2f}%\n")
                
                f.write("\n")
        
        f.write("\nDetailed Stock Listings by Category\n")
        f.write("================================\n\n")
        
        for factor, factor_listings in stock_listings.items():
            if not isinstance(factor_listings, dict):
                continue
                
            f.write(f"\n{factor} Categories:\n")
            f.write("----------------\n")
            
            for category, stocks in factor_listings.items():
                if not isinstance(stocks, list):
                    continue
                    
                f.write(f"\n{category} ({len(stocks)} stocks):\n")
                
                try:
                    sorted_stocks = sorted(stocks, key=lambda x: str(x.get('symbol', '')))
                except (TypeError, AttributeError):
                    sorted_stocks = stocks
                
                for stock in sorted_stocks:
                    if not isinstance(stock, dict):
                        continue
                        
                    try:
                        f.write(f"  {stock.get('symbol', 'N/A')}: {stock.get('name', 'N/A')}\n")
                        if 'market_cap' in stock:
                            market_cap = stock['market_cap']
                            if isinstance(market_cap, (int, float)) and market_cap > 0:
                                f.write(f"    Market Cap: ${market_cap:,.2f}\n")
                        if 'volume' in stock:
                            volume = stock['volume']
                            if isinstance(volume, (int, float)) and volume > 0:
                                f.write(f"    Volume: {volume:,.0f}\n")
                        if 'sector' in stock:
                            f.write(f"    Sector: {stock['sector']}\n")
                    except Exception as e:
                        print(f"Error writing stock info: {e}")
                        continue
                f.write("\n")

def process_csv(filepath: str, info_filepath: str) -> Tuple[pd.DataFrame, dict, dict]:
    # Read both CSVs
    df = pd.read_csv(filepath)
    info_df = pd.read_csv(info_filepath)
    
    date = extract_date_from_filename(filepath)
    next_date = date + datetime.timedelta(days=1)
    
    l1_distances = []
    true_values = []
    predicted_values = []
    real_values = []
    
    # Process each row
    for idx, row in df.iterrows():
        stock_data_result = prepare_single_stock_data(
            ticker_symbol=row['name'],
            start_datetime=date,
            days=10,
            min_points=10
        )
        
        if stock_data_result is not None:
            stock_data, raw_latest_price = stock_data_result
            
            gt_value = returngroundtruthstock(row['name'], next_date)
            if gt_value.numel() == 0:
                print(f"No ground truth for {row['name']} on {next_date}")
                continue
                
            gt_value = float(gt_value.item())
            percentage_change = calculate_percentage_change(raw_latest_price, gt_value)
            
            # Store raw values and calculated metrics
            df.at[idx, 'real_value'] = gt_value
            df.at[idx, 'raw_latest_price'] = raw_latest_price
            df.at[idx, 'actual_percentage_change'] = percentage_change
            df.at[idx, 'prediction_error'] = abs(percentage_change - row['prediction'])
            
            l1_dist = float(abs(percentage_change - row['prediction']))
            l1_distances.append(l1_dist)
            
            true_values.append(percentage_change)
            predicted_values.append(float(row['prediction']))
            real_values.append(gt_value)
            
            df.at[idx, 'l1_distance'] = l1_dist
    
    if true_values and predicted_values:
        metrics = calculate_metrics(
            np.array(true_values, dtype=np.float32), 
            np.array(predicted_values, dtype=np.float32)
        )
        
        summary_row = {
            'name': 'SUMMARY',
            'prediction': np.nan,
            'real_value': np.mean(real_values),
            'raw_latest_price': np.nan,
            'actual_percentage_change': np.mean(true_values),
            'prediction_error': np.mean(np.abs(np.array(true_values) - np.array(predicted_values))),
            'l1_distance': np.mean(l1_distances)
        }
        
        for metric, value in metrics.items():
            if metric != 'l1_distance':
                df[metric] = np.nan
                summary_row[metric] = float(value)
        
        df = pd.concat([df, pd.DataFrame([summary_row])], ignore_index=True)
        
        # Perform factor analysis
        factor_analysis = {}
        stock_listings = {}
        for factor in ['Market Cap', 'Volume', 'Sector']:
            results, listings = analyze_by_factor(df, info_df, factor)
            factor_analysis[factor] = results.get(factor, {})
            stock_listings[factor] = listings
        
        return df, factor_analysis, stock_listings
    
    return df, {}, {}

def main():
    #filepath = "/Users/daniellavin/Desktop/proj/MoneyTrainer/TANDEM/XT/results/2025/01/07/results_20250107_180000.csv"
    pahts = ["/Users/daniellavin/Desktop/proj/MoneyTrainer/TANDEM/XT/results/2024/12/02/results_20241202_180000.csv","/Users/daniellavin/Desktop/proj/MoneyTrainer/TANDEM/XT/results/2024/12/03/results_20241203_180000.csv",
             "/Users/daniellavin/Desktop/proj/MoneyTrainer/TANDEM/XT/results/2024/12/04/results_20241204_180000.csv","/Users/daniellavin/Desktop/proj/MoneyTrainer/TANDEM/XT/results/2024/12/05/results_20241205_180000.csv",
             "/Users/daniellavin/Desktop/proj/MoneyTrainer/TANDEM/XT/results/2024/12/09/results_20241209_180000.csv","/Users/daniellavin/Desktop/proj/MoneyTrainer/TANDEM/XT/results/2024/12/10/results_20241210_180000.csv",
             "/Users/daniellavin/Desktop/proj/MoneyTrainer/TANDEM/XT/results/2024/12/11/results_20241211_180000.csv","/Users/daniellavin/Desktop/proj/MoneyTrainer/TANDEM/XT/results/2024/12/12/results_20241212_180000.csv",
             "/Users/daniellavin/Desktop/proj/MoneyTrainer/TANDEM/XT/results/2024/12/16/results_20241216_180000.csv","/Users/daniellavin/Desktop/proj/MoneyTrainer/TANDEM/XT/results/2024/12/17/results_20241217_180000.csv",
             "/Users/daniellavin/Desktop/proj/MoneyTrainer/TANDEM/XT/results/2024/12/18/results_20241218_180000.csv","/Users/daniellavin/Desktop/proj/MoneyTrainer/TANDEM/XT/results/2024/12/19/results_20241219_180000.csv"]
    


    info_filepath = "/Users/daniellavin/Desktop/proj/MoneyTrainer/Hybrid_stockscreen.csv"
    
    
    print(f"Using stock information from: {info_filepath}")
    for filepath in pahts:
        try:
            print(f"Processing results from: {filepath}")
            # Process CSV and get analysis
            result_df, factor_analysis, stock_listings = process_csv(filepath, info_filepath)
            
            # Save enhanced results CSV
            output_dir = Path(filepath).parent
            base_name = Path(filepath).stem
            
            # Save detailed CSV
            csv_output_path = output_dir / f"analyzed_{base_name}.csv"
            result_df.to_csv(csv_output_path, index=False)
            
            # Generate and save analysis report
            report_path = output_dir / f"analysis_report_{base_name}.txt"
            generate_analysis_report(factor_analysis, stock_listings, report_path)
            
            print(f"\nAnalysis complete.")
            print(f"Results saved to: {csv_output_path}")
            print(f"Detailed analysis report saved to: {report_path}")
            
        except Exception as e:
            print(f"\nError during analysis: {str(e)}")
            import traceback
            print(traceback.format_exc())

if __name__ == "__main__":
    main()