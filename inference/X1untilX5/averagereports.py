import os
import re
from collections import defaultdict
from typing import Dict, List, Tuple
from pathlib import Path

class ReportAggregator:
    def __init__(self):
        self.market_cap_data = defaultdict(list)
        self.volume_data = defaultdict(list)
        self.sector_data = defaultdict(list)
        self.total_reports = 0

    def parse_metrics(self, section_text: str) -> Dict[str, float]:
        """Parse metrics from a category section."""
        metrics = {}
        lines = section_text.split('\n')
        
        for line in lines:
            if 'Number of stocks:' in line:
                metrics['count'] = float(line.split(':')[1].strip())
            elif 'Average L1 Distance:' in line:
                metrics['avg_l1_distance'] = float(line.split(':')[1].strip())
            elif 'Average RMSE:' in line:
                metrics['avg_rmse'] = float(line.split(':')[1].strip())
            elif 'Prediction Error Std:' in line:
                metrics['prediction_error_std'] = float(line.split(':')[1].strip())
            elif 'Precision:' in line:
                metrics['precision'] = float(line.split(':')[1].strip())
        
        return metrics

    def parse_report(self, file_path: str) -> None:
        """Parse a single report file."""
        try:
            with open(file_path, 'r') as f:
                content = f.read()

            # Split content into major sections
            sections = content.split('\n\n')
            
            current_section = None
            current_data = None
            
            for section in sections:
                if 'Market Cap Analysis' in section:
                    current_section = 'market_cap'
                    current_data = self.market_cap_data
                elif 'Volume Analysis' in section:
                    current_section = 'volume'
                    current_data = self.volume_data
                elif 'Sector Analysis' in section:
                    current_section = 'sector'
                    current_data = self.sector_data
                
                if current_section and current_data is not None:
                    # Extract individual category data
                    categories = re.split(r'\n(?=Q[1-4]|[A-Za-z]+ \w+:)', section)
                    for category in categories:
                        if ':' in category:
                            category_name = category.split(':')[0].strip()
                            if category_name and not category_name.startswith('Key Insights'):
                                metrics = self.parse_metrics(category)
                                if metrics:
                                    current_data[category_name].append(metrics)

            self.total_reports += 1
            
        except Exception as e:
            print(f"Error processing {file_path}: {str(e)}")

    def calculate_averages(self) -> Tuple[Dict, Dict, Dict]:
        """Calculate averages for all sections."""
        def avg_metrics(metrics_list: List[Dict]) -> Dict:
            if not metrics_list:
                return {}
            
            avg = defaultdict(float)
            counts = defaultdict(int)
            
            for metrics in metrics_list:
                for key, value in metrics.items():
                    if isinstance(value, (int, float)):
                        avg[key] += value
                        counts[key] += 1
            
            return {k: avg[k] / counts[k] for k in avg}

        market_cap_avg = {k: avg_metrics(v) for k, v in self.market_cap_data.items()}
        volume_avg = {k: avg_metrics(v) for k, v in self.volume_data.items()}
        sector_avg = {k: avg_metrics(v) for k, v in self.sector_data.items()}

        return market_cap_avg, volume_avg, sector_avg

    def generate_summary_report(self, output_path: str) -> None:
        """Generate and save the summary report."""
        market_cap_avg, volume_avg, sector_avg = self.calculate_averages()
        
        with open(output_path, 'w') as f:
            f.write(f"Summary Report (Averaged across {self.total_reports} reports)\n")
            f.write("====================================================\n\n")
            
            sections = [
                ("Market Cap Analysis", market_cap_avg),
                ("Volume Analysis", volume_avg),
                ("Sector Analysis", sector_avg)
            ]
            
            for section_name, data in sections:
                f.write(f"\n{section_name}\n")
                f.write("-" * len(section_name) + "\n\n")
                
                sorted_categories = sorted(data.items(), 
                                        key=lambda x: x[1].get('avg_l1_distance', float('inf')))
                
                for category, metrics in sorted_categories:
                    f.write(f"\n{category}:\n")
                    if 'count' in metrics:
                        f.write(f"  Average Number of stocks: {metrics['count']:.1f}\n")
                    if 'avg_l1_distance' in metrics:
                        f.write(f"  Average L1 Distance: {metrics['avg_l1_distance']:.4f}\n")
                    if 'avg_rmse' in metrics:
                        f.write(f"  Average RMSE: {metrics['avg_rmse']:.4f}\n")
                    if 'prediction_error_std' in metrics:
                        f.write(f"  Average Prediction Error Std: {metrics['prediction_error_std']:.4f}\n")
                    if 'precision' in metrics:
                        f.write(f"  Average Precision: {metrics['precision']:.4f}\n")
                
                if sorted_categories:
                    f.write("\nKey Insights:\n")
                    best_category = sorted_categories[0][0]
                    worst_category = sorted_categories[-1][0]
                    f.write(f"- Best performing category: {best_category}\n")
                    f.write(f"- Worst performing category: {worst_category}\n")
                    
                    if ('avg_l1_distance' in sorted_categories[0][1] and 
                        'avg_l1_distance' in sorted_categories[-1][1]):
                        perf_diff = abs(sorted_categories[0][1]['avg_l1_distance'] - 
                                      sorted_categories[-1][1]['avg_l1_distance']) * 100
                        f.write(f"- Performance difference: {perf_diff:.2f}%\n")
                
                f.write("\n")

def main():
    # Replace with your main directory path
    main_dir = "/Users/daniellavin/Desktop/proj/MoneyTrainer/results/"
    
    # Create sum directory if it doesn't exist
    sum_dir = os.path.join(main_dir, "sum")
    os.makedirs(sum_dir, exist_ok=True)
    
    # Initialize aggregator
    aggregator = ReportAggregator()
    
    # Walk through all subdirectories
    for root, _, files in os.walk(main_dir):
        for file in files:
            if file.endswith('.txt') and 'analysis_report' in file:
                file_path = os.path.join(root, file)
                aggregator.parse_report(file_path)
    
    # Generate and save summary report
    summary_path = os.path.join(sum_dir, "summary_report.txt")
    aggregator.generate_summary_report(summary_path)
    print(f"Summary report generated at: {summary_path}")

if __name__ == "__main__":
    main()