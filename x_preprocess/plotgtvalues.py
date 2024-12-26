import torch
import glob
import os
import numpy as np
from scipy.stats import ks_2samp, skew, kurtosis
import matplotlib.pyplot as plt
from typing import List, Dict, Tuple
import seaborn as sns
from datetime import datetime

def load_ground_truths(folder_path: str) -> List[float]:
    """Load ground truth values from all PT files in the folder."""
    gt_values = []
    for file_path in glob.glob(os.path.join(folder_path, "*.pt")):
        try:
            tensor = torch.load(file_path, weights_only=True)
            gt_values.append(tensor[0].item())
        except Exception as e:
            print(f"Error loading {file_path}: {e}")
    return gt_values

def analyze_thresholds(values: List[float]) -> Dict:
    """Analyze data retention at different magnitude thresholds."""
    values_array = np.abs(np.array(values))
    total_samples = len(values)
    
    thresholds = [0.02, 0.03, 0.04, 0.05, 0.06, 0.07, 0.08, 0.09, 0.10, 0.15, 0.20]
    threshold_stats = {}
    
    for threshold in thresholds:
        samples_within = np.sum(values_array <= threshold)
        samples_outside = total_samples - samples_within
        
        threshold_stats[threshold] = {
            "samples_within": samples_within,
            "samples_outside": samples_outside,
            "percent_retained": (samples_within / total_samples) * 100,
            "percent_removed": (samples_outside / total_samples) * 100
        }
    
    return threshold_stats

def analyze_distribution(values: List[float]) -> Dict:
    """Analyze statistical properties of the distribution."""
    values_array = np.array(values)
    return {
        "count": len(values),
        "mean": np.mean(values),
        "median": np.median(values),
        "std": np.std(values),
        "min": np.min(values),
        "max": np.max(values),
        "positive_ratio": np.mean(values_array > 0),
        "negative_ratio": np.mean(values_array < 0),
        "abs_mean": np.mean(np.abs(values_array)),
        "skewness": skew(values_array),
        "kurtosis": kurtosis(values_array),
        ">5%": np.mean(values_array > 0.05),
        "<-5%": np.mean(values_array < -0.05),
        "±1%": np.mean(np.abs(values_array) <= 0.01),
        "±2%": np.mean(np.abs(values_array) <= 0.02),
        "±5%": np.mean(np.abs(values_array) <= 0.05),
    }

def plot_distributions(values1: List[float], values2: List[float], 
                      label1: str, label2: str, save_path: str):
    """Create and save visualization of the two distributions."""
    plt.figure(figsize=(15, 15))
    
    # Main distribution plot
    plt.subplot(3, 1, 1)
    sns.kdeplot(data=values1, label=label1, fill=True, alpha=0.5)
    sns.kdeplot(data=values2, label=label2, fill=True, alpha=0.5)
    plt.title("Ground Truth Distribution Comparison")
    plt.xlabel("Price Change (%)")
    plt.ylabel("Density")
    plt.legend()
    plt.grid(True)
    
    # Box plot comparison
    plt.subplot(3, 1, 2)
    plt.boxplot([values1, values2], labels=[label1, label2])
    plt.title("Box Plot Comparison")
    plt.ylabel("Price Change (%)")
    plt.grid(True)
    
    # Histogram comparison
    plt.subplot(3, 1, 3)
    plt.hist(values1, bins=50, alpha=0.5, label=label1, density=True)
    plt.hist(values2, bins=50, alpha=0.5, label=label2, density=True)
    plt.title("Histogram Comparison")
    plt.xlabel("Price Change (%)")
    plt.ylabel("Density")
    plt.legend()
    plt.grid(True)
    
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()

def compare_distributions(folder1: str, folder2: str, output_dir: str):
    """Compare ground truth distributions between two folders."""
    
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    # Load values
    values1 = load_ground_truths(folder1)
    values2 = load_ground_truths(folder2)
    
    # Get folder names for labels
    label1 = os.path.basename(folder1)
    label2 = os.path.basename(folder2)
    
    # Analyze distributions
    stats1 = analyze_distribution(values1)
    stats2 = analyze_distribution(values2)
    
    # Analyze thresholds
    thresholds1 = analyze_thresholds(values1)
    thresholds2 = analyze_thresholds(values2)
    
    # Create comparison report
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    report_path = os.path.join(output_dir, f"distribution_analysis_{timestamp}.txt")
    plot_path = os.path.join(output_dir, f"distribution_plot_{timestamp}.png")
    
    # Generate report
    with open(report_path, "w") as f:
        f.write(f"Ground Truth Distribution Analysis\n")
        f.write(f"Generated at: {datetime.now()}\n\n")
        
        f.write(f"Dataset Sizes:\n")
        f.write(f"{label1}: {stats1['count']} samples\n")
        f.write(f"{label2}: {stats2['count']} samples\n\n")
        
        f.write("Statistical Measures:\n")
        f.write(f"{'Metric':<15} {label1:>15} {label2:>15} {'Difference':>15}\n")
        f.write("-" * 60 + "\n")
        
        # Write comparison for each metric
        metrics_to_compare = [
            ("mean", "Mean"),
            ("median", "Median"),
            ("std", "Std Dev"),
            ("skewness", "Skewness"),
            ("kurtosis", "Kurtosis"),
            ("positive_ratio", "% Positive"),
            ("negative_ratio", "% Negative"),
            ("abs_mean", "Abs Mean"),
            (">5%", "> +5%"),
            ("<-5%", "< -5%"),
            ("±1%", "Within ±1%"),
            ("±2%", "Within ±2%"),
            ("±5%", "Within ±5%")
        ]
        
        for key, label in metrics_to_compare:
            val1 = stats1[key]
            val2 = stats2[key]
            diff = val2 - val1
            f.write(f"{label:<15} {val1:>15.4f} {val2:>15.4f} {diff:>15.4f}\n")
        
        # Write threshold analysis
        f.write("\nThreshold Analysis:\n")
        f.write("\nDataset 1 ({}):\n".format(label1))
        f.write(f"{'Threshold':<10} {'Samples Within':<15} {'Samples Outside':<15} {'% Retained':<12} {'% Removed':<12}\n")
        f.write("-" * 65 + "\n")
        
        for threshold, stats in thresholds1.items():
            f.write(f"±{threshold*100:>3.1f}% {stats['samples_within']:>13d} {stats['samples_outside']:>14d} "
                   f"{stats['percent_retained']:>11.2f}% {stats['percent_removed']:>11.2f}%\n")
        
        f.write(f"\nDataset 2 ({label2}):\n")
        f.write(f"{'Threshold':<10} {'Samples Within':<15} {'Samples Outside':<15} {'% Retained':<12} {'% Removed':<12}\n")
        f.write("-" * 65 + "\n")
        
        for threshold, stats in thresholds2.items():
            f.write(f"±{threshold*100:>3.1f}% {stats['samples_within']:>13d} {stats['samples_outside']:>14d} "
                   f"{stats['percent_retained']:>11.2f}% {stats['percent_removed']:>11.2f}%\n")
        
        # Perform KS test
        ks_statistic, p_value = ks_2samp(values1, values2)
        f.write(f"\nKolmogorov-Smirnov Test:\n")
        f.write(f"KS statistic: {ks_statistic:.4f}\n")
        f.write(f"p-value: {p_value:.4f}\n")
        
        if p_value < 0.05:
            f.write("The distributions are significantly different (p < 0.05)\n")
        else:
            f.write("No significant difference between distributions (p >= 0.05)\n")
    
    # Create visualization
    plot_distributions(values1, values2, label1, label2, plot_path)
    
    print(f"Analysis complete. Results saved to {report_path}")
    print(f"Plots saved to {plot_path}")

if __name__ == "__main__":
    # Update these paths
    folder1 = "/Users/daniellavin/Desktop/proj/MoneyTrainer/X_findataset"
    folder2 = "/Users/daniellavin/Desktop/proj/MoneyTrainer/minivals"
    output_dir = "/Users/daniellavin/Desktop/proj/MoneyTrainer/plots"
    
    compare_distributions(folder1, folder2, output_dir)