import torch
import numpy as np
from typing import Tuple, List, Dict
import json
from datetime import datetime
from dataclasses import dataclass
import logging
from pathlib import Path
from tqdm import tqdm

@dataclass
class AnalysisConfig:
    # GT regime configurations
    gt_min: float = -0.8
    gt_max: float = 0.8
    initial_regime_splits: int = 3
    max_regime_splits: int = 12 
    
    # Feature analysis configurations
    stock_threshold: float = 1    
    topic_threshold: float = 0.2
    
    # Analysis parameters
    min_samples_per_regime: int = 50 
    correlation_threshold: float = 0.4 
    precision_threshold: float = 0.7   

class TensorAnalyzer:
    def __init__(self, config: AnalysisConfig, report_path: str):
        self.config = config
        self.report_path = Path(report_path)
        self.report_path.mkdir(parents=True, exist_ok=True)
        self.logger = self._setup_logger()
        
    def _setup_logger(self) -> logging.Logger:
        logger = logging.getLogger('TensorAnalyzer')
        logger.setLevel(logging.INFO)
        fh = logging.FileHandler(self.report_path / 'analysis_log.txt')
        fh.setFormatter(logging.Formatter('%(asctime)s - %(message)s'))
        logger.addHandler(fh)
        return logger

    def load_tensor_directory(self, dir_path: str) -> torch.Tensor:
        """Load all tensors from a directory and stack them"""
        dir_path = Path(dir_path)
        all_tensors = []
        
        self.logger.info(f"Loading tensors from {dir_path}")
        
        # Get list of all .pt files
        tensor_files = list(dir_path.glob('*.pt'))
        
        for file_path in tqdm(tensor_files, desc="Loading tensors"):
            try:
                tensor = torch.load(file_path)
                all_tensors.append(tensor)
            except Exception as e:
                self.logger.warning(f"Error loading {file_path}: {str(e)}")
                continue
        
        if not all_tensors:
            raise ValueError(f"No valid tensors found in {dir_path}")
            
        # Stack all tensors
        return torch.stack(all_tensors)

    def analyze_regime(self, data: torch.Tensor, gt_min: float, gt_max: float) -> Dict:
        """Analyze a specific GT regime for patterns"""
        # Extract tensors within GT range
        mask = (data[:, 0] >= gt_min) & (data[:, 0] <= gt_max)
        regime_data = data[mask]
        
        if len(regime_data) < self.config.min_samples_per_regime:
            return None
        
        self.logger.info(f"Analyzing regime {gt_min:.3f} to {gt_max:.3f} with {len(regime_data)} samples")
        
        results = {
            'gt_range': (gt_min, gt_max),
            'n_samples': len(regime_data),
            'stock_patterns': self._analyze_stock_patterns(regime_data),
            'topic_patterns': self._analyze_topic_patterns(regime_data),
            'combined_patterns': self._analyze_combined_patterns(regime_data)
        }
        
        return results

    def _analyze_stock_patterns(self, data: torch.Tensor) -> List[Dict]:
        """Analyze stock data for patterns using statistical measures"""
        patterns = []
        stock_data = data[:, 1:81]  # 80 stock features
        gt = data[:, 0]
        
        # Look for strong deviations in stock behavior
        mean_values = torch.mean(stock_data, dim=0)
        std_values = torch.std(stock_data, dim=0)
        
        for i in range(80):
            if std_values[i] > self.config.stock_threshold:
                correlation = torch.corrcoef(torch.stack([stock_data[:, i], gt]))[0, 1]
                if abs(correlation) > self.config.correlation_threshold:
                    patterns.append({
                        'feature_idx': i,
                        'correlation': float(correlation),
                        'mean': float(mean_values[i]),
                        'std': float(std_values[i])
                    })
        
        return patterns

    def _analyze_topic_patterns(self, data: torch.Tensor) -> List[Dict]:
        """Analyze BERTopic probabilities for patterns"""
        patterns = []
        topic_data = data[:, 81:]  # 100 topic features
        gt = data[:, 0]
        
        # Look for topics with strong probabilities
        mean_probs = torch.mean(topic_data, dim=0)
        
        for i in range(100):
            if mean_probs[i] > self.config.topic_threshold:
                correlation = torch.corrcoef(torch.stack([topic_data[:, i], gt]))[0, 1]
                if abs(correlation) > self.config.correlation_threshold:
                    patterns.append({
                        'topic_idx': i,
                        'correlation': float(correlation),
                        'mean_prob': float(mean_probs[i])
                    })
        
        return patterns

    def _analyze_combined_patterns(self, data: torch.Tensor) -> List[Dict]:
        """Analyze combinations of strong stock and topic patterns"""
        stock_patterns = self._analyze_stock_patterns(data)
        topic_patterns = self._analyze_topic_patterns(data)
        
        combined_patterns = []
        
        # Only look at combinations of features that showed individual strength
        for stock in stock_patterns:
            for topic in topic_patterns:
                stock_idx = stock['feature_idx']
                topic_idx = topic['topic_idx']
                
                # Create combined feature
                combined_signal = data[:, stock_idx + 1] * data[:, topic_idx + 81]
                correlation = torch.corrcoef(torch.stack([combined_signal, data[:, 0]]))[0, 1]
                
                if abs(correlation) > max(abs(stock['correlation']), abs(topic['correlation'])):
                    combined_patterns.append({
                        'stock_idx': stock_idx,
                        'topic_idx': topic_idx,
                        'correlation': float(correlation)
                    })
        
        return combined_patterns

    def validate_patterns(self, train_results: Dict, val_data: torch.Tensor) -> Dict:
        """Validate discovered patterns on validation data"""
        validation_results = {}
        
        for gt_range, patterns in train_results.items():
            if not patterns:
                continue
                
            # Extract validation data in same GT range
            gt_min, gt_max = eval(gt_range)
            mask = (val_data[:, 0] >= gt_min) & (val_data[:, 0] <= gt_max)
            regime_val_data = val_data[mask]
            
            if len(regime_val_data) < self.config.min_samples_per_regime:
                continue
                
            validation_results[gt_range] = self._validate_regime_patterns(
                patterns, regime_val_data)
            
        return validation_results

    def _validate_regime_patterns(self, patterns: Dict, val_data: torch.Tensor) -> Dict:
        """Validate patterns for a specific regime"""
        validated = {
            'stock_patterns': [],
            'topic_patterns': [],
            'combined_patterns': []
        }
        
        # Validate each pattern type
        for pattern_type in validated.keys():
            for pattern in patterns[pattern_type]:
                validation_metrics = self._compute_validation_metrics(
                    pattern, pattern_type, val_data)
                
                if validation_metrics['precision'] > self.config.precision_threshold:
                    validated[pattern_type].append({
                        **pattern,
                        'validation': validation_metrics
                    })
        
        return validated

    def _compute_validation_metrics(self, pattern: Dict, pattern_type: str, 
                                 val_data: torch.Tensor) -> Dict:
        """Compute validation metrics for a pattern"""
        if pattern_type == 'stock_patterns':
            signal = val_data[:, pattern['feature_idx'] + 1]
        elif pattern_type == 'topic_patterns':
            signal = val_data[:, pattern['topic_idx'] + 81]
        else:  # combined_patterns
            signal = (val_data[:, pattern['stock_idx'] + 1] * 
                     val_data[:, pattern['topic_idx'] + 81])
        
        # Compute validation correlation
        val_correlation = torch.corrcoef(torch.stack([signal, val_data[:, 0]]))[0, 1]
        
        # Compute precision (how often signal predicts correct direction)
        pred_direction = torch.sign(signal)
        true_direction = torch.sign(val_data[:, 0])
        precision = torch.mean((pred_direction == true_direction).float())
        
        return {
            'correlation': float(val_correlation),
            'precision': float(precision)
        }

    def run_analysis(self, train_dir: str, val_dir: str):
        """Run complete analysis pipeline"""
        # Load data from directories
        self.logger.info("Starting analysis...")
        train_data = self.load_tensor_directory(train_dir)
        val_data = self.load_tensor_directory(val_dir)
        
        results = {}
        
        # Analyze different regime granularities
        for n_splits in range(self.config.initial_regime_splits, 
                            self.config.max_regime_splits + 1):
            
            self.logger.info(f"Analyzing with {n_splits} splits")
            gt_ranges = np.linspace(self.config.gt_min, self.config.gt_max, n_splits + 1)
            
            for i in range(len(gt_ranges) - 1):
                gt_min, gt_max = gt_ranges[i], gt_ranges[i + 1]
                regime_results = self.analyze_regime(train_data, gt_min, gt_max)
                
                if regime_results:
                    results[f"({gt_min:.3f}, {gt_max:.3f})"] = regime_results
        
        # Validate patterns
        self.logger.info("Validating patterns...")
        validation_results = self.validate_patterns(results, val_data)
        
        # Write report
        self._write_report(results, validation_results)
        self.logger.info("Analysis complete!")

    def _write_report(self, results: Dict, validation_results: Dict):
        """Write analysis results to report file"""
        report_time = datetime.now().strftime("%Y%m%d_%H%M%S")
        report_file = self.report_path / f'analysis_report_{report_time}.txt'
        
        with open(report_file, 'w') as f:
            f.write("=== Tensor Analysis Report ===\n\n")
            
            # Write best findings for each regime
            for gt_range, patterns in validation_results.items():
                if not any(patterns.values()):
                    continue
                    
                f.write(f"\nRegime {gt_range}:\n")
                
                # Find best pattern across all types
                best_pattern = self._find_best_pattern(patterns)
                if best_pattern:
                    f.write(f"Best Pattern:\n{json.dumps(best_pattern, indent=2)}\n")

    def _find_best_pattern(self, patterns: Dict) -> Dict:
        """Find pattern with highest validated correlation"""
        all_patterns = []
        for pattern_type, type_patterns in patterns.items():
            all_patterns.extend(type_patterns)
            
        if not all_patterns:
            return None
            
        return max(all_patterns, 
                  key=lambda x: abs(x['validation']['correlation']))

def main():
    # Setup paths
    train_dir = "/Users/daniellavin/Desktop/proj/MoneyTrainer/Y/y_100_train"
    val_dir = "/Users/daniellavin/Desktop/proj/MoneyTrainer/Y/y_100_val"
    report_path = "/Users/daniellavin/Desktop/proj/MoneyTrainer/Y/datafitting_exp/reports"
    
    # Create config
    config = AnalysisConfig()
    
    # Initialize and run analyzer
    analyzer = TensorAnalyzer(config, report_path)
    analyzer.run_analysis(train_dir, val_dir)

if __name__ == "__main__":
    main()