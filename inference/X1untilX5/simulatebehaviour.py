
import numpy as np
from dataclasses import dataclass
from typing import List, Dict
import random
from datetime import datetime, timedelta

@dataclass
class Category:
    name: str
    precision: float
    l1_distance: float
    avg_rmse: float
    std_error: float
    avg_stocks: float
    avg_spread: float = 0.001  # Average spread as percentage

@dataclass
class TradeResult:
    profit_loss: float
    win_rate: float
    max_drawdown: float
    total_trades: int
    successful_trades: int
    failed_trades: int
    roi_percent: float
    sharpe_ratio: float
    avg_trade_profit: float
    worst_trade: float
    best_trade: float

class TradingSimulator:
    def __init__(self, 
                 initial_capital: float = 400.0,
                 transaction_cost: float = 1.50,
                 trades_per_month: int = 20,
                 stop_loss_percent: float = 5.0,
                 take_profit_percent: float = 7.0,
                 avg_slippage_percent: float = 0.15):  # 0.15% average slippage
        self.initial_capital = initial_capital
        self.transaction_cost = transaction_cost
        self.trades_per_month = trades_per_month
        self.stop_loss_percent = stop_loss_percent
        self.take_profit_percent = take_profit_percent
        self.avg_slippage_percent = avg_slippage_percent
        
        # Initialize categories with more conservative estimations
        self.categories = {
            'market_cap': {
                'Q3_Large_Cap': Category('Q3 Large Cap', 0.6323, 0.0481, 0.0616, 0.0384, 142.5, 0.0008),
                'Q4_Mega_Cap': Category('Q4 Mega Cap', 0.6189, 0.0496, 0.0719, 0.0507, 143.2, 0.0005),
                'Q2_Mid_Cap': Category('Q2 Mid Cap', 0.5944, 0.0499, 0.0638, 0.0397, 142.7, 0.0012),
                'Q1_Small_Cap': Category('Q1 Small Cap', 0.5803, 0.0808, 0.3507, 0.3285, 143.5, 0.0020)
            },
            'volume': {
                'Q1_Low': Category('Q1 Low Volume', 0.5859, 0.0435, 0.0567, 0.0363, 143.5, 0.0020),
                'Q3_Med_High': Category('Q3 Medium-High', 0.5941, 0.0548, 0.0740, 0.0493, 142.8, 0.0012),
                'Q4_High': Category('Q4 High Volume', 0.6355, 0.0597, 0.0760, 0.0470, 143.3, 0.0008),
                'Q2_Med_Low': Category('Q2 Medium-Low', 0.6055, 0.0703, 0.3423, 0.3251, 142.8, 0.0015)
            },
            'sector': {
                'Industrials': Category('Industrials', 0.5554, 0.0444, 0.0582, 0.0376, 59.3, 0.0012),
                'Consumer_Staples': Category('Consumer Staples', 0.5290, 0.0452, 0.0565, 0.0342, 23.7, 0.0010),
                'Real_Estate': Category('Real Estate', 0.6472, 0.0468, 0.0561, 0.0313, 21.5, 0.0015),
                'Health_Care': Category('Health Care', 0.4900, 0.0472, 0.0597, 0.0367, 55.2, 0.0010),
                'Consumer_Discretionary': Category('Consumer Discretionary', 0.6003, 0.0477, 0.0617, 0.0392, 155.7, 0.0012),
                'Basic_Materials': Category('Basic Materials', 0.7667*0.85, 0.0570, 0.0656, 0.0333, 15.7, 0.0018),  # Reduced precision due to small sample
                'Energy': Category('Energy', 0.6932, 0.0589, 0.0722, 0.0415, 30.2, 0.0015),
                'Technology': Category('Technology', 0.6787, 0.0589, 0.0764, 0.0488, 106.7, 0.0010),
                'Utilities': Category('Utilities', 0.5819, 0.0577, 0.0898, 0.0683, 27.7, 0.0012),
                'Finance': Category('Finance', 0.6380, 0.0998, 0.4632, 0.4391, 64.2, 0.0010)
            }
        }

    def calculate_total_costs(self, position_size: float, category: Category) -> float:
        """Calculate total costs including spread and slippage."""
        transaction_costs = self.transaction_cost * 2  # Buy and sell
        spread_cost = position_size * category.avg_spread
        slippage_cost = position_size * (self.avg_slippage_percent / 100)
        return transaction_costs + spread_cost + slippage_cost

    def simulate_trade(self, category: Category, position_size: float) -> float:
        """Simulate a single trade with realistic market friction."""
        # Calculate total costs
        total_costs = self.calculate_total_costs(position_size, category)
        
        # Add random market noise
        market_noise = np.random.normal(0, 0.001)  # 0.1% standard deviation noise
        
        # Determine if prediction is correct based on precision
        is_correct_prediction = random.random() < category.precision
        
        # Calculate base return
        if is_correct_prediction:
            # Generate positive return with more conservative estimation
            return_multiplier = abs(np.random.normal(category.l1_distance * 0.8, category.std_error))
            if return_multiplier > self.take_profit_percent / 100:
                return_multiplier = self.take_profit_percent / 100
        else:
            # Generate negative return
            return_multiplier = -abs(np.random.normal(category.l1_distance * 1.2, category.std_error))
            if return_multiplier < -self.stop_loss_percent / 100:
                return_multiplier = -self.stop_loss_percent / 100
        
        # Apply market noise
        return_multiplier += market_noise
        
        # Calculate final profit/loss
        gross_profit = position_size * return_multiplier
        return gross_profit - total_costs

    def run_simulation(self, category: Category, num_months: int = 12) -> TradeResult:
        """Run a complete simulation for a given category."""
        capital = self.initial_capital
        trades_history = []
        trade_returns = []
        successful_trades = 0
        failed_trades = 0
        
        total_trades = self.trades_per_month * num_months
        monthly_returns = []
        
        for _ in range(total_trades):
            if capital <= 0:
                break
                
            # More conservative position sizing
            position_size = min(capital * 0.8, capital - 10)  # Keep some buffer
            
            # Simulate trade
            trade_result = self.simulate_trade(category, position_size)
            capital += trade_result
            trades_history.append(capital)
            trade_returns.append(trade_result)
            
            if trade_result > 0:
                successful_trades += 1
            else:
                failed_trades += 1
            
            if len(trades_history) % self.trades_per_month == 0:
                monthly_return = (capital / trades_history[-self.trades_per_month] - 1)
                monthly_returns.append(monthly_return)
        
        # Calculate metrics
        final_profit = capital - self.initial_capital
        win_rate = successful_trades / total_trades if total_trades > 0 else 0
        max_drawdown = self._calculate_max_drawdown(trades_history)
        roi_percent = (final_profit / self.initial_capital) * 100
        sharpe_ratio = self._calculate_sharpe_ratio(monthly_returns)
        
        return TradeResult(
            profit_loss=final_profit,
            win_rate=win_rate,
            max_drawdown=max_drawdown,
            total_trades=total_trades,
            successful_trades=successful_trades,
            failed_trades=failed_trades,
            roi_percent=roi_percent,
            sharpe_ratio=sharpe_ratio,
            avg_trade_profit=np.mean(trade_returns) if trade_returns else 0,
            worst_trade=min(trade_returns) if trade_returns else 0,
            best_trade=max(trade_returns) if trade_returns else 0
        )

    def _calculate_max_drawdown(self, trades_history: List[float]) -> float:
        """Calculate maximum drawdown from trades history."""
        peak = trades_history[0]
        max_drawdown = 0
        
        for value in trades_history:
            if value > peak:
                peak = value
            drawdown = (peak - value) / peak
            max_drawdown = max(max_drawdown, drawdown)
            
        return max_drawdown

    def _calculate_sharpe_ratio(self, returns: List[float], risk_free_rate: float = 0.02) -> float:
        """Calculate Sharpe ratio from monthly returns."""
        if not returns:
            return 0
        
        returns_array = np.array(returns)
        excess_returns = returns_array - (risk_free_rate / 12)
        if len(returns_array) <= 1:
            return 0
        
        return np.mean(excess_returns) / np.std(excess_returns, ddof=1) * np.sqrt(12)

    def run_all_simulations(self, num_simulations: int = 100) -> Dict:
        """Run multiple simulations for all categories and average results."""
        results = {}
        
        for category_type, categories in self.categories.items():
            results[category_type] = {}
            print(f"\n=== {category_type.upper()} CATEGORY SIMULATIONS ===")
            
            for name, category in categories.items():
                # Run multiple simulations and average results
                sim_results = []
                for _ in range(num_simulations):
                    sim_results.append(self.run_simulation(category))
                
                # Average results
                avg_result = TradeResult(
                    profit_loss=np.mean([r.profit_loss for r in sim_results]),
                    win_rate=np.mean([r.win_rate for r in sim_results]),
                    max_drawdown=np.mean([r.max_drawdown for r in sim_results]),
                    total_trades=sim_results[0].total_trades,
                    successful_trades=int(np.mean([r.successful_trades for r in sim_results])),
                    failed_trades=int(np.mean([r.failed_trades for r in sim_results])),
                    roi_percent=np.mean([r.roi_percent for r in sim_results]),
                    sharpe_ratio=np.mean([r.sharpe_ratio for r in sim_results]),
                    avg_trade_profit=np.mean([r.avg_trade_profit for r in sim_results]),
                    worst_trade=min([r.worst_trade for r in sim_results]),
                    best_trade=max([r.best_trade for r in sim_results])
                )
                
                results[category_type][name] = avg_result
                
                print(f"\n{category.name}:")
                print(f"Final P/L: €{avg_result.profit_loss:.2f}")
                print(f"ROI: {avg_result.roi_percent:.2f}%")
                print(f"Win Rate: {avg_result.win_rate*100:.2f}%")
                print(f"Max Drawdown: {avg_result.max_drawdown*100:.2f}%")
                print(f"Sharpe Ratio: {avg_result.sharpe_ratio:.2f}")
                print(f"Avg Trade Profit: €{avg_result.avg_trade_profit:.2f}")
                print(f"Worst Trade: €{avg_result.worst_trade:.2f}")
                print(f"Best Trade: €{avg_result.best_trade:.2f}")
                print(f"Total Trades: {avg_result.total_trades}")
                print(f"Successful/Failed: {avg_result.successful_trades}/{avg_result.failed_trades}")
        
        return results

def main():
    # Initialize simulator with conservative parameters
    simulator = TradingSimulator(
        initial_capital=400.0,
        transaction_cost=1.50,
        trades_per_month=20,
        stop_loss_percent=5.0,
        take_profit_percent=7.0,
        avg_slippage_percent=0.15
    )
    
    # Run multiple simulations and average results
    results = simulator.run_all_simulations(num_simulations=100)
    
    # Print best performing category from each type
    print("\n=== BEST PERFORMERS (Averaged over 100 simulations) ===")
    for category_type, categories in results.items():
        best_category = max(categories.items(), key=lambda x: x[1].sharpe_ratio)
        print(f"\nBest {category_type}: {best_category[0]}")
        print(f"ROI: {best_category[1].roi_percent:.2f}%")
        print(f"Sharpe Ratio: {best_category[1].sharpe_ratio:.2f}")
        print(f"Max Drawdown: {best_category[1].max_drawdown*100:.2f}%")
        print(f"Win Rate: {best_category[1].win_rate*100:.2f}%")

if __name__ == "__main__":
    main()
