#!/usr/bin/env python3
"""
Advanced Portfolio Optimizer for Indian Stocks
==============================================
"""

import numpy as np
import pandas as pd
from datetime import datetime, timedelta
import yfinance as yf
from scipy import optimize
import warnings
warnings.filterwarnings('ignore')

class AdvancedPortfolioOptimizer:
    """Advanced Portfolio Optimization for Indian Stocks"""
    
    def __init__(self, symbols, risk_free_rate=0.06):
        self.symbols = symbols
        self.risk_free_rate = risk_free_rate
        self.returns_data = None
        
        print(f"🎯 Portfolio Optimizer initialized for {len(symbols)} Indian stocks")
        print(f"💹 Risk-free rate: {risk_free_rate:.2%}")
    
    def load_data(self, start_date='2020-01-01', end_date='2024-12-31'):
        """Load historical data and calculate returns"""
        print(f"📊 Loading data for portfolio optimization...")
        
        self.price_data = {}
        self.returns_data = {}
        
        for symbol in self.symbols:
            try:
                stock = yf.Ticker(symbol)
                df = stock.history(start=start_date, end=end_date)
                if not df.empty and len(df) > 252:
                    self.price_data[symbol] = df
                    returns = df['Close'].pct_change().dropna()
                    self.returns_data[symbol] = returns
                    print(f"✅ {symbol}: {len(returns)} return observations")
            except Exception as e:
                print(f"❌ Failed to load {symbol}: {e}")
        
        if self.returns_data:
            self.returns_matrix = pd.DataFrame(self.returns_data)
            self.returns_matrix = self.returns_matrix.dropna()
            
            # Calculate statistics
            self.mean_returns = self.returns_matrix.mean() * 252  # Annualized
            self.covariance_matrix = self.returns_matrix.cov() * 252  # Annualized
            
            print(f"✅ Portfolio data loaded: {len(self.returns_matrix)} observations")
        
        return self.returns_matrix
    
    def calculate_portfolio_metrics(self, weights):
        """Calculate portfolio return, risk, and Sharpe ratio"""
        portfolio_return = np.sum(weights * self.mean_returns)
        portfolio_variance = np.dot(weights.T, np.dot(self.covariance_matrix, weights))
        portfolio_std = np.sqrt(portfolio_variance)
        sharpe_ratio = (portfolio_return - self.risk_free_rate) / portfolio_std
        
        return portfolio_return, portfolio_std, sharpe_ratio
    
    def optimize_maximum_sharpe(self):
        """Optimize for maximum Sharpe ratio"""
        print("🎯 Optimizing for Maximum Sharpe Ratio...")
        
        n_assets = len(self.symbols)
        
        def negative_sharpe_objective(weights):
            _, _, sharpe = self.calculate_portfolio_metrics(weights)
            return -sharpe
        
        constraints = [
            {'type': 'eq', 'fun': lambda x: np.sum(x) - 1},
        ]
        
        bounds = tuple((0, 1) for _ in range(n_assets))
        initial_weights = np.array([1/n_assets] * n_assets)
        
        result = optimize.minimize(
            negative_sharpe_objective,
            initial_weights,
            method='SLSQP',
            bounds=bounds,
            constraints=constraints
        )
        
        if result.success:
            optimal_weights = result.x
            portfolio_return, portfolio_risk, sharpe_ratio = self.calculate_portfolio_metrics(optimal_weights)
            
            print(f"✅ Optimization completed")
            print(f"📈 Expected Return: {portfolio_return:.2%}")
            print(f"📊 Portfolio Risk: {portfolio_risk:.2%}")
            print(f"🎯 Sharpe Ratio: {sharpe_ratio:.3f}")
            
            return {
                'weights': optimal_weights,
                'return': portfolio_return,
                'risk': portfolio_risk,
                'sharpe_ratio': sharpe_ratio,
                'symbols': self.symbols
            }
        else:
            print(f"❌ Optimization failed: {result.message}")
            return None
    
    def optimize_equal_weight(self):
        """Equal weight baseline portfolio"""
        n_assets = len(self.symbols)
        weights = np.array([1/n_assets] * n_assets)
        portfolio_return, portfolio_risk, sharpe_ratio = self.calculate_portfolio_metrics(weights)
        
        print(f"📊 Equal Weight Portfolio:")
        print(f"📈 Expected Return: {portfolio_return:.2%}")
        print(f"📊 Portfolio Risk: {portfolio_risk:.2%}")
        print(f"🎯 Sharpe Ratio: {sharpe_ratio:.3f}")
        
        return {
            'weights': weights,
            'return': portfolio_return,
            'risk': portfolio_risk,
            'sharpe_ratio': sharpe_ratio,
            'symbols': self.symbols
        }
    
    def generate_allocation_report(self, portfolio):
        """Generate detailed allocation report"""
        if not portfolio:
            return
        
        print(f"\n📋 Portfolio Allocation Report:")
        print(f"={'='*60}")
        
        allocation_df = pd.DataFrame({
            'Symbol': portfolio['symbols'],
            'Weight': portfolio['weights'],
            'Weight_Pct': portfolio['weights'] * 100
        }).sort_values('Weight', ascending=False)
        
        for _, row in allocation_df.iterrows():
            print(f"{row['Symbol']:<12} | {row['Weight_Pct']:>6.2f}% | ₹{row['Weight']*1000000:>8,.0f}")
        
        print(f"{'='*60}")
        print(f"Expected Annual Return: {portfolio['return']:.2%}")
        print(f"Expected Annual Risk:   {portfolio['risk']:.2%}")
        print(f"Sharpe Ratio:          {portfolio['sharpe_ratio']:.3f}")
        
        return allocation_df

if __name__ == "__main__":
    # Test with top Indian stocks
    indian_stocks = [
        'RELIANCE.NS', 'TCS.NS', 'HDFCBANK.NS', 'INFY.NS', 'ICICIBANK.NS',
        'HINDUNILVR.NS', 'ITC.NS', 'SBIN.NS', 'BHARTIARTL.NS', 'KOTAKBANK.NS'
    ]
    
    optimizer = AdvancedPortfolioOptimizer(indian_stocks)
    
    # Load data
    returns_data = optimizer.load_data()
    
    if not returns_data.empty:
        # Compare strategies
        equal_weight = optimizer.optimize_equal_weight()
        max_sharpe = optimizer.optimize_maximum_sharpe()
        
        if max_sharpe:
            optimizer.generate_allocation_report(max_sharpe)
        
        print("\n🎉 Portfolio optimization completed!")
    else:
        print("❌ No data available for optimization") 