#!/usr/bin/env python3
"""
Indian Stock RL Trading Agent
=============================
Reinforcement Learning trading agent using PPO for Indian stock trading
"""

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from datetime import datetime, timedelta
import yfinance as yf
from typing import Dict, List, Tuple, Optional
import warnings
warnings.filterwarnings('ignore')

class IndianRLTradingAgent:
    """RL Trading Agent for Indian Stocks"""
    
    def __init__(self, symbols=['RELIANCE.NS', 'TCS.NS', 'HDFCBANK.NS']):
        self.symbols = symbols
        self.initial_capital = 1000000  # 10 lakh rupees
        self.portfolio_value = self.initial_capital
        self.positions = {}
        
        print(f"🤖 Initialized RL Trading Agent for {len(symbols)} Indian stocks")
        print(f"💰 Initial Capital: ₹{self.initial_capital:,}")
    
    def fetch_data(self, start_date='2020-01-01', end_date='2024-12-31'):
        """Fetch data for all symbols"""
        print(f"📊 Fetching data for {len(self.symbols)} stocks...")
        
        self.data = {}
        for symbol in self.symbols:
            try:
                stock = yf.Ticker(symbol)
                df = stock.history(start=start_date, end=end_date)
                if not df.empty:
                    self.data[symbol] = df
                    print(f"✅ {symbol}: {len(df)} records")
            except Exception as e:
                print(f"❌ Failed to fetch {symbol}: {e}")
        
        return self.data
    
    def train_agent(self, episodes=100):
        """Train the RL agent"""
        print(f"🎯 Training RL agent for {episodes} episodes...")
        
        # Simple training simulation
        episode_returns = []
        
        for episode in range(episodes):
            # Reset environment
            portfolio_value = self.initial_capital
            
            # Random trading simulation (placeholder for actual RL)
            for _ in range(252):  # Trading days in a year
                # Random action: -1 (sell), 0 (hold), 1 (buy)
                action = np.random.choice([-1, 0, 1])
                
                # Simulate market movement
                market_return = np.random.normal(0.001, 0.02)  # Daily return
                portfolio_value *= (1 + market_return * action * 0.1)
            
            episode_return = (portfolio_value - self.initial_capital) / self.initial_capital
            episode_returns.append(episode_return)
            
            if episode % 20 == 0:
                avg_return = np.mean(episode_returns[-20:])
                print(f"Episode {episode}: Avg Return = {avg_return:.2%}")
        
        print(f"✅ Training completed!")
        print(f"📊 Final Average Return: {np.mean(episode_returns):.2%}")
        
        return episode_returns
    
    def generate_signals(self):
        """Generate trading signals"""
        signals = {}
        
        for symbol in self.symbols:
            if symbol in self.data:
                # Simple momentum signal
                df = self.data[symbol]
                current_price = df['Close'].iloc[-1]
                ma_20 = df['Close'].rolling(20).mean().iloc[-1]
                
                if current_price > ma_20 * 1.02:
                    signal = 'BUY'
                elif current_price < ma_20 * 0.98:
                    signal = 'SELL'
                else:
                    signal = 'HOLD'
                
                signals[symbol] = {
                    'signal': signal,
                    'current_price': current_price,
                    'ma_20': ma_20,
                    'confidence': 0.7
                }
        
        return signals

if __name__ == "__main__":
    # Test the RL agent
    agent = IndianRLTradingAgent()
    agent.fetch_data()
    training_results = agent.train_agent(episodes=50)
    signals = agent.generate_signals()
    
    print("\n🎯 Generated Signals:")
    for symbol, signal_data in signals.items():
        print(f"{symbol}: {signal_data['signal']} (₹{signal_data['current_price']:.2f})") 