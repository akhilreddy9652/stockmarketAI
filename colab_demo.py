
# 🚀 Quick Demo: Maximum Historical Data Analysis in Google Colab

# Setup
!pip install -q yfinance pandas numpy scikit-learn plotly

import pandas as pd
import numpy as np
import yfinance as yf
from datetime import datetime
import matplotlib.pyplot as plt

# Check GPU
import torch
print(f"🔥 GPU Available: {torch.cuda.is_available()}")

# Load our maximum historical data
print("📊 Loading TITAN.NS (25+ years of data)...")
titan_data = pd.read_parquet('stock_data/TITAN_NS_max_data.parquet')

print(f"✅ Loaded {len(titan_data)} records")
print(f"📅 Date Range: {titan_data['Date'].min()} to {titan_data['Date'].max()}")

# Quick analysis
years_of_data = (titan_data['Date'].max() - titan_data['Date'].min()).days / 365.25
initial_price = titan_data['Close'].iloc[0]
final_price = titan_data['Close'].iloc[-1]
total_return = ((final_price / initial_price) - 1) * 100

print(f"⏳ Years of Data: {years_of_data:.1f}")
print(f"💰 Total Return: {total_return:+.1f}%")
print(f"📈 CAGR: {(((final_price/initial_price)**(1/years_of_data))-1)*100:.1f}%")

# Plot the journey
plt.figure(figsize=(12, 6))
plt.plot(titan_data['Date'], titan_data['Close'], linewidth=2)
plt.title('TITAN.NS: 25+ Years Price Journey (2000-2025)')
plt.xlabel('Year')
plt.ylabel('Price (₹)')
plt.grid(True, alpha=0.3)
plt.show()

print("🏆 This is the power of maximum historical data in Google Colab!")
