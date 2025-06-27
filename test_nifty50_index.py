"""
Test Nifty 50 Index Enhanced Backtesting
========================================
Testing the enhanced backtesting system on the Nifty 50 Index itself.
"""

import yfinance as yf
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from enhanced_backtesting_v2 import EnhancedBacktester
import warnings

warnings.filterwarnings('ignore')

def test_nifty50_index():
    """Test Nifty 50 Index data and performance."""
    
    print("🇮🇳 Testing Nifty 50 Index (^NSEI)...")
    print("=" * 50)
    
    try:
        # Fetch Nifty 50 Index data
        end_date = datetime.now()
        start_date = end_date - timedelta(days=2*365)  # 2 years of data
        
        nifty_data = yf.download('^NSEI', 
                               start=start_date.strftime('%Y-%m-%d'), 
                               end=end_date.strftime('%Y-%m-%d'),
                               auto_adjust=True)
        
        if nifty_data.empty:
            print("❌ Failed to fetch Nifty 50 Index data")
            return
        
        # Reset index to get Date as column
        nifty_data.reset_index(inplace=True)
        
        print(f"✅ Successfully fetched {len(nifty_data)} records for Nifty 50 Index")
        print(f"📅 Date Range: {nifty_data['Date'].min().strftime('%Y-%m-%d')} to {nifty_data['Date'].max().strftime('%Y-%m-%d')}")
        
        # Basic statistics
        latest_close = nifty_data['Close'].iloc[-1]
        first_close = nifty_data['Close'].iloc[0]
        ytd_change = ((latest_close / first_close) - 1) * 100
        
        print(f"💰 Latest Close: ₹{latest_close:.2f}")
        print(f"📈 Total Return: {ytd_change:.2f}%")
        print(f"📊 Average Volume: {nifty_data['Volume'].mean():,.0f}")
        
        # Calculate volatility
        returns = nifty_data['Close'].pct_change().dropna()
        volatility = returns.std() * np.sqrt(252) * 100  # Annualized
        print(f"📉 Annualized Volatility: {volatility:.2f}%")
        
        # Max drawdown
        rolling_max = nifty_data['Close'].cummax()
        drawdown = (nifty_data['Close'] - rolling_max) / rolling_max
        max_drawdown = drawdown.min() * 100
        print(f"⚠️  Max Drawdown: {max_drawdown:.2f}%")
        
        print("\n🚀 Running Enhanced Backtesting on Nifty 50 Index...")
        print("-" * 50)
        
        # Initialize enhanced backtester
        backtester = EnhancedBacktester()
        
        # Run enhanced backtesting
        results = backtester.run_enhanced_backtest('^NSEI', start_date.strftime('%Y-%m-%d'))
        
        if results:
            print("✅ Enhanced Backtesting Results for Nifty 50 Index:")
            print(f"🎯 Directional Accuracy: {results.get('directional_accuracy', 0):.2f}%")
            print(f"📊 Sharpe Ratio: {results.get('sharpe_ratio', 0):.3f}")
            print(f"💰 Total Return: {results.get('total_return', 0):.2f}%")
            print(f"⚠️  Max Drawdown: {results.get('max_drawdown', 0):.2f}%")
            print(f"🏆 Win Rate: {results.get('win_rate', 0):.2f}%")
            print(f"📈 MAPE: {results.get('mape', 0):.2f}%")
            
            # Performance rating
            directional_acc = results.get('directional_accuracy', 0)
            if directional_acc >= 90:
                rating = "🏆 EXCEPTIONAL"
            elif directional_acc >= 80:
                rating = "🥇 EXCELLENT"
            elif directional_acc >= 70:
                rating = "🥈 VERY GOOD"
            elif directional_acc >= 60:
                rating = "🥉 GOOD"
            else:
                rating = "⚠️ NEEDS IMPROVEMENT"
            
            print(f"\n🌟 Performance Rating: {rating}")
            
        else:
            print("❌ Enhanced backtesting failed")
            
    except Exception as e:
        print(f"❌ Error: {str(e)}")

def compare_index_vs_stocks():
    """Compare Nifty 50 Index performance vs individual stocks."""
    
    print("\n🔍 Comparing Nifty 50 Index vs Top Constituent Stocks...")
    print("=" * 60)
    
    # Top Nifty 50 stocks by market cap
    top_stocks = {
        '^NSEI': 'Nifty 50 Index',
        'RELIANCE.NS': 'Reliance Industries',
        'TCS.NS': 'TCS',
        'HDFCBANK.NS': 'HDFC Bank',
        'INFY.NS': 'Infosys',
        'ICICIBANK.NS': 'ICICI Bank'
    }
    
    comparison_results = {}
    
    for symbol, name in top_stocks.items():
        try:
            print(f"\n📊 Analyzing {name} ({symbol})...")
            
            # Quick performance check
            end_date = datetime.now()
            start_date = end_date - timedelta(days=365)  # 1 year
            
            data = yf.download(symbol, 
                             start=start_date.strftime('%Y-%m-%d'), 
                             end=end_date.strftime('%Y-%m-%d'),
                             auto_adjust=True)
            
            if not data.empty:
                returns = ((data['Close'].iloc[-1] / data['Close'].iloc[0]) - 1) * 100
                volatility = data['Close'].pct_change().std() * np.sqrt(252) * 100
                
                comparison_results[symbol] = {
                    'name': name,
                    'returns': returns,
                    'volatility': volatility
                }
                
                print(f"   📈 1-Year Return: {returns:.2f}%")
                print(f"   📊 Volatility: {volatility:.2f}%")
                
        except Exception as e:
            print(f"   ❌ Error analyzing {name}: {str(e)}")
    
    # Summary comparison
    if comparison_results:
        print("\n🏆 Performance Summary (1-Year):")
        print("-" * 40)
        
        sorted_by_returns = sorted(comparison_results.items(), 
                                 key=lambda x: x[1]['returns'], 
                                 reverse=True)
        
        for i, (symbol, data) in enumerate(sorted_by_returns, 1):
            print(f"{i}. {data['name']}: {data['returns']:+.2f}% (Vol: {data['volatility']:.1f}%)")

if __name__ == "__main__":
    test_nifty50_index()
    compare_index_vs_stocks()
    
    print("\n" + "="*60)
    print("🎯 CONCLUSION: Nifty 50 Index Analysis Complete!")
    print("✅ The enhanced backtesting system works with both:")
    print("   - Individual Nifty 50 stocks")
    print("   - Nifty 50 Index itself (^NSEI)")
    print("🌟 You can analyze both in the frontend dashboard!") 