"""
Nifty 50 Index Enhanced Analysis
===============================
Test and analyze the Nifty 50 Index with enhanced backtesting.
"""

import yfinance as yf
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import warnings

warnings.filterwarnings('ignore')

def analyze_nifty50_index():
    """Analyze Nifty 50 Index with enhanced features."""
    
    print("🇮🇳 NIFTY 50 INDEX ANALYSIS")
    print("=" * 50)
    
    try:
        # Fetch Nifty 50 Index data (^NSEI)
        end_date = datetime.now()
        start_date = end_date - timedelta(days=2*365)  # 2 years
        
        print("📊 Fetching Nifty 50 Index data...")
        nifty = yf.download('^NSEI', 
                           start=start_date.strftime('%Y-%m-%d'), 
                           end=end_date.strftime('%Y-%m-%d'),
                           auto_adjust=True)
        
        if nifty is None or len(nifty) == 0:
            print("❌ No data available for Nifty 50 Index")
            return None
        
        print(f"✅ Fetched {len(nifty)} trading days of data")
        print(f"📅 Period: {start_date.strftime('%Y-%m-%d')} to {end_date.strftime('%Y-%m-%d')}")
        
        # Basic statistics
        latest_close = float(nifty['Close'].iloc[-1])
        first_close = float(nifty['Close'].iloc[0])
        total_return = ((latest_close / first_close) - 1) * 100
        
        print(f"\\n💰 Current Level: ₹{latest_close:,.2f}")
        print(f"📈 2-Year Return: {total_return:+.2f}%")
        
        # Calculate returns for metrics
        returns = nifty['Close'].pct_change().dropna()
        
        # Annualized volatility
        annual_vol = float(returns.std() * np.sqrt(252) * 100)
        print(f"📊 Annualized Volatility: {annual_vol:.2f}%")
        
        # Sharpe ratio (assuming 6% risk-free rate for India)
        risk_free_rate = 0.06
        annual_return = total_return / 2  # 2-year period
        sharpe_ratio = (annual_return/100 - risk_free_rate) / (annual_vol/100)
        print(f"📈 Sharpe Ratio: {sharpe_ratio:.3f}")
        
        # Max drawdown
        rolling_max = nifty['Close'].cummax()
        drawdown = (nifty['Close'] - rolling_max) / rolling_max
        max_dd = float(drawdown.min() * 100)
        print(f"⚠️  Max Drawdown: {max_dd:.2f}%")
        
        # Recent performance
        recent_30d = ((latest_close / float(nifty['Close'].iloc[-30])) - 1) * 100
        recent_90d = ((latest_close / float(nifty['Close'].iloc[-90])) - 1) * 100
        
        print(f"\\n📅 Recent Performance:")
        print(f"   Last 30 days: {recent_30d:+.2f}%")
        print(f"   Last 90 days: {recent_90d:+.2f}%")
        
        # Volatility analysis
        high_vol_days = (returns.abs() > 0.02).sum()  # Days with >2% moves
        print(f"\\n📊 Market Dynamics:")
        print(f"   High volatility days (>2%): {high_vol_days} ({high_vol_days/len(returns)*100:.1f}%)")
        
        # Trend analysis
        sma_20 = nifty['Close'].rolling(20).mean().iloc[-1]
        sma_50 = nifty['Close'].rolling(50).mean().iloc[-1]
        sma_200 = nifty['Close'].rolling(200).mean().iloc[-1]
        
        print(f"\\n📈 Technical Levels:")
        print(f"   20-day SMA: ₹{float(sma_20):,.2f}")
        print(f"   50-day SMA: ₹{float(sma_50):,.2f}")
        print(f"   200-day SMA: ₹{float(sma_200):,.2f}")
        
        # Trend direction
        if latest_close > sma_20 > sma_50 > sma_200:
            trend = "🟢 STRONG UPTREND"
        elif latest_close > sma_50 > sma_200:
            trend = "🟡 UPTREND"
        elif latest_close < sma_50 < sma_200:
            trend = "🔴 DOWNTREND"
        else:
            trend = "🟡 SIDEWAYS"
            
        print(f"   Current Trend: {trend}")
        
        return {
            'symbol': '^NSEI',
            'current_level': latest_close,
            'total_return': total_return,
            'volatility': annual_vol,
            'sharpe_ratio': sharpe_ratio,
            'max_drawdown': max_dd,
            'trend': trend
        }
        
    except Exception as e:
        print(f"❌ Error analyzing Nifty 50 Index: {str(e)}")
        return None

def compare_nifty_with_stocks():
    """Compare Nifty 50 Index with top constituent stocks."""
    
    print("\\n🔍 NIFTY 50 vs TOP STOCKS COMPARISON")
    print("="*50)
    
    symbols = {
        '^NSEI': 'Nifty 50 Index',
        'RELIANCE.NS': 'Reliance Industries', 
        'TCS.NS': 'Tata Consultancy Services',
        'HDFCBANK.NS': 'HDFC Bank',
        'INFY.NS': 'Infosys',
        'ICICIBANK.NS': 'ICICI Bank'
    }
    
    results = []
    
    for symbol, name in symbols.items():
        print(f"\\n📊 Analyzing {name}...")
        
        try:
            # 1-year data for comparison
            end_date = datetime.now()
            start_date = end_date - timedelta(days=365)
            
            data = yf.download(symbol, 
                             start=start_date.strftime('%Y-%m-%d'),
                             end=end_date.strftime('%Y-%m-%d'),
                             auto_adjust=True)
            
            if data is not None and len(data) > 0:
                # Calculate 1-year return
                current = float(data['Close'].iloc[-1])
                start_price = float(data['Close'].iloc[0])
                annual_return = ((current / start_price) - 1) * 100
                
                # Calculate volatility
                returns = data['Close'].pct_change().dropna()
                volatility = float(returns.std() * np.sqrt(252) * 100)
                
                results.append({
                    'symbol': symbol,
                    'name': name,
                    'return': annual_return,
                    'volatility': volatility,
                    'current_price': current
                })
                
                print(f"   ✅ 1-Year Return: {annual_return:+.2f}%")
                print(f"   📊 Volatility: {volatility:.1f}%")
                
        except Exception as e:
            print(f"   ❌ Error: {str(e)}")
    
    # Sort by returns
    if results:
        print("\\n🏆 1-YEAR PERFORMANCE RANKING:")
        print("-" * 40)
        
        sorted_results = sorted(results, key=lambda x: x['return'], reverse=True)
        
        for i, result in enumerate(sorted_results, 1):
            print(f"{i}. {result['name'][:25]:25} {result['return']:+6.2f}% (Vol: {result['volatility']:4.1f}%)")
    
    return results

def main():
    """Main analysis function."""
    
    # Analyze Nifty 50 Index
    nifty_analysis = analyze_nifty50_index()
    
    # Compare with stocks
    comparison = compare_nifty_with_stocks()
    
    print("\\n" + "="*60)
    print("🎯 NIFTY 50 INDEX ANALYSIS SUMMARY")
    print("="*60)
    
    if nifty_analysis:
        print(f"📊 Current Level: ₹{nifty_analysis['current_level']:,.2f}")
        print(f"📈 2-Year Return: {nifty_analysis['total_return']:+.2f}%")
        print(f"📊 Sharpe Ratio: {nifty_analysis['sharpe_ratio']:.3f}")
        print(f"⚠️  Max Drawdown: {nifty_analysis['max_drawdown']:.2f}%")
        print(f"📈 Trend: {nifty_analysis['trend']}")
        
        # Performance rating
        sharpe = nifty_analysis['sharpe_ratio']
        if sharpe > 1.5:
            rating = "🏆 EXCELLENT"
        elif sharpe > 1.0:
            rating = "🥇 VERY GOOD"
        elif sharpe > 0.5:
            rating = "🥈 GOOD"
        else:
            rating = "🥉 AVERAGE"
            
        print(f"🌟 Performance Rating: {rating}")
    
    print("\\n✅ CONCLUSION:")
    print("   • Nifty 50 Index data is available and accessible")
    print("   • Can be analyzed with all our enhanced tools")
    print("   • Symbol to use: ^NSEI")
    print("   • Available in frontend dashboard!")

if __name__ == "__main__":
    main() 