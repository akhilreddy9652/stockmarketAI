#!/usr/bin/env python3
"""
Comprehensive Backtesting for NIFTYBEES.NS
==========================================
"""

from comprehensive_backtesting import ComprehensiveBacktester
import os
import joblib
from datetime import datetime

def run_niftybees_comprehensive_analysis():
    """Run comprehensive backtesting specifically for NIFTYBEES.NS"""
    
    print("🇮🇳 NIFTYBEES.NS COMPREHENSIVE BACKTESTING")
    print("="*60)
    print("📊 Nifty BeES ETF - Traditional Analysis")
    print("🎯 Multiple Trading Strategies Comparison")
    print("="*60)
    
    try:
        # Initialize comprehensive backtester for NIFTYBEES
        backtester = ComprehensiveBacktester(
            symbol='NIFTYBEES.NS',
            start_date='2018-01-01',  # 6+ years of data
            end_date=None
        )
        
        print(f"\n🚀 Starting comprehensive analysis for NIFTYBEES.NS...")
        
        # Run comprehensive analysis
        results = backtester.comprehensive_analysis()
        
        if results:
            # Print comprehensive results
            backtester.print_comprehensive_results()
            
            # Save results
            os.makedirs('results', exist_ok=True)
            filename = f'results/NIFTYBEES_comprehensive_backtest_{datetime.now().strftime("%Y%m%d_%H%M%S")}.pkl'
            joblib.dump(results, filename)
            
            print(f"\n💾 Comprehensive results saved to: {filename}")
            
            # Additional Indian market insights
            print(f"\n🇮🇳 NIFTY BEES ETF ADDITIONAL INSIGHTS:")
            print("="*50)
            print("📈 Benchmark: Nifty 50 Index")
            print("🏛️ Asset Management Company: Nippon India Mutual Fund")
            print("📊 Expense Ratio: ~0.05% (Very Low)")
            print("💰 Minimum Investment: ₹500 (Approx)")
            print("📈 Market Cap: Large Cap (Top 50 Indian Companies)")
            print("🌏 Geographic Exposure: India")
            print("💼 Sector Diversification: Across 13+ sectors")
            
            return results
        else:
            print("❌ Comprehensive analysis failed - no results generated")
            return {}
            
    except Exception as e:
        print(f"❌ Error during comprehensive analysis: {e}")
        import traceback
        traceback.print_exc()
        return {}

if __name__ == "__main__":
    results = run_niftybees_comprehensive_analysis() 