#!/usr/bin/env python3
"""
Run Comprehensive Indian Stock Analysis
=======================================
Execute full analysis and display results
"""

from comprehensive_indian_analysis import ComprehensiveIndianAnalysis
import json
from datetime import datetime

def main():
    print("🚀 LAUNCHING COMPREHENSIVE INDIAN STOCK ANALYSIS")
    print("=" * 70)
    
    # Initialize analyzer for 50 stocks (manageable for demo)
    analyzer = ComprehensiveIndianAnalysis(
        analysis_type='COMPREHENSIVE',
        max_stocks=50
    )
    
    print(f"\n📊 Analysis Configuration:")
    print(f"   Stock Universe: {len(analyzer.stock_universe)} stocks")
    print(f"   Sectors Covered: {len(analyzer._get_sectors_covered())} sectors")
    print(f"   Analysis Type: {analyzer.analysis_type}")
    
    # Run comprehensive analysis
    print(f"\n🔥 Starting comprehensive analysis...")
    results = analyzer.run_comprehensive_analysis()
    
    if results:
        print_summary(results)
    else:
        print("❌ Analysis failed")

def print_summary(results):
    """Print analysis summary"""
    print("\n" + "=" * 70)
    print("📊 COMPREHENSIVE ANALYSIS RESULTS")
    print("=" * 70)
    
    summary = results['summary']
    
    print(f"📈 MARKET OVERVIEW:")
    print(f"   Stocks Analyzed: {summary['total_stocks_analyzed']}")
    print(f"   Buy Signals: {summary['buy_signals_generated']}")
    print(f"   Signal Ratio: {summary['signal_ratio']}")
    
    print(f"\n🏆 TOP 5 STOCKS:")
    for i, stock in enumerate(results['top_10_stocks'][:5], 1):
        score = results['stock_rankings'][stock]['composite_score']
        returns = results['stock_rankings'][stock]['returns_1m']
        print(f"   {i}. {stock:<15} Score: {score:6.1f} | Returns: {returns:+6.1f}%")
    
    print(f"\n🏭 SECTOR PERFORMANCE:")
    best_sector = summary['best_performing_sector']
    worst_sector = summary['worst_performing_sector']
    print(f"   🥇 Best:  {best_sector['name']:<20} (+{best_sector['return_1m']:5.1f}%)")
    print(f"   🥉 Worst: {worst_sector['name']:<20} ({worst_sector['return_1m']:+5.1f}%)")
    
    print(f"\n💼 PORTFOLIO RECOMMENDATIONS:")
    portfolios = summary['portfolio_expected_returns']
    print(f"   Conservative: {portfolios['conservative']:+6.2f}% expected return")
    print(f"   Moderate:     {portfolios['moderate']:+6.2f}% expected return")
    print(f"   Aggressive:   {portfolios['aggressive']:+6.2f}% expected return")
    
    print(f"\n🎯 TRADING SIGNALS:")
    signal_counts = {}
    for signal_data in results['trading_signals'].values():
        signal = signal_data['signal']
        signal_counts[signal] = signal_counts.get(signal, 0) + 1
    
    for signal, count in signal_counts.items():
        print(f"   {signal:<12}: {count:2d} stocks")
    
    print(f"\n💎 TOP RECOMMENDATIONS:")
    for i, (symbol, signal_data) in enumerate(list(results['trading_signals'].items())[:5], 1):
        if signal_data['signal'] in ['STRONG_BUY', 'BUY']:
            price = signal_data['current_price']
            confidence = signal_data['confidence']
            print(f"   {i}. {symbol:<15} {signal_data['signal']:<10} @ ₹{price:7.2f} ({confidence:.0%} confidence)")
    
    print(f"\n💾 Full results saved to JSON file")
    print("=" * 70)

if __name__ == "__main__":
    main() 