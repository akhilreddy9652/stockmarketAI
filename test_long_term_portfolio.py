#!/usr/bin/env python3
"""
Long-Term Portfolio Testing Script
=================================
Test the long-term investment system across multiple assets
and generate a comprehensive comparison report.
"""

import os
import pandas as pd
from datetime import datetime, timedelta
from long_term_investment_system import LongTermInvestmentSystem
import warnings
warnings.filterwarnings('ignore')

def test_long_term_portfolio():
    """Test long-term investment system on diversified portfolio"""
    
    print("🎯 LONG-TERM PORTFOLIO ANALYSIS")
    print("=" * 50)
    
    # Test portfolio - mix of US and Indian assets
    test_portfolio = {
        "🇺🇸 US Large Cap": ["AAPL", "MSFT", "GOOGL"],
        "🇺🇸 US ETFs": ["SPY", "QQQ"],
        "🇮🇳 Indian Large Cap": ["RELIANCE.NS", "TCS.NS"],
        "🇮🇳 Indian ETFs": ["NIFTYBEES.NS", "JUNIORBEES.NS"]
    }
    
    results = {}
    total_assets = sum(len(assets) for assets in test_portfolio.values())
    current_asset = 0
    
    print(f"📊 Testing {total_assets} assets for long-term performance...")
    print()
    
    for category, symbols in test_portfolio.items():
        print(f"\n🔍 Analyzing {category}:")
        print("-" * 30)
        
        for symbol in symbols:
            current_asset += 1
            print(f"[{current_asset}/{total_assets}] {symbol}...", end=" ")
            
            try:
                # Initialize long-term system
                lt_system = LongTermInvestmentSystem(
                    symbol=symbol,
                    start_date='2015-01-01',  # 9+ years
                    initial_capital=100000
                )
                
                # Run analysis
                analysis = lt_system.run_complete_analysis()
                
                if analysis:
                    signals = analysis['current_signals']
                    backtest = analysis['backtest_results']
                    strategy_perf = backtest['strategy_performance']
                    buy_hold_perf = backtest['buy_hold_performance']
                    
                    results[symbol] = {
                        'category': category,
                        'signal': signals['Overall']['signal'],
                        'confidence': signals['Overall']['confidence'],
                        'current_price': signals['Latest_Data']['Price'],
                        'annual_return': signals['Latest_Data']['Annual_Return'],
                        'strategy_cagr': strategy_perf['cagr'],
                        'buyhold_cagr': buy_hold_perf['cagr'],
                        'excess_cagr': strategy_perf['cagr'] - buy_hold_perf['cagr'],
                        'sharpe_ratio': strategy_perf['sharpe_ratio'],
                        'max_drawdown': strategy_perf['max_drawdown'],
                        'transactions': strategy_perf['total_transactions'],
                        'final_value': strategy_perf['final_value']
                    }
                    
                    print(f"✅ CAGR: {strategy_perf['cagr']:.1f}%")
                else:
                    print("❌ Failed")
                    
            except Exception as e:
                print(f"❌ Error: {str(e)[:50]}...")
    
    return results

def generate_portfolio_report(results):
    """Generate comprehensive portfolio analysis report"""
    
    if not results:
        print("❌ No results to analyze")
        return
    
    print("\n" + "=" * 60)
    print("📊 LONG-TERM PORTFOLIO ANALYSIS REPORT")
    print("=" * 60)
    
    # Convert to DataFrame for analysis
    df = pd.DataFrame.from_dict(results, orient='index')
    df.index.name = 'Symbol'
    df = df.reset_index()
    
    # Overall portfolio statistics
    print(f"\n📈 PORTFOLIO OVERVIEW:")
    print(f"Total Assets Analyzed: {len(df)}")
    print(f"Average Strategy CAGR: {df['strategy_cagr'].mean():.2f}%")
    print(f"Average Buy & Hold CAGR: {df['buyhold_cagr'].mean():.2f}%")
    print(f"Average Excess CAGR: {df['excess_cagr'].mean():.2f}%")
    print(f"Average Sharpe Ratio: {df['sharpe_ratio'].mean():.2f}")
    
    # Performance by category
    print(f"\n📊 PERFORMANCE BY CATEGORY:")
    category_stats = df.groupby('category').agg({
        'strategy_cagr': 'mean',
        'buyhold_cagr': 'mean',
        'excess_cagr': 'mean',
        'sharpe_ratio': 'mean',
        'transactions': 'mean'
    }).round(2)
    
    for category, stats in category_stats.iterrows():
        print(f"\n{category}:")
        print(f"  Strategy CAGR: {stats['strategy_cagr']:.2f}%")
        print(f"  Buy & Hold CAGR: {stats['buyhold_cagr']:.2f}%")
        print(f"  Excess CAGR: {stats['excess_cagr']:+.2f}%")
        print(f"  Avg Sharpe: {stats['sharpe_ratio']:.2f}")
        print(f"  Avg Transactions: {stats['transactions']:.1f}")
    
    # Best performers
    print(f"\n🏆 TOP PERFORMERS:")
    top_performers = df.nlargest(3, 'strategy_cagr')[['Symbol', 'category', 'strategy_cagr', 'excess_cagr', 'signal']]
    for _, performer in top_performers.iterrows():
        print(f"  {performer['Symbol']:12} ({performer['category'][:10]}): {performer['strategy_cagr']:6.2f}% CAGR, {performer['excess_cagr']:+5.2f}% excess, {performer['signal']}")
    
    # Current signals
    print(f"\n🎯 CURRENT SIGNALS:")
    buy_signals = df[df['signal'] == 'BUY']
    sell_signals = df[df['signal'] == 'SELL']
    hold_signals = df[df['signal'] == 'HOLD']
    
    print(f"  🟢 BUY:  {len(buy_signals):2d} assets")
    print(f"  🔴 SELL: {len(sell_signals):2d} assets")
    print(f"  🟡 HOLD: {len(hold_signals):2d} assets")
    
    if len(buy_signals) > 0:
        print(f"\n🟢 BUY RECOMMENDATIONS:")
        for _, asset in buy_signals.iterrows():
            print(f"  {asset['Symbol']:12} - {asset['confidence']:.0%} confidence, {asset['annual_return']:+5.1f}% annual return")
    
    # Strategy effectiveness
    print(f"\n📊 STRATEGY EFFECTIVENESS:")
    outperforming = len(df[df['excess_cagr'] > 0])
    conservative = len(df[df['transactions'] <= 5])
    high_sharpe = len(df[df['sharpe_ratio'] > 1.0])
    
    print(f"  Outperforming Buy & Hold: {outperforming}/{len(df)} ({outperforming/len(df)*100:.1f}%)")
    print(f"  Conservative (≤5 trades): {conservative}/{len(df)} ({conservative/len(df)*100:.1f}%)")
    print(f"  High Sharpe (>1.0): {high_sharpe}/{len(df)} ({high_sharpe/len(df)*100:.1f}%)")
    
    # Long-term investment insights
    print(f"\n💡 LONG-TERM INVESTMENT INSIGHTS:")
    
    if df['excess_cagr'].mean() > 0:
        print("  ✅ Strategy shows positive alpha generation on average")
    else:
        print("  ⚠️ Buy & Hold outperforms on average - consider passive approach")
    
    if df['transactions'].mean() < 5:
        print("  ✅ Low transaction frequency - tax and cost efficient")
    else:
        print("  ⚠️ Higher transaction frequency - consider tax implications")
    
    if df['sharpe_ratio'].mean() > 0.8:
        print("  ✅ Good risk-adjusted returns across portfolio")
    else:
        print("  ⚠️ Moderate risk-adjusted returns - review risk management")
    
    # Portfolio allocation suggestions
    print(f"\n💼 SUGGESTED LONG-TERM ALLOCATION:")
    
    # Strong buy signals with high confidence
    strong_buys = df[(df['signal'] == 'BUY') & (df['confidence'] > 0.7)]
    if len(strong_buys) > 0:
        print("  Core Holdings (20-30% each):")
        for _, asset in strong_buys.head(2).iterrows():
            print(f"    {asset['Symbol']:12} - {asset['strategy_cagr']:.1f}% CAGR, {asset['confidence']:.0%} confidence")
    
    # Good performers
    good_performers = df[df['strategy_cagr'] > 10]
    if len(good_performers) > 0:
        print("  Growth Positions (10-15% each):")
        for _, asset in good_performers.head(3).iterrows():
            print(f"    {asset['Symbol']:12} - {asset['strategy_cagr']:.1f}% CAGR")
    
    print(f"\n📋 LONG-TERM STRATEGY RECOMMENDATIONS:")
    print("  • Rebalance quarterly to maintain target allocations")
    print("  • Focus on CAGR over short-term volatility")
    print("  • Consider tax implications when making changes")
    print("  • Maintain diversification across geographies and sectors")
    print("  • Use dollar-cost averaging for new positions")
    
    # Save detailed results
    df.to_csv('results/long_term_portfolio_analysis.csv', index=False)
    print(f"\n💾 Detailed results saved to: results/long_term_portfolio_analysis.csv")
    
    print("=" * 60)

def main():
    """Main function"""
    print("🚀 Starting Long-Term Portfolio Analysis")
    print("Focus: CAGR, Quality, Sustainable Growth")
    print()
    
    # Create results directory
    os.makedirs('results', exist_ok=True)
    
    # Test portfolio
    results = test_long_term_portfolio()
    
    # Generate report
    generate_portfolio_report(results)
    
    print("\n🎉 Long-term portfolio analysis completed!")
    print("💡 Use these insights for your long-term investment strategy")

if __name__ == "__main__":
    main() 