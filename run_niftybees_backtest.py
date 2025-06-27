#!/usr/bin/env python3
"""
Run Advanced Backtesting for NIFTYBEES.NS
=========================================
"""

import sys
import os
from datetime import datetime
from ultra_enhanced_backtesting import UltraEnhancedBacktester

def run_niftybees_analysis():
    """Run advanced backtesting specifically for NIFTYBEES.NS"""
    
    print("🇮🇳 NIFTYBEES.NS ADVANCED BACKTESTING")
    print("="*50)
    print("📊 Nifty BeES ETF Analysis")
    print("🎯 Ultra-Enhanced ML Models")
    print("="*50)
    
    # Initialize backtester for NIFTYBEES
    backtester = UltraEnhancedBacktester(
        symbol='NIFTYBEES.NS',
        start_date='2018-01-01',  # 6+ years of data for robust analysis
        end_date=None
    )
    
    try:
        print(f"\n🚀 Starting analysis for NIFTYBEES.NS...")
        
        # Run ultra-enhanced walk-forward test
        results = backtester.ultra_walk_forward_test()
        
        if results:
            print(f"\n{'='*80}")
            print("🏆 NIFTYBEES.NS FINAL PERFORMANCE SUMMARY")
            print(f"{'='*80}")
            
            print(f"📊 Symbol: {results['symbol']}")
            print(f"📈 Total Steps: {results['total_steps']}")
            print(f"🎯 Average RMSE: ₹{results['avg_rmse']:.2f}")
            print(f"📊 Average MAPE: {results['avg_mape']:.1f}%")
            print(f"🎯 Directional Accuracy: {results['avg_directional_accuracy']:.1f}%")
            print(f"💰 Total Return: {results['total_return']:.2%}")
            print(f"📊 Average Sharpe: {results['avg_sharpe']:.3f}")
            print(f"🎯 Win Rate: {results['win_rate']:.1f}%")
            print(f"📊 Average Confidence: {results['avg_confidence']:.1%}")
            
            # Performance assessment
            print(f"\n🏆 PERFORMANCE ASSESSMENT:")
            
            if results['avg_directional_accuracy'] > 60:
                print("✅ Excellent directional accuracy (>60%)")
            elif results['avg_directional_accuracy'] > 55:
                print("🟡 Good directional accuracy (55-60%)")
            else:
                print("🔴 Directional accuracy needs improvement (<55%)")
            
            if results['avg_sharpe'] > 1.0:
                print("✅ Excellent risk-adjusted returns (Sharpe >1.0)")
            elif results['avg_sharpe'] > 0.5:
                print("🟡 Good risk-adjusted returns (Sharpe 0.5-1.0)")
            else:
                print("🔴 Poor risk-adjusted returns (Sharpe <0.5)")
            
            if results['win_rate'] > 60:
                print("✅ High win rate (>60%)")
            elif results['win_rate'] > 50:
                print("🟡 Moderate win rate (50-60%)")
            else:
                print("🔴 Low win rate (<50%)")
            
            # Indian ETF specific insights
            print(f"\n🇮🇳 NIFTY BEES ETF INSIGHTS:")
            print("📈 NIFTYBEES tracks the Nifty 50 index")
            print("💰 Currency: Indian Rupees (₹)")
            print("🏛️ Managed by Nippon India Mutual Fund")
            print("📊 Provides exposure to top 50 Indian companies")
            
            # Save results
            import joblib
            os.makedirs('results', exist_ok=True)
            filename = f'results/NIFTYBEES_advanced_backtest_{datetime.now().strftime("%Y%m%d_%H%M%S")}.pkl'
            joblib.dump(results, filename)
            print(f"\n💾 Results saved to: {filename}")
            
            return results
        else:
            print("❌ Analysis failed - no results generated")
            return {}
            
    except Exception as e:
        print(f"❌ Error during analysis: {e}")
        import traceback
        traceback.print_exc()
        return {}

if __name__ == "__main__":
    results = run_niftybees_analysis() 