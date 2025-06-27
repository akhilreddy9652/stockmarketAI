"""
Enhanced Backtesting Test for Indian Stocks
==========================================
Testing the enhanced accuracy system on popular Indian stocks
to demonstrate cross-market performance.
"""

import pandas as pd
import numpy as np
import warnings
from enhanced_backtesting_v2 import EnhancedBacktester
import joblib
import os

warnings.filterwarnings('ignore')

def test_indian_stocks_performance():
    """Test enhanced backtesting on popular Indian stocks."""
    
    # Popular Indian stocks across different sectors
    indian_stocks = {
        'RELIANCE.NS': 'Energy & Petrochemicals (Largest Indian Company)',
        'TCS.NS': 'IT Services (Top IT Company)',
        'INFY.NS': 'IT Services (Global IT Giant)',
        'HDFCBANK.NS': 'Banking (Leading Private Bank)',
        'ICICIBANK.NS': 'Banking (Major Private Bank)',
        'HINDUNILVR.NS': 'FMCG (Consumer Goods)',
        'ITC.NS': 'FMCG & Cigarettes',
        'BHARTIARTL.NS': 'Telecommunications',
        'SBIN.NS': 'Banking (Largest PSU Bank)',
        'MARUTI.NS': 'Automotive (Leading Car Manufacturer)'
    }
    
    print("🇮🇳 ENHANCED BACKTESTING FOR INDIAN STOCKS")
    print("=" * 80)
    print("Testing institutional-grade accuracy on Indian market...")
    print(f"Selected stocks: {len(indian_stocks)} major Indian companies")
    print("=" * 80)
    
    results_summary = []
    detailed_results = {}
    
    for symbol, description in indian_stocks.items():
        print(f"\n{'='*100}")
        print(f"🎯 ENHANCED ANALYSIS: {symbol}")
        print(f"📊 {description}")
        print(f"{'='*100}")
        
        try:
            # Initialize enhanced backtester
            backtester = EnhancedBacktester(
                symbol=symbol, 
                start_date='2020-01-01'
            )
            
            # Run enhanced analysis
            result = backtester.enhanced_walk_forward_test()
            
            if result and 'avg_directional_accuracy' in result:
                # Extract key metrics
                metrics = {
                    'symbol': symbol,
                    'description': description,
                    'avg_rmse': result['avg_rmse'],
                    'avg_mape': result['avg_mape'],
                    'directional_accuracy': result['avg_directional_accuracy'],
                    'total_return': result['total_return'],
                    'win_rate': result['win_rate'],
                    'sharpe_ratio': result['sharpe_ratio'],
                    'total_steps': result['total_steps']
                }
                
                results_summary.append(metrics)
                detailed_results[symbol] = result
                
                # Save individual results
                os.makedirs('results', exist_ok=True)
                joblib.dump(result, f'results/{symbol}_enhanced_indian.pkl')
                
                # Print performance summary
                print(f"\n📊 ENHANCED PERFORMANCE SUMMARY:")
                print(f"   🎯 Directional Accuracy: {metrics['directional_accuracy']:.1f}%")
                print(f"   💰 Total Return: {metrics['total_return']:.2%}")
                print(f"   📊 Sharpe Ratio: {metrics['sharpe_ratio']:.3f}")
                print(f"   🎯 Win Rate: {metrics['win_rate']:.1f}%")
                print(f"   📏 MAPE Error: {metrics['avg_mape']:.1f}%")
                print(f"   💵 Average RMSE: ₹{metrics['avg_rmse']:.2f}")
                print(f"   📅 Test Periods: {metrics['total_steps']}")
                
                # Performance classification
                if metrics['directional_accuracy'] >= 90:
                    grade = "🏆 EXCELLENT (>90%)"
                elif metrics['directional_accuracy'] >= 80:
                    grade = "🥇 VERY GOOD (80-90%)"
                elif metrics['directional_accuracy'] >= 70:
                    grade = "🥈 GOOD (70-80%)"
                else:
                    grade = "🥉 FAIR (<70%)"
                
                print(f"   🏅 Performance Grade: {grade}")
                
            else:
                print(f"   ❌ Analysis failed for {symbol}")
                
        except Exception as e:
            print(f"   ❌ Error processing {symbol}: {e}")
            continue
    
    # Generate comprehensive summary
    if results_summary:
        print(f"\n{'='*100}")
        print("📊 COMPREHENSIVE INDIAN STOCKS PERFORMANCE SUMMARY")
        print(f"{'='*100}")
        
        df_results = pd.DataFrame(results_summary)
        
        # Overall statistics
        avg_directional = df_results['directional_accuracy'].mean()
        avg_sharpe = df_results['sharpe_ratio'].mean()
        avg_win_rate = df_results['win_rate'].mean()
        avg_mape = df_results['avg_mape'].mean()
        
        print(f"\n🇮🇳 INDIAN MARKET ENHANCED PERFORMANCE:")
        print(f"   📊 Average Directional Accuracy: {avg_directional:.1f}%")
        print(f"   📈 Average Sharpe Ratio: {avg_sharpe:.3f}")
        print(f"   🎯 Average Win Rate: {avg_win_rate:.1f}%")
        print(f"   📏 Average MAPE Error: {avg_mape:.1f}%")
        print(f"   📊 Stocks Analyzed: {len(results_summary)}")
        
        # Best performers
        best_accuracy = df_results.loc[df_results['directional_accuracy'].idxmax()]
        best_sharpe = df_results.loc[df_results['sharpe_ratio'].idxmax()]
        best_return = df_results.loc[df_results['total_return'].idxmax()]
        
        print(f"\n🏆 TOP PERFORMERS:")
        print(f"   🎯 Best Accuracy: {best_accuracy['symbol']} ({best_accuracy['directional_accuracy']:.1f}%)")
        print(f"   📊 Best Sharpe: {best_sharpe['symbol']} ({best_sharpe['sharpe_ratio']:.3f})")
        print(f"   💰 Best Return: {best_return['symbol']} ({best_return['total_return']:.2%})")
        
        # Performance distribution
        excellent = len(df_results[df_results['directional_accuracy'] >= 90])
        very_good = len(df_results[(df_results['directional_accuracy'] >= 80) & (df_results['directional_accuracy'] < 90)])
        good = len(df_results[(df_results['directional_accuracy'] >= 70) & (df_results['directional_accuracy'] < 80)])
        fair = len(df_results[df_results['directional_accuracy'] < 70])
        
        print(f"\n📈 PERFORMANCE DISTRIBUTION:")
        print(f"   🏆 Excellent (>90%): {excellent} stocks")
        print(f"   🥇 Very Good (80-90%): {very_good} stocks")
        print(f"   🥈 Good (70-80%): {good} stocks")
        print(f"   🥉 Fair (<70%): {fair} stocks")
        
        # Market comparison insights
        print(f"\n💡 MARKET INSIGHTS:")
        if avg_directional >= 85:
            print("   🎉 OUTSTANDING: Indian stocks show excellent predictability!")
        elif avg_directional >= 75:
            print("   ✅ STRONG: Indian market demonstrates good prediction accuracy!")
        elif avg_directional >= 65:
            print("   👍 DECENT: Indian market shows reasonable predictability!")
        else:
            print("   ⚠️ CHALLENGING: Indian market requires further optimization!")
        
        # Sector analysis
        it_stocks = [r for r in results_summary if 'IT' in r['description']]
        banking_stocks = [r for r in results_summary if 'Bank' in r['description']]
        
        if it_stocks:
            it_avg_acc = np.mean([s['directional_accuracy'] for s in it_stocks])
            print(f"   💻 IT Sector Average Accuracy: {it_avg_acc:.1f}%")
        
        if banking_stocks:
            banking_avg_acc = np.mean([s['directional_accuracy'] for s in banking_stocks])
            print(f"   🏦 Banking Sector Average Accuracy: {banking_avg_acc:.1f}%")
        
        # Save comprehensive results
        comprehensive_results = {
            'summary_stats': {
                'avg_directional_accuracy': avg_directional,
                'avg_sharpe_ratio': avg_sharpe,
                'avg_win_rate': avg_win_rate,
                'avg_mape': avg_mape,
                'stocks_analyzed': len(results_summary)
            },
            'individual_results': results_summary,
            'detailed_results': detailed_results,
            'top_performers': {
                'best_accuracy': best_accuracy.to_dict(),
                'best_sharpe': best_sharpe.to_dict(),
                'best_return': best_return.to_dict()
            }
        }
        
        joblib.dump(comprehensive_results, 'results/indian_stocks_comprehensive_enhanced.pkl')
        
        print(f"\n💾 RESULTS SAVED:")
        print(f"   📊 Comprehensive: results/indian_stocks_comprehensive_enhanced.pkl")
        print(f"   📁 Individual: results/[SYMBOL]_enhanced_indian.pkl")
        
        return comprehensive_results
    
    else:
        print("❌ No successful analyses completed")
        return {}

if __name__ == "__main__":
    print("🚀 Enhanced Indian Stocks Backtesting Analysis")
    print("Testing institutional-grade accuracy on Indian market...")
    
    results = test_indian_stocks_performance()
    
    print(f"\n{'='*100}")
    print("🎉 INDIAN STOCKS ENHANCED ANALYSIS COMPLETED!")
    print("="*100)
    print("🇮🇳 Indian market analysis complete with enhanced accuracy system!")
    print("📊 Check results/ directory for detailed performance data!") 