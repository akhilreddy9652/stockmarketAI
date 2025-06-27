#!/usr/bin/env python3
"""
Indian ETF Portfolio Analysis System
===================================
Comprehensive analysis of Indian ETFs for portfolio construction
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from ultra_enhanced_backtesting import UltraEnhancedBacktester
import joblib
import os
import warnings
warnings.filterwarnings('ignore')

class IndianETFPortfolioAnalyzer:
    """Comprehensive Indian ETF Portfolio Analysis System"""
    
    def __init__(self, start_date='2020-01-01', end_date=None):
        self.start_date = start_date
        self.end_date = end_date or datetime.now().strftime('%Y-%m-%d')
        self.etf_results = {}
        
        # Popular Indian ETFs
        self.indian_etfs = {
            'NIFTYBEES.NS': {
                'name': 'Nifty BeES',
                'benchmark': 'Nifty 50',
                'category': 'Large Cap',
                'description': 'Tracks Nifty 50 Index (Top 50 companies)',
                'fund_house': 'Nippon India Mutual Fund'
            },
            'JUNIORBEES.NS': {
                'name': 'Junior BeES', 
                'benchmark': 'Nifty Next 50',
                'category': 'Mid Cap',
                'description': 'Tracks Nifty Next 50 Index',
                'fund_house': 'Nippon India Mutual Fund'
            },
            'BANKBEES.NS': {
                'name': 'Bank BeES',
                'benchmark': 'Nifty Bank',
                'category': 'Sector - Banking',
                'description': 'Tracks Nifty Bank Index',
                'fund_house': 'Nippon India Mutual Fund'
            },
            'ICICIB22.NS': {
                'name': 'ICICI Prudential Nifty Bank ETF',
                'benchmark': 'Nifty Bank',
                'category': 'Sector - Banking', 
                'description': 'Banking sector exposure',
                'fund_house': 'ICICI Prudential'
            },
            'SETFNIF50.NS': {
                'name': 'SBI ETF Nifty 50',
                'benchmark': 'Nifty 50',
                'category': 'Large Cap',
                'description': 'SBI Nifty 50 ETF',
                'fund_house': 'SBI Mutual Fund'
            },
            'ITBEES.NS': {
                'name': 'IT BeES',
                'benchmark': 'Nifty IT',
                'category': 'Sector - IT',
                'description': 'Tracks Nifty IT Index',
                'fund_house': 'Nippon India Mutual Fund'
            }
        }
    
    def analyze_single_etf(self, symbol):
        """Analyze a single Indian ETF using ultra-enhanced backtesting."""
        print(f"\n🔍 Analyzing {symbol} ({self.indian_etfs[symbol]['name']})...")
        
        try:
            # Initialize ultra-enhanced backtester
            backtester = UltraEnhancedBacktester(
                symbol=symbol,
                start_date=self.start_date,
                end_date=self.end_date
            )
            
            # Run analysis
            results = backtester.ultra_walk_forward_test()
            
            if results:
                # Add ETF metadata
                results['etf_info'] = self.indian_etfs[symbol]
                results['analysis_date'] = datetime.now().isoformat()
                
                print(f"✅ {symbol}: Dir={results['avg_directional_accuracy']:.1f}%, "
                      f"MAPE={results['avg_mape']:.1f}%, Ret={results['total_return']:.1f}%")
                
                return results
            else:
                print(f"❌ Failed to analyze {symbol}")
                return None
                
        except Exception as e:
            print(f"❌ Error analyzing {symbol}: {e}")
            return None
    
    def run_portfolio_analysis(self):
        """Run comprehensive analysis on all Indian ETFs."""
        print("🇮🇳 INDIAN ETF PORTFOLIO ANALYSIS")
        print("="*60)
        print("📊 Analyzing multiple Indian ETFs for portfolio construction")
        print("🎯 Ultra-Enhanced ML Models for each ETF")
        print("="*60)
        
        results_summary = []
        
        for symbol in self.indian_etfs.keys():
            result = self.analyze_single_etf(symbol)
            
            if result:
                self.etf_results[symbol] = result
                
                # Create summary entry
                summary_entry = {
                    'Symbol': symbol,
                    'Name': self.indian_etfs[symbol]['name'],
                    'Category': self.indian_etfs[symbol]['category'],
                    'Benchmark': self.indian_etfs[symbol]['benchmark'],
                    'Directional_Accuracy': result['avg_directional_accuracy'],
                    'MAPE': result['avg_mape'],
                    'Total_Return': result['total_return'],
                    'Sharpe_Ratio': result['avg_sharpe'],
                    'Win_Rate': result['win_rate'],
                    'Confidence': result['avg_confidence'],
                    'Steps': result['total_steps']
                }
                results_summary.append(summary_entry)
        
        return pd.DataFrame(results_summary)
    
    def create_portfolio_recommendations(self, summary_df):
        """Create portfolio recommendations based on analysis results."""
        print(f"\n📊 PORTFOLIO RECOMMENDATIONS")
        print("="*50)
        
        if summary_df.empty:
            print("❌ No valid results for portfolio construction")
            return {}
        
        # Score each ETF based on multiple criteria
        summary_df['Performance_Score'] = (
            summary_df['Directional_Accuracy'] * 0.4 +  # 40% weight on accuracy
            (100 - summary_df['MAPE']) * 0.3 +         # 30% weight on prediction precision
            summary_df['Sharpe_Ratio'] * 20 +          # 20% weight on risk-adjusted returns
            summary_df['Win_Rate'] * 0.1               # 10% weight on win rate
        )
        
        # Sort by performance score
        summary_df = summary_df.sort_values('Performance_Score', ascending=False)
        
        # Portfolio construction
        portfolio_recommendations = {
            'core_holdings': [],
            'satellite_holdings': [],
            'weights': {},
            'risk_profile': 'Moderate',
            'total_etfs': len(summary_df)
        }
        
        # Core holdings (top performers)
        top_etfs = summary_df.head(3)
        for _, etf in top_etfs.iterrows():
            portfolio_recommendations['core_holdings'].append({
                'symbol': etf['Symbol'],
                'name': etf['Name'],
                'category': etf['Category'],
                'performance_score': etf['Performance_Score'],
                'recommended_weight': 0.4 if etf['Category'] == 'Large Cap' else 0.3
            })
        
        # Satellite holdings (sector/specialized)
        sector_etfs = summary_df[summary_df['Category'].str.contains('Sector')].head(2)
        for _, etf in sector_etfs.iterrows():
            portfolio_recommendations['satellite_holdings'].append({
                'symbol': etf['Symbol'], 
                'name': etf['Name'],
                'category': etf['Category'],
                'performance_score': etf['Performance_Score'],
                'recommended_weight': 0.15
            })
        
        return portfolio_recommendations
    
    def print_comprehensive_results(self, summary_df, portfolio_rec):
        """Print comprehensive portfolio analysis results."""
        print(f"\n{'='*80}")
        print("🏆 INDIAN ETF PORTFOLIO ANALYSIS RESULTS")
        print(f"{'='*80}")
        
        # Overall statistics
        if not summary_df.empty:
            print(f"\n📊 PORTFOLIO STATISTICS:")
            print(f"   Total ETFs Analyzed: {len(summary_df)}")
            print(f"   Average Directional Accuracy: {summary_df['Directional_Accuracy'].mean():.1f}%")
            print(f"   Average MAPE: {summary_df['MAPE'].mean():.1f}%")
            print(f"   Average Sharpe Ratio: {summary_df['Sharpe_Ratio'].mean():.2f}")
            print(f"   Average Win Rate: {summary_df['Win_Rate'].mean():.1f}%")
            
            # Top performers
            print(f"\n🏆 TOP PERFORMING ETFs:")
            top_3 = summary_df.head(3)
            for i, (_, etf) in enumerate(top_3.iterrows(), 1):
                print(f"   {i}. {etf['Symbol']} ({etf['Name']})")
                print(f"      📈 Directional Accuracy: {etf['Directional_Accuracy']:.1f}%")
                print(f"      📊 MAPE: {etf['MAPE']:.1f}% | Sharpe: {etf['Sharpe_Ratio']:.2f}")
                print(f"      🎯 Category: {etf['Category']}")
            
            # Portfolio recommendations
            if portfolio_rec and portfolio_rec['core_holdings']:
                print(f"\n💼 RECOMMENDED PORTFOLIO ALLOCATION:")
                print(f"   Risk Profile: {portfolio_rec['risk_profile']}")
                
                print(f"\n   🎯 CORE HOLDINGS (70-80%):")
                for holding in portfolio_rec['core_holdings']:
                    print(f"     • {holding['symbol']} ({holding['name']})")
                    print(f"       Weight: {holding['recommended_weight']*100:.0f}% | Category: {holding['category']}")
                
                if portfolio_rec['satellite_holdings']:
                    print(f"\n   🛰️ SATELLITE HOLDINGS (20-30%):")
                    for holding in portfolio_rec['satellite_holdings']:
                        print(f"     • {holding['symbol']} ({holding['name']})")
                        print(f"       Weight: {holding['recommended_weight']*100:.0f}% | Category: {holding['category']}")
            
            # Diversification analysis
            print(f"\n🌈 DIVERSIFICATION ANALYSIS:")
            category_counts = summary_df['Category'].value_counts()
            for category, count in category_counts.items():
                print(f"   {category}: {count} ETF(s)")
            
            # Risk assessment
            print(f"\n⚠️ RISK ASSESSMENT:")
            high_accuracy = len(summary_df[summary_df['Directional_Accuracy'] > 70])
            low_mape = len(summary_df[summary_df['MAPE'] < 5])
            
            print(f"   High Accuracy ETFs (>70%): {high_accuracy}/{len(summary_df)}")
            print(f"   Low Prediction Error (<5% MAPE): {low_mape}/{len(summary_df)}")
            
            if high_accuracy >= 2 and low_mape >= 2:
                print("   ✅ Portfolio Risk: LOW (Multiple high-quality ETFs)")
            elif high_accuracy >= 1 or low_mape >= 1:
                print("   🟡 Portfolio Risk: MODERATE (Some high-quality ETFs)")
            else:
                print("   🔴 Portfolio Risk: HIGH (Limited high-quality options)")
    
    def save_results(self):
        """Save all analysis results."""
        os.makedirs('results/portfolio', exist_ok=True)
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # Save individual ETF results
        for symbol, result in self.etf_results.items():
            filename = f'results/portfolio/{symbol}_analysis_{timestamp}.pkl'
            joblib.dump(result, filename)
        
        # Save portfolio analysis
        portfolio_filename = f'results/portfolio/indian_etf_portfolio_{timestamp}.pkl'
        joblib.dump(self.etf_results, portfolio_filename)
        
        print(f"\n💾 Results saved to: results/portfolio/")
        print(f"   Individual ETF results: {len(self.etf_results)} files")
        print(f"   Portfolio analysis: {portfolio_filename}")

def main():
    """Main function to run Indian ETF portfolio analysis."""
    
    # Initialize analyzer
    analyzer = IndianETFPortfolioAnalyzer(
        start_date='2020-01-01',  # 4+ years of data
        end_date=None
    )
    
    try:
        # Run comprehensive analysis
        summary_df = analyzer.run_portfolio_analysis()
        
        # Create portfolio recommendations
        portfolio_recommendations = analyzer.create_portfolio_recommendations(summary_df)
        
        # Print results
        analyzer.print_comprehensive_results(summary_df, portfolio_recommendations)
        
        # Save results
        analyzer.save_results()
        
        return {
            'summary': summary_df,
            'portfolio_recommendations': portfolio_recommendations,
            'individual_results': analyzer.etf_results
        }
        
    except Exception as e:
        print(f"❌ Error in portfolio analysis: {e}")
        import traceback
        traceback.print_exc()
        return {}

if __name__ == "__main__":
    results = main() 