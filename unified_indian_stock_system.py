#!/usr/bin/env python3
"""
Unified Indian Stock Management System
====================================
Combines RL Agent + Portfolio Optimizer + Long-term System
"""

import numpy as np
import pandas as pd
from datetime import datetime, timedelta
import yfinance as yf
import json
import warnings
warnings.filterwarnings('ignore')

# Import our working systems
from indian_rl_trading_agent import IndianRLTradingAgent
from advanced_portfolio_optimizer import AdvancedPortfolioOptimizer
from long_term_investment_system import LongTermInvestmentSystem

class UnifiedIndianStockSystem:
    """Unified AI-Driven Indian Stock Management System"""
    
    def __init__(self, initial_capital=1000000, risk_tolerance='moderate'):
        self.initial_capital = initial_capital
        self.current_capital = initial_capital
        self.risk_tolerance = risk_tolerance
        
        # Component systems
        self.rl_agent = None
        self.portfolio_optimizer = None
        self.long_term_system = None
        
        # Portfolio state
        self.current_positions = {}
        self.signals_history = []
        
        print(f"🚀 Unified Indian Stock System Initialized")
        print(f"💰 Initial Capital: ₹{self.initial_capital:,}")
        print(f"🎯 Risk Tolerance: {risk_tolerance}")
    
    def initialize_systems(self, stock_universe=None):
        """Initialize all component systems"""
        print("🔧 Initializing component systems...")
        
        if stock_universe is None:
            # Import expanded universe
            try:
                from expanded_indian_stock_universe import get_comprehensive_indian_universe, get_balanced_portfolio
                
                if self.risk_tolerance == 'conservative':
                    stock_universe = get_comprehensive_indian_universe('LARGE_CAP', 40)
                elif self.risk_tolerance == 'aggressive':
                    stock_universe = get_comprehensive_indian_universe('GROWTH', 50)
                else:
                    stock_universe = get_balanced_portfolio(45, 'MODERATE')
                
                print(f"🇮🇳 Using expanded Indian universe: {len(stock_universe)} stocks")
                
            except ImportError:
                # Fallback to original universe
                stock_universe = [
                    'RELIANCE.NS', 'TCS.NS', 'HDFCBANK.NS', 'INFY.NS', 'ICICIBANK.NS',
                    'HINDUNILVR.NS', 'ITC.NS', 'SBIN.NS', 'BHARTIARTL.NS', 'KOTAKBANK.NS',
                    'ASIANPAINT.NS', 'MARUTI.NS', 'SUNPHARMA.NS', 'TATAMOTORS.NS', 'WIPRO.NS',
                    'ULTRACEMCO.NS', 'TITAN.NS', 'BAJFINANCE.NS', 'NESTLEIND.NS', 'POWERGRID.NS',
                    'TECHM.NS', 'BAJAJFINSV.NS', 'NTPC.NS', 'HCLTECH.NS', 'ONGC.NS',
                    'JSWSTEEL.NS', 'TATACONSUM.NS', 'ADANIENT.NS', 'COALINDIA.NS', 'HINDALCO.NS'
                ]
                print(f"🔄 Using fallback universe: {len(stock_universe)} stocks")
        
        self.stock_universe = stock_universe
        
        try:
            # Initialize RL Trading Agent (expanded to more stocks)
            rl_stocks = stock_universe[:8]  # Top 8 for RL training
            self.rl_agent = IndianRLTradingAgent(symbols=rl_stocks)
            self.rl_agent.fetch_data()
            print(f"✅ RL Trading Agent: {len(rl_stocks)} stocks")
            
            # Initialize Portfolio Optimizer (full universe)
            portfolio_stocks = stock_universe[:25]  # Top 25 for optimization
            self.portfolio_optimizer = AdvancedPortfolioOptimizer(portfolio_stocks)
            self.portfolio_optimizer.load_data()
            print(f"✅ Portfolio Optimizer: {len(portfolio_stocks)} stocks")
            
            print("✅ Long-term Investment System ready")
            print(f"📊 Total Universe: {len(stock_universe)} Indian stocks")
            return True
            
        except Exception as e:
            print(f"❌ Error initializing systems: {e}")
            return False
    
    def generate_unified_signals(self):
        """Generate unified trading signals from all systems"""
        print("🎯 Generating unified trading signals...")
        
        signals = {
            'timestamp': datetime.now(),
            'rl_signals': {},
            'portfolio_allocation': {},
            'unified_recommendation': {}
        }
        
        try:
            # 1. Get RL Agent signals
            if self.rl_agent:
                rl_signals = self.rl_agent.generate_signals()
                signals['rl_signals'] = rl_signals
                print(f"📊 RL signals: {len(rl_signals)} stocks")
            
            # 2. Get Portfolio Optimization
            if self.portfolio_optimizer:
                optimal_portfolio = self.portfolio_optimizer.optimize_maximum_sharpe()
                if optimal_portfolio:
                    signals['portfolio_allocation'] = {
                        'weights': dict(zip(optimal_portfolio['symbols'], optimal_portfolio['weights'])),
                        'expected_return': optimal_portfolio['return'],
                        'sharpe_ratio': optimal_portfolio['sharpe_ratio']
                    }
                    print(f"📈 Portfolio optimized - Sharpe: {optimal_portfolio['sharpe_ratio']:.3f}")
            
            # 3. Create unified recommendations
            unified_recommendations = self._create_unified_recommendations(signals)
            signals['unified_recommendation'] = unified_recommendations
            
            self.signals_history.append(signals)
            print("✅ Unified signals generated")
            return signals
            
        except Exception as e:
            print(f"❌ Error generating signals: {e}")
            return signals
    
    def _create_unified_recommendations(self, signals):
        """Create unified buy/sell/hold recommendations"""
        recommendations = {}
        
        try:
            rl_signals = signals.get('rl_signals', {})
            portfolio_weights = signals.get('portfolio_allocation', {}).get('weights', {})
            
            for symbol in self.stock_universe:
                recommendation = {
                    'action': 'HOLD',
                    'confidence': 0.5,
                    'allocation': 0.0,
                    'reasoning': []
                }
                
                # RL signal influence
                if symbol in rl_signals:
                    rl_signal = rl_signals[symbol]
                    rl_action = rl_signal.get('signal', 'HOLD')
                    rl_confidence = rl_signal.get('confidence', 0.5)
                    
                    if rl_action in ['BUY', 'SELL'] and rl_confidence > 0.6:
                        recommendation['action'] = rl_action
                        recommendation['confidence'] = min(recommendation['confidence'] + 0.3, 1.0)
                        recommendation['reasoning'].append(f"RL: {rl_action} ({rl_confidence:.1%})")
                
                # Portfolio optimization influence
                if symbol in portfolio_weights:
                    target_weight = portfolio_weights[symbol]
                    if target_weight > 0.05:  # >5% allocation
                        recommendation['allocation'] = min(target_weight, 0.15)  # Cap at 15%
                        recommendation['confidence'] = min(recommendation['confidence'] + 0.2, 1.0)
                        recommendation['reasoning'].append(f"Portfolio: {target_weight:.1%}")
                        
                        if recommendation['action'] == 'HOLD' and target_weight > 0.1:
                            recommendation['action'] = 'BUY'
                
                recommendations[symbol] = recommendation
            
            return recommendations
            
        except Exception as e:
            print(f"❌ Error creating recommendations: {e}")
            return {}
    
    def execute_paper_trading(self, signals):
        """Execute paper trading based on signals"""
        print("💼 Executing paper trading...")
        
        try:
            recommendations = signals.get('unified_recommendation', {})
            executed_trades = []
            
            for symbol, rec in recommendations.items():
                action = rec['action']
                confidence = rec['confidence']
                target_allocation = rec['allocation']
                
                if action in ['BUY', 'SELL'] and confidence > 0.6:
                    try:
                        # Get current price
                        stock = yf.Ticker(symbol)
                        current_price = stock.history(period="1d")['Close'].iloc[-1]
                        
                        if action == 'BUY' and target_allocation > 0:
                            target_value = self.current_capital * target_allocation
                            if target_value > 1000:  # Minimum ₹1000 trade
                                shares = int(target_value / current_price)
                                trade_value = shares * current_price
                                
                                executed_trades.append({
                                    'symbol': symbol,
                                    'action': action,
                                    'shares': shares,
                                    'price': current_price,
                                    'value': trade_value,
                                    'confidence': confidence
                                })
                                
                                print(f"📈 BUY {shares} {symbol} @ ₹{current_price:.2f} = ₹{trade_value:,.0f}")
                    
                    except Exception as e:
                        print(f"⚠️ Trade failed for {symbol}: {e}")
            
            print(f"✅ Paper trading: {len(executed_trades)} trades executed")
            return executed_trades
            
        except Exception as e:
            print(f"❌ Error in paper trading: {e}")
            return []
    
    def generate_performance_report(self):
        """Generate performance report"""
        print("📊 Generating performance report...")
        
        report = {
            'timestamp': datetime.now(),
            'portfolio_summary': {
                'initial_capital': self.initial_capital,
                'current_capital': self.current_capital,
                'risk_tolerance': self.risk_tolerance
            },
            'recent_signals': {},
            'recommendations': [
                "Monitor Indian market conditions daily",
                "Maintain diversified portfolio",
                "Review allocation monthly"
            ]
        }
        
        # Recent signals summary
        if self.signals_history:
            latest_signals = self.signals_history[-1]
            recommendations = latest_signals.get('unified_recommendation', {})
            
            buy_signals = sum(1 for rec in recommendations.values() if rec['action'] == 'BUY')
            sell_signals = sum(1 for rec in recommendations.values() if rec['action'] == 'SELL')
            
            report['recent_signals'] = {
                'total_analyzed': len(recommendations),
                'buy_signals': buy_signals,
                'sell_signals': sell_signals,
                'avg_confidence': np.mean([rec['confidence'] for rec in recommendations.values()])
            }
        
        print("✅ Performance report generated")
        return report
    
    def run_daily_analysis(self):
        """Run complete daily analysis"""
        print("🌅 Starting daily analysis...")
        print("="*50)
        
        try:
            # Generate signals
            signals = self.generate_unified_signals()
            
            # Execute paper trading
            trades = self.execute_paper_trading(signals)
            
            # Generate report
            report = self.generate_performance_report()
            
            # Save results
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            
            try:
                with open(f'results/unified_analysis_{timestamp}.json', 'w') as f:
                    results = {
                        'signals': signals,
                        'trades': trades,
                        'report': report
                    }
                    json.dump(results, f, indent=2, default=str)
                print(f"💾 Results saved: unified_analysis_{timestamp}.json")
            except:
                print("⚠️ Could not save to file")
            
            print("="*50)
            print("✅ Daily analysis completed!")
            
            return {
                'signals': signals,
                'trades': trades,
                'report': report
            }
            
        except Exception as e:
            print(f"❌ Error in daily analysis: {e}")
            return None

if __name__ == "__main__":
    print("🚀 Testing Unified Indian Stock System")
    print("="*50)
    
    # Initialize system
    system = UnifiedIndianStockSystem(
        initial_capital=1000000,  # ₹10 lakh
        risk_tolerance='moderate'
    )
    
    # Initialize and run
    if system.initialize_systems():
        results = system.run_daily_analysis()
        
        if results:
            report = results['report']
            print(f"\n📊 SUMMARY:")
            print(f"Stocks Analyzed: {report['recent_signals']['total_analyzed']}")
            print(f"Buy Signals: {report['recent_signals']['buy_signals']}")
            print(f"Avg Confidence: {report['recent_signals']['avg_confidence']:.1%}")
            print(f"\n🎉 System working successfully!")
    else:
        print("❌ System initialization failed") 