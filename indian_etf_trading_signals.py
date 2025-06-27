#!/usr/bin/env python3
"""
Indian ETF Automated Trading Signals & Alert System
==================================================
Automated signal generation and alerts for Indian ETF portfolio
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import yfinance as yf
import joblib
import os
import json
from data_ingestion import fetch_yfinance
from feature_engineering import add_technical_indicators, get_trading_signals
import warnings
warnings.filterwarnings('ignore')

class IndianETFTradingSignals:
    """Automated trading signals for Indian ETF portfolio"""
    
    def __init__(self):
        self.portfolio_etfs = {
            'NIFTYBEES.NS': {'name': 'Nifty BeES', 'weight': 25, 'category': 'Large Cap'},
            'JUNIORBEES.NS': {'name': 'Junior BeES', 'weight': 30, 'category': 'Mid Cap'},
            'BANKBEES.NS': {'name': 'Bank BeES', 'weight': 25, 'category': 'Banking'},
            'ICICIB22.NS': {'name': 'ICICI Bank ETF', 'weight': 15, 'category': 'Banking'},
            'ITBEES.NS': {'name': 'IT BeES', 'weight': 5, 'category': 'IT Sector'}
        }
        
        # Performance thresholds from backtesting
        self.confidence_thresholds = {
            'NIFTYBEES.NS': 0.991,  # 99.1% confidence from backtesting
            'JUNIORBEES.NS': 0.988,  # 98.8% confidence
            'BANKBEES.NS': 0.992,   # 99.2% confidence
            'ICICIB22.NS': 0.988,   # 98.8% confidence
            'ITBEES.NS': 0.991      # 99.1% confidence
        }
        
        self.signal_history = []
        
    def generate_enhanced_signals(self, symbol):
        """Generate enhanced trading signals with multiple confirmations"""
        try:
            # Fetch recent data (3 months for robust signals)
            end_date = datetime.now()
            start_date = end_date - timedelta(days=90)
            
            df = fetch_yfinance(symbol, start_date.strftime('%Y-%m-%d'), end_date.strftime('%Y-%m-%d'))
            
            if df.empty:
                return {'status': 'error', 'message': 'No data available'}
            
            # Add technical indicators
            df = add_technical_indicators(df)
            
            # Get basic trading signals
            basic_signals = get_trading_signals(df)
            
            # Enhanced signal analysis
            latest = df.iloc[-1]
            prev = df.iloc[-2] if len(df) > 1 else latest
            
            # Technical indicators
            rsi = latest.get('RSI_14', 50)
            macd = latest.get('MACD', 0)
            macd_signal = latest.get('MACD_Signal', 0)
            bb_position = latest.get('BB_Position', 0.5)
            volume_ratio = latest.get('Volume_Ratio', 1.0)
            
            # Price momentum
            price_change_1d = (latest['Close'] - prev['Close']) / prev['Close'] * 100
            price_change_5d = latest.get('Price_Change_5d', 0) * 100
            
            # Enhanced signal generation
            signals = []
            confidence_factors = []
            
            # RSI Signal
            if rsi < 30:
                signals.append('BUY')
                confidence_factors.append(0.8)
            elif rsi > 70:
                signals.append('SELL')
                confidence_factors.append(0.8)
            else:
                signals.append('NEUTRAL')
                confidence_factors.append(0.3)
            
            # MACD Signal
            if macd > macd_signal and macd > 0:
                signals.append('BUY')
                confidence_factors.append(0.7)
            elif macd < macd_signal and macd < 0:
                signals.append('SELL')
                confidence_factors.append(0.7)
            else:
                signals.append('NEUTRAL')
                confidence_factors.append(0.4)
            
            # Bollinger Bands Signal
            if bb_position < 0.2:
                signals.append('BUY')
                confidence_factors.append(0.6)
            elif bb_position > 0.8:
                signals.append('SELL')
                confidence_factors.append(0.6)
            else:
                signals.append('NEUTRAL')
                confidence_factors.append(0.3)
            
            # Volume Confirmation
            volume_signal = 'STRONG' if volume_ratio > 1.5 else 'WEAK'
            volume_factor = 1.2 if volume_ratio > 1.5 else 0.8
            
            # Consensus Signal
            buy_votes = signals.count('BUY')
            sell_votes = signals.count('SELL')
            neutral_votes = signals.count('NEUTRAL')
            
            if buy_votes >= 2:
                consensus = 'BUY'
                base_confidence = np.mean([cf for i, cf in enumerate(confidence_factors) if signals[i] == 'BUY'])
            elif sell_votes >= 2:
                consensus = 'SELL'
                base_confidence = np.mean([cf for i, cf in enumerate(confidence_factors) if signals[i] == 'SELL'])
            else:
                consensus = 'HOLD'
                base_confidence = 0.5
            
            # Apply volume factor and historical confidence
            final_confidence = min(base_confidence * volume_factor, 1.0)
            
            # Apply ETF-specific confidence threshold
            etf_threshold = self.confidence_thresholds.get(symbol, 0.95)
            meets_threshold = final_confidence >= etf_threshold * 0.8  # 80% of backtested confidence
            
            # Signal strength
            if final_confidence > 0.8:
                strength = 'STRONG'
            elif final_confidence > 0.6:
                strength = 'MODERATE'
            else:
                strength = 'WEAK'
            
            # Generate recommendation
            if consensus != 'HOLD' and meets_threshold:
                recommendation = consensus
            else:
                recommendation = 'HOLD'
            
            return {
                'status': 'success',
                'symbol': symbol,
                'timestamp': datetime.now().isoformat(),
                'recommendation': recommendation,
                'confidence': final_confidence,
                'strength': strength,
                'technical_details': {
                    'rsi': rsi,
                    'macd': macd,
                    'bb_position': bb_position,
                    'volume_signal': volume_signal,
                    'price_change_1d': price_change_1d,
                    'price_change_5d': price_change_5d
                },
                'signal_breakdown': {
                    'rsi_signal': signals[0],
                    'macd_signal': signals[1],
                    'bb_signal': signals[2],
                    'volume_factor': volume_factor,
                    'consensus': consensus
                },
                'meets_threshold': meets_threshold,
                'basic_signals': basic_signals
            }
            
        except Exception as e:
            return {'status': 'error', 'message': str(e)}
    
    def scan_portfolio_signals(self):
        """Scan all ETFs in portfolio for trading signals"""
        print("🔍 Scanning Indian ETF Portfolio for Trading Signals")
        print("="*60)
        
        portfolio_signals = {}
        signal_summary = {'BUY': 0, 'SELL': 0, 'HOLD': 0}
        high_confidence_signals = []
        
        for symbol, info in self.portfolio_etfs.items():
            print(f"\n📊 Analyzing {symbol} ({info['name']})...")
            
            signals = self.generate_enhanced_signals(symbol)
            
            if signals['status'] == 'success':
                portfolio_signals[symbol] = signals
                
                # Count signals
                signal_summary[signals['recommendation']] += 1
                
                # High confidence signals
                if signals['confidence'] > 0.7 and signals['recommendation'] != 'HOLD':
                    high_confidence_signals.append({
                        'symbol': symbol,
                        'name': info['name'],
                        'signal': signals['recommendation'],
                        'confidence': signals['confidence'],
                        'strength': signals['strength']
                    })
                
                # Print signal details
                print(f"   Signal: {signals['recommendation']} ({signals['strength']})")
                print(f"   Confidence: {signals['confidence']:.1%}")
                print(f"   RSI: {signals['technical_details']['rsi']:.1f}")
                print(f"   Price Change (1D): {signals['technical_details']['price_change_1d']:+.2f}%")
                
            else:
                print(f"   ❌ Error: {signals['message']}")
        
        return {
            'portfolio_signals': portfolio_signals,
            'signal_summary': signal_summary,
            'high_confidence_signals': high_confidence_signals,
            'scan_time': datetime.now().isoformat()
        }
    
    def generate_portfolio_recommendations(self, scan_results):
        """Generate portfolio-level recommendations"""
        signals = scan_results['portfolio_signals']
        
        # Calculate portfolio signal strength
        total_weight = 0
        weighted_signal = 0
        
        for symbol, signal_data in signals.items():
            if signal_data['status'] == 'success':
                weight = self.portfolio_etfs[symbol]['weight'] / 100
                signal_value = 1 if signal_data['recommendation'] == 'BUY' else -1 if signal_data['recommendation'] == 'SELL' else 0
                confidence_weight = signal_data['confidence']
                
                weighted_signal += signal_value * weight * confidence_weight
                total_weight += weight
        
        # Portfolio recommendation
        if weighted_signal > 0.3:
            portfolio_action = 'INCREASE_EXPOSURE'
        elif weighted_signal < -0.3:
            portfolio_action = 'REDUCE_EXPOSURE'
        else:
            portfolio_action = 'MAINTAIN_CURRENT'
        
        # Risk assessment
        high_conf_signals = len(scan_results['high_confidence_signals'])
        total_signals = len([s for s in signals.values() if s['status'] == 'success'])
        
        if high_conf_signals >= 3:
            risk_level = 'LOW'
        elif high_conf_signals >= 1:
            risk_level = 'MODERATE'
        else:
            risk_level = 'HIGH'
        
        return {
            'portfolio_action': portfolio_action,
            'weighted_signal_strength': weighted_signal,
            'risk_level': risk_level,
            'high_confidence_ratio': high_conf_signals / total_signals if total_signals > 0 else 0,
            'recommendation_summary': {
                'action': portfolio_action,
                'confidence': abs(weighted_signal),
                'risk': risk_level
            }
        }
    
    def save_signals(self, scan_results, recommendations):
        """Save signals to file for historical tracking"""
        os.makedirs('signals', exist_ok=True)
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f'signals/indian_etf_signals_{timestamp}.json'
        
        data = {
            'scan_results': scan_results,
            'portfolio_recommendations': recommendations,
            'generated_at': datetime.now().isoformat()
        }
        
        with open(filename, 'w') as f:
            json.dump(data, f, indent=2, default=str)
        
        return filename
    
    def print_comprehensive_report(self, scan_results, recommendations):
        """Print comprehensive trading signals report"""
        print(f"\n{'='*80}")
        print("🇮🇳 INDIAN ETF PORTFOLIO TRADING SIGNALS REPORT")
        print(f"{'='*80}")
        
        print(f"📅 Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S IST')}")
        print(f"🎯 Portfolio ETFs: {len(self.portfolio_etfs)}")
        
        # Signal Summary
        print(f"\n📊 SIGNAL SUMMARY:")
        summary = scan_results['signal_summary']
        print(f"   🟢 BUY Signals: {summary['BUY']}")
        print(f"   🔴 SELL Signals: {summary['SELL']}")
        print(f"   🟡 HOLD Signals: {summary['HOLD']}")
        
        # High Confidence Signals
        high_conf = scan_results['high_confidence_signals']
        if high_conf:
            print(f"\n🎯 HIGH CONFIDENCE SIGNALS ({len(high_conf)}):")
            for signal in high_conf:
                print(f"   • {signal['name']} ({signal['symbol']})")
                print(f"     Signal: {signal['signal']} | Confidence: {signal['confidence']:.1%} | Strength: {signal['strength']}")
        else:
            print(f"\n⚠️ No high confidence signals detected")
        
        # Portfolio Recommendations
        print(f"\n💼 PORTFOLIO RECOMMENDATIONS:")
        rec = recommendations['recommendation_summary']
        print(f"   Action: {rec['action']}")
        print(f"   Confidence Level: {rec['confidence']:.1%}")
        print(f"   Risk Level: {rec['risk']}")
        print(f"   High Confidence Ratio: {recommendations['high_confidence_ratio']:.1%}")
        
        # Individual ETF Details
        print(f"\n🔍 INDIVIDUAL ETF ANALYSIS:")
        for symbol, signal_data in scan_results['portfolio_signals'].items():
            if signal_data['status'] == 'success':
                info = self.portfolio_etfs[symbol]
                print(f"\n   📈 {info['name']} ({symbol}):")
                print(f"      Signal: {signal_data['recommendation']} ({signal_data['strength']})")
                print(f"      Confidence: {signal_data['confidence']:.1%}")
                print(f"      Weight: {info['weight']}%")
                
                tech = signal_data['technical_details']
                print(f"      RSI: {tech['rsi']:.1f} | BB Position: {tech['bb_position']:.2f}")
                print(f"      Price Change (1D): {tech['price_change_1d']:+.2f}%")
                print(f"      Volume: {tech['volume_signal']}")
        
        # Trading Actions
        print(f"\n🎯 RECOMMENDED ACTIONS:")
        if recommendations['portfolio_action'] == 'INCREASE_EXPOSURE':
            print("   ✅ Consider increasing portfolio exposure")
            print("   ✅ Focus on high-confidence BUY signals")
        elif recommendations['portfolio_action'] == 'REDUCE_EXPOSURE':
            print("   ⚠️ Consider reducing portfolio exposure")
            print("   ⚠️ Review SELL signals and risk management")
        else:
            print("   🔄 Maintain current portfolio allocation")
            print("   👀 Monitor for signal changes")
        
        print(f"\n📊 RISK MANAGEMENT:")
        if recommendations['risk_level'] == 'LOW':
            print("   ✅ Low risk environment - Multiple high-confidence signals")
        elif recommendations['risk_level'] == 'MODERATE':
            print("   🟡 Moderate risk - Some high-confidence signals available")
        else:
            print("   🔴 High risk - Limited high-confidence signals")
        
        print(f"\n💡 NEXT STEPS:")
        print("   1. Review individual ETF signals and confirmations")
        print("   2. Implement recommended portfolio actions with proper risk management")
        print("   3. Monitor signal changes and adjust positions accordingly")
        print("   4. Set stop-losses and take-profit levels based on technical analysis")

def run_automated_signal_scan():
    """Run automated signal scanning for Indian ETF portfolio"""
    print("🚀 Indian ETF Automated Trading Signals System")
    print("="*50)
    
    # Initialize signal generator
    signal_generator = IndianETFTradingSignals()
    
    try:
        # Scan portfolio for signals
        scan_results = signal_generator.scan_portfolio_signals()
        
        # Generate portfolio recommendations
        recommendations = signal_generator.generate_portfolio_recommendations(scan_results)
        
        # Print comprehensive report
        signal_generator.print_comprehensive_report(scan_results, recommendations)
        
        # Save signals
        filename = signal_generator.save_signals(scan_results, recommendations)
        print(f"\n💾 Signals saved to: {filename}")
        
        return {
            'scan_results': scan_results,
            'recommendations': recommendations,
            'saved_file': filename
        }
        
    except Exception as e:
        print(f"❌ Error in signal generation: {e}")
        import traceback
        traceback.print_exc()
        return {}

if __name__ == "__main__":
    results = run_automated_signal_scan() 