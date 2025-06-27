#!/usr/bin/env python3
"""
Enhanced Unified Indian Stock System
===================================
Advanced AI system with expanded Indian stock universe (194 stocks)
"""

import numpy as np
import pandas as pd
from datetime import datetime, timedelta
import yfinance as yf
import json
import warnings
warnings.filterwarnings('ignore')

# Import our expanded universe
from expanded_indian_stock_universe import (
    get_comprehensive_indian_universe, 
    get_balanced_portfolio,
    SECTOR_MAPPING,
    RISK_CATEGORIES
)

class EnhancedUnifiedIndianSystem:
    """
    Enhanced AI-Driven Indian Stock Management System
    Supports 194+ Indian stocks with advanced portfolio management
    """
    
    def __init__(self, initial_capital=1000000, risk_tolerance='moderate', universe_size=50):
        self.initial_capital = initial_capital
        self.current_capital = initial_capital
        self.risk_tolerance = risk_tolerance
        self.universe_size = universe_size
        
        # Enhanced stock universe
        self.stock_universe = self._initialize_stock_universe()
        
        # Portfolio state
        self.current_positions = {}
        self.signals_history = []
        self.sector_allocation = {}
        
        # Performance tracking
        self.performance_metrics = {}
        
        print(f"🚀 Enhanced Unified Indian Stock System Initialized")
        print(f"💰 Initial Capital: ₹{self.initial_capital:,}")
        print(f"🎯 Risk Tolerance: {risk_tolerance}")
        print(f"📊 Stock Universe: {len(self.stock_universe)} Indian stocks")
        print(f"🏭 Sectors Covered: {len(self._get_sector_distribution())} sectors")
        
    def _initialize_stock_universe(self):
        """Initialize expanded stock universe based on risk tolerance"""
        if self.risk_tolerance == 'conservative':
            universe = get_comprehensive_indian_universe('LARGE_CAP', self.universe_size)
        elif self.risk_tolerance == 'aggressive':
            universe = get_comprehensive_indian_universe('GROWTH', self.universe_size)
        else:
            universe = get_balanced_portfolio(self.universe_size, 'MODERATE')
        
        return universe
    
    def _get_sector_distribution(self):
        """Get sector distribution of current universe"""
        sector_count = {}
        for stock in self.stock_universe:
            for sector, stocks in SECTOR_MAPPING.items():
                if stock in stocks:
                    sector_count[sector] = sector_count.get(sector, 0) + 1
                    break
        return sector_count
    
    def fetch_market_data(self, days=365):
        """Fetch market data for entire universe"""
        print(f"📊 Fetching market data for {len(self.stock_universe)} stocks...")
        
        market_data = {}
        successful_fetches = 0
        
        for symbol in self.stock_universe:
            try:
                # Get stock data
                stock = yf.Ticker(symbol)
                end_date = datetime.now()
                start_date = end_date - timedelta(days=days)
                
                data = stock.history(
                    start=start_date.strftime('%Y-%m-%d'),
                    end=end_date.strftime('%Y-%m-%d')
                )
                
                if not data.empty and len(data) > 50:  # Minimum data requirement
                    market_data[symbol] = {
                        'data': data,
                        'current_price': data['Close'].iloc[-1],
                        'change_1d': ((data['Close'].iloc[-1] - data['Close'].iloc[-2]) / data['Close'].iloc[-2]) * 100,
                        'change_30d': ((data['Close'].iloc[-1] - data['Close'].iloc[-30]) / data['Close'].iloc[-30]) * 100,
                        'volatility': data['Close'].pct_change().std() * np.sqrt(252),
                        'volume_avg': data['Volume'].mean(),
                        'market_cap_proxy': data['Close'].iloc[-1] * data['Volume'].iloc[-1]
                    }
                    successful_fetches += 1
                
            except Exception as e:
                print(f"⚠️ Failed to fetch {symbol}: {str(e)[:50]}")
                continue
        
        print(f"✅ Successfully fetched data for {successful_fetches}/{len(self.stock_universe)} stocks")
        return market_data
    
    def calculate_enhanced_features(self, market_data):
        """Calculate enhanced technical and fundamental features"""
        print("🔧 Calculating enhanced features...")
        
        enhanced_data = {}
        
        for symbol, data_dict in market_data.items():
            try:
                data = data_dict['data']
                
                # Technical indicators
                data['SMA_20'] = data['Close'].rolling(20).mean()
                data['SMA_50'] = data['Close'].rolling(50).mean()
                data['RSI'] = self._calculate_rsi(data['Close'])
                data['BB_Upper'], data['BB_Lower'] = self._calculate_bollinger_bands(data['Close'])
                data['MACD'], data['MACD_Signal'] = self._calculate_macd(data['Close'])
                
                # Enhanced metrics
                latest = data.iloc[-1]
                enhanced_features = {
                    'technical_score': self._calculate_technical_score(data),
                    'momentum_score': self._calculate_momentum_score(data),
                    'volatility_score': self._calculate_volatility_score(data),
                    'volume_score': self._calculate_volume_score(data),
                    'trend_strength': self._calculate_trend_strength(data),
                    'risk_adjusted_return': self._calculate_risk_adjusted_return(data),
                    'current_price': latest['Close'],
                    'rsi': latest['RSI'],
                    'macd': latest['MACD'],
                    'price_vs_sma20': (latest['Close'] - latest['SMA_20']) / latest['SMA_20'] if pd.notna(latest['SMA_20']) else 0,
                    'price_vs_sma50': (latest['Close'] - latest['SMA_50']) / latest['SMA_50'] if pd.notna(latest['SMA_50']) else 0
                }
                
                enhanced_data[symbol] = enhanced_features
                
            except Exception as e:
                print(f"⚠️ Feature calculation failed for {symbol}: {str(e)[:50]}")
                continue
        
        print(f"✅ Enhanced features calculated for {len(enhanced_data)} stocks")
        return enhanced_data
    
    def _calculate_rsi(self, prices, period=14):
        """Calculate RSI"""
        delta = prices.diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
        rs = gain / loss
        return 100 - (100 / (1 + rs))
    
    def _calculate_bollinger_bands(self, prices, period=20, std_dev=2):
        """Calculate Bollinger Bands"""
        sma = prices.rolling(period).mean()
        std = prices.rolling(period).std()
        upper = sma + (std * std_dev)
        lower = sma - (std * std_dev)
        return upper, lower
    
    def _calculate_macd(self, prices, fast=12, slow=26, signal=9):
        """Calculate MACD"""
        exp1 = prices.ewm(span=fast).mean()
        exp2 = prices.ewm(span=slow).mean()
        macd = exp1 - exp2
        macd_signal = macd.ewm(span=signal).mean()
        return macd, macd_signal
    
    def _calculate_technical_score(self, data):
        """Calculate composite technical score (0-100)"""
        latest = data.iloc[-1]
        score = 0
        
        # RSI contribution (30%)
        if pd.notna(latest['RSI']):
            if 30 <= latest['RSI'] <= 70:
                score += 30
            elif latest['RSI'] < 30:
                score += 15  # Oversold - potential buy
            else:
                score += 10  # Overbought
        
        # MACD contribution (25%)
        if pd.notna(latest['MACD']) and pd.notna(latest['MACD_Signal']):
            if latest['MACD'] > latest['MACD_Signal']:
                score += 25
            else:
                score += 10
        
        # Price vs SMA contribution (25%)
        if pd.notna(latest['SMA_20']) and pd.notna(latest['SMA_50']):
            if latest['Close'] > latest['SMA_20'] > latest['SMA_50']:
                score += 25
            elif latest['Close'] > latest['SMA_20']:
                score += 15
            else:
                score += 5
        
        # Volume trend contribution (20%)
        recent_volume = data['Volume'].tail(5).mean()
        historical_volume = data['Volume'].mean()
        if recent_volume > historical_volume * 1.2:
            score += 20
        elif recent_volume > historical_volume:
            score += 10
        else:
            score += 5
        
        return min(score, 100)
    
    def _calculate_momentum_score(self, data):
        """Calculate momentum score"""
        returns_1w = (data['Close'].iloc[-1] / data['Close'].iloc[-7] - 1) * 100
        returns_1m = (data['Close'].iloc[-1] / data['Close'].iloc[-21] - 1) * 100
        returns_3m = (data['Close'].iloc[-1] / data['Close'].iloc[-63] - 1) * 100
        
        # Weight recent performance more
        momentum = (returns_1w * 0.5 + returns_1m * 0.3 + returns_3m * 0.2)
        return max(0, min(100, 50 + momentum))  # Normalize to 0-100
    
    def _calculate_volatility_score(self, data):
        """Calculate volatility score (lower volatility = higher score)"""
        volatility = data['Close'].pct_change().std() * np.sqrt(252)
        # Invert and normalize (lower vol = higher score)
        return max(0, min(100, 100 - volatility * 200))
    
    def _calculate_volume_score(self, data):
        """Calculate volume score"""
        recent_volume = data['Volume'].tail(10).mean()
        historical_volume = data['Volume'].mean()
        volume_ratio = recent_volume / historical_volume if historical_volume > 0 else 1
        return max(0, min(100, 50 + (volume_ratio - 1) * 50))
    
    def _calculate_trend_strength(self, data):
        """Calculate trend strength"""
        if len(data) < 50:
            return 50
        
        sma_20 = data['Close'].rolling(20).mean().iloc[-1]
        sma_50 = data['Close'].rolling(50).mean().iloc[-1]
        current_price = data['Close'].iloc[-1]
        
        if pd.notna(sma_20) and pd.notna(sma_50):
            if current_price > sma_20 > sma_50:
                return 80
            elif current_price > sma_20:
                return 60
            elif sma_20 > sma_50:
                return 40
            else:
                return 20
        return 50
    
    def _calculate_risk_adjusted_return(self, data):
        """Calculate Sharpe-like ratio for individual stock"""
        returns = data['Close'].pct_change().dropna()
        if len(returns) < 30:
            return 0
        
        avg_return = returns.mean() * 252  # Annualized
        volatility = returns.std() * np.sqrt(252)
        
        if volatility > 0:
            return avg_return / volatility
        return 0
    
    def generate_enhanced_signals(self, enhanced_data):
        """Generate enhanced trading signals for all stocks"""
        print("🎯 Generating enhanced trading signals...")
        
        signals = {}
        
        for symbol, features in enhanced_data.items():
            # Composite scoring
            composite_score = (
                features['technical_score'] * 0.3 +
                features['momentum_score'] * 0.25 +
                features['volatility_score'] * 0.15 +
                features['volume_score'] * 0.15 +
                features['trend_strength'] * 0.15
            )
            
            # Signal generation
            if composite_score >= 75:
                signal = 'STRONG_BUY'
                confidence = min(0.95, composite_score / 100)
            elif composite_score >= 60:
                signal = 'BUY'
                confidence = min(0.85, composite_score / 100)
            elif composite_score >= 40:
                signal = 'HOLD'
                confidence = 0.5
            elif composite_score >= 25:
                signal = 'SELL'
                confidence = min(0.75, (100 - composite_score) / 100)
            else:
                signal = 'STRONG_SELL'
                confidence = min(0.90, (100 - composite_score) / 100)
            
            signals[symbol] = {
                'signal': signal,
                'confidence': confidence,
                'composite_score': composite_score,
                'technical_score': features['technical_score'],
                'momentum_score': features['momentum_score'],
                'current_price': features['current_price'],
                'risk_adjusted_return': features['risk_adjusted_return']
            }
        
        print(f"✅ Generated signals for {len(signals)} stocks")
        return signals
    
    def optimize_portfolio_allocation(self, signals, max_positions=25):
        """Optimize portfolio allocation across selected stocks"""
        print(f"📊 Optimizing portfolio allocation (max {max_positions} positions)...")
        
        # Filter for buy signals
        buy_candidates = {
            symbol: data for symbol, data in signals.items() 
            if data['signal'] in ['BUY', 'STRONG_BUY'] and data['confidence'] > 0.6
        }
        
        if not buy_candidates:
            print("⚠️ No strong buy candidates found")
            return {}
        
        # Sort by composite score and confidence
        sorted_candidates = sorted(
            buy_candidates.items(),
            key=lambda x: x[1]['composite_score'] * x[1]['confidence'],
            reverse=True
        )
        
        # Select top candidates
        selected_stocks = dict(sorted_candidates[:max_positions])
        
        # Equal weight allocation with adjustments for confidence
        base_weight = 1.0 / len(selected_stocks)
        total_confidence = sum(data['confidence'] for data in selected_stocks.values())
        
        allocations = {}
        for symbol, data in selected_stocks.items():
            # Adjust weight based on confidence
            confidence_adjustment = data['confidence'] / (total_confidence / len(selected_stocks))
            weight = base_weight * confidence_adjustment
            allocations[symbol] = min(weight, 0.15)  # Cap at 15% per stock
        
        # Normalize weights
        total_weight = sum(allocations.values())
        if total_weight > 0:
            allocations = {symbol: weight/total_weight for symbol, weight in allocations.items()}
        
        print(f"✅ Portfolio optimized: {len(allocations)} positions")
        return allocations
    
    def execute_enhanced_paper_trading(self, allocations, signals):
        """Execute enhanced paper trading with position sizing"""
        print("💼 Executing enhanced paper trading...")
        
        executed_trades = []
        total_allocated = 0
        
        for symbol, allocation in allocations.items():
            if allocation > 0.01:  # Minimum 1% allocation
                try:
                    signal_data = signals[symbol]
                    current_price = signal_data['current_price']
                    
                    # Calculate position size
                    position_value = self.current_capital * allocation
                    shares = int(position_value / current_price)
                    
                    if shares > 0:
                        trade_value = shares * current_price
                        
                        executed_trades.append({
                            'symbol': symbol,
                            'action': 'BUY',
                            'shares': shares,
                            'price': current_price,
                            'value': trade_value,
                            'allocation': allocation,
                            'confidence': signal_data['confidence'],
                            'composite_score': signal_data['composite_score']
                        })
                        
                        total_allocated += trade_value
                        print(f"📈 BUY {shares} {symbol} @ ₹{current_price:.2f} = ₹{trade_value:,.0f} ({allocation:.1%})")
                
                except Exception as e:
                    print(f"⚠️ Trade failed for {symbol}: {e}")
        
        print(f"✅ Enhanced paper trading: {len(executed_trades)} trades, ₹{total_allocated:,.0f} allocated")
        return executed_trades
    
    def generate_comprehensive_report(self, signals, allocations, trades):
        """Generate comprehensive performance report"""
        print("📊 Generating comprehensive report...")
        
        # Signal analysis
        signal_summary = {}
        for signal_type in ['STRONG_BUY', 'BUY', 'HOLD', 'SELL', 'STRONG_SELL']:
            count = sum(1 for s in signals.values() if s['signal'] == signal_type)
            signal_summary[signal_type] = count
        
        # Sector analysis
        sector_allocation = {}
        for symbol, allocation in allocations.items():
            for sector, stocks in SECTOR_MAPPING.items():
                if symbol in stocks:
                    sector_allocation[sector] = sector_allocation.get(sector, 0) + allocation
                    break
        
        # Top performers
        top_stocks = sorted(
            signals.items(),
            key=lambda x: x[1]['composite_score'],
            reverse=True
        )[:10]
        
        report = {
            'timestamp': datetime.now(),
            'universe_size': len(self.stock_universe),
            'signals_generated': len(signals),
            'signal_summary': signal_summary,
            'portfolio_positions': len(allocations),
            'total_trades': len(trades),
            'total_invested': sum(trade['value'] for trade in trades),
            'cash_remaining': self.current_capital - sum(trade['value'] for trade in trades),
            'sector_allocation': sector_allocation,
            'top_stocks': [(symbol, data['composite_score']) for symbol, data in top_stocks],
            'avg_confidence': np.mean([s['confidence'] for s in signals.values()]),
            'risk_profile': self.risk_tolerance,
            'diversification_score': len(sector_allocation)
        }
        
        print(f"✅ Comprehensive report generated")
        return report
    
    def run_complete_analysis(self):
        """Run complete enhanced analysis"""
        print("🌅 Starting Enhanced Indian Stock Analysis")
        print("=" * 60)
        
        try:
            # Step 1: Fetch market data
            market_data = self.fetch_market_data()
            
            if len(market_data) < 10:
                print("❌ Insufficient market data")
                return None
            
            # Step 2: Calculate enhanced features
            enhanced_data = self.calculate_enhanced_features(market_data)
            
            # Step 3: Generate signals
            signals = self.generate_enhanced_signals(enhanced_data)
            
            # Step 4: Optimize portfolio
            allocations = self.optimize_portfolio_allocation(signals)
            
            # Step 5: Execute paper trading
            trades = self.execute_enhanced_paper_trading(allocations, signals)
            
            # Step 6: Generate report
            report = self.generate_comprehensive_report(signals, allocations, trades)
            
            # Save results
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            results = {
                'signals': signals,
                'allocations': allocations,
                'trades': trades,
                'report': report
            }
            
            with open(f'results/enhanced_indian_analysis_{timestamp}.json', 'w') as f:
                json.dump(results, f, indent=2, default=str)
            
            print("=" * 60)
            print("✅ Enhanced analysis completed successfully!")
            
            return results
            
        except Exception as e:
            print(f"❌ Error in enhanced analysis: {e}")
            return None

    def analyze_expanded_universe(self):
        """Quick analysis of expanded universe"""
        print(f"📊 Analyzing {len(self.stock_universe)} stocks...")
        
        # Sample analysis
        sample_data = {}
        successful = 0
        
        for symbol in self.stock_universe[:10]:  # Sample first 10
            try:
                stock = yf.Ticker(symbol)
                info = stock.history(period="1mo")
                if not info.empty:
                    current_price = info['Close'].iloc[-1]
                    change_1m = ((current_price - info['Close'].iloc[0]) / info['Close'].iloc[0]) * 100
                    
                    sample_data[symbol] = {
                        'price': current_price,
                        'change_1m': change_1m,
                        'volume': info['Volume'].mean()
                    }
                    successful += 1
            except:
                continue
        
        print(f"✅ Sample analysis: {successful}/10 stocks processed")
        return sample_data

if __name__ == "__main__":
    print("🚀 Testing Enhanced Unified Indian Stock System")
    print("=" * 60)
    
    # Test with different configurations
    configs = [
        {'risk': 'conservative', 'size': 30},
        {'risk': 'moderate', 'size': 50},
        {'risk': 'aggressive', 'size': 40}
    ]
    
    for config in configs:
        print(f"\n🧪 Testing {config['risk']} profile with {config['size']} stocks")
        
        system = EnhancedUnifiedIndianSystem(
            initial_capital=2000000,  # ₹20 lakh
            risk_tolerance=config['risk'],
            universe_size=config['size']
        )
        
        # Quick demo of universe
        print(f"Sample stocks: {system.stock_universe[:5]}")
        print(f"Sector distribution: {system._get_sector_distribution()}")
        break  # Run only first config for demo
    
    print(f"\n🎉 Enhanced system ready for {len(system.stock_universe)} Indian stocks!")
    
    # Quick analysis
    sample_data = system.analyze_expanded_universe()
    print(f"\n📊 Sample Data:")
    for symbol, data in sample_data.items():
        print(f"  {symbol}: ₹{data['price']:.2f} ({data['change_1m']:+.1f}%)") 