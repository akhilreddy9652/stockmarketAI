#!/usr/bin/env python3
"""
Comprehensive Indian Stock Analysis System
==========================================
Analyze 100+ Indian stocks with sector diversification and risk management
"""

import numpy as np
import pandas as pd
import yfinance as yf
from datetime import datetime, timedelta
import json
import warnings
warnings.filterwarnings('ignore')

from expanded_indian_stock_universe import (
    get_comprehensive_indian_universe,
    SECTOR_MAPPING,
    RISK_CATEGORIES,
    NIFTY_50,
    NIFTY_NEXT_50,
    IT_TECHNOLOGY,
    PHARMA_HEALTHCARE,
    FINTECH_BANKING,
    MANUFACTURING_AUTO,
    ENERGY_UTILITIES
)

class ComprehensiveIndianAnalysis:
    """
    Comprehensive Indian Stock Analysis System
    Handles 100+ stocks with advanced portfolio optimization
    """
    
    def __init__(self, analysis_type='COMPREHENSIVE', max_stocks=100):
        self.analysis_type = analysis_type
        self.max_stocks = max_stocks
        
        # Initialize comprehensive universe
        self.stock_universe = self._build_comprehensive_universe()
        self.sector_data = {}
        self.market_data = {}
        self.analysis_results = {}
        
        print(f"🚀 Comprehensive Indian Analysis System")
        print(f"📊 Analysis Type: {analysis_type}")
        print(f"🇮🇳 Stock Universe: {len(self.stock_universe)} Indian stocks")
        print(f"🏭 Sectors: {len(self._get_sectors_covered())} covered")
        
    def _build_comprehensive_universe(self):
        """Build comprehensive stock universe"""
        universe = []
        
        if self.analysis_type == 'COMPREHENSIVE':
            # Mix of all categories
            universe.extend(NIFTY_50)
            universe.extend(NIFTY_NEXT_50[:30])
            universe.extend(IT_TECHNOLOGY[:15])
            universe.extend(PHARMA_HEALTHCARE[:12])
            universe.extend(FINTECH_BANKING[:10])
            universe.extend(MANUFACTURING_AUTO[:8])
            universe.extend(ENERGY_UTILITIES[:8])
            
        elif self.analysis_type == 'GROWTH':
            universe.extend(IT_TECHNOLOGY)
            universe.extend(PHARMA_HEALTHCARE[:10])
            universe.extend(NIFTY_50[:30])
            
        elif self.analysis_type == 'VALUE':
            universe.extend(ENERGY_UTILITIES)
            universe.extend(MANUFACTURING_AUTO)
            universe.extend(FINTECH_BANKING[:8])
            
        else:
            universe = NIFTY_50 + NIFTY_NEXT_50
        
        # Remove duplicates and limit
        universe = list(dict.fromkeys(universe))
        return universe[:self.max_stocks]
    
    def _get_sectors_covered(self):
        """Get list of sectors covered"""
        sectors = set()
        for stock in self.stock_universe:
            for sector, stocks in SECTOR_MAPPING.items():
                if stock in stocks:
                    sectors.add(sector)
        return list(sectors)
    
    def fetch_comprehensive_data(self, days=365):
        """Fetch data for all stocks in universe"""
        print(f"📊 Fetching data for {len(self.stock_universe)} stocks...")
        
        successful_fetches = 0
        failed_fetches = []
        
        for i, symbol in enumerate(self.stock_universe, 1):
            try:
                print(f"  Progress: {i}/{len(self.stock_universe)} - {symbol}", end='\r')
                
                # Fetch stock data
                stock = yf.Ticker(symbol)
                end_date = datetime.now()
                start_date = end_date - timedelta(days=days)
                
                data = stock.history(
                    start=start_date.strftime('%Y-%m-%d'),
                    end=end_date.strftime('%Y-%m-%d')
                )
                
                if not data.empty and len(data) > 100:  # Minimum data requirement
                    # Calculate additional metrics
                    returns = data['Close'].pct_change().dropna()
                    
                    self.market_data[symbol] = {
                        'data': data,
                        'current_price': float(data['Close'].iloc[-1]),
                        'returns_1d': float(((data['Close'].iloc[-1] / data['Close'].iloc[-2]) - 1) * 100),
                        'returns_1w': float(((data['Close'].iloc[-1] / data['Close'].iloc[-7]) - 1) * 100),
                        'returns_1m': float(((data['Close'].iloc[-1] / data['Close'].iloc[-21]) - 1) * 100),
                        'returns_3m': float(((data['Close'].iloc[-1] / data['Close'].iloc[-63]) - 1) * 100),
                        'volatility': float(returns.std() * np.sqrt(252)),
                        'sharpe_ratio': float((returns.mean() * 252) / (returns.std() * np.sqrt(252))) if returns.std() > 0 else 0,
                        'max_drawdown': float(self._calculate_max_drawdown(data['Close'])),
                        'volume_trend': float(data['Volume'].tail(10).mean() / data['Volume'].mean()),
                        'market_cap_proxy': float(data['Close'].iloc[-1] * data['Volume'].mean())
                    }
                    successful_fetches += 1
                    
                else:
                    failed_fetches.append(symbol)
            
            except Exception as e:
                failed_fetches.append(symbol)
                continue
        
        print(f"\n✅ Data fetched: {successful_fetches}/{len(self.stock_universe)} stocks")
        if failed_fetches:
            print(f"⚠️ Failed: {len(failed_fetches)} stocks")
        
        return successful_fetches
    
    def _calculate_max_drawdown(self, prices):
        """Calculate maximum drawdown"""
        peak = prices.expanding().max()
        drawdown = (prices - peak) / peak
        return drawdown.min()
    
    def analyze_sectors(self):
        """Analyze performance by sector"""
        print("🏭 Analyzing sector performance...")
        
        sector_performance = {}
        
        for sector, stocks in SECTOR_MAPPING.items():
            sector_stocks = [s for s in stocks if s in self.market_data]
            
            if sector_stocks:
                returns_1m = []
                returns_3m = []
                volatilities = []
                sharpe_ratios = []
                
                for stock in sector_stocks:
                    data = self.market_data[stock]
                    returns_1m.append(data['returns_1m'])
                    returns_3m.append(data['returns_3m'])
                    volatilities.append(data['volatility'])
                    sharpe_ratios.append(data['sharpe_ratio'])
                
                sector_performance[sector] = {
                    'stock_count': len(sector_stocks),
                    'avg_return_1m': np.mean(returns_1m),
                    'avg_return_3m': np.mean(returns_3m),
                    'avg_volatility': np.mean(volatilities),
                    'avg_sharpe': np.mean(sharpe_ratios),
                    'best_performer': max(sector_stocks, key=lambda s: self.market_data[s]['returns_1m']),
                    'worst_performer': min(sector_stocks, key=lambda s: self.market_data[s]['returns_1m']),
                    'stocks': sector_stocks[:5]  # Top 5 stocks in sector
                }
        
        self.sector_data = sector_performance
        print(f"✅ Sector analysis: {len(sector_performance)} sectors")
        return sector_performance
    
    def generate_stock_rankings(self):
        """Generate comprehensive stock rankings"""
        print("📈 Generating stock rankings...")
        
        rankings = {}
        
        for symbol, data in self.market_data.items():
            # Composite score calculation
            momentum_score = (data['returns_1w'] * 0.3 + 
                            data['returns_1m'] * 0.4 + 
                            data['returns_3m'] * 0.3)
            
            risk_score = max(0, 100 - (data['volatility'] * 100))
            sharpe_score = min(100, max(0, (data['sharpe_ratio'] + 1) * 50))
            volume_score = min(100, data['volume_trend'] * 50)
            
            composite_score = (momentum_score * 0.4 + 
                             risk_score * 0.2 + 
                             sharpe_score * 0.3 + 
                             volume_score * 0.1)
            
            rankings[symbol] = {
                'composite_score': composite_score,
                'momentum_score': momentum_score,
                'risk_score': risk_score,
                'sharpe_score': sharpe_score,
                'current_price': data['current_price'],
                'returns_1m': data['returns_1m'],
                'volatility': data['volatility'],
                'sharpe_ratio': data['sharpe_ratio']
            }
        
        # Sort by composite score
        sorted_rankings = dict(sorted(rankings.items(), 
                                    key=lambda x: x[1]['composite_score'], 
                                    reverse=True))
        
        print(f"✅ Rankings generated for {len(rankings)} stocks")
        return sorted_rankings
    
    def create_optimized_portfolios(self, rankings):
        """Create optimized portfolios for different risk profiles"""
        print("💼 Creating optimized portfolios...")
        
        portfolios = {}
        
        # Get top performers
        top_stocks = list(rankings.keys())[:50]
        
        # Conservative Portfolio (Low Risk)
        conservative_stocks = [s for s in top_stocks if rankings[s]['risk_score'] > 70][:20]
        conservative_portfolio = self._create_equal_weight_portfolio(conservative_stocks, 'conservative')
        
        # Moderate Portfolio (Balanced)
        moderate_stocks = [s for s in top_stocks if 50 < rankings[s]['risk_score'] <= 80][:25]
        moderate_portfolio = self._create_equal_weight_portfolio(moderate_stocks, 'moderate')
        
        # Aggressive Portfolio (High Growth)
        aggressive_stocks = [s for s in top_stocks if rankings[s]['momentum_score'] > 10][:30]
        aggressive_portfolio = self._create_equal_weight_portfolio(aggressive_stocks, 'aggressive')
        
        portfolios = {
            'conservative': conservative_portfolio,
            'moderate': moderate_portfolio,
            'aggressive': aggressive_portfolio
        }
        
        print(f"✅ Created 3 optimized portfolios")
        return portfolios
    
    def _create_equal_weight_portfolio(self, stocks, risk_profile):
        """Create equal weight portfolio"""
        if not stocks:
            return {'stocks': [], 'total_value': 0, 'expected_return': 0, 'risk': 0}
        
        weight_per_stock = 1.0 / len(stocks)
        total_return = 0
        total_risk = 0
        
        portfolio_stocks = []
        for stock in stocks:
            data = self.market_data[stock]
            stock_data = {
                'symbol': stock,
                'weight': weight_per_stock,
                'current_price': data['current_price'],
                'expected_return': data['returns_1m'],
                'risk': data['volatility']
            }
            portfolio_stocks.append(stock_data)
            total_return += data['returns_1m'] * weight_per_stock
            total_risk += data['volatility'] * weight_per_stock
        
        return {
            'stocks': portfolio_stocks,
            'stock_count': len(stocks),
            'expected_return': total_return,
            'risk': total_risk,
            'sharpe_estimate': total_return / total_risk if total_risk > 0 else 0,
            'risk_profile': risk_profile
        }
    
    def generate_trading_signals(self, rankings):
        """Generate trading signals for top stocks"""
        print("🎯 Generating trading signals...")
        
        signals = {}
        
        for symbol, rank_data in list(rankings.items())[:30]:  # Top 30 stocks
            score = rank_data['composite_score']
            
            if score >= 70:
                signal = 'STRONG_BUY'
                confidence = min(0.95, score / 100)
            elif score >= 50:
                signal = 'BUY'
                confidence = min(0.80, score / 100)
            elif score >= 30:
                signal = 'HOLD'
                confidence = 0.5
            elif score >= 15:
                signal = 'SELL'
                confidence = min(0.75, (100 - score) / 100)
            else:
                signal = 'STRONG_SELL'
                confidence = min(0.90, (100 - score) / 100)
            
            signals[symbol] = {
                'signal': signal,
                'confidence': confidence,
                'composite_score': score,
                'current_price': rank_data['current_price'],
                'returns_1m': rank_data['returns_1m'],
                'recommendation': self._get_recommendation(signal, rank_data)
            }
        
        print(f"✅ Generated signals for {len(signals)} stocks")
        return signals
    
    def _get_recommendation(self, signal, data):
        """Get detailed recommendation"""
        if signal in ['STRONG_BUY', 'BUY']:
            return f"Strong fundamentals with {data['returns_1m']:.1f}% monthly return"
        elif signal == 'HOLD':
            return f"Stable performance, monitor for opportunities"
        else:
            return f"Weak performance with {data['returns_1m']:.1f}% monthly return"
    
    def run_comprehensive_analysis(self):
        """Run complete comprehensive analysis"""
        print("🌅 Starting Comprehensive Indian Stock Analysis")
        print("=" * 70)
        
        try:
            # Step 1: Fetch market data
            success_count = self.fetch_comprehensive_data()
            
            if success_count < 20:
                print("❌ Insufficient data for analysis")
                return None
            
            # Step 2: Analyze sectors
            sector_analysis = self.analyze_sectors()
            
            # Step 3: Generate rankings
            rankings = self.generate_stock_rankings()
            
            # Step 4: Create portfolios
            portfolios = self.create_optimized_portfolios(rankings)
            
            # Step 5: Generate signals
            signals = self.generate_trading_signals(rankings)
            
            # Step 6: Compile results
            results = {
                'timestamp': datetime.now(),
                'analysis_type': self.analysis_type,
                'stocks_analyzed': len(self.market_data),
                'sectors_covered': len(sector_analysis),
                'top_10_stocks': list(rankings.keys())[:10],
                'sector_performance': sector_analysis,
                'stock_rankings': dict(list(rankings.items())[:20]),  # Top 20
                'optimized_portfolios': portfolios,
                'trading_signals': signals,
                'summary': self._generate_summary(rankings, sector_analysis, portfolios, signals)
            }
            
            # Save results
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f'results/comprehensive_indian_analysis_{timestamp}.json'
            with open(filename, 'w') as f:
                json.dump(results, f, indent=2, default=str)
            
            print("=" * 70)
            print("✅ Comprehensive analysis completed!")
            print(f"💾 Results saved: {filename}")
            
            return results
            
        except Exception as e:
            print(f"❌ Error in comprehensive analysis: {e}")
            return None
    
    def _generate_summary(self, rankings, sectors, portfolios, signals):
        """Generate analysis summary"""
        buy_signals = sum(1 for s in signals.values() if s['signal'] in ['BUY', 'STRONG_BUY'])
        
        best_sector = max(sectors.items(), key=lambda x: x[1]['avg_return_1m'])
        worst_sector = min(sectors.items(), key=lambda x: x[1]['avg_return_1m'])
        
        return {
            'total_stocks_analyzed': len(self.market_data),
            'buy_signals_generated': buy_signals,
            'signal_ratio': f"{buy_signals}/{len(signals)}",
            'best_performing_sector': {
                'name': best_sector[0],
                'return_1m': best_sector[1]['avg_return_1m']
            },
            'worst_performing_sector': {
                'name': worst_sector[0], 
                'return_1m': worst_sector[1]['avg_return_1m']
            },
            'top_stock': list(rankings.keys())[0],
            'portfolio_expected_returns': {
                'conservative': portfolios['conservative']['expected_return'],
                'moderate': portfolios['moderate']['expected_return'],
                'aggressive': portfolios['aggressive']['expected_return']
            }
        }

if __name__ == "__main__":
    print("🚀 Testing Comprehensive Indian Analysis")
    print("=" * 60)
    
    # Test comprehensive analysis
    analyzer = ComprehensiveIndianAnalysis(
        analysis_type='COMPREHENSIVE',
        max_stocks=75  # Analyze 75 stocks
    )
    
    print(f"\n📊 Sample Universe:")
    for i, stock in enumerate(analyzer.stock_universe[:10], 1):
        print(f"  {i:2d}. {stock}")
    
    print(f"\n🏭 Sectors Covered:")
    for i, sector in enumerate(analyzer._get_sectors_covered(), 1):
        print(f"  {i}. {sector}")
    
    # Run quick demo analysis
    print(f"\n🎉 Ready to analyze {len(analyzer.stock_universe)} Indian stocks!")
    print("Run analyzer.run_comprehensive_analysis() for full analysis") 