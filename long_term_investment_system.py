#!/usr/bin/env python3
"""
Long-Term Investment System
===========================
Advanced algorithms and backtesting designed specifically for long-term investing:
- Buy-and-hold strategies
- Fundamental analysis integration
- Multi-year trend identification
- Position sizing for long-term holds
- Focus on CAGR over short-term gains
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import yfinance as yf
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error
import joblib
import os
import warnings
warnings.filterwarnings('ignore')

class LongTermInvestmentSystem:
    """Comprehensive Long-Term Investment System"""
    
    def __init__(self, symbol, start_date='2015-01-01', end_date=None, initial_capital=100000):
        self.symbol = symbol
        # Ensure dates are strings for yfinance compatibility
        if hasattr(start_date, 'strftime'):  # If it's a date/datetime object
            self.start_date = start_date.strftime('%Y-%m-%d')
        else:
            self.start_date = start_date
        
        if end_date is None:
            self.end_date = datetime.now().strftime('%Y-%m-%d')
        elif hasattr(end_date, 'strftime'):  # If it's a date/datetime object
            self.end_date = end_date.strftime('%Y-%m-%d')
        else:
            self.end_date = end_date
            
        self.initial_capital = initial_capital
        self.models = {}
        self.scalers = {}
        self.results = {}
        
    def fetch_long_term_data(self):
        """Fetch comprehensive data for long-term analysis"""
        print(f"📊 Fetching long-term data for {self.symbol}...")
        
        # Fetch main data
        ticker = yf.Ticker(self.symbol)
        df = ticker.history(start=self.start_date, end=self.end_date, interval='1d')
        
        if df.empty:
            raise ValueError(f"No data available for {self.symbol}")
        
        df = df.reset_index()
        df['Date'] = pd.to_datetime(df['Date'])
        
        # Get fundamental data
        try:
            info = ticker.info
            self.fundamental_data = {
                'market_cap': info.get('marketCap', 0),
                'pe_ratio': info.get('trailingPE', 0),
                'pb_ratio': info.get('priceToBook', 0),
                'debt_to_equity': info.get('debtToEquity', 0),
                'roe': info.get('returnOnEquity', 0),
                'dividend_yield': info.get('dividendYield', 0),
                'sector': info.get('sector', 'Unknown')
            }
        except:
            self.fundamental_data = {}
        
        print(f"✅ Fetched {len(df)} days of data")
        return df
    
    def create_long_term_features(self, df):
        """Create features specifically designed for long-term investing"""
        print("🔧 Creating long-term investment features...")
        
        # Long-term moving averages
        df['MA_50'] = df['Close'].rolling(window=50).mean()
        df['MA_100'] = df['Close'].rolling(window=100).mean()
        df['MA_200'] = df['Close'].rolling(window=200).mean()
        df['MA_365'] = df['Close'].rolling(window=365).mean()
        
        # Long-term trend indicators
        df['Trend_50_200'] = (df['MA_50'] - df['MA_200']) / df['MA_200']
        df['Trend_100_365'] = (df['MA_100'] - df['MA_365']) / df['MA_365']
        df['Price_vs_MA200'] = (df['Close'] - df['MA_200']) / df['MA_200']
        
        # Long-term momentum (quarterly and yearly)
        df['Momentum_3M'] = df['Close'].pct_change(periods=63)  # ~3 months
        df['Momentum_6M'] = df['Close'].pct_change(periods=126)  # ~6 months
        df['Momentum_1Y'] = df['Close'].pct_change(periods=252)  # ~1 year
        df['Momentum_2Y'] = df['Close'].pct_change(periods=504)  # ~2 years
        
        # Volatility (long-term)
        df['Volatility_30D'] = df['Close'].rolling(window=30).std()
        df['Volatility_90D'] = df['Close'].rolling(window=90).std()
        df['Volatility_365D'] = df['Close'].rolling(window=365).std()
        
        # Support and resistance levels
        df['Support_1Y'] = df['Low'].rolling(window=252).min()
        df['Resistance_1Y'] = df['High'].rolling(window=252).max()
        df['Position_in_Range'] = (df['Close'] - df['Support_1Y']) / (df['Resistance_1Y'] - df['Support_1Y'])
        
        # Seasonal patterns
        df['Month'] = df['Date'].dt.month
        df['Quarter'] = df['Date'].dt.quarter
        df['Year'] = df['Date'].dt.year
        df['Day_of_Year'] = df['Date'].dt.dayofyear
        
        # Economic cycle indicators
        df['Long_Term_Growth'] = df['Close'].rolling(window=252).apply(lambda x: np.polyfit(range(len(x)), x, 1)[0])
        df['Price_Strength'] = df['Close'] / df['Close'].rolling(window=252).max()
        df['Drawdown_1Y'] = (df['Close'] - df['Close'].rolling(window=252).max()) / df['Close'].rolling(window=252).max()
        
        # Volume analysis (long-term)
        df['Volume_MA_50'] = df['Volume'].rolling(window=50).mean()
        df['Volume_Ratio'] = df['Volume'] / df['Volume_MA_50']
        df['Volume_Trend'] = df['Volume'].rolling(window=100).apply(lambda x: np.polyfit(range(len(x)), x, 1)[0])
        
        # Long-term price targets
        df['Target_1M'] = df['Close'].shift(-21)  # 1 month ahead
        df['Target_3M'] = df['Close'].shift(-63)  # 3 months ahead
        df['Target_6M'] = df['Close'].shift(-126)  # 6 months ahead
        df['Target_1Y'] = df['Close'].shift(-252)  # 1 year ahead
        
        # Clean data
        df = df.dropna()
        
        print(f"✅ Created {len([col for col in df.columns if col not in ['Date', 'Open', 'High', 'Low', 'Close', 'Volume']])} long-term features")
        return df
    
    def get_long_term_signals(self, df):
        """Generate long-term investment signals"""
        print("🎯 Generating long-term investment signals...")
        
        signals = {}
        latest = df.iloc[-1]
        
        # Trend Analysis
        trend_signals = []
        if latest['MA_50'] > latest['MA_200']:
            trend_signals.append('BULLISH_TREND')
        if latest['Trend_50_200'] > 0.05:  # 5% above 200-day MA
            trend_signals.append('STRONG_UPTREND')
        if latest['Price_vs_MA200'] > 0.1:  # 10% above 200-day MA
            trend_signals.append('MOMENTUM_BUY')
        
        # Long-term momentum
        momentum_signals = []
        if latest['Momentum_1Y'] > 0.2:  # 20% annual return
            momentum_signals.append('STRONG_ANNUAL_GROWTH')
        if latest['Momentum_6M'] > 0.1:  # 10% semi-annual return
            momentum_signals.append('POSITIVE_MOMENTUM')
        
        # Position in range
        position_signals = []
        if latest['Position_in_Range'] < 0.3:  # In lower 30% of range
            position_signals.append('VALUE_OPPORTUNITY')
        elif latest['Position_in_Range'] > 0.7:  # In upper 30% of range
            position_signals.append('RESISTANCE_AREA')
        
        # Overall signal
        buy_signals = len(trend_signals) + len([s for s in momentum_signals if 'GROWTH' in s or 'POSITIVE' in s])
        sell_signals = len([s for s in position_signals if 'RESISTANCE' in s])
        
        if buy_signals >= 2:
            overall_signal = 'BUY'
            confidence = min(0.9, 0.5 + (buy_signals * 0.1))
        elif sell_signals >= 1:
            overall_signal = 'SELL'
            confidence = 0.6
        else:
            overall_signal = 'HOLD'
            confidence = 0.5
        
        signals = {
            'Overall': {'signal': overall_signal, 'confidence': confidence},
            'Trend': trend_signals,
            'Momentum': momentum_signals,
            'Position': position_signals,
            'Latest_Data': {
                'Price': latest['Close'],
                'MA_200': latest['MA_200'],
                'Annual_Return': latest['Momentum_1Y'] * 100,
                'Position_in_Range': latest['Position_in_Range']
            }
        }
        
        return signals
    
    def create_long_term_model(self, df):
        """Create ML model for long-term predictions"""
        print("🤖 Training long-term prediction model...")
        
        # Features for long-term model
        feature_columns = [
            'MA_50', 'MA_100', 'MA_200', 'MA_365',
            'Trend_50_200', 'Trend_100_365', 'Price_vs_MA200',
            'Momentum_3M', 'Momentum_6M', 'Momentum_1Y',
            'Volatility_30D', 'Volatility_90D', 'Volatility_365D',
            'Position_in_Range', 'Long_Term_Growth', 'Price_Strength',
            'Volume_Ratio', 'Volume_Trend', 'Month', 'Quarter'
        ]
        
        # Prepare data
        X = df[feature_columns].copy()
        y_1y = df['Target_1Y'].copy()
        
        # Remove NaN values
        valid_indices = ~(X.isnull().any(axis=1) | y_1y.isnull())
        X = X[valid_indices]
        y_1y = y_1y[valid_indices]
        
        if len(X) < 50:
            print("⚠️ Insufficient data for model training")
            return None, None
        
        # Split data (80% train, 20% test)
        split_idx = int(len(X) * 0.8)
        X_train, X_test = X[:split_idx], X[split_idx:]
        y_train, y_test = y_1y[:split_idx], y_1y[split_idx:]
        
        # Scale features
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)
        
        # Train ensemble model
        models = {
            'RandomForest': RandomForestRegressor(n_estimators=200, max_depth=10, random_state=42),
            'GradientBoosting': GradientBoostingRegressor(n_estimators=200, max_depth=6, random_state=42)
        }
        
        best_model = None
        best_score = float('inf')
        
        for name, model in models.items():
            model.fit(X_train_scaled, y_train)
            predictions = model.predict(X_test_scaled)
            mse = mean_squared_error(y_test, predictions)
            
            if mse < best_score:
                best_score = mse
                best_model = model
        
        # Model performance
        predictions = best_model.predict(X_test_scaled)
        mae = mean_absolute_error(y_test, predictions)
        mape = np.mean(np.abs((y_test - predictions) / y_test)) * 100
        
        print(f"✅ Model trained - MAE: ${mae:.2f}, MAPE: {mape:.2f}%")
        
        return best_model, scaler
    
    def long_term_backtesting(self, df, model, scaler):
        """Comprehensive long-term backtesting"""
        print("📈 Running long-term backtesting...")
        
        # Backtesting parameters
        initial_capital = self.initial_capital
        capital = initial_capital
        shares = 0
        transactions = []
        portfolio_values = []
        
        # Feature columns
        feature_columns = [
            'MA_50', 'MA_100', 'MA_200', 'MA_365',
            'Trend_50_200', 'Trend_100_365', 'Price_vs_MA200',
            'Momentum_3M', 'Momentum_6M', 'Momentum_1Y',
            'Volatility_30D', 'Volatility_90D', 'Volatility_365D',
            'Position_in_Range', 'Long_Term_Growth', 'Price_Strength',
            'Volume_Ratio', 'Volume_Trend', 'Month', 'Quarter'
        ]
        
        # Long-term rebalancing (quarterly)
        rebalance_frequency = 63  # ~3 months
        
        for i in range(365, len(df) - 252, rebalance_frequency):  # Start after 1 year, leave 1 year buffer
            current_row = df.iloc[i]
            current_date = current_row['Date']
            current_price = current_row['Close']
            
            # Get features
            try:
                features = current_row[feature_columns].values.reshape(1, -1)
                features = features.astype(float)
                
                # Skip if any NaN values
                if np.isnan(features).any():
                    continue
            except (ValueError, TypeError):
                continue
            
            # Scale features
            features_scaled = scaler.transform(features)
            
            # Get prediction (1 year ahead)
            try:
                if model is not None:
                    predicted_price = model.predict(features_scaled)[0]
                    expected_return = (predicted_price - current_price) / current_price
                else:
                    continue
            except:
                continue
            
            # Generate signals
            signals = self.get_long_term_signals(df.iloc[:i+1])
            signal = signals['Overall']['signal']
            confidence = signals['Overall']['confidence']
            
            # Long-term investment decisions
            action = 'HOLD'
            
            if signal == 'BUY' and expected_return > 0.15 and confidence > 0.7:  # Expect 15%+ return
                if shares == 0:  # Buy
                    shares = capital / current_price
                    capital = 0
                    action = 'BUY'
                    
            elif signal == 'SELL' and (expected_return < -0.1 or confidence < 0.4):  # Expect -10% or low confidence
                if shares > 0:  # Sell
                    capital = shares * current_price
                    shares = 0
                    action = 'SELL'
            
            # Record transaction
            if action in ['BUY', 'SELL']:
                transactions.append({
                    'Date': current_date,
                    'Action': action,
                    'Price': current_price,
                    'Shares': shares if action == 'BUY' else 0,
                    'Capital': capital,
                    'Expected_Return': expected_return,
                    'Confidence': confidence
                })
            
            # Calculate portfolio value
            portfolio_value = capital + (shares * current_price)
            portfolio_values.append({
                'Date': current_date,
                'Portfolio_Value': portfolio_value,
                'Price': current_price,
                'Shares': shares,
                'Capital': capital
            })
        
        # Final portfolio value
        final_price = df.iloc[-1]['Close']
        final_value = capital + (shares * final_price)
        
        # Calculate returns
        total_return = (final_value - initial_capital) / initial_capital * 100
        years = (pd.to_datetime(df.iloc[-1]['Date']) - pd.to_datetime(df.iloc[365]['Date'])).days / 365.25
        cagr = ((final_value / initial_capital) ** (1/years) - 1) * 100
        
        # Buy and hold comparison
        buy_hold_shares = initial_capital / df.iloc[365]['Close']
        buy_hold_value = buy_hold_shares * final_price
        buy_hold_return = (buy_hold_value - initial_capital) / initial_capital * 100
        buy_hold_cagr = ((buy_hold_value / initial_capital) ** (1/years) - 1) * 100
        
        # Calculate metrics
        portfolio_df = pd.DataFrame(portfolio_values)
        if len(portfolio_df) > 0:
            portfolio_df['Daily_Return'] = portfolio_df['Portfolio_Value'].pct_change()
            portfolio_df = portfolio_df.dropna()
            
            if len(portfolio_df) > 0:
                volatility = portfolio_df['Daily_Return'].std() * np.sqrt(252) * 100
                sharpe_ratio = (cagr - 2) / volatility if volatility > 0 else 0  # Assuming 2% risk-free rate
                
                # Maximum drawdown
                rolling_max = portfolio_df['Portfolio_Value'].expanding().max()
                drawdown = (portfolio_df['Portfolio_Value'] - rolling_max) / rolling_max
                max_drawdown = drawdown.min() * 100
            else:
                volatility = 0
                sharpe_ratio = 0
                max_drawdown = 0
        else:
            volatility = 0
            sharpe_ratio = 0
            max_drawdown = 0
        
        results = {
            'strategy_performance': {
                'total_return': total_return,
                'cagr': cagr,
                'volatility': volatility,
                'sharpe_ratio': sharpe_ratio,
                'max_drawdown': max_drawdown,
                'final_value': final_value,
                'total_transactions': len(transactions)
            },
            'buy_hold_performance': {
                'total_return': buy_hold_return,
                'cagr': buy_hold_cagr,
                'final_value': buy_hold_value
            },
            'outperformance': {
                'excess_return': total_return - buy_hold_return,
                'excess_cagr': cagr - buy_hold_cagr
            },
            'transactions': transactions,
            'portfolio_history': portfolio_values
        }
        
        print(f"✅ Backtesting completed")
        print(f"📊 Strategy CAGR: {cagr:.2f}% vs Buy & Hold: {buy_hold_cagr:.2f}%")
        print(f"🎯 Excess CAGR: {cagr - buy_hold_cagr:+.2f}%")
        
        return results
    
    def run_complete_analysis(self):
        """Run complete long-term investment analysis"""
        print("🚀 Starting Complete Long-Term Investment Analysis")
        print("=" * 60)
        
        try:
            # 1. Fetch data
            df = self.fetch_long_term_data()
            
            # 2. Create features
            df = self.create_long_term_features(df)
            
            # 3. Train model
            model, scaler = self.create_long_term_model(df)
            
            if model is None:
                print("❌ Model training failed")
                return None
            
            # 4. Generate current signals
            signals = self.get_long_term_signals(df)
            
            # 5. Run backtesting
            backtest_results = self.long_term_backtesting(df, model, scaler)
            
            # 6. Save results
            self.results = {
                'symbol': self.symbol,
                'analysis_date': datetime.now().isoformat(),
                'data_period': f"{self.start_date} to {self.end_date}",
                'fundamental_data': self.fundamental_data,
                'current_signals': signals,
                'backtest_results': backtest_results,
                'model_type': 'Long-Term Investment System'
            }
            
            # Save to file
            os.makedirs('results', exist_ok=True)
            filename = f"results/{self.symbol}_long_term_analysis_{datetime.now().strftime('%Y%m%d_%H%M%S')}.pkl"
            joblib.dump(self.results, filename)
            
            print(f"💾 Results saved to {filename}")
            return self.results
            
        except Exception as e:
            print(f"❌ Error in analysis: {str(e)}")
            return None
    
    def print_summary(self):
        """Print detailed summary of long-term analysis"""
        if not self.results:
            print("❌ No results available")
            return
        
        print("\n" + "="*60)
        print(f"📊 LONG-TERM INVESTMENT ANALYSIS: {self.symbol}")
        print("="*60)
        
        # Current signals
        signals = self.results['current_signals']
        print(f"\n🎯 CURRENT INVESTMENT SIGNALS:")
        print(f"Overall Signal: {signals['Overall']['signal']} (Confidence: {signals['Overall']['confidence']:.1%})")
        print(f"Current Price: ${signals['Latest_Data']['Price']:.2f}")
        print(f"200-Day MA: ${signals['Latest_Data']['MA_200']:.2f}")
        print(f"Annual Return: {signals['Latest_Data']['Annual_Return']:+.1f}%")
        
        # Backtest results
        strategy = self.results['backtest_results']['strategy_performance']
        buy_hold = self.results['backtest_results']['buy_hold_performance']
        
        print(f"\n📈 LONG-TERM PERFORMANCE COMPARISON:")
        print(f"Strategy CAGR: {strategy['cagr']:.2f}%")
        print(f"Buy & Hold CAGR: {buy_hold['cagr']:.2f}%")
        print(f"Excess CAGR: {strategy['cagr'] - buy_hold['cagr']:+.2f}%")
        print(f"Strategy Sharpe Ratio: {strategy['sharpe_ratio']:.2f}")
        print(f"Max Drawdown: {strategy['max_drawdown']:.1f}%")
        print(f"Total Transactions: {strategy['total_transactions']}")
        
        # Investment recommendation
        overall_signal = signals['Overall']['signal']
        confidence = signals['Overall']['confidence']
        
        print(f"\n💡 LONG-TERM INVESTMENT RECOMMENDATION:")
        if overall_signal == 'BUY' and confidence > 0.7:
            print("🟢 STRONG BUY - Excellent long-term opportunity")
        elif overall_signal == 'BUY':
            print("🟢 BUY - Good long-term potential")
        elif overall_signal == 'SELL':
            print("🔴 SELL - Consider reducing position")
        else:
            print("🟡 HOLD - Monitor for better entry/exit points")
        
        print("="*60)

def main():
    """Main function to run long-term analysis"""
    import sys
    
    if len(sys.argv) > 1:
        symbol = sys.argv[1]
    else:
        symbol = input("Enter stock symbol (e.g., AAPL, NIFTYBEES.NS): ").upper()
    
    print(f"🎯 Running Long-Term Investment Analysis for {symbol}")
    
    # Initialize system
    lt_system = LongTermInvestmentSystem(
        symbol=symbol,
        start_date='2015-01-01',  # 9+ years of data
        initial_capital=100000
    )
    
    # Run analysis
    results = lt_system.run_complete_analysis()
    
    if results:
        # Print summary
        lt_system.print_summary()
        
        print(f"\n🎉 Long-term investment analysis completed successfully!")
        print(f"📁 Detailed results saved in results/ directory")
    else:
        print("❌ Analysis failed. Please check the symbol and try again.")

if __name__ == "__main__":
    main() 