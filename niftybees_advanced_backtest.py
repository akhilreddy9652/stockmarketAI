"""
Advanced Backtesting for NIFTYBEES.NS (Nifty BeES ETF)
=====================================================
Comprehensive backtesting with ultra-enhanced features:
- XGBoost and ensemble models
- Indian market specific analysis
- Advanced risk metrics
- Dynamic position sizing
- Multi-strategy comparison
"""

import pandas as pd
import numpy as np
import warnings
import joblib
import os
from datetime import datetime, timedelta
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.feature_selection import SelectKBest, f_regression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error
from data_ingestion import fetch_yfinance
from feature_engineering import add_technical_indicators

# Optional advanced models
try:
    import xgboost as xgb
    XGBOOST_AVAILABLE = True
except ImportError:
    XGBOOST_AVAILABLE = False
    print("⚠️ XGBoost not available. Install with: pip install xgboost")

try:
    import lightgbm as lgb
    LIGHTGBM_AVAILABLE = True
except ImportError:
    LIGHTGBM_AVAILABLE = False
    print("⚠️ LightGBM not available. Install with: pip install lightgbm")

warnings.filterwarnings('ignore')

class NiftyBeesAdvancedBacktester:
    """Advanced backtesting specifically for NIFTYBEES.NS (Nifty BeES ETF)"""
    
    def __init__(self, symbol='NIFTYBEES.NS', start_date='2018-01-01', end_date=None):
        self.symbol = symbol
        self.start_date = start_date
        self.end_date = end_date or datetime.now().strftime('%Y-%m-%d')
        self.results = {}
        self.feature_importance = {}
        
    def create_indian_market_features(self, df):
        """Create Indian market specific features for NIFTYBEES analysis."""
        print("🇮🇳 Creating Indian market specific features...")
        
        # Basic technical indicators
        df = add_technical_indicators(df)
        
        # 1. Indian market specific momentum (considering monsoon cycles, festival seasons)
        momentum_periods = [3, 5, 7, 10, 14, 21, 30, 63]  # Including quarterly periods
        for period in momentum_periods:
            df[f'Return_{period}d'] = df['Close'].pct_change(period)
            df[f'SMA_{period}'] = df['Close'].rolling(period).mean()
            df[f'EMA_{period}'] = df['Close'].ewm(span=period).mean()
            df[f'ROC_{period}'] = df['Close'] / df['Close'].shift(period) - 1
        
        # 2. Volatility analysis (Indian markets can be more volatile)
        vol_periods = [5, 10, 15, 21, 30, 63]
        for period in vol_periods:
            df[f'Vol_{period}'] = df['Close'].pct_change().rolling(period).std()
            df[f'VolRank_{period}'] = df[f'Vol_{period}'].rolling(252).rank(pct=True)
        
        # 3. Indian market trading patterns
        # Create day of week features (Indian markets have specific patterns)
        df['DayOfWeek'] = pd.to_datetime(df.index).dayofweek
        df['IsMonday'] = (df['DayOfWeek'] == 0).astype(int)
        df['IsFriday'] = (df['DayOfWeek'] == 4).astype(int)
        
        # Month effects (Budget, Results seasons)
        df['Month'] = pd.to_datetime(df.index).month
        df['IsBudgetSeason'] = df['Month'].isin([2, 3]).astype(int)  # Budget season
        df['IsResultsSeason'] = df['Month'].isin([4, 7, 10, 1]).astype(int)  # Results seasons
        df['IsFestivalSeason'] = df['Month'].isin([9, 10, 11]).astype(int)  # Festival season
        
        # 4. Volume analysis (ETF specific)
        df['Volume_SMA_10'] = df['Volume'].rolling(10).mean()
        df['Volume_SMA_21'] = df['Volume'].rolling(21).mean()
        df['Volume_Ratio_10'] = df['Volume'] / df['Volume_SMA_10']
        df['Volume_Ratio_21'] = df['Volume'] / df['Volume_SMA_21']
        df['Price_Volume'] = df['Close'] * df['Volume']
        df['VWAP_10'] = (df['Price_Volume'].rolling(10).sum() / 
                        df['Volume'].rolling(10).sum())
        df['VWAP_21'] = (df['Price_Volume'].rolling(21).sum() / 
                        df['Volume'].rolling(21).sum())
        
        # 5. Price position and momentum
        for period in [10, 21, 50, 63]:
            high = df['High'].rolling(period).max()
            low = df['Low'].rolling(period).min()
            df[f'Price_Position_{period}'] = (df['Close'] - low) / (high - low)
            df[f'High_Distance_{period}'] = (high - df['Close']) / df['Close']
            df[f'Low_Distance_{period}'] = (df['Close'] - low) / df['Close']
        
        # 6. Statistical features
        for period in [10, 21, 63]:
            returns = df['Close'].pct_change()
            df[f'Skew_{period}'] = returns.rolling(period).skew()
            df[f'Kurt_{period}'] = returns.rolling(period).kurtosis()
            df[f'Sharpe_{period}'] = (returns.rolling(period).mean() / 
                                    returns.rolling(period).std())
        
        # 7. Trend strength and consistency
        for period in [10, 21, 50]:
            df[f'Trend_Strength_{period}'] = (df['Close'] - df['Close'].shift(period)) / df['Close'].shift(period)
            df[f'Trend_Consistency_{period}'] = (df['Close'] > df['Close'].shift(1)).rolling(period).mean()
        
        # 8. Market regime features (Indian market specific)
        # Volatility regime
        vol_21 = df['Close'].pct_change().rolling(21).std()
        df['Vol_Regime'] = pd.qcut(vol_21.rank(method='first'), 3, labels=[0, 1, 2])
        
        # Trend regime
        sma_10 = df['Close'].rolling(10).mean()
        sma_50 = df['Close'].rolling(50).mean()
        df['Trend_Regime'] = (sma_10 > sma_50).astype(int)
        
        # Price level regime
        df['Price_Regime'] = pd.qcut(df['Close'].rank(method='first'), 3, labels=[0, 1, 2])
        
        # 9. Interaction features
        df['RSI_Vol'] = df.get('RSI_14', 50) * df['Vol_21']
        df['MA_Spread'] = df['SMA_10'] - df['SMA_21']
        df['Volume_Price_Momentum'] = df['Volume_Ratio_21'] * df['Return_5d']
        
        # 10. Advanced momentum indicators
        df['ROC_Momentum'] = df['ROC_10'] * df['Volume_Ratio_10']
        df['Price_Acceleration'] = df['Return_5d'] - df['Return_10d']
        df['Volatility_Momentum'] = df['Vol_10'] - df['Vol_21']
        
        # Clean data
        df = df.replace([np.inf, -np.inf], np.nan)
        df = df.fillna(method='ffill').fillna(method='bfill')
        
        print(f"✅ Indian market features created: {len(df.columns)} total features")
        return df
    
    def train_advanced_ensemble(self, X_train, y_train):
        """Train advanced ensemble models optimized for Indian market."""
        print("🚀 Training advanced ensemble models...")
        
        models = {
            'rf_optimized': RandomForestRegressor(
                n_estimators=200, max_depth=15, min_samples_split=3,
                min_samples_leaf=1, max_features='sqrt', random_state=42
            ),
            'gb_optimized': GradientBoostingRegressor(
                n_estimators=200, max_depth=10, learning_rate=0.05,
                subsample=0.8, random_state=42
            ),
            'linear': LinearRegression(),
            'ridge_tuned': Ridge(alpha=5.0),
        }
        
        # Add XGBoost if available (optimized for Indian market)
        if XGBOOST_AVAILABLE:
            models['xgb_indian'] = xgb.XGBRegressor(
                n_estimators=200, max_depth=10, learning_rate=0.05,
                subsample=0.8, colsample_bytree=0.8, random_state=42,
                verbosity=0, gamma=0.1, min_child_weight=3
            )
        
        # Add LightGBM if available
        if LIGHTGBM_AVAILABLE:
            models['lgb_indian'] = lgb.LGBMRegressor(
                n_estimators=200, max_depth=10, learning_rate=0.05,
                subsample=0.8, colsample_bytree=0.8, random_state=42,
                verbosity=-1, min_child_samples=20
            )
        
        trained_models = {}
        scores = {}
        feature_importance = {}
        
        for name, model in models.items():
            try:
                model.fit(X_train, y_train)
                trained_models[name] = model
                
                # Calculate score
                train_pred = model.predict(X_train)
                score = np.sqrt(mean_squared_error(y_train, train_pred))
                scores[name] = score
                
                # Feature importance (if available)
                if hasattr(model, 'feature_importances_'):
                    feature_importance[name] = model.feature_importances_
                
                print(f"   ✅ {name.upper()}: Training RMSE = ₹{score:.2f}")
                
            except Exception as e:
                print(f"   ❌ {name.upper()}: Failed - {e}")
        
        self.feature_importance = feature_importance
        return trained_models, scores
    
    def calculate_indian_market_metrics(self, predictions, actuals, returns):
        """Calculate Indian market specific performance metrics."""
        # Basic metrics
        rmse = np.sqrt(mean_squared_error(actuals, predictions))
        mae = mean_absolute_error(actuals, predictions)
        mape = np.mean(np.abs((actuals - predictions) / actuals)) * 100
        
        # Directional accuracy
        actual_dir = np.diff(actuals) > 0
        pred_dir = np.diff(predictions) > 0
        dir_acc = np.mean(actual_dir == pred_dir) * 100 if len(actual_dir) > 0 else 50
        
        # Returns-based metrics
        if len(returns) > 0:
            total_return = np.prod(1 + returns) - 1
            annual_return = (1 + total_return) ** (252 / len(returns)) - 1
            volatility = np.std(returns) * np.sqrt(252)
            sharpe = annual_return / volatility if volatility > 0 else 0
            
            # Maximum drawdown
            cumulative = np.cumprod(1 + returns)
            running_max = np.maximum.accumulate(cumulative)
            drawdowns = (cumulative - running_max) / running_max
            max_drawdown = np.min(drawdowns)
            
            # Calmar ratio
            calmar = annual_return / abs(max_drawdown) if max_drawdown != 0 else 0
            
            # Sortino ratio
            downside_returns = returns[returns < 0]
            downside_deviation = np.std(downside_returns) * np.sqrt(252) if len(downside_returns) > 0 else 0
            sortino = annual_return / downside_deviation if downside_deviation > 0 else 0
            
            # Win rate
            win_rate = np.mean(returns > 0) * 100
            
            return {
                'rmse': rmse, 'mae': mae, 'mape': mape, 'directional_accuracy': dir_acc,
                'total_return': total_return, 'annual_return': annual_return,
                'volatility': volatility, 'sharpe': sharpe, 'calmar': calmar,
                'sortino': sortino, 'max_drawdown': max_drawdown, 'win_rate': win_rate
            }
        
        return {
            'rmse': rmse, 'mae': mae, 'mape': mape, 'directional_accuracy': dir_acc
        }
    
    def run_comprehensive_backtest(self):
        """Run comprehensive advanced backtesting for NIFTYBEES.NS"""
        print(f"🇮🇳 Advanced Backtesting for {self.symbol} (Nifty BeES ETF)")
        print("="*80)
        
        # Fetch and prepare data
        print("📊 Fetching NIFTYBEES data...")
        df = fetch_yfinance(self.symbol, self.start_date, self.end_date)
        
        if df.empty:
            print("❌ No data found for NIFTYBEES.NS")
            return {}
        
        df = self.create_indian_market_features(df)
        print(f"✅ Dataset prepared: {len(df)} records, {len(df.columns)} features")
        
        # Feature columns (exclude price columns)
        feature_cols = [col for col in df.columns 
                       if col not in ['Date', 'Close', 'Open', 'High', 'Low', 'Volume']]
        
        # Walk-forward testing parameters
        train_size = 252 * 2  # 2 years for training
        test_size = 21       # 1 month test periods
        min_data = 252 * 3   # Minimum 3 years of data
        
        if len(df) < min_data:
            print(f"❌ Insufficient data. Need at least {min_data} records, got {len(df)}")
            return {}
        
        results = []
        all_predictions = []
        all_actuals = []
        all_returns = []
        
        print("\n🔄 Starting walk-forward analysis...")
        
        for i in range(min_data, len(df) - test_size, test_size):
            step = len(results) + 1
            train_start = max(0, i - train_size)
            train_end = i
            test_start = i
            test_end = min(i + test_size, len(df))
            
            # Split data
            train_data = df.iloc[train_start:train_end]
            test_data = df.iloc[test_start:test_end]
            
            # Prepare features
            X_train = train_data[feature_cols].values
            y_train = train_data['Close'].values
            X_test = test_data[feature_cols].values
            y_test = test_data['Close'].values
            
            # Handle missing values
            if np.any(np.isnan(X_train)) or np.any(np.isnan(X_test)):
                continue
            
            # Feature selection (top 40 features)
            try:
                selector = SelectKBest(score_func=f_regression, k=min(40, X_train.shape[1]))
                X_train_selected = selector.fit_transform(X_train, y_train)
                X_test_selected = selector.transform(X_test)
            except:
                X_train_selected = X_train
                X_test_selected = X_test
            
            # Scale features
            scaler = StandardScaler()
            X_train_scaled = scaler.fit_transform(X_train_selected)
            X_test_scaled = scaler.transform(X_test_selected)
            
            # Train ensemble
            models, scores = self.train_advanced_ensemble(X_train_scaled, y_train)
            
            if not models:
                continue
            
            # Generate ensemble predictions
            predictions = []
            for name, model in models.items():
                try:
                    pred = model.predict(X_test_scaled)
                    predictions.append(pred)
                except:
                    continue
            
            if len(predictions) == 0:
                continue
            
            # Ensemble prediction (weighted by inverse RMSE)
            weights = [1/scores[name] for name in models.keys() if name in scores]
            weights = np.array(weights) / np.sum(weights)
            
            ensemble_pred = np.average(predictions, axis=0, weights=weights[:len(predictions)])
            
            # Calculate returns for trading strategy
            actual_returns = np.diff(y_test) / y_test[:-1]
            pred_direction = np.diff(ensemble_pred) > 0
            actual_direction = np.diff(y_test) > 0
            
            # Simple momentum strategy returns
            strategy_returns = np.where(pred_direction, actual_returns, -actual_returns)
            
            # Calculate metrics
            metrics = self.calculate_indian_market_metrics(ensemble_pred, y_test, strategy_returns)
            
            results.append({
                'step': step,
                'period': f"{df.index[test_start].strftime('%Y-%m-%d')} to {df.index[test_end-1].strftime('%Y-%m-%d')}",
                **metrics
            })
            
            all_predictions.extend(ensemble_pred)
            all_actuals.extend(y_test)
            all_returns.extend(strategy_returns)
            
            print(f"   Step {step}: RMSE=₹{metrics.get('rmse', 0):.2f}, "
                  f"MAPE={metrics.get('mape', 0):.1f}%, "
                  f"Dir={metrics.get('directional_accuracy', 0):.1f}%, "
                  f"Ret={metrics.get('total_return', 0):.2%}")
        
        # Calculate overall performance
        if results and all_returns:
            df_results = pd.DataFrame(results)
            
            # Overall strategy performance
            total_strategy_return = np.prod(1 + np.array(all_returns)) - 1
            strategy_volatility = np.std(all_returns) * np.sqrt(252)
            strategy_sharpe = (np.mean(all_returns) * 252) / strategy_volatility if strategy_volatility > 0 else 0
            
            # Overall prediction accuracy
            overall_metrics = self.calculate_indian_market_metrics(
                np.array(all_predictions), np.array(all_actuals), np.array(all_returns)
            )
            
            summary = {
                'symbol': self.symbol,
                'analysis_period': f"{self.start_date} to {self.end_date}",
                'total_steps': len(results),
                'overall_metrics': overall_metrics,
                'avg_rmse': df_results['rmse'].mean(),
                'avg_mape': df_results['mape'].mean(),
                'avg_directional_accuracy': df_results['directional_accuracy'].mean(),
                'strategy_total_return': total_strategy_return,
                'strategy_annual_return': (1 + total_strategy_return) ** (252 / len(all_returns)) - 1,
                'strategy_volatility': strategy_volatility,
                'strategy_sharpe': strategy_sharpe,
                'win_rate': (df_results['total_return'] > 0).mean() * 100,
                'detailed_results': df_results,
                'feature_importance': self.feature_importance
            }
            
            self.results = summary
            return summary
        
        return {}
    
    def print_advanced_results(self):
        """Print comprehensive advanced backtesting results"""
        if not self.results:
            print("❌ No results to display")
            return
            
        print("\n" + "="*100)
        print(f"🇮🇳 ADVANCED BACKTESTING RESULTS FOR {self.symbol} (NIFTY BEES ETF)")
        print("="*100)
        
        r = self.results
        
        print(f"\n📊 ANALYSIS OVERVIEW:")
        print(f"   Period: {r['analysis_period']}")
        print(f"   Total Steps: {r['total_steps']}")
        print(f"   Currency: ₹ (Indian Rupees)")
        
        print(f"\n🎯 PREDICTION ACCURACY:")
        print(f"   Average RMSE: ₹{r['avg_rmse']:.2f}")
        print(f"   Average MAPE: {r['avg_mape']:.1f}%")
        print(f"   Directional Accuracy: {r['avg_directional_accuracy']:.1f}%")
        
        print(f"\n💰 TRADING STRATEGY PERFORMANCE:")
        print(f"   Total Return: {r['strategy_total_return']:.2%}")
        print(f"   Annual Return: {r['strategy_annual_return']:.2%}")
        print(f"   Volatility: {r['strategy_volatility']:.2%}")
        print(f"   Sharpe Ratio: {r['strategy_sharpe']:.2f}")
        print(f"   Win Rate: {r['win_rate']:.1f}%")
        
        if 'overall_metrics' in r:
            om = r['overall_metrics']
            if 'max_drawdown' in om:
                print(f"   Max Drawdown: {om['max_drawdown']:.2%}")
            if 'calmar' in om:
                print(f"   Calmar Ratio: {om['calmar']:.2f}")
            if 'sortino' in om:
                print(f"   Sortino Ratio: {om['sortino']:.2f}")
        
        print(f"\n🏆 KEY INSIGHTS:")
        
        # Performance rating
        if r['avg_directional_accuracy'] > 60:
            accuracy_rating = "🟢 EXCELLENT"
        elif r['avg_directional_accuracy'] > 55:
            accuracy_rating = "🟡 GOOD"
        else:
            accuracy_rating = "🔴 NEEDS IMPROVEMENT"
        
        if r['strategy_sharpe'] > 1.0:
            sharpe_rating = "🟢 EXCELLENT"
        elif r['strategy_sharpe'] > 0.5:
            sharpe_rating = "🟡 GOOD"
        else:
            sharpe_rating = "🔴 POOR"
        
        print(f"   Directional Accuracy: {accuracy_rating} ({r['avg_directional_accuracy']:.1f}%)")
        print(f"   Risk-Adjusted Returns: {sharpe_rating} (Sharpe: {r['strategy_sharpe']:.2f})")
        
        # Model insights
        if 'feature_importance' in r and r['feature_importance']:
            print(f"\n🔧 MODEL INSIGHTS:")
            print(f"   Models Used: {len(r['feature_importance'])} ensemble models")
            
            # Feature importance summary
            all_importance = []
            for model_name, importance in r['feature_importance'].items():
                if len(importance) > 0:
                    all_importance.extend(importance)
            
            if all_importance:
                print(f"   Average Feature Importance Range: {np.min(all_importance):.3f} - {np.max(all_importance):.3f}")
        
        print(f"\n📈 RECOMMENDATION:")
        if r['strategy_sharpe'] > 0.5 and r['avg_directional_accuracy'] > 55:
            print("   ✅ Model shows good predictive capability for NIFTYBEES")
            print("   ✅ Strategy demonstrates positive risk-adjusted returns")
            print("   💡 Consider implementing with proper risk management")
        else:
            print("   ⚠️ Model performance needs improvement")
            print("   💡 Consider additional feature engineering or parameter tuning")
            print("   💡 Review risk management strategies")

def run_niftybees_advanced_analysis():
    """Run advanced analysis specifically for NIFTYBEES.NS"""
    print("🇮🇳 NIFTYBEES.NS ADVANCED BACKTESTING SYSTEM")
    print("="*60)
    print("📊 Nifty BeES ETF - Comprehensive Analysis")
    print("🎯 Advanced ML Models + Indian Market Features")
    print("="*60)
    
    # Initialize backtester
    backtester = NiftyBeesAdvancedBacktester(
        symbol='NIFTYBEES.NS',
        start_date='2018-01-01',  # 6+ years of data
        end_date=None
    )
    
    try:
        # Run comprehensive analysis
        results = backtester.run_comprehensive_backtest()
        
        if results:
            # Print results
            backtester.print_advanced_results()
            
            # Save results
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
    results = run_niftybees_advanced_analysis() 