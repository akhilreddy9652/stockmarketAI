"""
Ultra-Enhanced Backtesting System v3.0
=====================================
Further improvements for maximum accuracy:
- XGBoost and LightGBM models
- Feature selection optimization
- Advanced confidence scoring
- Dynamic position sizing
- Regime-aware predictions
"""

import pandas as pd
import numpy as np
import warnings
import joblib
import os
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.feature_selection import SelectKBest, f_regression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error
from data_ingestion import fetch_yfinance
from feature_engineering import add_technical_indicators

# Optional advanced models (install if available)
try:
    import xgboost as xgb
    XGBOOST_AVAILABLE = True
except ImportError:
    XGBOOST_AVAILABLE = False

try:
    import lightgbm as lgb
    LIGHTGBM_AVAILABLE = True
except ImportError:
    LIGHTGBM_AVAILABLE = False

warnings.filterwarnings('ignore')

class UltraEnhancedBacktester:
    """Ultra-enhanced backtesting with maximum accuracy optimizations."""
    
    def __init__(self, symbol='AAPL', start_date='2020-01-01', end_date=None):
        self.symbol = symbol
        self.start_date = start_date
        self.end_date = end_date or pd.Timestamp.now().strftime('%Y-%m-%d')
        self.results = {}
        self.feature_importance = {}
        
    def create_ultra_features(self, df):
        """Create ultra-comprehensive feature set."""
        print("🔧 Creating ultra-enhanced features...")
        
        # Basic technical indicators
        df = add_technical_indicators(df)
        
        # 1. Advanced momentum features
        momentum_periods = [3, 5, 7, 10, 14, 15, 20, 30]
        for period in momentum_periods:
            df[f'Return_{period}d'] = df['Close'].pct_change(period)
            df[f'SMA_{period}'] = df['Close'].rolling(period).mean()
            df[f'EMA_{period}'] = df['Close'].ewm(span=period).mean()
            df[f'ROC_{period}'] = df['Close'] / df['Close'].shift(period) - 1
        
        # 2. Volatility and risk features
        vol_periods = [5, 10, 15, 20, 30]
        for period in vol_periods:
            df[f'Vol_{period}'] = df['Close'].pct_change().rolling(period).std()
            df[f'VolRank_{period}'] = df[f'Vol_{period}'].rolling(100).rank(pct=True)
        
        # 3. Volume analysis
        df['Volume_SMA_10'] = df['Volume'].rolling(10).mean()
        df['Volume_SMA_20'] = df['Volume'].rolling(20).mean()
        df['Volume_Ratio_10'] = df['Volume'] / df['Volume_SMA_10']
        df['Volume_Ratio_20'] = df['Volume'] / df['Volume_SMA_20']
        df['Price_Volume'] = df['Close'] * df['Volume']
        df['VWAP_10'] = (df['Price_Volume'].rolling(10).sum() / 
                        df['Volume'].rolling(10).sum())
        
        # 4. Price position and momentum
        for period in [10, 20, 50]:
            high = df['High'].rolling(period).max()
            low = df['Low'].rolling(period).min()
            df[f'Price_Position_{period}'] = (df['Close'] - low) / (high - low)
            df[f'High_Distance_{period}'] = (high - df['Close']) / df['Close']
            df[f'Low_Distance_{period}'] = (df['Close'] - low) / df['Close']
        
        # 5. Statistical features
        for period in [10, 20, 30]:
            returns = df['Close'].pct_change()
            df[f'Skew_{period}'] = returns.rolling(period).skew()
            df[f'Kurt_{period}'] = returns.rolling(period).kurt()
            df[f'Sharpe_{period}'] = (returns.rolling(period).mean() / 
                                    returns.rolling(period).std())
        
        # 6. Trend strength
        for period in [10, 20]:
            df[f'Trend_Strength_{period}'] = (df['Close'] - df['Close'].shift(period)) / df['Close'].shift(period)
            df[f'Trend_Consistency_{period}'] = (df['Close'] > df['Close'].shift(1)).rolling(period).mean()
        
        # 7. Market regime features
        # Volatility regime
        vol_20 = df['Close'].pct_change().rolling(20).std()
        df['Vol_Regime'] = pd.qcut(vol_20.rank(method='first'), 3, labels=[0, 1, 2])
        
        # Trend regime
        sma_10 = df['Close'].rolling(10).mean()
        sma_50 = df['Close'].rolling(50).mean()
        df['Trend_Regime'] = (sma_10 > sma_50).astype(int)
        
        # Price level regime
        df['Price_Regime'] = pd.qcut(df['Close'].rank(method='first'), 3, labels=[0, 1, 2])
        
        # 8. Interaction features
        df['RSI_Vol'] = df.get('RSI_14', 50) * df['Vol_20']
        df['MA_Spread'] = df['SMA_10'] - df['SMA_20']
        df['Volume_Price_Momentum'] = df['Volume_Ratio_20'] * df['Return_5d']
        
        # Clean data
        df = df.replace([np.inf, -np.inf], np.nan)
        df = df.fillna(method='ffill').fillna(method='bfill')
        
        print(f"✅ Ultra features created: {len(df.columns)} total features")
        return df
    
    def select_best_features(self, X, y, k=30):
        """Select top K features using statistical tests."""
        print(f"🎯 Selecting top {k} features...")
        
        selector = SelectKBest(score_func=f_regression, k=min(k, X.shape[1]))
        X_selected = selector.fit_transform(X, y)
        
        # Get selected feature names
        feature_names = [f"feature_{i}" for i in range(X.shape[1])]
        selected_features = [feature_names[i] for i in selector.get_support(indices=True)]
        
        print(f"✅ Selected {X_selected.shape[1]} best features")
        return X_selected, selector, selected_features
    
    def train_ultra_ensemble(self, X_train, y_train):
        """Train ultra-enhanced ensemble with advanced models."""
        print("🚀 Training ultra-enhanced ensemble...")
        
        models = {
            'rf_tuned': RandomForestRegressor(
                n_estimators=100, max_depth=12, min_samples_split=5,
                min_samples_leaf=2, random_state=42
            ),
            'gb_tuned': GradientBoostingRegressor(
                n_estimators=100, max_depth=8, learning_rate=0.1,
                subsample=0.8, random_state=42
            ),
            'linear': LinearRegression(),
            'ridge_tuned': Ridge(alpha=10.0),
        }
        
        # Add XGBoost if available
        if XGBOOST_AVAILABLE:
            models['xgb'] = xgb.XGBRegressor(
                n_estimators=100, max_depth=8, learning_rate=0.1,
                subsample=0.8, colsample_bytree=0.8, random_state=42,
                verbosity=0
            )
        
        # Add LightGBM if available
        if LIGHTGBM_AVAILABLE:
            models['lgb'] = lgb.LGBMRegressor(
                n_estimators=100, max_depth=8, learning_rate=0.1,
                subsample=0.8, colsample_bytree=0.8, random_state=42,
                verbosity=-1
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
                
                print(f"   ✅ {name.upper()}: Training RMSE = {score:.4f}")
                
            except Exception as e:
                print(f"   ❌ {name.upper()}: Failed - {e}")
        
        self.feature_importance = feature_importance
        return trained_models, scores
    
    def calculate_prediction_confidence(self, models, X):
        """Calculate advanced confidence scores."""
        predictions = {}
        
        for name, model in models.items():
            try:
                pred = model.predict(X)
                predictions[name] = pred
            except:
                continue
        
        if len(predictions) < 2:
            return np.ones(len(X)), np.mean(list(predictions.values()), axis=0)
        
        pred_array = np.array(list(predictions.values()))
        
        # Calculate ensemble prediction
        ensemble_pred = np.mean(pred_array, axis=0)
        
        # Calculate confidence based on model agreement
        pred_std = np.std(pred_array, axis=0)
        pred_mean = np.abs(ensemble_pred)
        cv = pred_std / (pred_mean + 1e-8)
        
        # Convert to confidence (0-1 scale)
        confidence = 1 / (1 + cv)
        confidence = np.clip(confidence, 0.1, 1.0)  # Minimum 10% confidence
        
        return confidence, ensemble_pred
    
    def dynamic_position_sizing(self, predictions, actuals, confidence):
        """Calculate dynamic position sizes based on confidence and risk."""
        # Calculate expected returns
        expected_returns = (predictions - actuals) / actuals
        
        # Volatility-based risk adjustment
        vol = pd.Series(actuals).pct_change().rolling(20).std().fillna(0.02)
        
        # Kelly criterion with confidence adjustment
        win_prob = confidence
        loss_prob = 1 - confidence
        
        # Position size (Kelly fraction with confidence)
        kelly_fraction = np.where(
            win_prob > 0.5,
            (win_prob - loss_prob) / np.abs(expected_returns + 1e-8),
            0
        )
        
        # Risk-adjusted position size
        position_size = kelly_fraction * confidence
        position_size = np.clip(position_size, 0, 0.25)  # Max 25% position
        
        return position_size
    
    def ultra_walk_forward_test(self):
        """Ultra-enhanced walk-forward testing."""
        print(f"🔍 Ultra-enhanced walk-forward test for {self.symbol}")
        
        # Fetch and prepare data
        df = fetch_yfinance(self.symbol, self.start_date, self.end_date)
        df = self.create_ultra_features(df)
        
        print(f"✅ Dataset prepared: {len(df)} records, {len(df.columns)} features")
        
        # Feature columns (exclude price columns)
        feature_cols = [col for col in df.columns 
                       if col not in ['Date', 'Close', 'Open', 'High', 'Low', 'Volume']]
        
        results = []
        all_predictions = []
        all_actuals = []
        all_confidence = []
        
        # Walk-forward parameters
        train_size = 300  # Longer training for stability
        test_size = 21    # 1 month test
        min_data = 600    # Minimum data needed
        
        for i in range(min_data, len(df) - test_size, test_size):
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
            
            # Feature selection
            X_train_selected, feature_selector, selected_features = self.select_best_features(X_train, y_train)
            X_test_selected = feature_selector.transform(X_test)
            
            # Scale features
            scaler = StandardScaler()
            X_train_scaled = scaler.fit_transform(X_train_selected)
            X_test_scaled = scaler.transform(X_test_selected)
            
            # Train ultra ensemble
            models, scores = self.train_ultra_ensemble(X_train_scaled, y_train)
            
            if not models:
                continue
            
            # Generate predictions with confidence
            confidence, ensemble_pred = self.calculate_prediction_confidence(models, X_test_scaled)
            
            # Dynamic position sizing
            position_sizes = self.dynamic_position_sizing(ensemble_pred, y_test, confidence)
            
            # Calculate metrics
            rmse = np.sqrt(mean_squared_error(y_test, ensemble_pred))
            mae = mean_absolute_error(y_test, ensemble_pred)
            mape = np.mean(np.abs((y_test - ensemble_pred) / y_test)) * 100
            
            # Directional accuracy
            actual_dir = np.diff(y_test) > 0
            pred_dir = np.diff(ensemble_pred) > 0
            dir_acc = np.mean(actual_dir == pred_dir) * 100 if len(actual_dir) > 0 else 50
            
            # Enhanced trading performance
            returns = np.diff(y_test) / y_test[:-1]
            signals = np.where(pred_dir, 1, -1)
            position_weighted_returns = signals * returns * position_sizes[:-1]
            total_return = np.prod(1 + position_weighted_returns) - 1
            
            # Risk metrics
            volatility = np.std(position_weighted_returns)
            sharpe = np.mean(position_weighted_returns) / volatility if volatility > 0 else 0
            
            results.append({
                'step': len(results) + 1,
                'rmse': rmse,
                'mae': mae,
                'mape': mape,
                'directional_accuracy': dir_acc,
                'total_return': total_return,
                'sharpe': sharpe,
                'avg_confidence': confidence.mean(),
                'avg_position_size': position_sizes.mean()
            })
            
            all_predictions.extend(ensemble_pred)
            all_actuals.extend(y_test)
            all_confidence.extend(confidence)
            
            print(f"   Step {len(results)}: RMSE=${rmse:.2f}, MAPE={mape:.1f}%, "
                  f"Dir={dir_acc:.1f}%, Ret={total_return:.2%}, Conf={confidence.mean():.3f}")
        
        # Calculate overall metrics
        if results:
            df_results = pd.DataFrame(results)
            
            overall_metrics = {
                'symbol': self.symbol,
                'total_steps': len(results),
                'avg_rmse': df_results['rmse'].mean(),
                'avg_mae': df_results['mae'].mean(),
                'avg_mape': df_results['mape'].mean(),
                'avg_directional_accuracy': df_results['directional_accuracy'].mean(),
                'total_return': (1 + df_results['total_return']).prod() - 1,
                'avg_sharpe': df_results['sharpe'].mean(),
                'win_rate': (df_results['total_return'] > 0).mean() * 100,
                'avg_confidence': df_results['avg_confidence'].mean(),
                'detailed_results': df_results,
                'feature_importance': self.feature_importance
            }
            
            print(f"\n📊 ULTRA-ENHANCED RESULTS SUMMARY:")
            print(f"   🎯 Average RMSE: ${overall_metrics['avg_rmse']:.2f}")
            print(f"   📊 Average MAPE: {overall_metrics['avg_mape']:.1f}%")
            print(f"   🎯 Directional Accuracy: {overall_metrics['avg_directional_accuracy']:.1f}%")
            print(f"   💰 Total Return: {overall_metrics['total_return']:.2%}")
            print(f"   📊 Average Sharpe: {overall_metrics['avg_sharpe']:.3f}")
            print(f"   🎯 Win Rate: {overall_metrics['win_rate']:.1f}%")
            print(f"   📊 Average Confidence: {overall_metrics['avg_confidence']:.3f}")
            
            return overall_metrics
        
        return {}

def run_ultra_enhanced_analysis():
    """Run ultra-enhanced analysis for key stocks."""
    stocks = ['AAPL', 'MSFT']  # Start with 2 for testing
    
    results = {}
    
    for stock in stocks:
        print(f"\n{'='*80}")
        print(f"🚀 ULTRA-ENHANCED ANALYSIS: {stock}")
        print(f"{'='*80}")
        
        try:
            backtester = UltraEnhancedBacktester(symbol=stock)
            result = backtester.ultra_walk_forward_test()
            
            if result:
                results[stock] = result
                
                # Save results
                os.makedirs('results', exist_ok=True)
                joblib.dump(result, f'results/{stock}_ultra_enhanced.pkl')
                
                print(f"\n🏆 FINAL ULTRA PERFORMANCE FOR {stock}:")
                print(f"   📈 Directional Accuracy: {result['avg_directional_accuracy']:.1f}%")
                print(f"   💰 Total Return: {result['total_return']:.2%}")
                print(f"   📊 Sharpe Ratio: {result['avg_sharpe']:.3f}")
                print(f"   🎯 Confidence Level: {result['avg_confidence']:.1%}")
            
        except Exception as e:
            print(f"❌ Error with {stock}: {e}")
    
    return results

if __name__ == "__main__":
    print("🚀 Ultra-Enhanced Backtesting System v3.0")
    print("="*50)
    
    results = run_ultra_enhanced_analysis()
    
    print(f"\n{'='*80}")
    print("🎉 ULTRA-ENHANCED BACKTESTING COMPLETED!")
    print("="*80)
    print("🚀 New Features Added:")
    print("✅ XGBoost and LightGBM models")
    print("✅ Feature selection optimization")
    print("✅ Advanced confidence scoring")
    print("✅ Dynamic position sizing")
    print("✅ Regime-aware predictions") 