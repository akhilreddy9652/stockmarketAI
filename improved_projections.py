#!/usr/bin/env python3
"""
Improved Projections System - Fixes Current Performance Issues
"""

import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout, BatchNormalization
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
import warnings
import yfinance as yf
from datetime import datetime, timedelta
from sklearn.preprocessing import RobustScaler
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error
import joblib
import os

warnings.filterwarnings('ignore')
tf.get_logger().setLevel('ERROR')

class ImprovedProjections:
    def __init__(self, symbol: str = "RELIANCE.NS"):
        self.symbol = symbol
        self.sequence_length = 30
        self.models = {}
        self.scalers = {}
        
        os.makedirs('models', exist_ok=True)
        os.makedirs('results', exist_ok=True)
        
        print(f"🚀 Initializing Improved Projections for {symbol}")
    
    def fetch_clean_data(self, start_date: str, end_date: str) -> pd.DataFrame:
        """Fetch and clean data with robust error handling"""
        print(f"📊 Fetching clean data for {self.symbol}...")
        
        try:
            ticker = yf.Ticker(self.symbol)
            df = ticker.history(start=start_date, end=end_date, interval='1d')
            
            if df.empty:
                raise ValueError(f"No data found for {self.symbol}")
            
            df.reset_index(inplace=True)
            df = df.dropna()
            df = df[df['Close'] > 0]
            df = df[df['Volume'] > 0]
            
            # Remove extreme outliers
            price_change = df['Close'].pct_change().abs()
            df = df[price_change < 0.3]  # Remove 30%+ daily changes
            
            print(f"✅ Fetched {len(df)} clean records")
            return df
            
        except Exception as e:
            print(f"❌ Error fetching data: {str(e)}")
            raise
    
    def create_enhanced_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Create enhanced features for better predictions"""
        print("🔧 Creating enhanced features...")
        
        featured_df = df.copy()
        
        # Price features
        featured_df['Returns'] = featured_df['Close'].pct_change()
        featured_df['Log_Returns'] = np.log(featured_df['Close'] / featured_df['Close'].shift(1))
        
        # Moving averages
        for period in [5, 10, 20, 50]:
            featured_df[f'MA_{period}'] = featured_df['Close'].rolling(period).mean()
            featured_df[f'Price_to_MA_{period}'] = featured_df['Close'] / featured_df[f'MA_{period}']
        
        # Volatility
        for window in [5, 10, 20]:
            featured_df[f'Volatility_{window}'] = featured_df['Returns'].rolling(window).std()
        
        # Volume features
        featured_df['Volume_MA_20'] = featured_df['Volume'].rolling(20).mean()
        featured_df['Volume_Ratio'] = featured_df['Volume'] / featured_df['Volume_MA_20']
        
        # Price patterns
        featured_df['HL_Ratio'] = featured_df['High'] / featured_df['Low']
        featured_df['OC_Ratio'] = featured_df['Open'] / featured_df['Close']
        featured_df['Price_Range'] = (featured_df['High'] - featured_df['Low']) / featured_df['Close']
        
        # Lag features
        for lag in [1, 2, 3, 5]:
            featured_df[f'Close_Lag_{lag}'] = featured_df['Close'].shift(lag)
            featured_df[f'Returns_Lag_{lag}'] = featured_df['Returns'].shift(lag)
        
        # Rolling statistics
        for window in [5, 10]:
            featured_df[f'Close_Std_{window}'] = featured_df['Close'].rolling(window).std()
            featured_df[f'Returns_Mean_{window}'] = featured_df['Returns'].rolling(window).mean()
        
        # Technical indicators (simple versions to avoid errors)
        featured_df['RSI_14'] = self.calculate_rsi(featured_df['Close'], 14)
        featured_df['MACD'] = self.calculate_macd(featured_df['Close'])
        
        # Clean data
        featured_df = featured_df.dropna()
        featured_df = featured_df.replace([np.inf, -np.inf], np.nan).dropna()
        
        print(f"✅ Created {len(featured_df.columns)} features")
        return featured_df
    
    def calculate_rsi(self, prices: pd.Series, window: int = 14) -> pd.Series:
        """Calculate RSI indicator"""
        delta = prices.diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=window).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=window).mean()
        rs = gain / loss
        rsi = 100 - (100 / (1 + rs))
        return rsi
    
    def calculate_macd(self, prices: pd.Series, fast: int = 12, slow: int = 26) -> pd.Series:
        """Calculate MACD indicator"""
        ema_fast = prices.ewm(span=fast).mean()
        ema_slow = prices.ewm(span=slow).mean()
        macd = ema_fast - ema_slow
        return macd
    
    def create_improved_model(self, input_shape: tuple) -> Sequential:
        """Create improved LSTM model"""
        model = Sequential([
            LSTM(64, return_sequences=True, input_shape=input_shape),
            Dropout(0.2),
            BatchNormalization(),
            
            LSTM(32, return_sequences=False),
            Dropout(0.2),
            BatchNormalization(),
            
            Dense(16, activation='relu'),
            Dropout(0.1),
            Dense(1)
        ])
        
        model.compile(
            optimizer=Adam(learning_rate=0.001),
            loss='huber',
            metrics=['mae']
        )
        
        return model
    
    def prepare_sequences(self, df: pd.DataFrame) -> tuple:
        """Prepare sequences for training"""
        print("📊 Preparing sequences...")
        
        exclude_cols = ['Date', 'Close']
        feature_cols = [col for col in df.columns 
                       if col not in exclude_cols and df[col].dtype in [np.float64, np.int64]]
        
        features = df[feature_cols].values
        target = df['Close'].values
        
        X, y = [], []
        for i in range(self.sequence_length, len(features)):
            X.append(features[i-self.sequence_length:i])
            y.append(target[i])
        
        X = np.array(X)
        y = np.array(y)
        
        print(f"✅ Created {len(X)} sequences with {X.shape[2]} features")
        return X, y, feature_cols
    
    def train_improved_system(self) -> dict:
        """Train the improved system"""
        print(f"🚀 Training improved system for {self.symbol}")
        
        # Fetch data
        end_date = datetime.now()
        start_date = end_date - timedelta(days=730)
        
        df = self.fetch_clean_data(start_date.strftime('%Y-%m-%d'), end_date.strftime('%Y-%m-%d'))
        featured_df = self.create_enhanced_features(df)
        
        # Prepare sequences
        X, y, feature_cols = self.prepare_sequences(featured_df)
        
        # Split data
        split_idx = int(len(X) * 0.8)
        X_train, X_val = X[:split_idx], X[split_idx:]
        y_train, y_val = y[:split_idx], y[split_idx:]
        
        # Scale data
        feature_scaler = RobustScaler()
        X_train_scaled = feature_scaler.fit_transform(X_train.reshape(-1, X_train.shape[-1])).reshape(X_train.shape)
        X_val_scaled = feature_scaler.transform(X_val.reshape(-1, X_val.shape[-1])).reshape(X_val.shape)
        
        target_scaler = RobustScaler()
        y_train_scaled = target_scaler.fit_transform(y_train.reshape(-1, 1)).flatten()
        y_val_scaled = target_scaler.transform(y_val.reshape(-1, 1)).flatten()
        
        # Save scalers
        self.scalers = {
            'feature_scaler': feature_scaler,
            'target_scaler': target_scaler,
            'feature_cols': feature_cols
        }
        
        # Train LSTM model
        print("🏋️ Training LSTM model...")
        lstm_model = self.create_improved_model((X_train.shape[1], X_train.shape[2]))
        
        callbacks = [
            EarlyStopping(patience=15, restore_best_weights=True),
            ReduceLROnPlateau(factor=0.5, patience=8, min_lr=1e-6)
        ]
        
        lstm_history = lstm_model.fit(
            X_train_scaled, y_train_scaled,
            validation_data=(X_val_scaled, y_val_scaled),
            epochs=100,
            batch_size=32,
            callbacks=callbacks,
            verbose=0
        )
        
        # Train Random Forest for comparison
        print("🌲 Training Random Forest...")
        X_train_flat = X_train_scaled.reshape(X_train_scaled.shape[0], -1)
        X_val_flat = X_val_scaled.reshape(X_val_scaled.shape[0], -1)
        
        rf_model = RandomForestRegressor(n_estimators=100, max_depth=15, random_state=42, n_jobs=-1)
        rf_model.fit(X_train_flat, y_train_scaled)
        
        # Store models
        self.models = {
            'lstm': lstm_model,
            'random_forest': rf_model
        }
        
        # Evaluate models
        results = {}
        
        # LSTM evaluation
        lstm_pred = lstm_model.predict(X_val_scaled, verbose=0).flatten()
        lstm_pred = target_scaler.inverse_transform(lstm_pred.reshape(-1, 1)).flatten()
        y_val_original = target_scaler.inverse_transform(y_val_scaled.reshape(-1, 1)).flatten()
        
        lstm_mse = mean_squared_error(y_val_original, lstm_pred)
        lstm_mae = mean_absolute_error(y_val_original, lstm_pred)
        lstm_mape = np.mean(np.abs((y_val_original - lstm_pred) / y_val_original)) * 100
        
        # Directional accuracy
        actual_direction = np.diff(y_val_original) > 0
        pred_direction = np.diff(lstm_pred) > 0
        lstm_directional = np.mean(actual_direction == pred_direction) * 100
        
        # Random Forest evaluation
        rf_pred = rf_model.predict(X_val_flat)
        rf_pred = target_scaler.inverse_transform(rf_pred.reshape(-1, 1)).flatten()
        
        rf_mse = mean_squared_error(y_val_original, rf_pred)
        rf_mae = mean_absolute_error(y_val_original, rf_pred)
        rf_mape = np.mean(np.abs((y_val_original - rf_pred) / y_val_original)) * 100
        
        rf_pred_direction = np.diff(rf_pred) > 0
        rf_directional = np.mean(actual_direction == rf_pred_direction) * 100
        
        results = {
            'lstm': {
                'mse': lstm_mse,
                'mae': lstm_mae,
                'rmse': np.sqrt(lstm_mse),
                'mape': lstm_mape,
                'directional_accuracy': lstm_directional
            },
            'random_forest': {
                'mse': rf_mse,
                'mae': rf_mae,
                'rmse': np.sqrt(rf_mse),
                'mape': rf_mape,
                'directional_accuracy': rf_directional
            }
        }
        
        print(f"✅ LSTM - RMSE: ${np.sqrt(lstm_mse):.2f}, MAPE: {lstm_mape:.2f}%, Directional: {lstm_directional:.1f}%")
        print(f"✅ Random Forest - RMSE: ${np.sqrt(rf_mse):.2f}, MAPE: {rf_mape:.2f}%, Directional: {rf_directional:.1f}%")
        
        # Save system
        self.save_system()
        
        return results
    
    def generate_forecast(self, forecast_days: int = 30) -> dict:
        """Generate improved forecast"""
        print(f"🔮 Generating {forecast_days}-day forecast...")
        
        # Fetch latest data
        end_date = datetime.now()
        start_date = end_date - timedelta(days=365)
        
        df = self.fetch_clean_data(start_date.strftime('%Y-%m-%d'), end_date.strftime('%Y-%m-%d'))
        featured_df = self.create_enhanced_features(df)
        
        # Prepare last sequence
        features = featured_df[self.scalers['feature_cols']].values
        last_sequence = features[-self.sequence_length:]
        
        # Scale features
        feature_scaler = self.scalers['feature_scaler']
        target_scaler = self.scalers['target_scaler']
        
        last_sequence_scaled = feature_scaler.transform(
            last_sequence.reshape(-1, features.shape[-1])
        ).reshape(1, self.sequence_length, -1)
        
        # Generate predictions
        lstm_pred = self.models['lstm'].predict(last_sequence_scaled, verbose=0)[0, 0]
        rf_pred = self.models['random_forest'].predict(last_sequence_scaled.reshape(1, -1))[0]
        
        # Inverse transform
        lstm_pred = target_scaler.inverse_transform([[lstm_pred]])[0, 0]
        rf_pred = target_scaler.inverse_transform([[rf_pred]])[0, 0]
        
        # Ensemble prediction (weighted average)
        ensemble_pred = (lstm_pred * 0.7 + rf_pred * 0.3)  # Weight LSTM more
        
        # Current price
        current_price = df['Close'].iloc[-1]
        
        # Analysis
        price_change = ensemble_pred - current_price
        price_change_pct = (price_change / current_price) * 100
        
        # Generate recommendation
        if price_change_pct > 5:
            recommendation = "🟢 Strong Buy"
        elif price_change_pct > 2:
            recommendation = "🟢 Buy"
        elif price_change_pct > -2:
            recommendation = "🟡 Hold"
        elif price_change_pct > -5:
            recommendation = "🔴 Sell"
        else:
            recommendation = "🔴 Strong Sell"
        
        results = {
            'symbol': self.symbol,
            'current_price': current_price,
            'lstm_prediction': lstm_pred,
            'rf_prediction': rf_pred,
            'ensemble_prediction': ensemble_pred,
            'price_change': price_change,
            'price_change_pct': price_change_pct,
            'recommendation': recommendation,
            'forecast_date': datetime.now().strftime('%Y-%m-%d'),
            'forecast_horizon_days': forecast_days
        }
        
        print(f"✅ Forecast generated:")
        print(f"   Current Price: ${current_price:.2f}")
        print(f"   Predicted Price: ${ensemble_pred:.2f}")
        print(f"   Expected Change: {price_change_pct:+.2f}%")
        print(f"   Recommendation: {recommendation}")
        
        return results
    
    def save_system(self):
        """Save the trained system"""
        joblib.dump(self.scalers, f'models/{self.symbol}_improved_scalers.pkl')
        self.models['lstm'].save(f'models/{self.symbol}_improved_lstm.keras')
        joblib.dump(self.models['random_forest'], f'models/{self.symbol}_improved_rf.pkl')
        print("✅ System saved")
    
    def run_analysis(self, symbol: str = None) -> dict:
        """Run complete improved analysis"""
        if symbol:
            self.symbol = symbol
        
        print(f"🚀 Running improved analysis for {self.symbol}")
        
        # Train system
        training_results = self.train_improved_system()
        
        # Generate forecast
        forecast_results = self.generate_forecast()
        
        # Save results
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        joblib.dump({
            'training_results': training_results,
            'forecast_results': forecast_results
        }, f'results/{self.symbol}_improved_analysis_{timestamp}.pkl')
        
        return {
            'training_results': training_results,
            'forecast_results': forecast_results
        }

if __name__ == "__main__":
    # Test with Indian stock
    projector = ImprovedProjections("RELIANCE.NS")
    results = projector.run_analysis()
    
    print(f"\n🎯 Improved Analysis Results:")
    print(f"Symbol: {results['forecast_results']['symbol']}")
    print(f"Current Price: ₹{results['forecast_results']['current_price']:.2f}")
    print(f"Predicted Price: ₹{results['forecast_results']['ensemble_prediction']:.2f}")
    print(f"Expected Change: {results['forecast_results']['price_change_pct']:+.2f}%")
    print(f"Recommendation: {results['forecast_results']['recommendation']}") 