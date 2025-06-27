#!/usr/bin/env python3
"""
Ultra-Enhanced Forecasting System
Advanced ML models with state-of-the-art architectures for superior projections
"""

import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras.models import Model, Sequential
from tensorflow.keras.layers import (
    LSTM, GRU, Dense, Dropout, Input, Conv1D, MaxPooling1D,
    Bidirectional, Attention, MultiHeadAttention, LayerNormalization,
    GlobalAveragePooling1D, Concatenate, BatchNormalization,
    SeparableConv1D, DepthwiseConv1D, Add, LeakyReLU
)
from tensorflow.keras.optimizers import Adam, AdamW, RMSprop
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau, ModelCheckpoint
from tensorflow.keras.regularizers import l1_l2
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.preprocessing import StandardScaler, MinMaxScaler, RobustScaler
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import joblib
import warnings
import yfinance as yf
from datetime import datetime, timedelta
import talib
from typing import Dict, List, Tuple, Optional, Any
import optuna
import logging

# Suppress warnings
warnings.filterwarnings('ignore')
tf.get_logger().setLevel('ERROR')

class UltraEnhancedForecaster:
    """
    Ultra-Enhanced Forecasting System with state-of-the-art ML architectures
    """
    
    def __init__(self, symbol: str = "AAPL", lookback_days: int = 252):
        self.symbol = symbol
        self.lookback_days = lookback_days
        self.sequence_length = 60
        self.models = {}
        self.scalers = {}
        self.feature_importance = {}
        self.performance_metrics = {}
        
        # Set up logging
        logging.basicConfig(level=logging.INFO)
        self.logger = logging.getLogger(__name__)
        
        print(f"🚀 Initializing Ultra-Enhanced Forecaster for {symbol}")
    
    def fetch_advanced_data(self, start_date: str, end_date: str) -> pd.DataFrame:
        """
        Fetch and prepare advanced dataset with comprehensive features
        """
        print(f"📊 Fetching advanced data for {self.symbol}...")
        
        try:
            # Download main stock data
            ticker = yf.Ticker(self.symbol)
            df = ticker.history(start=start_date, end=end_date, interval='1d')
            
            if df.empty:
                raise ValueError(f"No data found for {self.symbol}")
            
            # Reset index and clean data
            df.reset_index(inplace=True)
            df = df.dropna()
            
            print(f"✅ Fetched {len(df)} records")
            return df
            
        except Exception as e:
            self.logger.error(f"Error fetching data: {str(e)}")
            raise
    
    def create_ultra_advanced_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Create ultra-advanced features using cutting-edge financial engineering
        """
        print("🔧 Creating ultra-advanced features...")
        
        featured_df = df.copy()
        
        # Price-based features
        featured_df['Returns'] = featured_df['Close'].pct_change()
        featured_df['Log_Returns'] = np.log(featured_df['Close'] / featured_df['Close'].shift(1))
        featured_df['Volatility_5'] = featured_df['Returns'].rolling(5).std()
        featured_df['Volatility_20'] = featured_df['Returns'].rolling(20).std()
        featured_df['Volatility_60'] = featured_df['Returns'].rolling(60).std()
        
        # Advanced Price Patterns
        featured_df['Price_Position'] = (featured_df['Close'] - featured_df['Low']) / (featured_df['High'] - featured_df['Low'])
        featured_df['HL_Ratio'] = featured_df['High'] / featured_df['Low']
        featured_df['OC_Ratio'] = featured_df['Open'] / featured_df['Close']
        
        # Multi-timeframe Moving Averages
        for period in [5, 10, 20, 50, 100, 200]:
            featured_df[f'SMA_{period}'] = featured_df['Close'].rolling(period).mean()
            featured_df[f'EMA_{period}'] = featured_df['Close'].ewm(span=period).mean()
            featured_df[f'Price_to_SMA_{period}'] = featured_df['Close'] / featured_df[f'SMA_{period}']
            featured_df[f'Price_to_EMA_{period}'] = featured_df['Close'] / featured_df[f'EMA_{period}']
        
        # Technical Indicators (TALib)
        try:
            # Momentum Indicators
            featured_df['RSI_14'] = talib.RSI(featured_df['Close'].values, 14)
            featured_df['RSI_30'] = talib.RSI(featured_df['Close'].values, 30)
            featured_df['STOCH_K'], featured_df['STOCH_D'] = talib.STOCH(
                featured_df['High'].values, featured_df['Low'].values, featured_df['Close'].values
            )
            featured_df['WILLR'] = talib.WILLR(
                featured_df['High'].values, featured_df['Low'].values, featured_df['Close'].values
            )
            featured_df['MOM_10'] = talib.MOM(featured_df['Close'].values, 10)
            featured_df['ROC_10'] = talib.ROC(featured_df['Close'].values, 10)
            
            # Volatility Indicators
            featured_df['ATR_14'] = talib.ATR(
                featured_df['High'].values, featured_df['Low'].values, featured_df['Close'].values, 14
            )
            featured_df['NATR'] = talib.NATR(
                featured_df['High'].values, featured_df['Low'].values, featured_df['Close'].values
            )
            
            # Volume Indicators
            featured_df['OBV'] = talib.OBV(featured_df['Close'].values, featured_df['Volume'].values)
            featured_df['AD'] = talib.AD(
                featured_df['High'].values, featured_df['Low'].values, 
                featured_df['Close'].values, featured_df['Volume'].values
            )
            
            # Trend Indicators
            featured_df['ADX'] = talib.ADX(
                featured_df['High'].values, featured_df['Low'].values, featured_df['Close'].values
            )
            featured_df['CCI'] = talib.CCI(
                featured_df['High'].values, featured_df['Low'].values, featured_df['Close'].values
            )
            featured_df['AROON_UP'], featured_df['AROON_DOWN'] = talib.AROON(
                featured_df['High'].values, featured_df['Low'].values
            )
            
            # MACD
            featured_df['MACD'], featured_df['MACD_Signal'], featured_df['MACD_Hist'] = talib.MACD(
                featured_df['Close'].values
            )
            
            # Bollinger Bands
            featured_df['BB_Upper'], featured_df['BB_Middle'], featured_df['BB_Lower'] = talib.BBANDS(
                featured_df['Close'].values
            )
            featured_df['BB_Width'] = (featured_df['BB_Upper'] - featured_df['BB_Lower']) / featured_df['BB_Middle']
            featured_df['BB_Position'] = (featured_df['Close'] - featured_df['BB_Lower']) / (featured_df['BB_Upper'] - featured_df['BB_Lower'])
            
        except Exception as e:
            self.logger.warning(f"Error calculating TALib indicators: {str(e)}")
        
        # Advanced Volume Analysis
        featured_df['Volume_SMA_20'] = featured_df['Volume'].rolling(20).mean()
        featured_df['Volume_Ratio'] = featured_df['Volume'] / featured_df['Volume_SMA_20']
        featured_df['VWAP'] = (featured_df['Volume'] * featured_df['Close']).cumsum() / featured_df['Volume'].cumsum()
        featured_df['Price_to_VWAP'] = featured_df['Close'] / featured_df['VWAP']
        
        # Market Microstructure Features
        featured_df['Spread'] = featured_df['High'] - featured_df['Low']
        featured_df['Body'] = abs(featured_df['Close'] - featured_df['Open'])
        featured_df['Upper_Shadow'] = featured_df['High'] - np.maximum(featured_df['Open'], featured_df['Close'])
        featured_df['Lower_Shadow'] = np.minimum(featured_df['Open'], featured_df['Close']) - featured_df['Low']
        featured_df['Shadow_Ratio'] = (featured_df['Upper_Shadow'] + featured_df['Lower_Shadow']) / featured_df['Body']
        
        # Fractal and Chaos Features
        featured_df['Hurst_Exponent'] = self.calculate_hurst_exponent(featured_df['Close'])
        featured_df['Fractal_Dimension'] = 2 - featured_df['Hurst_Exponent']
        
        # Regime Detection Features
        featured_df['Trend_Strength'] = self.calculate_trend_strength(featured_df['Close'])
        featured_df['Market_Regime'] = self.detect_market_regime(featured_df)
        
        # Seasonal and Calendar Features
        featured_df['DayOfWeek'] = featured_df['Date'].dt.dayofweek
        featured_df['Month'] = featured_df['Date'].dt.month
        featured_df['Quarter'] = featured_df['Date'].dt.quarter
        featured_df['DayOfMonth'] = featured_df['Date'].dt.day
        featured_df['IsMonthEnd'] = featured_df['Date'].dt.is_month_end.astype(int)
        featured_df['IsQuarterEnd'] = featured_df['Date'].dt.is_quarter_end.astype(int)
        
        # Lag Features
        for lag in [1, 2, 3, 5, 10]:
            featured_df[f'Close_Lag_{lag}'] = featured_df['Close'].shift(lag)
            featured_df[f'Returns_Lag_{lag}'] = featured_df['Returns'].shift(lag)
            featured_df[f'Volume_Lag_{lag}'] = featured_df['Volume'].shift(lag)
        
        # Rolling Statistics
        for window in [5, 10, 20]:
            featured_df[f'Close_Mean_{window}'] = featured_df['Close'].rolling(window).mean()
            featured_df[f'Close_Std_{window}'] = featured_df['Close'].rolling(window).std()
            featured_df[f'Close_Skew_{window}'] = featured_df['Close'].rolling(window).skew()
            featured_df[f'Close_Kurt_{window}'] = featured_df['Close'].rolling(window).kurt()
            featured_df[f'Volume_Mean_{window}'] = featured_df['Volume'].rolling(window).mean()
            featured_df[f'Volume_Std_{window}'] = featured_df['Volume'].rolling(window).std()
        
        # Advanced Momentum Features
        for window in [5, 10, 20]:
            featured_df[f'Momentum_{window}'] = featured_df['Close'] / featured_df['Close'].shift(window) - 1
            featured_df[f'Acceleration_{window}'] = featured_df[f'Momentum_{window}'] - featured_df[f'Momentum_{window}'].shift(1)
        
        # Clean data
        featured_df = featured_df.dropna()
        featured_df = featured_df.replace([np.inf, -np.inf], np.nan).dropna()
        
        print(f"✅ Created {len(featured_df.columns)} features")
        return featured_df
    
    def calculate_hurst_exponent(self, price_series: pd.Series, max_lag: int = 20) -> pd.Series:
        """
        Calculate Hurst Exponent for fractal analysis
        """
        def hurst_single(ts):
            try:
                if len(ts) < max_lag * 2:
                    return 0.5
                
                lags = range(2, max_lag)
                tau = [np.sqrt(np.std(np.subtract(ts[lag:], ts[:-lag]))) for lag in lags]
                poly = np.polyfit(np.log(lags), np.log(tau), 1)
                return poly[0] * 2.0
            except:
                return 0.5
        
        return price_series.rolling(60).apply(hurst_single, raw=True).fillna(0.5)
    
    def calculate_trend_strength(self, price_series: pd.Series, window: int = 20) -> pd.Series:
        """
        Calculate trend strength using linear regression
        """
        def trend_strength_single(ts):
            try:
                if len(ts) < 3:
                    return 0
                x = np.arange(len(ts))
                slope, _ = np.polyfit(x, ts, 1)
                return slope / np.mean(ts)
            except:
                return 0
        
        return price_series.rolling(window).apply(trend_strength_single, raw=True).fillna(0)
    
    def detect_market_regime(self, df: pd.DataFrame) -> pd.Series:
        """
        Detect market regime (Bull/Bear/Sideways)
        """
        try:
            # Calculate regime based on multiple factors
            sma_20 = df['Close'].rolling(20).mean()
            sma_50 = df['Close'].rolling(50).mean()
            volatility = df['Returns'].rolling(20).std()
            
            conditions = [
                (df['Close'] > sma_20) & (sma_20 > sma_50) & (volatility < volatility.quantile(0.3)),  # Bull
                (df['Close'] < sma_20) & (sma_20 < sma_50) & (volatility < volatility.quantile(0.3)),  # Bear
            ]
            choices = [2, 0]  # Bull=2, Bear=0, Sideways=1
            
            regime = np.select(conditions, choices, default=1)
            return pd.Series(regime, index=df.index)
        except:
            return pd.Series(1, index=df.index)  # Default to sideways
    
    def create_transformer_model(self, input_shape: Tuple[int, int]) -> Model:
        """
        Create Transformer-based model for time series forecasting
        """
        inputs = Input(shape=input_shape)
        
        # Multi-head attention layers
        attention1 = MultiHeadAttention(num_heads=8, key_dim=64)(inputs, inputs)
        attention1 = LayerNormalization()(attention1 + inputs)
        
        attention2 = MultiHeadAttention(num_heads=4, key_dim=32)(attention1, attention1)
        attention2 = LayerNormalization()(attention2 + attention1)
        
        # Global pooling
        pooled = GlobalAveragePooling1D()(attention2)
        
        # Dense layers
        dense1 = Dense(128, activation='relu')(pooled)
        dense1 = Dropout(0.3)(dense1)
        dense1 = BatchNormalization()(dense1)
        
        dense2 = Dense(64, activation='relu')(dense1)
        dense2 = Dropout(0.2)(dense2)
        
        outputs = Dense(1)(dense2)
        
        model = Model(inputs=inputs, outputs=outputs)
        model.compile(optimizer=AdamW(learning_rate=0.001), loss='huber', metrics=['mae'])
        
        return model
    
    def create_conv_lstm_model(self, input_shape: Tuple[int, int]) -> Model:
        """
        Create CNN-LSTM hybrid model
        """
        inputs = Input(shape=input_shape)
        
        # Convolutional layers
        conv1 = Conv1D(filters=64, kernel_size=3, activation='relu', padding='same')(inputs)
        conv1 = BatchNormalization()(conv1)
        conv1 = Dropout(0.2)(conv1)
        
        conv2 = SeparableConv1D(filters=128, kernel_size=3, activation='relu', padding='same')(conv1)
        conv2 = BatchNormalization()(conv2)
        conv2 = Dropout(0.2)(conv2)
        
        # LSTM layers
        lstm1 = LSTM(100, return_sequences=True)(conv2)
        lstm1 = Dropout(0.3)(lstm1)
        
        lstm2 = LSTM(50, return_sequences=False)(lstm1)
        lstm2 = Dropout(0.3)(lstm2)
        
        # Dense layers
        dense1 = Dense(64, activation='relu')(lstm2)
        dense1 = Dropout(0.2)(dense1)
        
        outputs = Dense(1)(dense1)
        
        model = Model(inputs=inputs, outputs=outputs)
        model.compile(optimizer=Adam(learning_rate=0.001), loss='huber', metrics=['mae'])
        
        return model
    
    def create_advanced_lstm_model(self, input_shape: Tuple[int, int]) -> Model:
        """
        Create advanced LSTM model with residual connections
        """
        inputs = Input(shape=input_shape)
        
        # First LSTM block
        lstm1 = LSTM(128, return_sequences=True, kernel_regularizer=l1_l2(0.01, 0.01))(inputs)
        lstm1 = BatchNormalization()(lstm1)
        lstm1 = Dropout(0.3)(lstm1)
        
        # Second LSTM block with residual connection
        lstm2 = LSTM(128, return_sequences=True, kernel_regularizer=l1_l2(0.01, 0.01))(lstm1)
        lstm2 = BatchNormalization()(lstm2)
        lstm2 = Add()([lstm1, lstm2])  # Residual connection
        lstm2 = Dropout(0.3)(lstm2)
        
        # Third LSTM block
        lstm3 = LSTM(64, return_sequences=False)(lstm2)
        lstm3 = BatchNormalization()(lstm3)
        lstm3 = Dropout(0.3)(lstm3)
        
        # Dense layers
        dense1 = Dense(64, activation='relu')(lstm3)
        dense1 = Dropout(0.2)(dense1)
        
        dense2 = Dense(32, activation='relu')(dense1)
        dense2 = Dropout(0.2)(dense2)
        
        outputs = Dense(1)(dense2)
        
        model = Model(inputs=inputs, outputs=outputs)
        model.compile(optimizer=Adam(learning_rate=0.001), loss='huber', metrics=['mae'])
        
        return model
    
    def create_ensemble_models(self, input_shape: Tuple[int, int]) -> Dict[str, Model]:
        """
        Create ensemble of different model architectures
        """
        models = {}
        
        print("🏗️ Creating ensemble models...")
        
        # 1. Transformer Model
        models['transformer'] = self.create_transformer_model(input_shape)
        print("✅ Transformer model created")
        
        # 2. CNN-LSTM Hybrid
        models['conv_lstm'] = self.create_conv_lstm_model(input_shape)
        print("✅ CNN-LSTM model created")
        
        # 3. Advanced LSTM
        models['advanced_lstm'] = self.create_advanced_lstm_model(input_shape)
        print("✅ Advanced LSTM model created")
        
        # 4. Bidirectional GRU
        models['bi_gru'] = Sequential([
            Bidirectional(GRU(64, return_sequences=True), input_shape=input_shape),
            Dropout(0.3),
            Bidirectional(GRU(32, return_sequences=False)),
            Dropout(0.3),
            Dense(32, activation='relu'),
            Dropout(0.2),
            Dense(1)
        ])
        models['bi_gru'].compile(optimizer=Adam(learning_rate=0.001), loss='huber', metrics=['mae'])
        print("✅ Bidirectional GRU model created")
        
        return models
    
    def prepare_sequences(self, df: pd.DataFrame, target_col: str = 'Close') -> Tuple[np.ndarray, np.ndarray, List[str]]:
        """
        Prepare sequences for training
        """
        print("📊 Preparing sequences for training...")
        
        # Select features (exclude non-numeric and target)
        exclude_cols = ['Date', target_col]
        feature_cols = [col for col in df.columns if col not in exclude_cols and df[col].dtype in [np.float64, np.int64]]
        
        # Prepare data
        features = df[feature_cols].values
        target = df[target_col].values
        
        # Create sequences
        X, y = [], []
        for i in range(self.sequence_length, len(features)):
            X.append(features[i-self.sequence_length:i])
            y.append(target[i])
        
        X = np.array(X)
        y = np.array(y)
        
        print(f"✅ Created {len(X)} sequences with {X.shape[2]} features")
        return X, y, feature_cols
    
    def optimize_hyperparameters(self, X_train: np.ndarray, y_train: np.ndarray, 
                                 X_val: np.ndarray, y_val: np.ndarray) -> Dict:
        """
        Optimize hyperparameters using Optuna
        """
        print("🎯 Optimizing hyperparameters...")
        
        def objective(trial):
            # Suggest hyperparameters
            learning_rate = trial.suggest_float('learning_rate', 1e-5, 1e-2, log=True)
            lstm_units = trial.suggest_int('lstm_units', 32, 256, step=32)
            dropout_rate = trial.suggest_float('dropout_rate', 0.1, 0.5)
            batch_size = trial.suggest_categorical('batch_size', [16, 32, 64])
            
            # Create model
            model = Sequential([
                LSTM(lstm_units, return_sequences=True, input_shape=(X_train.shape[1], X_train.shape[2])),
                Dropout(dropout_rate),
                LSTM(lstm_units // 2, return_sequences=False),
                Dropout(dropout_rate),
                Dense(32, activation='relu'),
                Dense(1)
            ])
            
            model.compile(optimizer=Adam(learning_rate=learning_rate), loss='huber', metrics=['mae'])
            
            # Train model
            try:
                history = model.fit(
                    X_train, y_train,
                    validation_data=(X_val, y_val),
                    epochs=20,
                    batch_size=batch_size,
                    verbose=0,
                    callbacks=[EarlyStopping(patience=5, restore_best_weights=True)]
                )
                
                return min(history.history['val_loss'])
            except:
                return float('inf')
        
        # Run optimization
        study = optuna.create_study(direction='minimize')
        study.optimize(objective, n_trials=20, timeout=1800)  # 30 minutes max
        
        print(f"✅ Best hyperparameters: {study.best_params}")
        return study.best_params
    
    def train_ensemble(self, df: pd.DataFrame) -> Dict:
        """
        Train ensemble of models
        """
        print("🚀 Training ultra-enhanced ensemble...")
        
        # Create features
        featured_df = self.create_ultra_advanced_features(df)
        
        # Prepare sequences
        X, y, feature_cols = self.prepare_sequences(featured_df)
        
        # Split data
        split_idx = int(len(X) * 0.8)
        X_train, X_val = X[:split_idx], X[split_idx:]
        y_train, y_val = y[:split_idx], y[split_idx:]
        
        # Scale features
        scaler = RobustScaler()
        X_train_scaled = scaler.fit_transform(X_train.reshape(-1, X_train.shape[-1])).reshape(X_train.shape)
        X_val_scaled = scaler.transform(X_val.reshape(-1, X_val.shape[-1])).reshape(X_val.shape)
        
        # Scale target
        target_scaler = RobustScaler()
        y_train_scaled = target_scaler.fit_transform(y_train.reshape(-1, 1)).flatten()
        y_val_scaled = target_scaler.transform(y_val.reshape(-1, 1)).flatten()
        
        # Save scalers
        self.scalers['feature_scaler'] = scaler
        self.scalers['target_scaler'] = target_scaler
        
        # Optimize hyperparameters
        best_params = self.optimize_hyperparameters(X_train_scaled, y_train_scaled, X_val_scaled, y_val_scaled)
        
        # Create ensemble models
        ensemble_models = self.create_ensemble_models((X_train_scaled.shape[1], X_train_scaled.shape[2]))
        
        # Train each model
        results = {}
        for name, model in ensemble_models.items():
            print(f"🏋️ Training {name} model...")
            
            callbacks = [
                EarlyStopping(patience=20, restore_best_weights=True, monitor='val_loss'),
                ReduceLROnPlateau(factor=0.5, patience=10, min_lr=1e-7),
                ModelCheckpoint(f'models/{name}_best.keras', save_best_only=True, monitor='val_loss')
            ]
            
            history = model.fit(
                X_train_scaled, y_train_scaled,
                validation_data=(X_val_scaled, y_val_scaled),
                epochs=100,
                batch_size=best_params.get('batch_size', 32),
                callbacks=callbacks,
                verbose=0
            )
            
            # Evaluate model
            val_predictions = model.predict(X_val_scaled, verbose=0)
            val_predictions = target_scaler.inverse_transform(val_predictions.reshape(-1, 1)).flatten()
            y_val_original = target_scaler.inverse_transform(y_val_scaled.reshape(-1, 1)).flatten()
            
            # Calculate metrics
            mse = mean_squared_error(y_val_original, val_predictions)
            mae = mean_absolute_error(y_val_original, val_predictions)
            rmse = np.sqrt(mse)
            mape = np.mean(np.abs((y_val_original - val_predictions) / y_val_original)) * 100
            r2 = r2_score(y_val_original, val_predictions)
            
            # Directional accuracy
            actual_direction = np.diff(y_val_original) > 0
            pred_direction = np.diff(val_predictions) > 0
            directional_accuracy = np.mean(actual_direction == pred_direction) * 100
            
            results[name] = {
                'model': model,
                'mse': mse,
                'mae': mae,
                'rmse': rmse,
                'mape': mape,
                'r2': r2,
                'directional_accuracy': directional_accuracy,
                'history': history.history
            }
            
            print(f"✅ {name} - RMSE: ${rmse:.2f}, MAPE: {mape:.2f}%, Directional: {directional_accuracy:.1f}%")
        
        # Train traditional ML models for comparison
        print("🌲 Training traditional ML models...")
        
        # Prepare data for traditional ML
        X_train_flat = X_train_scaled.reshape(X_train_scaled.shape[0], -1)
        X_val_flat = X_val_scaled.reshape(X_val_scaled.shape[0], -1)
        
        # Random Forest
        rf_model = RandomForestRegressor(n_estimators=200, max_depth=10, random_state=42, n_jobs=-1)
        rf_model.fit(X_train_flat, y_train_scaled)
        rf_predictions = rf_model.predict(X_val_flat)
        rf_predictions = target_scaler.inverse_transform(rf_predictions.reshape(-1, 1)).flatten()
        
        rf_mae = mean_absolute_error(y_val_original, rf_predictions)
        rf_mape = np.mean(np.abs((y_val_original - rf_predictions) / y_val_original)) * 100
        
        results['random_forest'] = {
            'model': rf_model,
            'mae': rf_mae,
            'mape': rf_mape,
            'predictions': rf_predictions
        }
        
        # Gradient Boosting
        gb_model = GradientBoostingRegressor(n_estimators=200, max_depth=6, random_state=42)
        gb_model.fit(X_train_flat, y_train_scaled)
        gb_predictions = gb_model.predict(X_val_flat)
        gb_predictions = target_scaler.inverse_transform(gb_predictions.reshape(-1, 1)).flatten()
        
        gb_mae = mean_absolute_error(y_val_original, gb_predictions)
        gb_mape = np.mean(np.abs((y_val_original - gb_predictions) / y_val_original)) * 100
        
        results['gradient_boosting'] = {
            'model': gb_model,
            'mae': gb_mae,
            'mape': gb_mape,
            'predictions': gb_predictions
        }
        
        print(f"✅ Random Forest - MAE: ${rf_mae:.2f}, MAPE: {rf_mape:.2f}%")
        print(f"✅ Gradient Boosting - MAE: ${gb_mae:.2f}, MAPE: {gb_mape:.2f}%")
        
        # Save models and results
        self.models = results
        self.performance_metrics = {name: {k: v for k, v in model_results.items() if k != 'model'} 
                                   for name, model_results in results.items()}
        
        # Save everything
        joblib.dump(self.scalers, f'models/{self.symbol}_ultra_scalers.pkl')
        joblib.dump(self.performance_metrics, f'models/{self.symbol}_ultra_metrics.pkl')
        
        print("✅ Ultra-enhanced ensemble training completed!")
        return results
    
    def generate_ensemble_forecast(self, df: pd.DataFrame, forecast_days: int = 30) -> Dict:
        """
        Generate ensemble forecast with confidence intervals
        """
        print(f"🔮 Generating {forecast_days}-day ensemble forecast...")
        
        if not self.models:
            raise ValueError("Models not trained. Call train_ensemble first.")
        
        # Create features
        featured_df = self.create_ultra_advanced_features(df)
        
        # Prepare last sequence
        _, _, feature_cols = self.prepare_sequences(featured_df)
        features = featured_df[feature_cols].values
        
        # Get last sequence
        last_sequence = features[-self.sequence_length:]
        
        # Scale features
        scaler = self.scalers['feature_scaler']
        target_scaler = self.scalers['target_scaler']
        
        last_sequence_scaled = scaler.transform(last_sequence.reshape(-1, features.shape[-1])).reshape(1, self.sequence_length, -1)
        
        # Generate forecasts from each model
        forecasts = {}
        ensemble_predictions = []
        
        for name, model_data in self.models.items():
            if 'model' in model_data:
                model = model_data['model']
                
                if hasattr(model, 'predict'):  # Deep learning models
                    prediction = model.predict(last_sequence_scaled, verbose=0)
                    prediction = target_scaler.inverse_transform(prediction.reshape(-1, 1)).flatten()[0]
                else:  # Traditional ML models
                    prediction = model.predict(last_sequence_scaled.reshape(1, -1))
                    prediction = target_scaler.inverse_transform(prediction.reshape(-1, 1)).flatten()[0]
                
                forecasts[name] = prediction
                ensemble_predictions.append(prediction)
        
        # Calculate ensemble statistics
        ensemble_mean = np.mean(ensemble_predictions)
        ensemble_std = np.std(ensemble_predictions)
        ensemble_median = np.median(ensemble_predictions)
        
        # Confidence intervals
        confidence_95_lower = ensemble_mean - 1.96 * ensemble_std
        confidence_95_upper = ensemble_mean + 1.96 * ensemble_std
        confidence_80_lower = ensemble_mean - 1.28 * ensemble_std
        confidence_80_upper = ensemble_mean + 1.28 * ensemble_std
        
        # Model weights based on performance
        weights = {}
        for name in forecasts.keys():
            if name in self.performance_metrics:
                # Weight based on inverse MAPE (better performance = higher weight)
                mape = self.performance_metrics[name].get('mape', 100)
                weights[name] = 1 / (1 + mape)
        
        # Normalize weights
        total_weight = sum(weights.values())
        if total_weight > 0:
            weights = {k: v/total_weight for k, v in weights.items()}
            weighted_prediction = sum(forecasts[name] * weights.get(name, 0) for name in forecasts.keys())
        else:
            weighted_prediction = ensemble_mean
        
        # Current price for comparison
        current_price = df['Close'].iloc[-1]
        
        results = {
            'current_price': current_price,
            'individual_forecasts': forecasts,
            'ensemble_mean': ensemble_mean,
            'ensemble_median': ensemble_median,
            'ensemble_std': ensemble_std,
            'weighted_prediction': weighted_prediction,
            'confidence_intervals': {
                '95%': {'lower': confidence_95_lower, 'upper': confidence_95_upper},
                '80%': {'lower': confidence_80_lower, 'upper': confidence_80_upper}
            },
            'model_weights': weights,
            'price_change': weighted_prediction - current_price,
            'price_change_pct': ((weighted_prediction - current_price) / current_price) * 100,
            'forecast_date': (datetime.now() + timedelta(days=forecast_days)).strftime('%Y-%m-%d'),
            'model_performance': self.performance_metrics
        }
        
        print(f"✅ Forecast generated:")
        print(f"   Current Price: ${current_price:.2f}")
        print(f"   Predicted Price: ${weighted_prediction:.2f}")
        print(f"   Expected Change: {results['price_change_pct']:+.2f}%")
        print(f"   95% Confidence: ${confidence_95_lower:.2f} - ${confidence_95_upper:.2f}")
        
        return results
    
    def run_comprehensive_analysis(self, symbol: str, forecast_days: int = 30) -> Dict:
        """
        Run comprehensive analysis and forecasting
        """
        self.symbol = symbol
        print(f"🚀 Starting comprehensive analysis for {symbol}")
        
        # Fetch data
        end_date = datetime.now()
        start_date = end_date - timedelta(days=self.lookback_days * 2)  # Extra data for features
        
        df = self.fetch_advanced_data(start_date.strftime('%Y-%m-%d'), end_date.strftime('%Y-%m-%d'))
        
        # Train ensemble
        training_results = self.train_ensemble(df)
        
        # Generate forecast
        forecast_results = self.generate_ensemble_forecast(df, forecast_days)
        
        # Combine results
        comprehensive_results = {
            'symbol': symbol,
            'analysis_date': datetime.now().isoformat(),
            'data_points': len(df),
            'training_results': {name: {k: v for k, v in results.items() if k != 'model'} 
                               for name, results in training_results.items()},
            'forecast_results': forecast_results,
            'recommendations': self.generate_recommendations(forecast_results)
        }
        
        # Save results
        joblib.dump(comprehensive_results, f'results/{symbol}_ultra_analysis_{datetime.now().strftime("%Y%m%d_%H%M%S")}.pkl')
        
        print("✅ Comprehensive analysis completed!")
        return comprehensive_results
    
    def generate_recommendations(self, forecast_results: Dict) -> List[str]:
        """
        Generate investment recommendations based on forecast
        """
        recommendations = []
        
        price_change_pct = forecast_results['price_change_pct']
        confidence_width = forecast_results['confidence_intervals']['95%']['upper'] - forecast_results['confidence_intervals']['95%']['lower']
        current_price = forecast_results['current_price']
        
        # Price direction recommendation
        if price_change_pct > 5:
            recommendations.append("🟢 Strong Buy Signal - Expected significant upside")
        elif price_change_pct > 2:
            recommendations.append("🟢 Buy Signal - Positive momentum expected")
        elif price_change_pct > -2:
            recommendations.append("🟡 Hold Signal - Neutral outlook")
        elif price_change_pct > -5:
            recommendations.append("🔴 Sell Signal - Negative momentum expected")
        else:
            recommendations.append("🔴 Strong Sell Signal - Significant downside risk")
        
        # Confidence assessment
        confidence_pct = (confidence_width / current_price) * 100
        if confidence_pct < 5:
            recommendations.append("🎯 High Confidence - Low forecast uncertainty")
        elif confidence_pct < 10:
            recommendations.append("📊 Medium Confidence - Moderate forecast uncertainty")
        else:
            recommendations.append("⚠️ Low Confidence - High forecast uncertainty")
        
        # Risk assessment
        if abs(price_change_pct) > 10:
            recommendations.append("⚠️ High Volatility Expected - Use position sizing")
        
        # Model agreement
        individual_forecasts = list(forecast_results['individual_forecasts'].values())
        forecast_std = np.std(individual_forecasts)
        if forecast_std / current_price < 0.02:
            recommendations.append("🤝 Strong Model Agreement - Consensus forecast")
        elif forecast_std / current_price < 0.05:
            recommendations.append("📊 Moderate Model Agreement")
        else:
            recommendations.append("🔄 Mixed Model Signals - Consider additional analysis")
        
        return recommendations

if __name__ == "__main__":
    # Example usage
    forecaster = UltraEnhancedForecaster()
    results = forecaster.run_comprehensive_analysis("AAPL", forecast_days=30)
    print(f"Analysis completed for AAPL: {results}") 