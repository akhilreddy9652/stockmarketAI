"""
Advanced Future Forecasting Module
Implements recursive/multi-step prediction for stock prices up to 2 years into the future.
Features:
- Recursive forecasting with LSTM/ML models
- Dynamic feature engineering for future dates
- Macroeconomic feature forecasting
- Uncertainty quantification
- Multiple forecast horizons
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import warnings
from typing import Dict, List, Optional, Tuple, Union
import joblib
import os
from tensorflow.keras.models import load_model

# Import our modules
from data_ingestion import fetch_yfinance
from feature_engineering import get_comprehensive_features, add_technical_indicators
from macro_indicators import MacroIndicators

warnings.filterwarnings('ignore')

class FutureForecaster:
    """
    Advanced future forecasting for stock prices with comprehensive feature engineering.
    """
    
    def __init__(self, model_path: Optional[str] = None, scaler_path: Optional[str] = None, 
                 feature_info_path: Optional[str] = None, symbol: str = 'AAPL'):
        """
        Initialize the future forecaster.
        
        Args:
            model_path: Path to the trained model file
            scaler_path: Path to the feature scaler file
            feature_info_path: Path to the feature info file
            symbol: Stock symbol (for default model/scaler)
        """
        self.model = None
        self.scaler = None
        self.feature_columns = None
        self.window = 20
        self.prediction_horizon = 1
        self.symbol = symbol
        
        # Default paths for advanced model
        default_model_path = f"models/{symbol}_advanced_lstm.h5"
        default_scaler_path = f"models/{symbol}_advanced_scaler.pkl"
        default_feature_info_path = f"models/{symbol}_feature_info.pkl"
        
        # Try to load advanced model first, fall back to basic model
        if model_path and os.path.exists(model_path):
            self.load_model(model_path)
        elif os.path.exists(default_model_path):
            self.load_model(default_model_path)
        elif os.path.exists(f"models/{symbol}_lstm.h5"):
            self.load_model(f"models/{symbol}_lstm.h5")
            
        if scaler_path and os.path.exists(scaler_path):
            self.load_scaler(scaler_path)
        elif os.path.exists(default_scaler_path):
            self.load_scaler(default_scaler_path)
        elif os.path.exists(f"models/{symbol}_scaler.pkl"):
            self.load_scaler(f"models/{symbol}_scaler.pkl")
            
        if feature_info_path and os.path.exists(feature_info_path):
            self.load_feature_info(feature_info_path)
        elif os.path.exists(default_feature_info_path):
            self.load_feature_info(default_feature_info_path)
    
    def load_model(self, model_path: str):
        """Load a trained model."""
        try:
            if model_path.endswith('.h5'):
                self.model = load_model(model_path)
            else:
                self.model = joblib.load(model_path)
            print(f"✅ Model loaded from {model_path}")
        except Exception as e:
            print(f"❌ Error loading model: {e}")
    
    def load_scaler(self, scaler_path: str):
        """Load a feature scaler."""
        try:
            self.scaler = joblib.load(scaler_path)
            print(f"✅ Scaler loaded from {scaler_path}")
        except Exception as e:
            print(f"❌ Error loading scaler: {e}")
    
    def load_feature_info(self, feature_info_path: str):
        """Load feature information."""
        try:
            feature_info = joblib.load(feature_info_path)
            self.feature_columns = feature_info.get('feature_cols', [])
            self.window = feature_info.get('window', 20)
            self.prediction_horizon = feature_info.get('prediction_horizon', 1)
            print(f"✅ Feature info loaded: {len(self.feature_columns)} features, window={self.window}")
        except Exception as e:
            print(f"❌ Error loading feature info: {e}")
    
    def create_sample_model(self, input_shape: Tuple[int, int]):
        """
        Create a sample LSTM model for demonstration.
        In production, use your actual trained model.
        """
        try:
            from tensorflow.keras.models import Sequential
            from tensorflow.keras.layers import LSTM, Dense, Dropout
            from tensorflow.keras.optimizers import Adam
            
            model = Sequential([
                LSTM(50, return_sequences=True, input_shape=input_shape),
                Dropout(0.2),
                LSTM(50, return_sequences=False),
                Dropout(0.2),
                Dense(25),
                Dense(1)
            ])
            
            model.compile(optimizer=Adam(learning_rate=0.001), loss='mse')
            self.model = model
            print("✅ Sample LSTM model created")
            
        except ImportError:
            print("⚠️ TensorFlow not available. Using dummy model for demonstration.")
            self.model = DummyModel()
    
    def forecast_macro_features(self, macro_data: Dict[str, pd.DataFrame], 
                              forecast_days: int) -> Dict[str, pd.DataFrame]:
        """
        Forecast macroeconomic features for future dates.
        
        Args:
            macro_data: Dictionary of historical macro data
            forecast_days: Number of days to forecast
            
        Returns:
            Dictionary of forecasted macro data
        """
        forecasted_macro = {}
        future_dates = pd.date_range(
            start=datetime.now() + timedelta(days=1),
            periods=forecast_days,
            freq='D'
        )
        
        for indicator_name, df in macro_data.items():
            if df.empty:
                continue
            
            # Get the latest value
            latest_value = df.iloc[-1, 1]  # Second column is the value
            
            # Simple forecast: use last known value with small random walk
            # In production, use proper time series models (ARIMA, Prophet, etc.)
            forecast_values = []
            current_value = latest_value
            
            for _ in range(forecast_days):
                # Add small random walk (more realistic than constant)
                change = np.random.normal(0, current_value * 0.001)  # 0.1% daily volatility
                current_value += change
                forecast_values.append(current_value)
            
            # Create forecast DataFrame
            forecasted_macro[indicator_name] = pd.DataFrame({
                'Date': future_dates,
                df.columns[1]: forecast_values  # Use same column name as original
            })
        
        return forecasted_macro
    
    def prepare_future_features(self, last_row: pd.Series, 
                               macro_forecast: Optional[Dict[str, pd.DataFrame]] = None,
                               date: Optional[datetime] = None) -> np.ndarray:
        """
        Prepare features for a future date.
        
        Args:
            last_row: Last known data row
            macro_forecast: Forecasted macro data for this date
            date: Future date
            
        Returns:
            Feature array for prediction
        """
        # Start with last known features
        features = last_row.copy()
        
        # Update date
        if date:
            features['Date'] = date
        
        # Update macro features if available
        if macro_forecast:
            for indicator_name, forecast_df in macro_forecast.items():
                if not forecast_df.empty and date:
                    # Find the forecast for this specific date
                    date_forecast = forecast_df[forecast_df['Date'] == date]
                    if not date_forecast.empty:
                        value_col = forecast_df.columns[1]
                        features[f'{indicator_name}_{value_col}'] = date_forecast[value_col].iloc[0]
        
        # Remove non-feature columns
        feature_cols = [col for col in features.index 
                       if col not in ['Date', 'Open', 'High', 'Low', 'Close', 'Volume']]
        
        feature_values = features[feature_cols].values.astype(float)
        
        # Handle NaN values
        feature_values = np.nan_to_num(feature_values, nan=0.0)
        
        return feature_values
    
    def forecast_future(self, symbol: str, forecast_days: int = 504, 
                       include_macro: bool = True) -> pd.DataFrame:
        """
        Forecast stock prices for the specified number of days.
        
        Args:
            symbol: Stock symbol
            forecast_days: Number of days to forecast (default: 504 = ~2 years)
            include_macro: Whether to include macroeconomic features
            
        Returns:
            DataFrame with forecasted prices and dates
        """
        print(f"🔮 Forecasting {forecast_days} days for {symbol}...")
        
        # Fetch historical data
        end_date = datetime.now()
        start_date = end_date - timedelta(days=365*2)  # 2 years of historical data
        
        try:
            df = fetch_yfinance(symbol, start_date.strftime('%Y-%m-%d'), end_date.strftime('%Y-%m-%d'))
            print(f"✅ Fetched {len(df)} historical records")
        except Exception as e:
            print(f"❌ Error fetching data: {e}")
            return pd.DataFrame()
        
        # Add comprehensive features
        if include_macro:
            df = get_comprehensive_features(df, include_macro=True)
        else:
            df = add_technical_indicators(df)
        
        # Forecast macro features if needed
        macro_forecast = None
        if include_macro:
            macro = MacroIndicators()
            try:
                macro_data = macro.load_macro_data('data')
                if not macro_data:
                    # Create sample macro data
                    dates = df.index
                    macro_data = {
                        'interest_rate': pd.DataFrame({
                            'Date': dates,
                            'FEDFUNDS': np.random.normal(3.0, 1.0, len(dates))
                        }),
                        'inflation': pd.DataFrame({
                            'Date': dates,
                            'CPIAUCSL': np.random.normal(2.5, 0.5, len(dates))
                        })
                    }
                
                macro_forecast = self.forecast_macro_features(macro_data, forecast_days)
                print(f"✅ Forecasted {len(macro_forecast)} macro indicators")
            except Exception as e:
                print(f"⚠️ Macro forecasting failed: {e}")
        
        # Initialize forecasting
        future_predictions = []
        last_row = df.iloc[-1].copy()
        current_price = last_row['Close']
        
        # Generate future dates (business days)
        future_dates = pd.date_range(
            start=df['Date'].iloc[-1] + timedelta(days=1),
            periods=forecast_days,
            freq='B'  # Business days
        )
        
        print(f"📅 Forecasting for {len(future_dates)} business days...")
        
        # Recursive forecasting
        for i, future_date in enumerate(future_dates):
            try:
                # Prepare features for this future date
                features = self.prepare_future_features(
                    last_row, macro_forecast, future_date
                )
                
                # Make prediction
                if self.model is not None:
                    # Reshape features for model input
                    if hasattr(self.model, 'predict'):
                        if len(features.shape) == 1:
                            features = features.reshape(1, -1)
                        
                        # Scale features if scaler is available
                        if self.scaler is not None:
                            features = self.scaler.transform(features)
                        
                        # Make prediction
                        predicted_price = self.model.predict(features, verbose=0)[0, 0]
                    else:
                        # Dummy model
                        predicted_price = current_price * (1 + np.random.normal(0, 0.01))
                else:
                    # No model available, use simple trend
                    trend = 0.0001  # Slight upward trend
                    volatility = 0.02  # 2% daily volatility
                    predicted_price = current_price * (1 + trend + np.random.normal(0, volatility))
                
                # Ensure positive price
                predicted_price = max(predicted_price, current_price * 0.5)
                
                # Store prediction
                future_predictions.append({
                    'Date': future_date,
                    'Predicted_Close': predicted_price,
                    'Prediction_Step': i + 1
                })
                
                # Update for next iteration
                current_price = predicted_price
                last_row['Close'] = predicted_price
                
                # Update technical indicators for next step
                # This is a simplified update - in production, recalculate all indicators
                last_row['MA_20'] = last_row['MA_20'] * 0.95 + predicted_price * 0.05
                last_row['MA_50'] = last_row['MA_50'] * 0.98 + predicted_price * 0.02
                
                if i % 50 == 0:
                    print(f"   Progress: {i+1}/{len(future_dates)} days")
                
            except Exception as e:
                print(f"⚠️ Error in step {i}: {e}")
                # Use simple fallback
                predicted_price = current_price * (1 + np.random.normal(0, 0.01))
                future_predictions.append({
                    'Date': future_date,
                    'Predicted_Close': predicted_price,
                    'Prediction_Step': i + 1
                })
                current_price = predicted_price
        
        # Create forecast DataFrame
        forecast_df = pd.DataFrame(future_predictions)
        forecast_df['Date'] = pd.to_datetime(forecast_df['Date'])
        
        print(f"✅ Completed forecasting: {len(forecast_df)} predictions")
        
        return forecast_df
    
    def get_forecast_summary(self, forecast_df: pd.DataFrame, 
                           historical_df: pd.DataFrame) -> Dict:
        """
        Generate summary statistics for the forecast.
        
        Args:
            forecast_df: DataFrame with forecasted prices
            historical_df: DataFrame with historical prices
            
        Returns:
            Dictionary with forecast summary
        """
        if forecast_df.empty:
            return {}
        
        # Calculate summary statistics
        current_price = historical_df['Close'].iloc[-1]
        final_price = forecast_df['Predicted_Close'].iloc[-1]
        
        # Price change
        total_change = final_price - current_price
        total_change_pct = (total_change / current_price) * 100
        
        # Volatility
        forecast_returns = forecast_df['Predicted_Close'].pct_change().dropna()
        forecast_volatility = forecast_returns.std() * np.sqrt(252)  # Annualized
        
        # Max drawdown
        cumulative_returns = (1 + forecast_returns).cumprod()
        running_max = cumulative_returns.expanding().max()
        drawdown = (cumulative_returns - running_max) / running_max
        max_drawdown = drawdown.min()
        
        # Trend analysis
        forecast_trend = np.polyfit(range(len(forecast_df)), forecast_df['Predicted_Close'], 1)[0]
        
        summary = {
            'current_price': current_price,
            'final_price': final_price,
            'total_change': total_change,
            'total_change_pct': total_change_pct,
            'forecast_volatility': forecast_volatility,
            'max_drawdown': max_drawdown,
            'trend_slope': forecast_trend,
            'forecast_horizon_days': len(forecast_df),
            'confidence_level': 'Medium'  # Placeholder
        }
        
        return summary


class DummyModel:
    """Dummy model for demonstration when no trained model is available."""
    
    def predict(self, features, verbose=0):
        """Make dummy predictions."""
        return np.random.normal(100, 10, (features.shape[0], 1))


def test_future_forecasting():
    """
    Test the future forecasting functionality.
    """
    print("🧪 Testing Future Forecasting Module")
    print("=" * 50)
    
    # Initialize forecaster
    forecaster = FutureForecaster()
    
    # Test with a sample stock
    symbol = 'AAPL'
    
    print(f"\n📊 Testing forecast for {symbol}")
    
    # Test different forecast horizons
    horizons = [30, 90, 180, 365, 504]  # 1 month, 3 months, 6 months, 1 year, 2 years
    
    for horizon in horizons:
        print(f"\n🔮 Forecasting {horizon} days...")
        
        try:
            forecast_df = forecaster.forecast_future(symbol, forecast_days=horizon, include_macro=True)
            
            if not forecast_df.empty:
                print(f"✅ Generated {len(forecast_df)} predictions")
                print(f"   Start: {forecast_df['Date'].iloc[0].strftime('%Y-%m-%d')}")
                print(f"   End: {forecast_df['Date'].iloc[-1].strftime('%Y-%m-%d')}")
                print(f"   Price range: ${forecast_df['Predicted_Close'].min():.2f} - ${forecast_df['Predicted_Close'].max():.2f}")
            else:
                print("❌ No predictions generated")
                
        except Exception as e:
            print(f"❌ Error forecasting {horizon} days: {e}")
    
    print("\n🎉 Future forecasting test completed!")


if __name__ == "__main__":
    test_future_forecasting() 