"""
Handles all data retrieval tasks:
- Connects to live market feeds (Yahoo Finance, Alpha Vantage, NSEpy)
- Downloads OHLCV (Open, High, Low, Close, Volume) data for specified tickers and date ranges
- Provides unified DataFrame outputs for downstream processing
"""
import yfinance as yf
import pandas as pd
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

# Optional imports with fallbacks
try:
    from alpha_vantage.timeseries import TimeSeries
    ALPHA_VANTAGE_AVAILABLE = True
except ImportError:
    ALPHA_VANTAGE_AVAILABLE = False
    print("Warning: alpha_vantage not available. Install with: pip install alpha-vantage")

try:
    from nsepy import get_history
    NSEPY_AVAILABLE = True
except ImportError:
    NSEPY_AVAILABLE = False
    print("Warning: nsepy not available. Install with: pip install nsepy")

def fetch_yfinance(ticker: str, start: str, end: str) -> pd.DataFrame:
    """
    Fetch data from Yahoo Finance with robust error handling
    """
    try:
        df = yf.download(ticker, start=start, end=end)
        if df is None or df.empty:
            raise ValueError(f"No data found for ticker {ticker}")
        
        df.reset_index(inplace=True)
        
        # Handle yfinance column structure
        if isinstance(df.columns, pd.MultiIndex):
            # For yfinance data, flatten the multi-level columns
            df.columns = [col[1] if col[1] else col[0] for col in df.columns]
        
        # Ensure we have the standard column names
        if len(df.columns) >= 6:  # Date + OHLCV
            df.columns = ['Date', 'Open', 'High', 'Low', 'Close', 'Volume']
        elif len(df.columns) >= 5:  # OHLCV without Date
            df.columns = ['Open', 'High', 'Low', 'Close', 'Volume']
            df.insert(0, 'Date', df.index)
        
        # Clean the data
        df = df.dropna()
        df = df[df['Close'] > 0]  # Remove invalid prices
        
        return df
        
    except Exception as e:
        print(f"Error fetching data for {ticker}: {str(e)}")
        raise

def fetch_alpha_vantage(symbol: str, api_key: str, outputsize: str = 'full') -> pd.DataFrame:
    """
    Fetch data from Alpha Vantage with error handling
    """
    if not ALPHA_VANTAGE_AVAILABLE:
        raise ImportError("alpha_vantage library not available. Install with: pip install alpha-vantage")
    
    try:
        ts = TimeSeries(key=api_key, output_format='pandas')
        result = ts.get_daily(symbol=symbol, outputsize=outputsize)
        
        # Handle the tuple return from alpha_vantage
        if isinstance(result, tuple) and len(result) >= 2:
            data = result[0]  # First element is the DataFrame
        else:
            data = result
            
        if data is None or (hasattr(data, 'empty') and data.empty):
            raise ValueError(f"No data found for symbol {symbol}")
        
        # Ensure data is a DataFrame
        if not isinstance(data, pd.DataFrame):
            raise ValueError(f"Expected DataFrame but got {type(data)}")
        
        data.index = pd.to_datetime(data.index)
        data = data.rename(columns={
            '1. open': 'Open', '2. high': 'High', '3. low': 'Low',
            '4. close': 'Close', '5. volume': 'Volume'
        })
        data.reset_index(inplace=True)
        data = data.rename(columns={'index': 'Date'})
        return data
        
    except Exception as e:
        print(f"Error fetching Alpha Vantage data for {symbol}: {str(e)}")
        raise

def fetch_nsepy(symbol: str, start: str, end: str) -> pd.DataFrame:
    """
    Fetch data from NSEpy with error handling
    """
    if not NSEPY_AVAILABLE:
        raise ImportError("nsepy library not available. Install with: pip install nsepy")
    
    try:
        df = get_history(symbol=symbol, start=pd.to_datetime(start), end=pd.to_datetime(end))
        if df is None or df.empty:
            raise ValueError(f"No data found for symbol {symbol}")
        
        df.reset_index(inplace=True)
        return df
        
    except Exception as e:
        print(f"Error fetching NSEpy data for {symbol}: {str(e)}")
        raise

def get_data_source_status():
    """
    Check the availability of different data sources
    """
    return {
        'yfinance': True,  # Always available as it's a required dependency
        'alpha_vantage': ALPHA_VANTAGE_AVAILABLE,
        'nsepy': NSEPY_AVAILABLE
    }

# Test function to verify module is working
def test_data_ingestion():
    """
    Test the data ingestion module
    """
    try:
        # Test with a simple stock
        df = fetch_yfinance("AAPL", "2024-01-01", "2024-01-10")
        print(f"✅ Data ingestion test successful: {len(df)} records fetched")
        return True
    except Exception as e:
        print(f"❌ Data ingestion test failed: {str(e)}")
        return False

if __name__ == "__main__":
    print("Testing data ingestion module...")
    print(f"Data source status: {get_data_source_status()}")
    test_data_ingestion()
