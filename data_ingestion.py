"""
Handles all data retrieval tasks:
- Connects to live market feeds (Yahoo Finance, Alpha Vantage, NSEpy)
- Downloads OHLCV (Open, High, Low, Close, Volume) data for specified tickers and date ranges
- Provides unified DataFrame outputs for downstream processing
"""
import yfinance as yf
from alpha_vantage.timeseries import TimeSeries
from nsepy import get_history
import pandas as pd
from datetime import datetime

def fetch_yfinance(ticker: str, start: str, end: str) -> pd.DataFrame:
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
    
    return df

def fetch_alpha_vantage(symbol: str, api_key: str, outputsize: str = 'full') -> pd.DataFrame:
    ts = TimeSeries(key=api_key, output_format='pandas')
    data, meta_data = ts.get_daily(symbol=symbol, outputsize=outputsize)
    if data is None or data.empty:
        raise ValueError(f"No data found for symbol {symbol}")
    
    data.index = pd.to_datetime(data.index)
    data = data.rename(columns={
        '1. open': 'Open', '2. high': 'High', '3. low': 'Low',
        '4. close': 'Close', '5. volume': 'Volume'
    })
    data.reset_index(inplace=True)
    data = data.rename(columns={'index': 'Date'})
    return data

def fetch_nsepy(symbol: str, start: str, end: str) -> pd.DataFrame:
    df = get_history(symbol=symbol, start=pd.to_datetime(start), end=pd.to_datetime(end))
    if df is None or df.empty:
        raise ValueError(f"No data found for symbol {symbol}")
    
    df.reset_index(inplace=True)
    return df
