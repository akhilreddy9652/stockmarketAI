import streamlit as st
import pandas as pd
from datetime import datetime, timedelta
import numpy as np
import plotly.express as px
import plotly.graph_objs as go
from plotly.subplots import make_subplots

# Handle imports with graceful fallbacks
try:
    from data_ingestion import fetch_yfinance
except ImportError:
    st.error("Data ingestion module not available. Using fallback.")
    import yfinance as yf
    def fetch_yfinance(symbol, start_date, end_date):
        try:
            data = yf.download(symbol, start=start_date, end=end_date)
            if data is not None and not data.empty:
                data.reset_index(inplace=True)
                return data
            else:
                return pd.DataFrame()
        except Exception as e:
            st.error(f"Error fetching data: {str(e)}")
            return pd.DataFrame()

try:
    from feature_engineering import add_technical_indicators, get_trading_signals
except ImportError:
    st.warning("Advanced features not available. Using basic functionality.")
    def add_technical_indicators(df):
        # Basic RSI calculation
        delta = df['Close'].diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
        rs = gain / loss
        df['RSI_14'] = 100 - (100 / (1 + rs))
        
        # Basic moving averages - use consistent names
        df['MA_20'] = df['Close'].rolling(window=20).mean()
        df['MA_50'] = df['Close'].rolling(window=50).mean()
        df['SMA_20'] = df['MA_20']  # Alias for compatibility
        df['SMA_50'] = df['MA_50']  # Alias for compatibility
        
        # Basic Bollinger Bands
        df['BB_Middle'] = df['Close'].rolling(window=20).mean()
        bb_std = df['Close'].rolling(window=20).std()
        df['BB_Upper'] = df['BB_Middle'] + (bb_std * 2)
        df['BB_Lower'] = df['BB_Middle'] - (bb_std * 2)
        
        # Add other missing columns with default values
        df['Volume_Ratio'] = 1.0
        df['Volatility'] = df['Close'].pct_change().rolling(window=20).std()
        df['ATR_14'] = (df['High'] - df['Low']).rolling(window=14).mean()
        df['Williams_R'] = -50.0  # Neutral value
        df['Momentum'] = df['Close'] - df['Close'].shift(10)
        df['ROC'] = ((df['Close'] - df['Close'].shift(10)) / df['Close'].shift(10)) * 100
        
        # Fill NaN values
        df = df.ffill().bfill()
        
        return df
    
    def get_trading_signals(df):
        signals = {}
        latest = df.iloc[-1]
        
        # RSI Signal
        if latest['RSI_14'] > 70:
            signals['RSI'] = {'signal': 'SELL', 'confidence': 0.7}
        elif latest['RSI_14'] < 30:
            signals['RSI'] = {'signal': 'BUY', 'confidence': 0.7}
        else:
            signals['RSI'] = {'signal': 'HOLD', 'confidence': 0.5}
        
        # Moving Average Signal
        if latest['Close'] > latest['MA_20'] > latest['MA_50']:
            signals['MA'] = {'signal': 'BUY', 'confidence': 0.6}
        elif latest['Close'] < latest['MA_20'] < latest['MA_50']:
            signals['MA'] = {'signal': 'SELL', 'confidence': 0.6}
        else:
            signals['MA'] = {'signal': 'HOLD', 'confidence': 0.4}
        
        return signals

st.set_page_config(page_title="Stock Predictor Dashboard", layout="wide")
st.title("📈 Stock Market Predictor Dashboard")

def is_indian_stock(symbol: str) -> bool:
    """Check if the stock symbol is for an Indian stock."""
    return symbol.upper().endswith(('.NS', '.BO', '.NSE', '.BSE'))

def get_currency_symbol(symbol: str) -> str:
    """Get the appropriate currency symbol based on the stock."""
    if is_indian_stock(symbol):
        return "₹"
    else:
        return "$"

def format_currency(value: float, symbol: str) -> str:
    """Format currency value with appropriate symbol and formatting."""
    if symbol == "₹":
        if value >= 10000000:  # 1 crore
            return f"₹{value/10000000:.2f}Cr"
        elif value >= 100000:  # 1 lakh
            return f"₹{value/100000:.2f}L"
        else:
            return f"₹{value:,.2f}"
    else:
        return f"${value:,.2f}"

# Sidebar controls
st.sidebar.header("Stock Selection")

# Popular stocks dropdown
popular_stocks = {
    "US Stocks": ["AAPL", "MSFT", "GOOGL", "TSLA", "AMZN", "META", "NVDA", "NFLX"],
    "Indian Stocks": [
        "^NSEI", "RELIANCE.NS", "TCS.NS", "HDFCBANK.NS", "INFY.NS", "ICICIBANK.NS", 
        "HINDUNILVR.NS", "ITC.NS", "SBIN.NS", "BHARTIARTL.NS", "AXISBANK.NS",
        "KOTAKBANK.NS", "ASIANPAINT.NS", "MARUTI.NS", "SUNPHARMA.NS", "TATAMOTORS.NS"
    ]
}

stock_category = st.sidebar.selectbox(
    "Select Stock Category",
    ["US Stocks", "Indian Stocks", "Custom"]
)

if stock_category == "Custom":
    symbol = st.sidebar.text_input("Enter Stock Symbol (e.g., AAPL, MSFT, TSLA, RELIANCE.NS)", value="AAPL")
else:
    symbol = st.sidebar.selectbox(
        f"Select {stock_category}",
        popular_stocks[stock_category],
        index=0
    )

end_date = st.sidebar.date_input("End Date", value=datetime.now().date())
start_date = st.sidebar.date_input("Start Date", value=(datetime.now() - timedelta(days=365)).date())

# Show currency information
if symbol:
    currency_symbol = get_currency_symbol(symbol)
    is_indian = is_indian_stock(symbol)
    
    st.sidebar.markdown("---")
    st.sidebar.subheader("💰 Currency Information")
    
    if is_indian:
        st.sidebar.success(f"🇮🇳 Indian Stock - Currency: {currency_symbol} (Rupees)")
    else:
        st.sidebar.success(f"🇺🇸 US Stock - Currency: {currency_symbol} (Dollars)")

# Analysis button
if st.sidebar.button("Analyze") or symbol:
    try:
        with st.spinner("📊 Fetching stock data..."):
            # Data ingestion
            df = fetch_yfinance(symbol, start_date.strftime('%Y-%m-%d'), end_date.strftime('%Y-%m-%d'))
            
            # Check if data was fetched successfully
            if df.empty:
                st.error(f"❌ No data found for {symbol}. Please check the symbol and try again.")
                st.stop()
            
            # Add technical indicators
            df = add_technical_indicators(df)
            
        st.success(f"✅ Loaded {len(df)} records for {symbol}")

        # Get currency symbol for this stock
        currency_symbol = get_currency_symbol(symbol)

        # Create tabs for different analyses
        tab1, tab2, tab3 = st.tabs([
            "📊 Stock Analysis", 
            "📈 Technical Indicators", 
            "🎯 Trading Signals"
        ])

        with tab1:
            # Price chart
            st.subheader(f"📈 Price Chart for {symbol}")
            
            fig = go.Figure()
            fig.add_trace(go.Scatter(
                x=df['Date'],
                y=df['Close'],
                mode='lines',
                name='Close Price',
                line=dict(color='#1f77b4', width=2)
            ))
            
            fig.update_layout(
                title=f"{symbol} Stock Price",
                xaxis_title="Date",
                yaxis_title=f"Price ({currency_symbol})",
                hovermode='x unified',
                height=500
            )
            
            st.plotly_chart(fig, use_container_width=True)

            # Current price and market info
            st.subheader("📊 Current Market Information")
            latest = df.iloc[-1]
            current_price = latest['Close']
            price_change = df['Close'].iloc[-1] - df['Close'].iloc[-2] if len(df) > 1 else 0
            price_change_pct = (price_change / df['Close'].iloc[-2]) * 100 if len(df) > 1 else 0
            
            col1, col2, col3, col4 = st.columns(4)
            with col1:
                st.metric(
                    "Current Price",
                    format_currency(float(current_price), currency_symbol),
                    f"{float(price_change_pct):+.2f}%"
                )
            with col2:
                st.metric(
                    "Day High",
                    format_currency(float(latest['High']), currency_symbol)
                )
            with col3:
                st.metric(
                    "Day Low",
                    format_currency(float(latest['Low']), currency_symbol)
                )
            with col4:
                st.metric(
                    "Volume",
                    f"{float(latest['Volume']):,.0f}"
                )

        with tab2:
            st.subheader("📈 Technical Indicators")
            
            # Technical indicators chart
            fig = make_subplots(
                rows=3, cols=1,
                subplot_titles=['Price with Bollinger Bands', 'RSI', 'Moving Averages'],
                vertical_spacing=0.1,
                row_heights=[0.5, 0.25, 0.25]
            )
            
            # Price with Bollinger Bands
            fig.add_trace(go.Scatter(x=df['Date'], y=df['Close'], name='Close', line=dict(color='blue')), row=1, col=1)
            fig.add_trace(go.Scatter(x=df['Date'], y=df['BB_Upper'], name='BB Upper', line=dict(color='red', dash='dash')), row=1, col=1)
            fig.add_trace(go.Scatter(x=df['Date'], y=df['BB_Lower'], name='BB Lower', line=dict(color='red', dash='dash')), row=1, col=1)
            
            # RSI
            fig.add_trace(go.Scatter(x=df['Date'], y=df['RSI_14'], name='RSI', line=dict(color='purple')), row=2, col=1)
            # Add horizontal lines for RSI levels
            fig.add_shape(type="line", x0=df['Date'].iloc[0], y0=70, x1=df['Date'].iloc[-1], y1=70,
                         line=dict(color="red", width=1, dash="dash"), row=2, col=1)
            fig.add_shape(type="line", x0=df['Date'].iloc[0], y0=30, x1=df['Date'].iloc[-1], y1=30,
                         line=dict(color="green", width=1, dash="dash"), row=2, col=1)
            
            # Moving Averages
            fig.add_trace(go.Scatter(x=df['Date'], y=df['MA_20'], name='SMA 20', line=dict(color='orange')), row=3, col=1)
            fig.add_trace(go.Scatter(x=df['Date'], y=df['MA_50'], name='SMA 50', line=dict(color='red')), row=3, col=1)
            
            fig.update_layout(height=800, showlegend=True)
            st.plotly_chart(fig, use_container_width=True)
            
            # Show latest technical indicators
            st.subheader("📊 Latest Technical Indicators")
            cols = st.columns(3)
            with cols[0]:
                st.metric("RSI (14)", f"{float(latest['RSI_14']):.2f}")
                st.metric("SMA 20", format_currency(float(latest['MA_20']), currency_symbol))
            with cols[1]:
                st.metric("Bollinger Upper", format_currency(float(latest['BB_Upper']), currency_symbol))
                st.metric("Bollinger Lower", format_currency(float(latest['BB_Lower']), currency_symbol))
            with cols[2]:
                st.metric("SMA 50", format_currency(float(latest['MA_50']), currency_symbol))
                vol = df['Close'].pct_change().std() * np.sqrt(252)
                st.metric("Volatility (Annualized)", f"{float(vol):.2%}")

        with tab3:
            st.subheader("🎯 Trading Signals")
            
            # Get trading signals
            signals = get_trading_signals(df)
            
            st.write("**Technical Analysis Signals**")
            for indicator, signal_data in signals.items():
                emoji = "🟢" if signal_data['signal'] == 'BUY' else "🔴" if signal_data['signal'] == 'SELL' else "⚪"
                st.write(f"{emoji} **{indicator}**: {signal_data['signal']} (Confidence: {signal_data['confidence']:.0%})")
            
            # Overall recommendation
            buy_signals = sum(1 for s in signals.values() if s['signal'] == 'BUY')
            sell_signals = sum(1 for s in signals.values() if s['signal'] == 'SELL')
            
            if buy_signals > sell_signals:
                overall_signal = "BUY"
                overall_color = "green"
            elif sell_signals > buy_signals:
                overall_signal = "SELL" 
                overall_color = "red"
            else:
                overall_signal = "HOLD"
                overall_color = "orange"
            
            st.markdown(f"### 🎯 **Overall Signal:** :{overall_color}[{overall_signal}]")
            
            # Risk metrics
            st.subheader("⚠️ Risk Metrics")
            returns = df['Close'].pct_change().dropna()
            
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("Daily Volatility", f"{float(returns.std()):.2%}")
            with col2:
                st.metric("Max Daily Gain", f"{float(returns.max()):.2%}")
            with col3:
                st.metric("Max Daily Loss", f"{float(returns.min()):.2%}")

    except Exception as e:
        st.error(f"❌ Error: {str(e)}")
        st.error("Please check your internet connection and try again. If the error persists, try a different stock symbol.")
        st.exception(e)

# Footer
st.markdown("---")
st.markdown("**📈 Stock Market Predictor Dashboard** - Built with Streamlit")
st.markdown("*Disclaimer: This is for educational purposes only. Not financial advice.*") 