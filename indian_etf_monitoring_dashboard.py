#!/usr/bin/env python3
"""
Indian ETF Real-Time Monitoring Dashboard
=========================================
Live monitoring and trading signals for Indian ETF portfolio
"""

import streamlit as st
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import yfinance as yf
import plotly.graph_objs as go
from plotly.subplots import make_subplots
import plotly.express as px
import time
import joblib
import os
from data_ingestion import fetch_yfinance
from feature_engineering import add_technical_indicators, get_trading_signals
import warnings
warnings.filterwarnings('ignore')

# Page configuration
st.set_page_config(
    page_title="🇮🇳 Indian ETF Portfolio Monitor", 
    layout="wide",
    initial_sidebar_state="expanded"
)

class IndianETFMonitor:
    """Real-time monitoring system for Indian ETF portfolio"""
    
    def __init__(self):
        self.portfolio_etfs = {
            'NIFTYBEES.NS': {
                'name': 'Nifty BeES',
                'weight': 25,
                'category': 'Large Cap',
                'benchmark': 'Nifty 50',
                'fund_house': 'Nippon India'
            },
            'JUNIORBEES.NS': {
                'name': 'Junior BeES', 
                'weight': 30,
                'category': 'Mid Cap',
                'benchmark': 'Nifty Next 50',
                'fund_house': 'Nippon India'
            },
            'BANKBEES.NS': {
                'name': 'Bank BeES',
                'weight': 25,
                'category': 'Banking',
                'benchmark': 'Nifty Bank',
                'fund_house': 'Nippon India'
            },
            'ICICIB22.NS': {
                'name': 'ICICI Bank ETF',
                'weight': 15,
                'category': 'Banking',
                'benchmark': 'Nifty Bank',
                'fund_house': 'ICICI Prudential'
            },
            'ITBEES.NS': {
                'name': 'IT BeES',
                'weight': 5,
                'category': 'IT Sector',
                'benchmark': 'Nifty IT',
                'fund_house': 'Nippon India'
            }
        }
        
        # Performance data from analysis
        self.performance_data = {
            'NIFTYBEES.NS': {'accuracy': 86.4, 'mape': 0.7, 'sharpe': 0.98, 'confidence': 99.1},
            'JUNIORBEES.NS': {'accuracy': 93.3, 'mape': 1.1, 'sharpe': 1.12, 'confidence': 98.8},
            'BANKBEES.NS': {'accuracy': 86.9, 'mape': 0.8, 'sharpe': 1.07, 'confidence': 99.2},
            'ICICIB22.NS': {'accuracy': 89.2, 'mape': 1.4, 'sharpe': 1.07, 'confidence': 98.8},
            'ITBEES.NS': {'accuracy': 89.5, 'mape': 1.2, 'sharpe': 1.00, 'confidence': 99.1}
        }
    
    def get_live_data(self, symbol, period='5d'):
        """Fetch live market data for an ETF"""
        try:
            ticker = yf.Ticker(symbol)
            data = ticker.history(period=period, interval='1d')
            
            if not data.empty:
                # Calculate basic metrics
                current_price = data['Close'].iloc[-1]
                prev_close = data['Close'].iloc[-2] if len(data) > 1 else current_price
                change = current_price - prev_close
                change_pct = (change / prev_close) * 100
                
                volume = data['Volume'].iloc[-1]
                avg_volume = data['Volume'].mean()
                volume_ratio = volume / avg_volume if avg_volume > 0 else 1.0
                
                # High/Low
                high_52w = data['High'].max()
                low_52w = data['Low'].min()
                
                return {
                    'current_price': current_price,
                    'change': change,
                    'change_pct': change_pct,
                    'volume': volume,
                    'volume_ratio': volume_ratio,
                    'high_52w': high_52w,
                    'low_52w': low_52w,
                    'data': data,
                    'status': 'success'
                }
            else:
                return {'status': 'error', 'message': 'No data available'}
                
        except Exception as e:
            return {'status': 'error', 'message': str(e)}
    
    def generate_trading_signals(self, symbol):
        """Generate real-time trading signals for an ETF"""
        try:
            # Fetch recent data
            end_date = datetime.now()
            start_date = end_date - timedelta(days=100)
            
            df = fetch_yfinance(symbol, start_date.strftime('%Y-%m-%d'), end_date.strftime('%Y-%m-%d'))
            
            if df.empty:
                return {'status': 'error', 'message': 'No data for signal generation'}
            
            # Add technical indicators
            df = add_technical_indicators(df)
            
            # Get trading signals
            signals = get_trading_signals(df)
            
            # Get latest technical indicators
            latest = df.iloc[-1]
            
            technical_data = {
                'RSI': latest.get('RSI_14', 50),
                'MACD': latest.get('MACD', 0),
                'BB_Position': latest.get('BB_Position', 0.5),
                'Volume_Ratio': latest.get('Volume_Ratio', 1.0),
                'ATR': latest.get('ATR_14', 0)
            }
            
            return {
                'status': 'success',
                'signals': signals,
                'technical_data': technical_data,
                'recommendation': signals.get('Overall', {}).get('signal', 'HOLD'),
                'confidence': signals.get('Overall', {}).get('confidence', 0.5)
            }
            
        except Exception as e:
            return {'status': 'error', 'message': str(e)}
    
    def calculate_portfolio_metrics(self, live_data):
        """Calculate overall portfolio performance metrics"""
        total_value = 0
        weighted_change = 0
        portfolio_data = []
        
        for symbol, info in self.portfolio_etfs.items():
            if symbol in live_data and live_data[symbol]['status'] == 'success':
                data = live_data[symbol]
                weight = info['weight'] / 100
                
                value = data['current_price'] * weight * 100  # Assume 100 units base
                total_value += value
                weighted_change += data['change_pct'] * weight
                
                portfolio_data.append({
                    'Symbol': symbol,
                    'Name': info['name'],
                    'Weight': f"{info['weight']}%",
                    'Price': f"₹{data['current_price']:.2f}",
                    'Change': f"{data['change_pct']:+.2f}%",
                    'Volume_Ratio': f"{data['volume_ratio']:.2f}x",
                    'Category': info['category']
                })
        
        return {
            'total_value': total_value,
            'portfolio_change': weighted_change,
            'etf_data': portfolio_data
        }
    
    def create_portfolio_overview_chart(self, live_data):
        """Create portfolio overview visualization"""
        # Prepare data for chart
        symbols = []
        prices = []
        changes = []
        weights = []
        colors = []
        
        for symbol, info in self.portfolio_etfs.items():
            if symbol in live_data and live_data[symbol]['status'] == 'success':
                data = live_data[symbol]
                
                symbols.append(info['name'])
                prices.append(data['current_price'])
                changes.append(data['change_pct'])
                weights.append(info['weight'])
                
                # Color based on performance
                if data['change_pct'] > 1:
                    colors.append('darkgreen')
                elif data['change_pct'] > 0:
                    colors.append('green')
                elif data['change_pct'] > -1:
                    colors.append('orange')
                else:
                    colors.append('red')
        
        # Create subplots
        fig = make_subplots(
            rows=2, cols=2,
            subplot_titles=('Portfolio Weights', 'Price Changes (%)', 'Current Prices (₹)', 'Performance vs Benchmark'),
            specs=[[{"type": "pie"}, {"type": "bar"}],
                   [{"type": "bar"}, {"type": "scatter"}]]
        )
        
        # Portfolio weights pie chart
        fig.add_trace(
            go.Pie(labels=symbols, values=weights, name="Weights"),
            row=1, col=1
        )
        
        # Price changes bar chart
        fig.add_trace(
            go.Bar(x=symbols, y=changes, marker_color=colors, name="Change %"),
            row=1, col=2
        )
        
        # Current prices bar chart
        fig.add_trace(
            go.Bar(x=symbols, y=prices, name="Price ₹", marker_color='lightblue'),
            row=2, col=1
        )
        
        # Performance vs benchmark scatter
        accuracy_data = [self.performance_data[symbol]['accuracy'] for symbol in self.portfolio_etfs.keys() if symbol in live_data]
        sharpe_data = [self.performance_data[symbol]['sharpe'] for symbol in self.portfolio_etfs.keys() if symbol in live_data]
        
        fig.add_trace(
            go.Scatter(
                x=accuracy_data, y=sharpe_data, 
                mode='markers+text',
                text=[info['name'] for info in self.portfolio_etfs.values()],
                textposition="top center",
                marker=dict(size=10, color='purple'),
                name="Accuracy vs Sharpe"
            ),
            row=2, col=2
        )
        
        fig.update_layout(height=600, showlegend=False, title_text="🇮🇳 Indian ETF Portfolio Dashboard")
        return fig

def main():
    """Main dashboard function"""
    st.title("🇮🇳 Indian ETF Portfolio Real-Time Monitor")
    st.markdown("**Live monitoring and trading signals for optimized Indian ETF portfolio**")
    
    # Initialize monitor
    monitor = IndianETFMonitor()
    
    # Sidebar controls
    st.sidebar.header("⚙️ Dashboard Controls")
    
    # Auto-refresh toggle
    auto_refresh = st.sidebar.checkbox("🔄 Auto Refresh (30s)", value=False)
    
    if auto_refresh:
        time.sleep(30)
        st.rerun()
    
    # Manual refresh button
    if st.sidebar.button("🔄 Refresh Data"):
        st.rerun()
    
    # Risk threshold
    risk_threshold = st.sidebar.slider("⚠️ Risk Alert Threshold (%)", 1.0, 10.0, 5.0, 0.5)
    
    # Portfolio view options
    view_option = st.sidebar.selectbox(
        "📊 View Mode",
        ["Portfolio Overview", "Individual ETFs", "Trading Signals", "Risk Analysis"]
    )
    
    # Market status indicator
    current_time = datetime.now()
    market_open = current_time.replace(hour=9, minute=15)
    market_close = current_time.replace(hour=15, minute=30)
    
    if market_open <= current_time <= market_close and current_time.weekday() < 5:
        st.sidebar.success("🟢 Market: OPEN")
    else:
        st.sidebar.error("🔴 Market: CLOSED")
    
    # Fetch live data for all ETFs
    with st.spinner("📡 Fetching live market data..."):
        live_data = {}
        for symbol in monitor.portfolio_etfs.keys():
            live_data[symbol] = monitor.get_live_data(symbol)
    
    # Calculate portfolio metrics
    portfolio_metrics = monitor.calculate_portfolio_metrics(live_data)
    
    # Display based on selected view
    if view_option == "Portfolio Overview":
        # Portfolio summary metrics
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric(
                "Portfolio Value",
                f"₹{portfolio_metrics['total_value']:,.0f}",
                f"{portfolio_metrics['portfolio_change']:+.2f}%"
            )
        
        with col2:
            successful_etfs = sum(1 for data in live_data.values() if data['status'] == 'success')
            st.metric("ETFs Tracked", f"{successful_etfs}/5", "Live Data")
        
        with col3:
            avg_accuracy = np.mean([perf['accuracy'] for perf in monitor.performance_data.values()])
            st.metric("Avg Model Accuracy", f"{avg_accuracy:.1f}%", "ML Performance")
        
        with col4:
            avg_sharpe = np.mean([perf['sharpe'] for perf in monitor.performance_data.values()])
            st.metric("Avg Sharpe Ratio", f"{avg_sharpe:.2f}", "Risk-Adjusted")
        
        # Portfolio overview chart
        st.plotly_chart(monitor.create_portfolio_overview_chart(live_data), use_container_width=True)
        
        # ETF Performance Table
        st.subheader("📊 ETF Performance Summary")
        if portfolio_metrics['etf_data']:
            df_portfolio = pd.DataFrame(portfolio_metrics['etf_data'])
            st.dataframe(df_portfolio, use_container_width=True)
    
    elif view_option == "Individual ETFs":
        st.subheader("🔍 Individual ETF Analysis")
        
        selected_etf = st.selectbox(
            "Select ETF for Detailed Analysis",
            list(monitor.portfolio_etfs.keys()),
            format_func=lambda x: f"{monitor.portfolio_etfs[x]['name']} ({x})"
        )
        
        if selected_etf in live_data and live_data[selected_etf]['status'] == 'success':
            data = live_data[selected_etf]
            info = monitor.portfolio_etfs[selected_etf]
            perf = monitor.performance_data[selected_etf]
            
            # ETF metrics
            col1, col2, col3, col4 = st.columns(4)
            
            with col1:
                st.metric(
                    "Current Price",
                    f"₹{data['current_price']:.2f}",
                    f"{data['change_pct']:+.2f}%"
                )
            
            with col2:
                st.metric("Model Accuracy", f"{perf['accuracy']:.1f}%")
            
            with col3:
                st.metric("MAPE", f"{perf['mape']:.1f}%")
            
            with col4:
                st.metric("Sharpe Ratio", f"{perf['sharpe']:.2f}")
            
            # Price chart
            if 'data' in data:
                fig = go.Figure()
                fig.add_trace(go.Candlestick(
                    x=data['data'].index,
                    open=data['data']['Open'],
                    high=data['data']['High'],
                    low=data['data']['Low'],
                    close=data['data']['Close'],
                    name=info['name']
                ))
                fig.update_layout(
                    title=f"{info['name']} - 5 Day Chart",
                    yaxis_title="Price (₹)",
                    height=400
                )
                st.plotly_chart(fig, use_container_width=True)
    
    elif view_option == "Trading Signals":
        st.subheader("🎯 Live Trading Signals")
        
        signal_data = []
        for symbol in monitor.portfolio_etfs.keys():
            signals = monitor.generate_trading_signals(symbol)
            if signals['status'] == 'success':
                signal_data.append({
                    'ETF': monitor.portfolio_etfs[symbol]['name'],
                    'Symbol': symbol,
                    'Signal': signals['recommendation'],
                    'Confidence': f"{signals['confidence']:.1%}",
                    'RSI': f"{signals['technical_data']['RSI']:.1f}",
                    'MACD': f"{signals['technical_data']['MACD']:.3f}",
                    'BB_Position': f"{signals['technical_data']['BB_Position']:.2f}"
                })
        
        if signal_data:
            df_signals = pd.DataFrame(signal_data)
            st.dataframe(df_signals, use_container_width=True)
            
            # Signal summary
            buy_signals = len(df_signals[df_signals['Signal'] == 'BUY'])
            sell_signals = len(df_signals[df_signals['Signal'] == 'SELL'])
            hold_signals = len(df_signals[df_signals['Signal'] == 'HOLD'])
            
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("🟢 BUY Signals", buy_signals)
            with col2:
                st.metric("🔴 SELL Signals", sell_signals)
            with col3:
                st.metric("🟡 HOLD Signals", hold_signals)
    
    elif view_option == "Risk Analysis":
        st.subheader("⚠️ Portfolio Risk Analysis")
        
        # Risk alerts
        risk_alerts = []
        for symbol, data in live_data.items():
            if data['status'] == 'success' and abs(data['change_pct']) > risk_threshold:
                risk_alerts.append({
                    'ETF': monitor.portfolio_etfs[symbol]['name'],
                    'Change': f"{data['change_pct']:+.2f}%",
                    'Alert': 'High Volatility',
                    'Severity': 'HIGH' if abs(data['change_pct']) > risk_threshold * 2 else 'MEDIUM'
                })
        
        if risk_alerts:
            st.warning(f"⚠️ {len(risk_alerts)} Risk Alert(s) Detected")
            df_alerts = pd.DataFrame(risk_alerts)
            st.dataframe(df_alerts, use_container_width=True)
        else:
            st.success("✅ No risk alerts - Portfolio within normal parameters")
        
        # Risk metrics
        st.subheader("📊 Risk Metrics")
        
        portfolio_volatility = np.std([data['change_pct'] for data in live_data.values() if data['status'] == 'success'])
        avg_confidence = np.mean([perf['confidence'] for perf in monitor.performance_data.values()])
        
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Portfolio Volatility", f"{portfolio_volatility:.2f}%")
        with col2:
            st.metric("Model Confidence", f"{avg_confidence:.1f}%")
        with col3:
            diversification_score = len(set(info['category'] for info in monitor.portfolio_etfs.values()))
            st.metric("Diversification Score", f"{diversification_score}/4")
    
    # Footer
    st.markdown("---")
    st.markdown(f"**Last Updated:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S IST')} | **Data Source:** Yahoo Finance | **Models:** Ultra-Enhanced ML")

if __name__ == "__main__":
    main() 