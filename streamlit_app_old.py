import streamlit as st
import pandas as pd
from datetime import datetime, timedelta
from data_ingestion import fetch_yfinance
from feature_engineering import (
    add_technical_indicators, 
    get_trading_signals, 
    get_comprehensive_features,
    get_macro_trading_signals
)
import numpy as np
import plotly.express as px
import plotly.graph_objs as go
from plotly.subplots import make_subplots
from future_forecasting import FutureForecaster

# Import enhanced training system
from train_enhanced_system import EnhancedTrainingSystem
from backtesting import Backtester
from macro_indicators import MacroIndicators
import joblib
import os

st.set_page_config(page_title="Stock Predictor Enhanced Dashboard", layout="wide")
st.title("📈 Advanced Stock Predictor Dashboard with Macro Analysis")

# Helper functions for macro overlay charts
def create_macro_overlay_chart(stock_data, macro_data, selected_indicators, normalize_data, symbol):
    """Create an overlay chart with stock price and macro indicators on dual y-axes."""
    fig = go.Figure()
    
    # Add stock price (primary y-axis)
    fig.add_trace(go.Scatter(
        x=stock_data['Date'],
        y=stock_data['Close'],
        name=f'{symbol} Price',
        yaxis='y',
        line=dict(color='#1f77b4', width=2)
    ))
    
    # Add macro indicators (secondary y-axis)
    colors = ['#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#8c564b', '#e377c2']
    
    for i, indicator in enumerate(selected_indicators):
        if indicator in macro_data:
            data = macro_data[indicator]
            if not data.empty:
                y_data = data.iloc[:, 1]  # Assuming second column is the value
                
                if normalize_data:
                    # Normalize to 0-1 range
                    y_data = (y_data - y_data.min()) / (y_data.max() - y_data.min())
                    # Scale to stock price range
                    stock_range = stock_data['Close'].max() - stock_data['Close'].min()
                    y_data = y_data * stock_range + stock_data['Close'].min()
                
                fig.add_trace(go.Scatter(
                    x=data.iloc[:, 0],  # Date column
                    y=y_data,
                    name=indicator,
                    yaxis='y2',
                    line=dict(color=colors[i % len(colors)], width=1, dash='dot'),
                    opacity=0.7
                ))
    
    # Update layout
    fig.update_layout(
        title=f'{symbol} Price vs Macro Indicators (Overlay)',
        xaxis_title='Date',
        yaxis=dict(title=f'{symbol} Price', side='left'),
        yaxis2=dict(title='Macro Indicators', side='right', overlaying='y'),
        hovermode='x unified',
        height=600,
        showlegend=True
    )
    
    return fig

def create_macro_subplot_chart(stock_data, macro_data, selected_indicators, normalize_data, symbol):
    """Create a subplot chart with stock price and macro indicators."""
    n_indicators = len(selected_indicators)
    fig = make_subplots(
        rows=n_indicators + 1, cols=1,
        subplot_titles=[f'{symbol} Price'] + selected_indicators,
        vertical_spacing=0.05
    )
    
    # Add stock price
    fig.add_trace(
        go.Scatter(x=stock_data['Date'], y=stock_data['Close'], name=f'{symbol} Price'),
        row=1, col=1
    )
    
    # Add macro indicators
    for i, indicator in enumerate(selected_indicators):
        if indicator in macro_data:
            data = macro_data[indicator]
            if not data.empty:
                fig.add_trace(
                    go.Scatter(x=data.iloc[:, 0], y=data.iloc[:, 1], name=indicator),
                    row=i+2, col=1
                )
    
    fig.update_layout(height=200 * (n_indicators + 1), showlegend=False)
    return fig

def create_correlation_heatmap(stock_data, macro_data, selected_indicators, symbol):
    """Create a correlation heatmap between stock price and macro indicators."""
    # Prepare data for correlation
    correlation_data = {}
    correlation_data[f'{symbol}_Price'] = stock_data['Close']
    
    for indicator in selected_indicators:
        if indicator in macro_data:
            data = macro_data[indicator]
            if not data.empty:
                # Align dates and interpolate if necessary
                aligned_data = data.set_index(data.iloc[:, 0])[data.iloc[:, 1]]
                aligned_data = aligned_data.reindex(stock_data['Date']).interpolate()
                correlation_data[indicator] = aligned_data
    
    # Calculate correlation matrix
    corr_df = pd.DataFrame(correlation_data).corr()
    
    # Create heatmap
    fig = go.Figure(data=go.Heatmap(
        z=corr_df.values,
        x=corr_df.columns,
        y=corr_df.index,
        colorscale='RdBu',
        zmid=0,
        text=corr_df.round(3).values,
        texttemplate="%{text}",
        textfont={"size": 10}
    ))
    
    fig.update_layout(
        title=f'Correlation Heatmap: {symbol} vs Macro Indicators',
        height=500
    )
    
    return fig

def calculate_macro_correlations(stock_data, macro_data, selected_indicators):
    """Calculate correlation coefficients between stock price and macro indicators."""
    correlations = []
    
    for indicator in selected_indicators:
        if indicator in macro_data:
            data = macro_data[indicator]
            if not data.empty:
                # Align dates and interpolate
                aligned_data = data.set_index(data.iloc[:, 0])[data.iloc[:, 1]]
                aligned_data = aligned_data.reindex(stock_data['Date']).interpolate()
                
                # Calculate correlation
                correlation = stock_data['Close'].corr(aligned_data)
                correlations.append({
                    'Indicator': indicator,
                    'Correlation': correlation,
                    'Strength': 'Strong' if abs(correlation) > 0.7 else 'Moderate' if abs(correlation) > 0.3 else 'Weak',
                    'Direction': 'Positive' if correlation > 0 else 'Negative'
                })
    
    return pd.DataFrame(correlations).sort_values('Correlation', key=abs, ascending=False)

def generate_correlation_insights(correlation_df, symbol):
    """Generate insights from correlation analysis."""
    insights = []
    
    if not correlation_df.empty:
        # Strongest positive correlation
        strongest_pos = correlation_df[correlation_df['Correlation'] > 0].iloc[0] if len(correlation_df[correlation_df['Correlation'] > 0]) > 0 else None
        if strongest_pos is not None:
            insights.append(f"**{strongest_pos['Indicator']}** shows the strongest positive correlation ({strongest_pos['Correlation']:.3f}) with {symbol} price.")
        
        # Strongest negative correlation
        strongest_neg = correlation_df[correlation_df['Correlation'] < 0].iloc[0] if len(correlation_df[correlation_df['Correlation'] < 0]) > 0 else None
        if strongest_neg is not None:
            insights.append(f"**{strongest_neg['Indicator']}** shows the strongest negative correlation ({strongest_neg['Correlation']:.3f}) with {symbol} price.")
        
        # Overall correlation strength
        strong_correlations = len(correlation_df[correlation_df['Strength'] == 'Strong'])
        if strong_correlations > 0:
            insights.append(f"{strong_correlations} indicator(s) show strong correlation with {symbol} price.")
        
        # Trading implications
        if strongest_pos is not None and strongest_pos['Correlation'] > 0.7:
            insights.append(f"Consider monitoring **{strongest_pos['Indicator']}** as a leading indicator for {symbol} price movements.")
    
    return insights

def is_indian_stock(symbol: str) -> bool:
    """
    Check if the stock symbol is for an Indian stock.
    Indian stocks typically end with .NS (NSE) or .BO (BSE)
    """
    return symbol.upper().endswith(('.NS', '.BO', '.NSE', '.BSE'))

def get_currency_symbol(symbol: str) -> str:
    """
    Get the appropriate currency symbol based on the stock.
    """
    if is_indian_stock(symbol):
        return "₹"
    else:
        return "$"

def format_currency(value: float, symbol: str) -> str:
    """
    Format currency value with appropriate symbol and formatting.
    """
    if symbol == "₹":
        # Indian Rupee formatting
        if value >= 10000000:  # 1 crore
            return f"₹{value/10000000:.2f}Cr"
        elif value >= 100000:  # 1 lakh
            return f"₹{value/100000:.2f}L"
        else:
            return f"₹{value:,.2f}"
    else:
        # US Dollar formatting
        return f"${value:,.2f}"

# Sidebar controls
st.sidebar.header("Stock Selection")

# Popular stocks dropdown
popular_stocks = {
    "US Stocks": ["AAPL", "MSFT", "GOOGL", "TSLA", "AMZN", "META", "NVDA", "NFLX"],
    "Indian Stocks": [
        # Nifty 50 Index
        "^NSEI",
        
        # Nifty 50 - Complete List
        "RELIANCE.NS", "TCS.NS", "HDFCBANK.NS", "INFY.NS", "ICICIBANK.NS", 
        "HINDUNILVR.NS", "ITC.NS", "SBIN.NS", "BHARTIARTL.NS", "AXISBANK.NS",
        "KOTAKBANK.NS", "ASIANPAINT.NS", "MARUTI.NS", "SUNPHARMA.NS", "TATAMOTORS.NS",
        "WIPRO.NS", "ULTRACEMCO.NS", "TITAN.NS", "BAJFINANCE.NS", "NESTLEIND.NS",
        "POWERGRID.NS", "TECHM.NS", "BAJAJFINSV.NS", "NTPC.NS", "HCLTECH.NS",
        "ONGC.NS", "JSWSTEEL.NS", "TATACONSUM.NS", "ADANIENT.NS", "COALINDIA.NS",
        "HINDALCO.NS", "TATASTEEL.NS", "BRITANNIA.NS", "GRASIM.NS", "INDUSINDBK.NS",
        "M&M.NS", "BAJAJ-AUTO.NS", "VEDL.NS", "UPL.NS", "BPCL.NS",
        "SBILIFE.NS", "HDFCLIFE.NS", "DIVISLAB.NS", "CIPLA.NS", "EICHERMOT.NS",
        "HEROMOTOCO.NS", "SHREECEM.NS", "ADANIPORTS.NS", "DRREDDY.NS", "APOLLOHOSP.NS",
        "TATACONSUM.NS", "BAJFINANCE.NS", "HINDUNILVR.NS", "NESTLEIND.NS",
        
        # Nifty Next 50 - Complete List
        "ADANIENT.NS", "ADANIPORTS.NS", "APOLLOHOSP.NS", "BAJAJFINSV.NS", "BAJFINANCE.NS",
        "BHARTIARTL.NS", "BRITANNIA.NS", "CIPLA.NS", "COALINDIA.NS", "DIVISLAB.NS",
        "DRREDDY.NS", "EICHERMOT.NS", "GRASIM.NS", "HCLTECH.NS", "HDFCLIFE.NS",
        "HEROMOTOCO.NS", "HINDALCO.NS", "HINDUNILVR.NS", "ICICIBANK.NS", "INDUSINDBK.NS",
        "INFY.NS", "ITC.NS", "JSWSTEEL.NS", "KOTAKBANK.NS", "M&M.NS",
        "MARUTI.NS", "NESTLEIND.NS", "NTPC.NS", "ONGC.NS", "POWERGRID.NS",
        "RELIANCE.NS", "SBILIFE.NS", "SBIN.NS", "SHREECEM.NS", "SUNPHARMA.NS",
        "TATACONSUM.NS", "TATAMOTORS.NS", "TATASTEEL.NS", "TCS.NS", "TECHM.NS",
        "TITAN.NS", "ULTRACEMCO.NS", "UPL.NS", "VEDL.NS", "WIPRO.NS",
        
        # Banking & Financial Services
        "HDFC.NS", "KOTAKBANK.NS", "AXISBANK.NS", "ICICIBANK.NS", "SBIN.NS",
        "INDUSINDBK.NS", "FEDERALBNK.NS", "IDFCFIRSTB.NS", "BANDHANBNK.NS", "PNB.NS",
        "CANBK.NS", "UNIONBANK.NS", "BANKBARODA.NS", "IOB.NS", "UCOBANK.NS",
        "CENTRALBK.NS", "MAHABANK.NS", "INDIANB.NS", "PSB.NS", "ALLAHABAD.NS",
        "HDFCAMC.NS", "ICICIPRULI.NS", "SBICARD.NS", "BAJFINANCE.NS", "BAJAJFINSV.NS",
        "CHOLAFIN.NS", "MUTHOOTFIN.NS", "PEL.NS", "RECLTD.NS", "PFC.NS",
        
        # IT & Technology
        "TCS.NS", "INFY.NS", "WIPRO.NS", "HCLTECH.NS", "TECHM.NS",
        "MINDTREE.NS", "LTI.NS", "MPHASIS.NS", "PERSISTENT.NS", "COFORGE.NS",
        "L&TINFOTECH.NS", "HEXAWARE.NS", "NIITTECH.NS", "CYIENT.NS", "KPITTECH.NS",
        "SONATSOFTW.NS", "RAMCOSYS.NS", "INTELLECT.NS", "QUESS.NS", "TEAMLEASE.NS",
        "APLLTD.NS", "DATAPATTERNS.NS", "MAPMYINDIA.NS", "TATAELXSI.NS", "ZENSARTECH.NS",
        
        # Auto & Manufacturing
        "TATAMOTORS.NS", "MARUTI.NS", "EICHERMOT.NS", "HEROMOTOCO.NS", "BAJAJ-AUTO.NS",
        "M&M.NS", "ASHOKLEY.NS", "TVSMOTOR.NS", "ESCORTS.NS", "MRF.NS",
        "CEAT.NS", "APOLLOTYRE.NS", "JKTYRE.NS", "BALKRISIND.NS", "AMARAJABAT.NS",
        "EXIDEIND.NS", "SUNDARMFIN.NS", "BAJAJELEC.NS", "CROMPTON.NS", "HAVELLS.NS",
        "VOLTAS.NS", "BLUESTARCO.NS", "WHIRLPOOL.NS", "GODREJCP.NS", "MARICO.NS",
        
        # Pharma & Healthcare
        "SUNPHARMA.NS", "DRREDDY.NS", "CIPLA.NS", "DIVISLAB.NS", "BIOCON.NS",
        "APOLLOHOSP.NS", "FORTIS.NS", "ALKEM.NS", "TORNTPHARM.NS", "CADILAHC.NS",
        "LUPIN.NS", "AUROPHARMA.NS", "GLENMARK.NS", "NATCOPHARM.NS", "AJANTPHARM.NS",
        "LAURUSLABS.NS", "GRANULES.NS", "IPCA.NS", "PFIZER.NS", "SANOFI.NS",
        "ABBOTINDIA.NS", "ALKEM.NS", "TORNTPHARM.NS", "CADILAHC.NS",
        
        # Consumer Goods & FMCG
        "HINDUNILVR.NS", "ITC.NS", "NESTLEIND.NS", "BRITANNIA.NS", "MARICO.NS",
        "DABUR.NS", "COLPAL.NS", "GODREJCP.NS", "EMAMILTD.NS", "VBL.NS",
        "UBL.NS", "RADICO.NS", "UNILEVER.NS", "GILLETTE.NS", "MARICO.NS",
        "DABUR.NS", "COLPAL.NS", "GODREJCP.NS", "EMAMILTD.NS", "VBL.NS",
        "UBL.NS", "RADICO.NS", "UNILEVER.NS", "GILLETTE.NS", "MARICO.NS",
        
        # Energy & Oil
        "RELIANCE.NS", "ONGC.NS", "COALINDIA.NS", "NTPC.NS", "POWERGRID.NS",
        "BPCL.NS", "IOC.NS", "HPCL.NS", "ADANIGREEN.NS", "TATAPOWER.NS",
        "ADANITRANS.NS", "ADANIGAS.NS", "ADANIPOWER.NS", "ADANIENT.NS", "ADANIPORTS.NS",
        "GAIL.NS", "PETRONET.NS", "OIL.NS", "CONCOR.NS", "CONTAINER.NS",
        
        # Metals & Mining
        "TATASTEEL.NS", "JSWSTEEL.NS", "HINDALCO.NS", "VEDL.NS", "HINDCOPPER.NS",
        "NATIONALUM.NS", "WELCORP.NS", "JINDALSTEL.NS", "SAIL.NS", "NMDC.NS",
        "HINDZINC.NS", "VEDL.NS", "HINDALCO.NS", "TATASTEEL.NS", "JSWSTEEL.NS",
        "SAIL.NS", "NMDC.NS", "HINDZINC.NS", "VEDL.NS", "HINDALCO.NS",
        
        # Real Estate & Construction
        "DLF.NS", "GODREJPROP.NS", "SUNTV.NS", "PRESTIGE.NS", "BRIGADE.NS",
        "OBEROIRLTY.NS", "PHOENIXLTD.NS", "SOBHA.NS", "GODREJIND.NS", "KOLTEPATIL.NS",
        "LODHA.NS", "MACROTECH.NS", "GODREJPROP.NS", "DLF.NS", "SUNTV.NS",
        "PRESTIGE.NS", "BRIGADE.NS", "OBEROIRLTY.NS", "PHOENIXLTD.NS", "SOBHA.NS",
        
        # Telecom & Media
        "BHARTIARTL.NS", "IDEA.NS", "VODAFONE.NS", "MTNL.NS", "BSNL.NS",
        "SUNTV.NS", "ZEEL.NS", "PVR.NS", "INOXLEISURE.NS", "PVR.NS",
        "INOXLEISURE.NS", "SUNTV.NS", "ZEEL.NS", "PVR.NS", "INOXLEISURE.NS",
        
        # Cement & Construction
        "ULTRACEMCO.NS", "SHREECEM.NS", "ACC.NS", "AMBUJACEM.NS", "RAMCOCEM.NS",
        "HEIDELBERG.NS", "BIRLACORPN.NS", "JKLAKSHMI.NS", "ORIENTCEM.NS", "MANGALAM.NS",
        "ULTRACEMCO.NS", "SHREECEM.NS", "ACC.NS", "AMBUJACEM.NS", "RAMCOCEM.NS",
        
        # Chemicals & Fertilizers
        "UPL.NS", "COROMANDEL.NS", "CHAMBLFERT.NS", "GSFC.NS", "RCF.NS",
        "NATIONALUM.NS", "HINDALCO.NS", "VEDL.NS", "HINDCOPPER.NS", "NATIONALUM.NS",
        "UPL.NS", "COROMANDEL.NS", "CHAMBLFERT.NS", "GSFC.NS", "RCF.NS",
        
        # Aviation & Logistics
        "INDIGO.NS", "SPICEJET.NS", "JETAIRWAYS.NS", "AIRINDIA.NS", "VISTARA.NS",
        "CONCOR.NS", "CONTAINER.NS", "ADANIPORTS.NS", "ADANITRANS.NS", "ADANIGAS.NS",
        
        # E-commerce & Digital
        "FLIPKART.NS", "AMAZON.NS", "SNAPDEAL.NS", "PAYTM.NS", "ZOMATO.NS",
        "NYKAA.NS", "DELHIVERY.NS", "CARTRADE.NS", "EASEMYTRIP.NS", "MAPMYINDIA.NS",
        
        # Small Cap Gems
        "TATACOMM.NS", "TATACONSUM.NS", "TATAMOTORS.NS", "TATAPOWER.NS", "TATASTEEL.NS",
        "TATACONSUM.NS", "TATAMOTORS.NS", "TATAPOWER.NS", "TATASTEEL.NS", "TATACOMM.NS",
        
        # PSU Banks
        "SBIN.NS", "PNB.NS", "CANBK.NS", "UNIONBANK.NS", "BANKBARODA.NS",
        "IOB.NS", "UCOBANK.NS", "CENTRALBK.NS", "MAHABANK.NS", "INDIANB.NS",
        
        # Private Banks
        "HDFCBANK.NS", "ICICIBANK.NS", "AXISBANK.NS", "KOTAKBANK.NS", "INDUSINDBK.NS",
        "FEDERALBNK.NS", "IDFCFIRSTB.NS", "BANDHANBNK.NS", "RBLBANK.NS", "YESBANK.NS",
        
        # NBFCs & Financial Services
        "BAJFINANCE.NS", "BAJAJFINSV.NS", "HDFC.NS", "HDFCAMC.NS", "ICICIPRULI.NS",
        "SBICARD.NS", "CHOLAFIN.NS", "MUTHOOTFIN.NS", "PEL.NS", "RECLTD.NS",
        
        # Insurance
        "SBILIFE.NS", "HDFCLIFE.NS", "ICICIPRULI.NS", "MAXLIFE.NS", "BAJAJALLIANZ.NS",
        "SHRIRAM.NS", "CHOLAMANDALAM.NS", "BAJAJFINSV.NS", "HDFCAMC.NS", "ICICIPRULI.NS"
    ]
}

stock_category = st.sidebar.selectbox(
    "Select Stock Category",
    ["US Stocks", "Indian Stocks", "Custom"]
)

if stock_category == "Custom":
    symbol = st.sidebar.text_input("Enter Stock Symbol (e.g., AAPL, MSFT, TSLA, RELIANCE.NS)", value="AAPL")
else:
    if stock_category == "Indian Stocks":
        # Add search functionality for Indian stocks
        search_term = st.sidebar.text_input("🔍 Search Indian Stocks", placeholder="Type to search...")
        
        if search_term:
            # Filter stocks based on search term
            filtered_stocks = [stock for stock in popular_stocks[stock_category] 
                             if search_term.upper() in stock.upper()]
            if filtered_stocks:
                symbol = st.sidebar.selectbox(
                    f"Select {stock_category} (Filtered)",
                    filtered_stocks,
                    index=0
                )
            else:
                st.sidebar.warning("No stocks found matching your search.")
                symbol = st.sidebar.selectbox(
                    f"Select {stock_category}",
                    popular_stocks[stock_category],
                    index=0
                )
        else:
            symbol = st.sidebar.selectbox(
                f"Select {stock_category}",
                popular_stocks[stock_category],
                index=0
            )
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
        st.sidebar.info("💡 Indian stocks use Rupee (₹) currency with Indian number formatting.")
        
        # Show stock category information for Indian stocks
        if symbol == '^NSEI':
            st.sidebar.info("🏆 **Nifty 50 Index** - Benchmark Index (50 Largest Indian Companies)")
        else:
            stock_name = symbol.replace('.NS', '')
        
        # Nifty 50 stocks
        nifty50_stocks = ['RELIANCE', 'TCS', 'HDFCBANK', 'INFY', 'ICICIBANK', 'HINDUNILVR', 'ITC', 'SBIN', 
                         'BHARTIARTL', 'AXISBANK', 'KOTAKBANK', 'ASIANPAINT', 'MARUTI', 'SUNPHARMA', 'TATAMOTORS',
                         'WIPRO', 'ULTRACEMCO', 'TITAN', 'BAJFINANCE', 'NESTLEIND', 'POWERGRID', 'TECHM', 
                         'BAJAJFINSV', 'NTPC', 'HCLTECH', 'ONGC', 'JSWSTEEL', 'TATACONSUM', 'ADANIENT', 'COALINDIA',
                         'HINDALCO', 'TATASTEEL', 'BRITANNIA', 'GRASIM', 'INDUSINDBK', 'M&M', 'BAJAJ-AUTO', 
                         'VEDL', 'UPL', 'BPCL', 'SBILIFE', 'HDFCLIFE', 'DIVISLAB', 'CIPLA', 'EICHERMOT', 
                         'HEROMOTOCO', 'SHREECEM', 'ADANIPORTS', 'DRREDDY', 'APOLLOHOSP']
        
        # Banking & Financial
        banking_stocks = ['HDFC', 'KOTAKBANK', 'AXISBANK', 'ICICIBANK', 'SBIN', 'INDUSINDBK', 'FEDERALBNK', 
                         'IDFCFIRSTB', 'BANDHANBNK', 'PNB', 'CANBK', 'UNIONBANK', 'BANKBARODA', 'IOB', 
                         'UCOBANK', 'CENTRALBK', 'MAHABANK', 'INDIANB', 'PSB', 'ALLAHABAD', 'HDFCAMC', 
                         'ICICIPRULI', 'SBICARD', 'BAJFINANCE', 'BAJAJFINSV', 'CHOLAFIN', 'MUTHOOTFIN', 
                         'PEL', 'RECLTD', 'PFC']
        
        # IT & Technology
        it_stocks = ['TCS', 'INFY', 'WIPRO', 'HCLTECH', 'TECHM', 'MINDTREE', 'LTI', 'MPHASIS', 'PERSISTENT', 
                    'COFORGE', 'L&TINFOTECH', 'HEXAWARE', 'NIITTECH', 'CYIENT', 'KPITTECH', 'SONATSOFTW', 
                    'RAMCOSYS', 'INTELLECT', 'QUESS', 'TEAMLEASE', 'APLLTD', 'DATAPATTERNS', 'MAPMYINDIA', 
                    'TATAELXSI', 'ZENSARTECH']
        
        # Auto & Manufacturing
        auto_stocks = ['TATAMOTORS', 'MARUTI', 'EICHERMOT', 'HEROMOTOCO', 'BAJAJ-AUTO', 'M&M', 'ASHOKLEY', 
                      'TVSMOTOR', 'ESCORTS', 'MRF', 'CEAT', 'APOLLOTYRE', 'JKTYRE', 'BALKRISIND', 'AMARAJABAT', 
                      'EXIDEIND', 'SUNDARMFIN', 'BAJAJELEC', 'CROMPTON', 'HAVELLS', 'VOLTAS', 'BLUESTARCO', 
                      'WHIRLPOOL', 'GODREJCP', 'MARICO']
        
        # Pharma & Healthcare
        pharma_stocks = ['SUNPHARMA', 'DRREDDY', 'CIPLA', 'DIVISLAB', 'BIOCON', 'APOLLOHOSP', 'FORTIS', 
                        'ALKEM', 'TORNTPHARM', 'CADILAHC', 'LUPIN', 'AUROPHARMA', 'GLENMARK', 'NATCOPHARM', 
                        'AJANTPHARM', 'LAURUSLABS', 'GRANULES', 'IPCA', 'PFIZER', 'SANOFI', 'ABBOTINDIA']
        
        # Consumer Goods & FMCG
        fmcg_stocks = ['HINDUNILVR', 'ITC', 'NESTLEIND', 'BRITANNIA', 'MARICO', 'DABUR', 'COLPAL', 'GODREJCP', 
                      'EMAMILTD', 'VBL', 'UBL', 'RADICO', 'UNILEVER', 'GILLETTE']
        
        # Energy & Oil
        energy_stocks = ['RELIANCE', 'ONGC', 'COALINDIA', 'NTPC', 'POWERGRID', 'BPCL', 'IOC', 'HPCL', 
                        'ADANIGREEN', 'TATAPOWER', 'ADANITRANS', 'ADANIGAS', 'ADANIPOWER', 'ADANIENT', 
                        'ADANIPORTS', 'GAIL', 'PETRONET', 'OIL', 'CONCOR', 'CONTAINER']
        
        # Metals & Mining
        metals_stocks = ['TATASTEEL', 'JSWSTEEL', 'HINDALCO', 'VEDL', 'HINDCOPPER', 'NATIONALUM', 'WELCORP', 
                        'JINDALSTEL', 'SAIL', 'NMDC', 'HINDZINC']
        
        # Real Estate & Construction
        realty_stocks = ['DLF', 'GODREJPROP', 'SUNTV', 'PRESTIGE', 'BRIGADE', 'OBEROIRLTY', 'PHOENIXLTD', 
                        'SOBHA', 'GODREJIND', 'KOLTEPATIL', 'LODHA', 'MACROTECH']
        
        # Cement & Construction
        cement_stocks = ['ULTRACEMCO', 'SHREECEM', 'ACC', 'AMBUJACEM', 'RAMCOCEM', 'HEIDELBERG', 'BIRLACORPN', 
                        'JKLAKSHMI', 'ORIENTCEM', 'MANGALAM']
        
        # Telecom & Media
        telecom_stocks = ['BHARTIARTL', 'IDEA', 'VODAFONE', 'MTNL', 'BSNL', 'SUNTV', 'ZEEL', 'PVR', 'INOXLEISURE']
        
        # E-commerce & Digital
        digital_stocks = ['FLIPKART', 'AMAZON', 'SNAPDEAL', 'PAYTM', 'ZOMATO', 'NYKAA', 'DELHIVERY', 
                         'CARTRADE', 'EASEMYTRIP', 'MAPMYINDIA']
        
        # Insurance
        insurance_stocks = ['SBILIFE', 'HDFCLIFE', 'ICICIPRULI', 'MAXLIFE', 'BAJAJALLIANZ', 'SHRIRAM', 
                           'CHOLAMANDALAM']
        
        if stock_name in nifty50_stocks:
            st.sidebar.info("🏆 **Nifty 50** - Large Cap Stock")
        elif stock_name in banking_stocks:
            st.sidebar.info("🏦 **Banking & Financial** Sector")
        elif stock_name in it_stocks:
            st.sidebar.info("💻 **IT & Technology** Sector")
        elif stock_name in auto_stocks:
            st.sidebar.info("🚗 **Auto & Manufacturing** Sector")
        elif stock_name in pharma_stocks:
            st.sidebar.info("💊 **Pharma & Healthcare** Sector")
        elif stock_name in fmcg_stocks:
            st.sidebar.info("🛒 **Consumer Goods & FMCG** Sector")
        elif stock_name in energy_stocks:
            st.sidebar.info("⚡ **Energy & Oil** Sector")
        elif stock_name in metals_stocks:
            st.sidebar.info("🏭 **Metals & Mining** Sector")
        elif stock_name in realty_stocks:
            st.sidebar.info("🏢 **Real Estate & Construction** Sector")
        elif stock_name in cement_stocks:
            st.sidebar.info("🏗️ **Cement & Construction** Sector")
        elif stock_name in telecom_stocks:
            st.sidebar.info("📡 **Telecom & Media** Sector")
        elif stock_name in digital_stocks:
            st.sidebar.info("🌐 **E-commerce & Digital** Sector")
        elif stock_name in insurance_stocks:
            st.sidebar.info("🛡️ **Insurance** Sector")
        else:
            st.sidebar.info("📈 **Indian Stock** - Other Sector")
    else:
        st.sidebar.success(f"🇺🇸 US Stock - Currency: {currency_symbol} (Dollars)")
        st.sidebar.info("💡 US stocks use Dollar ($) currency with standard formatting.")

# Analysis options
st.sidebar.header("Analysis Options")
include_macro = st.sidebar.checkbox("Include Macroeconomic Analysis", value=True)
show_advanced = st.sidebar.checkbox("Show Advanced Features", value=True)

if st.sidebar.button("Analyze") or symbol:
    try:
        # Data ingestion
        df = fetch_yfinance(symbol, start_date.strftime('%Y-%m-%d'), end_date.strftime('%Y-%m-%d'))
        
        # Add comprehensive features
        if include_macro:
            df = get_comprehensive_features(df, include_macro=True)
        else:
            df = add_technical_indicators(df)
            
        st.success(f"Loaded {len(df)} records for {symbol}")

        # Get currency symbol for this stock
        currency_symbol = get_currency_symbol(symbol)
        is_indian = is_indian_stock(symbol)

        # Create tabs for different analyses
        tab1, tab2, tab3, tab4, tab5, tab6, tab7, tab8 = st.tabs([
            "📊 Stock Analysis", 
            "🔮 Future Forecasting",
            "🤖 Enhanced Training",
            "📈 Technical Indicators", 
            "📰 News Sentiment", 
            "💰 Insider Trading",
            "🌍 Macro Analysis",
            "⚙️ Settings"
        ])

        with tab1:
            # Price chart
            st.subheader(f"Price Chart for {symbol}")
            st.line_chart(df.set_index('Date')['Close'])

            # Show latest technical indicators
            st.subheader("Latest Technical Indicators")
            latest = df.iloc[-1]
            cols = st.columns(3)
            with cols[0]:
                st.metric("RSI (14)", f"{latest['RSI_14']:.2f}")
                st.metric("MACD", f"{latest['MACD']:.2f}")
                st.metric("MACD Signal", f"{latest['MACD_Signal']:.2f}")
            with cols[1]:
                st.metric("Bollinger Upper", format_currency(latest['BB_Upper'], currency_symbol))
                st.metric("Bollinger Lower", format_currency(latest['BB_Lower'], currency_symbol))
                st.metric("Stochastic K", f"{latest['Stoch_K']:.2f}")
            with cols[2]:
                st.metric("ATR (14)", format_currency(latest['ATR_14'], currency_symbol))
                st.metric("Williams %R", f"{latest['Williams_R']:.2f}")
                st.metric("Volatility", f"{latest['Volatility']:.4f}")

            # Current price and market info
            st.subheader("📈 Current Market Information")
            current_price = latest['Close']
            price_change = df['Close'].iloc[-1] - df['Close'].iloc[-2] if len(df) > 1 else 0
            price_change_pct = (price_change / df['Close'].iloc[-2]) * 100 if len(df) > 1 else 0
            
            col1, col2, col3, col4 = st.columns(4)
            with col1:
                st.metric(
                    "Current Price",
                    format_currency(current_price, currency_symbol),
                    f"{price_change_pct:+.2f}%"
                )
            with col2:
                st.metric(
                    "Day High",
                    format_currency(latest['High'], currency_symbol)
                )
            with col3:
                st.metric(
                    "Day Low",
                    format_currency(latest['Low'], currency_symbol)
                )
            with col4:
                st.metric(
                    "Volume",
                    f"{latest['Volume']:,.0f}"
                )

        with tab2:
            st.header("🔮 Future Price Forecasting")
            st.markdown("Forecast stock prices up to 2 years into the future using advanced ML models and macroeconomic indicators.")
            
            col1, col2, col3 = st.columns([2, 2, 1])
            
            with col1:
                forecast_symbol = st.selectbox(
                    "Select Stock for Forecasting",
                    options=[symbol],
                    index=0,
                    key="forecast_symbol"
                )
            
            with col2:
                forecast_horizon = st.selectbox(
                    "Forecast Horizon",
                    options=[
                        ("30 days", 30),
                        ("3 months", 90),
                        ("6 months", 180),
                        ("1 year", 365),
                        ("2 years", 504)
                    ],
                    format_func=lambda x: x[0],
                    index=3,  # Default to 1 year
                    key="forecast_horizon"
                )
            forecast_days = forecast_horizon[1]
            
            with col3:
                include_macro_forecast = st.checkbox(
                    "Include Macro Features",
                    value=True,
                    help="Include macroeconomic indicators in forecasting"
                )
            
            # Forecast button
            if st.button("🚀 Generate Forecast", type="primary", key="generate_forecast"):
                with st.spinner("🔮 Generating future forecast..."):
                    try:
                        # Initialize forecaster
                        forecaster = FutureForecaster()
                        
                        # Generate forecast
                        forecast_df = forecaster.forecast_future(
                            symbol=forecast_symbol,
                            forecast_days=forecast_days,
                            include_macro=include_macro_forecast
                        )
                        
                        if not forecast_df.empty:
                            st.success(f"✅ Generated {len(forecast_df)} predictions for {forecast_symbol}")
                            
                            # Get historical data for comparison
                            end_date = datetime.now()
                            start_date = end_date - timedelta(days=365)
                            historical_df = fetch_yfinance(forecast_symbol, start_date.strftime('%Y-%m-%d'), end_date.strftime('%Y-%m-%d'))
                            
                            # Get forecast summary
                            summary = forecaster.get_forecast_summary(forecast_df, historical_df)
                            
                            # Display summary metrics
                            col1, col2, col3, col4 = st.columns(4)
                            
                            with col1:
                                st.metric(
                                    "Current Price",
                                    format_currency(summary.get('current_price', 0), currency_symbol)
                                )
                            
                            with col2:
                                st.metric(
                                    "Final Price",
                                    format_currency(summary.get('final_price', 0), currency_symbol),
                                    f"{summary.get('total_change_pct', 0):+.2f}%"
                                )
                            
                            with col3:
                                st.metric(
                                    "Forecast Volatility",
                                    f"{summary.get('forecast_volatility', 0):.2%}"
                                )
                            
                            with col4:
                                st.metric(
                                    "Max Drawdown",
                                    f"{summary.get('max_drawdown', 0):.2%}"
                                )
                            
                            # Create combined chart
                            st.subheader("📈 Historical vs Forecasted Prices")
                            
                            # Prepare data for plotting
                            historical_plot = historical_df[['Date', 'Close']].copy()
                            historical_plot.columns = ['Date', 'Price']
                            historical_plot['Type'] = 'Historical'
                            
                            forecast_plot = forecast_df[['Date', 'Predicted_Close']].copy()
                            forecast_plot.columns = ['Date', 'Price']
                            forecast_plot['Type'] = 'Forecast'
                            
                            combined_df = pd.concat([historical_plot, forecast_plot], ignore_index=True)
                            
                            # Create chart
                            fig = px.line(
                                combined_df,
                                x='Date',
                                y='Price',
                                color='Type',
                                title=f"{forecast_symbol} Price Forecast ({forecast_horizon[0]})",
                                color_discrete_map={
                                    'Historical': '#1f77b4',
                                    'Forecast': '#ff7f0e'
                                }
                            )
                            
                            fig.update_layout(
                                xaxis_title="Date",
                                yaxis_title="Price ($)",
                                hovermode='x unified',
                                height=500
                            )
                            
                            st.plotly_chart(fig, use_container_width=True)
                            
                            # Forecast details
                            col1, col2 = st.columns([2, 1])
                            
                            with col1:
                                st.subheader("📊 Forecast Details")
                                
                                # Price trend analysis
                                trend_slope = summary.get('trend_slope', 0)
                                if trend_slope > 0:
                                    trend_direction = "📈 Bullish"
                                    trend_color = "green"
                                else:
                                    trend_direction = "📉 Bearish"
                                    trend_color = "red"
                                
                                st.markdown(f"""
                                **Trend Analysis:**
                                - Direction: {trend_direction}
                                - Slope: {trend_slope:.4f}
                                - Confidence: {summary.get('confidence_level', 'Medium')}
                                """)
                                
                                # Key milestones
                                st.subheader("🎯 Key Milestones")
                                
                                milestones = [30, 90, 180, 365]
                                if forecast_days > 365:
                                    milestones.append(504)
                                
                                milestone_data = []
                                for milestone in milestones:
                                    if milestone <= len(forecast_df):
                                        milestone_price = forecast_df.iloc[milestone-1]['Predicted_Close']
                                        milestone_date = forecast_df.iloc[milestone-1]['Date']
                                        milestone_data.append({
                                            'Days': milestone,
                                            'Date': milestone_date.strftime('%Y-%m-%d'),
                                            'Price': format_currency(milestone_price, currency_symbol)
                                        })
                                
                                if milestone_data:
                                    milestone_df = pd.DataFrame(milestone_data)
                                    st.dataframe(milestone_df, use_container_width=True)
                            
                            with col2:
                                st.subheader("📋 Forecast Summary")
                                
                                st.markdown(f"""
                                **Forecast Period:**
                                - Start: {forecast_df['Date'].iloc[0].strftime('%Y-%m-%d')}
                                - End: {forecast_df['Date'].iloc[-1].strftime('%Y-%m-%d')}
                                - Duration: {len(forecast_df)} business days
                                
                                **Price Statistics:**
                                - Min: {format_currency(float(forecast_df['Predicted_Close'].min()), currency_symbol)}
                                - Max: {format_currency(float(forecast_df['Predicted_Close'].max()), currency_symbol)}
                                - Mean: {format_currency(float(forecast_df['Predicted_Close'].mean()), currency_symbol)}
                                - Std: {format_currency(float(forecast_df['Predicted_Close'].std()), currency_symbol)}
                                """)
                                
                            # Download forecast data
                            st.subheader("💾 Download Forecast Data")
                            
                            csv_data = forecast_df.to_csv(index=False)
                            st.download_button(
                                label="📥 Download Forecast CSV",
                                data=csv_data,
                                file_name=f"{forecast_symbol}_forecast_{forecast_days}days.csv",
                                mime="text/csv"
                            )
                            
                        else:
                            st.error("❌ Failed to generate forecast. Please try again.")
                            
                    except Exception as e:
                        st.error(f"❌ Error generating forecast: {str(e)}")
                        st.exception(e)

        with tab3:
            st.header("🤖 Enhanced Model Training")
            st.markdown("Train advanced LSTM models with comprehensive features, hyperparameter optimization, and backtesting.")
            
            # Training configuration
            col1, col2, col3 = st.columns([2, 2, 1])
            
            with col1:
                training_symbol = st.selectbox(
                    "Select Stock for Training",
                    options=[symbol],
                    index=0,
                    key="training_symbol"
                )
                
                training_start_date = st.date_input(
                    "Training Start Date",
                    value=(datetime.now() - timedelta(days=730)).date(),
                    key="training_start"
                )
                
            with col2:
                training_end_date = st.date_input(
                    "Training End Date",
                    value=datetime.now().date(),
                    key="training_end"
                )
                
                initial_capital = st.number_input(
                    "Initial Capital ($)",
                    value=100000,
                    min_value=1000,
                    step=10000,
                    key="initial_capital"
                )
                
            with col3:
                run_optimization = st.checkbox(
                    "Run Hyperparameter Optimization",
                    value=True,
                    help="Use Bayesian optimization to find best parameters"
                )
                
                optimization_trials = st.number_input(
                    "Optimization Trials",
                    value=10,
                    min_value=5,
                    max_value=50,
                    help="Number of optimization trials"
                )
            
            # Training options
            st.subheader("🎯 Training Options")
            
            col1, col2, col3 = st.columns(3)
            
            with col1:
                include_advanced_features = st.checkbox(
                    "Advanced Features",
                    value=True,
                    help="Include sentiment, microstructure, and advanced technical indicators"
                )
                
                include_macro_features = st.checkbox(
                    "Macro Features",
                    value=True,
                    help="Include macroeconomic indicators"
                )
                
            with col2:
                run_backtest = st.checkbox(
                    "Run Backtesting",
                    value=True,
                    help="Perform walk-forward backtesting after training"
                )
                
                save_model = st.checkbox(
                    "Save Model",
                    value=True,
                    help="Save the trained model for future use"
                )
                
            with col3:
                show_training_plots = st.checkbox(
                    "Show Training Plots",
                    value=True,
                    help="Display training progress and metrics"
                )
                
                export_results = st.checkbox(
                    "Export Results",
                    value=True,
                    help="Export training results and backtesting data"
                )
            
            # Start training button
            if st.button("🚀 Start Enhanced Training", type="primary", key="start_training"):
                with st.spinner("🤖 Training enhanced model..."):
                    try:
                        # Initialize enhanced training system
                        training_system = EnhancedTrainingSystem(
                            symbol=training_symbol,
                            start_date=training_start_date.strftime('%Y-%m-%d'),
                            end_date=training_end_date.strftime('%Y-%m-%d'),
                            initial_capital=initial_capital
                        )
                        
                        # Create progress bar
                        progress_bar = st.progress(0)
                        status_text = st.empty()
                        
                        # Step 1: Fetch and prepare data
                        status_text.text("📊 Fetching and preparing data...")
                        df = training_system.fetch_and_prepare_data()
                        progress_bar.progress(20)
                        
                        # Step 2: Create enhanced features
                        status_text.text("🔧 Creating enhanced features...")
                        featured_df = training_system.create_enhanced_features(df)
                        progress_bar.progress(40)
                        
                        # Step 3: Optimize hyperparameters (if enabled)
                        if run_optimization:
                            status_text.text("🎯 Running hyperparameter optimization...")
                            optimization_results = training_system.optimize_hyperparameters(featured_df)
                            progress_bar.progress(60)
                            
                            # Display optimization results
                            st.subheader("📊 Optimization Results")
                            if optimization_results:
                                best_params = optimization_results.get('best_params', {})
                                best_value = optimization_results.get('best_value', 0)
                                
                                st.success(f"✅ Best MAPE: {best_value:.2f}%")
                                st.write("**Best Parameters:**")
                                for param, value in best_params.items():
                                    st.write(f"- {param}: {value}")
                        
                        # Step 4: Train model
                        status_text.text("🤖 Training enhanced LSTM model...")
                        model, scaler = training_system.train_enhanced_model(featured_df)
                        progress_bar.progress(80)
                        
                        # Step 5: Evaluate performance
                        status_text.text("📊 Evaluating system performance...")
                        performance = training_system.evaluate_system_performance(featured_df)
                        progress_bar.progress(100)
                        
                        # Display results
                        st.success("✅ Enhanced training completed successfully!")
                        
                        # Training metrics
                        st.subheader("📈 Training Results")
                        
                        if performance and 'backtest_results' in performance:
                            backtest_results = performance['backtest_results']
                            
                            col1, col2, col3, col4 = st.columns(4)
                            
                            with col1:
                                st.metric(
                                    "MAPE",
                                    f"{backtest_results.get('mape', 0):.2f}%"
                                )
                            
                            with col2:
                                st.metric(
                                    "Directional Accuracy",
                                    f"{backtest_results.get('directional_accuracy', 0):.2f}%"
                                )
                            
                            with col3:
                                st.metric(
                                    "Cumulative Return",
                                    f"{backtest_results.get('cumulative_return', 0):.2f}%"
                                )
                            
                            with col4:
                                st.metric(
                                    "Sharpe Ratio",
                                    f"{backtest_results.get('sharpe_ratio', 0):.2f}"
                                )
                        
                        # Model information
                        st.subheader("🤖 Model Information")
                        
                        if performance and 'model_info' in performance:
                            model_info = performance['model_info']
                            
                            col1, col2 = st.columns(2)
                            
                            with col1:
                                st.write(f"**Model Type:** {model_info.get('model_type', 'Enhanced LSTM')}")
                                st.write(f"**Features Used:** {model_info.get('features_used', 0)}")
                                st.write(f"**Data Points:** {model_info.get('data_points', 0)}")
                            
                            with col2:
                                st.write(f"**Training Period:** {training_start_date} to {training_end_date}")
                                st.write(f"**Symbol:** {training_symbol}")
                                st.write(f"**Initial Capital:** ${initial_capital:,}")
                        
                        # Save system state
                        if save_model:
                            training_system.save_system_state()
                            st.success("💾 Model and system state saved successfully!")
                        
                        # Export results
                        if export_results and performance:
                            st.subheader("💾 Export Results")
                            
                            # Create results summary
                            results_summary = {
                                'symbol': training_symbol,
                                'training_period': f"{training_start_date} to {training_end_date}",
                                'performance_metrics': performance.get('backtest_results', {}),
                                'model_info': performance.get('model_info', {}),
                                'training_date': datetime.now().isoformat()
                            }
                            
                            # Export as JSON
                            import json
                            json_data = json.dumps(results_summary, indent=2)
                            st.download_button(
                                label="📥 Download Results JSON",
                                data=json_data,
                                file_name=f"{training_symbol}_enhanced_training_results.json",
                                mime="application/json"
                            )
                        
                    except Exception as e:
                        st.error(f"❌ Error during enhanced training: {str(e)}")
                        st.exception(e)

        with tab4:
            st.subheader("Trading Signals")
            
            # Technical signals
            st.write("**Technical Analysis Signals**")
            signals = get_trading_signals(df)
            for indicator, signal_data in signals.items():
                if indicator != 'Overall':
                    emoji = "🟢" if signal_data['signal'] == 'BUY' else "🔴" if signal_data['signal'] == 'SELL' else "⚪"
                    st.write(f"{emoji} **{indicator}**: {signal_data['signal']} (Confidence: {signal_data['confidence']:.0%})")
            
            if 'Overall' in signals:
                st.markdown(f"### 🎯 **Overall Technical Signal:** {signals['Overall']['signal']} (Confidence: {signals['Overall']['confidence']:.0%})")

            # Macro signals
            if include_macro:
                st.write("**Macroeconomic Signals**")
                macro_signals = get_macro_trading_signals(df)
                if macro_signals:
                    for indicator, signal_data in macro_signals.items():
                        emoji = "🟢" if signal_data['signal'] in ['GROWTH', 'CYCLICAL', 'GROWTH_TECH'] else "🔴" if signal_data['signal'] in ['DEFENSIVE', 'INFLATION_HEDGE'] else "⚪"
                        st.write(f"{emoji} **{indicator}**: {signal_data['signal']} (Confidence: {signal_data['confidence']:.0%})")
                        st.write(f"   *{signal_data['reason']}*")
                else:
                    st.info("No macro signals available")

        with tab7:
            st.header("🌍 Macroeconomic Analysis")
            st.markdown("Analyze macroeconomic indicators and their relationship with stock prices.")
            
            # Macro superimposition section
            st.subheader("📊 Macro Indicator Overlay")
            st.markdown("Superimpose macroeconomic indicators on stock price charts to visualize correlations and trends.")
            
            # Macro indicator selection
            col1, col2 = st.columns([2, 1])
            
            with col1:
                # Available macro indicators
                macro_indicators = {
                    "Interest Rates": ["Federal Funds Rate", "10-Year Treasury Yield", "2-Year Treasury Yield", "Yield Curve Spread"],
                    "Inflation": ["Consumer Price Index", "Core CPI", "Producer Price Index", "Inflation Expectations"],
                    "Economic Growth": ["GDP Growth", "Industrial Production", "Retail Sales", "Manufacturing PMI"],
                    "Employment": ["Unemployment Rate", "Non-Farm Payrolls", "Job Openings", "Wage Growth"],
                    "Commodities": ["Oil Prices (WTI)", "Gold Prices", "Copper Prices", "Natural Gas"],
                    "Currencies": ["US Dollar Index", "EUR/USD", "USD/JPY", "USD/CNY"],
                    "Market Sentiment": ["VIX Volatility", "Consumer Confidence", "Business Confidence", "Housing Market Index"],
                    "Money Supply": ["M2 Money Supply", "M1 Money Supply", "Bank Reserves", "Credit Growth"]
                }
                
                selected_category = st.selectbox(
                    "Select Macro Category",
                    list(macro_indicators.keys()),
                    key="macro_category"
                )
                
                selected_indicators = st.multiselect(
                    "Select Indicators to Overlay",
                    macro_indicators[selected_category],
                    default=macro_indicators[selected_category][:2],
                    key="macro_indicators"
                )
            
            with col2:
                # Chart options
                st.subheader("📈 Chart Options")
                
                chart_type = st.selectbox(
                    "Chart Type",
                    ["Overlay (Dual Y-axis)", "Subplot", "Correlation Heatmap"],
                    key="chart_type"
                )
                
                date_range = st.selectbox(
                    "Date Range",
                    ["1 Year", "2 Years", "5 Years", "10 Years"],
                    index=1,
                    key="date_range"
                )
                
                normalize_data = st.checkbox(
                    "Normalize Data",
                    value=True,
                    help="Normalize indicators to same scale for better comparison"
                )
                
                show_correlation = st.checkbox(
                    "Show Correlation Analysis",
                    value=True,
                    help="Display correlation coefficients between stock and macro indicators"
                )
            
            # Generate macro overlay chart
            if st.button("🚀 Generate Macro Overlay Chart", type="primary"):
                with st.spinner("📊 Fetching macro data and generating overlay chart..."):
                    try:
                        # Initialize macro indicators fetcher
                        macro_fetcher = MacroIndicators()
                        
                        # Calculate date range
                        years_map = {"1 Year": 1, "2 Years": 2, "5 Years": 5, "10 Years": 10}
                        years = years_map[date_range]
                        start_date_macro = (datetime.now() - timedelta(days=365*years)).strftime('%Y-%m-%d')
                        end_date_macro = datetime.now().strftime('%Y-%m-%d')
                        
                        # Fetch macro data
                        macro_data = macro_fetcher.get_macro_indicators(start_date_macro, end_date_macro)
                        
                        if macro_data:
                            # Prepare stock data for the same period
                            stock_data = fetch_yfinance(symbol, start_date_macro, end_date_macro)
                            
                            if not stock_data.empty:
                                # Create the overlay chart based on selected type
                                if chart_type == "Overlay (Dual Y-axis)":
                                    fig = create_macro_overlay_chart(
                                        stock_data, macro_data, selected_indicators, 
                                        normalize_data, symbol
                                    )
                                    st.plotly_chart(fig, use_container_width=True)
                                    
                                elif chart_type == "Subplot":
                                    fig = create_macro_subplot_chart(
                                        stock_data, macro_data, selected_indicators, 
                                        normalize_data, symbol
                                    )
                                    st.plotly_chart(fig, use_container_width=True)
                                    
                                elif chart_type == "Correlation Heatmap":
                                    fig = create_correlation_heatmap(
                                        stock_data, macro_data, selected_indicators, symbol
                                    )
                                    st.plotly_chart(fig, use_container_width=True)
                                
                                # Show correlation analysis
                                if show_correlation:
                                    st.subheader("📊 Correlation Analysis")
                                    correlation_df = calculate_macro_correlations(
                                        stock_data, macro_data, selected_indicators
                                    )
                                    st.dataframe(correlation_df, use_container_width=True)
                                    
                                    # Correlation insights
                                    st.subheader("💡 Correlation Insights")
                                    insights = generate_correlation_insights(correlation_df, symbol)
                                    for insight in insights:
                                        st.write(f"• {insight}")
                                
                                # Macro trading signals
                                st.subheader("🎯 Macro Trading Signals")
                                macro_signals = get_macro_trading_signals(df)
                                if macro_signals:
                                    for indicator, signal_data in macro_signals.items():
                                        emoji = "🟢" if signal_data['signal'] in ['GROWTH', 'CYCLICAL', 'GROWTH_TECH'] else "🔴" if signal_data['signal'] in ['DEFENSIVE', 'INFLATION_HEDGE'] else "⚪"
                                        st.write(f"{emoji} **{indicator}**: {signal_data['signal']} (Confidence: {signal_data['confidence']:.0%})")
                                        st.write(f"   *{signal_data['reason']}*")
                                else:
                                    st.info("No macro signals available")
                                
                            else:
                                st.error("❌ Failed to fetch stock data for the selected period.")
                        else:
                            st.error("❌ Failed to fetch macroeconomic data. Please check your internet connection.")
                            
                    except Exception as e:
                        st.error(f"❌ Error generating macro overlay chart: {str(e)}")
                        st.exception(e)
            
            # Macro risk factors
            if include_macro and show_advanced:
                st.subheader("Macro Risk Factors")
                macro_risk_cols = st.columns(2)
                with macro_risk_cols[0]:
                    if 'economic_stress' in latest:
                        stress = latest['economic_stress']
                        stress_level = "🔴 High Risk" if stress > 0.6 else "🟡 Medium Risk" if stress > 0.3 else "🟢 Low Risk"
                        st.metric("Economic Stress Risk", stress_level, f"{stress:.3f}")
                    
                    if 'inflation_adjusted_volatility' in latest:
                        adj_vol = latest['inflation_adjusted_volatility']
                        vol_risk = "🔴 High" if adj_vol > 0.05 else "🟡 Medium" if adj_vol > 0.02 else "🟢 Low"
                        st.metric("Inflation Adjusted Vol Risk", vol_risk, f"{adj_vol:.4f}")
                
                with macro_risk_cols[1]:
                    if 'rate_adjusted_returns' in latest:
                        rate_adj_ret = latest['rate_adjusted_returns']
                        ret_signal = "🟢 Positive" if rate_adj_ret > 0 else "🔴 Negative"
                        st.metric("Rate Adjusted Returns", ret_signal, f"{rate_adj_ret:.4f}")

        with tab8:
            st.subheader("Settings")
            
            # Stock selection
            symbol = st.text_input("Enter Stock Symbol (e.g., AAPL, MSFT, TSLA)", value=symbol)
            
            # Date selection
            end_date = st.date_input("End Date", value=end_date)
            start_date = st.date_input("Start Date", value=start_date)
            
            # Analysis options
            include_macro = st.checkbox("Include Macroeconomic Analysis", value=include_macro)
            show_advanced = st.checkbox("Show Advanced Features", value=show_advanced)

    except Exception as e:
        st.error(f"Error: {e}")
        st.error("Please check your internet connection and try again.")
