#!/usr/bin/env python3
"""
Enhanced Long-Term Investment Streamlit Dashboard
================================================
Advanced interface for long-term investment strategies with:
- Multi-asset portfolio analysis
- Long-term vs short-term comparison
- CAGR-focused metrics
- Buy-and-hold strategy evaluation
"""

import streamlit as st
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import plotly.graph_objs as go
from plotly.subplots import make_subplots
import plotly.express as px
from long_term_investment_system import LongTermInvestmentSystem
import joblib
import os

# Page configuration
st.set_page_config(
    page_title="🎯 Long-Term Investment Dashboard", 
    layout="wide",
    initial_sidebar_state="expanded"
)

st.title("🎯 Advanced Long-Term Investment Analysis Dashboard")
st.markdown("**Focus on CAGR, Quality, and Sustainable Long-Term Growth**")

# Sidebar controls
st.sidebar.header("🎯 Long-Term Investment Settings")

# Asset selection
asset_categories = {
    "🇺🇸 US Large Cap": ["AAPL", "MSFT", "GOOGL", "AMZN", "TSLA", "META", "NVDA"],
    "🇺🇸 US ETFs": ["SPY", "QQQ", "VTI", "VOO", "VEA", "VWO"],
    "🇮🇳 Indian Large Cap": ["RELIANCE.NS", "TCS.NS", "HDFCBANK.NS", "INFY.NS", "ITC.NS"],
    "🇮🇳 Indian ETFs": ["NIFTYBEES.NS", "JUNIORBEES.NS", "BANKBEES.NS", "ICICIB22.NS", "ITBEES.NS"],
    "🏛️ Blue Chip Dividend": ["JNJ", "PG", "KO", "PEP", "WMT", "MCD"]
}

selected_category = st.sidebar.selectbox(
    "Select Asset Category",
    list(asset_categories.keys())
)

selected_symbol = st.sidebar.selectbox(
    "Select Asset",
    asset_categories[selected_category]
)

# Time horizon settings
st.sidebar.subheader("⏱️ Time Horizon")
analysis_start = st.sidebar.date_input(
    "Analysis Start Date",
    value=(datetime.now() - timedelta(days=3650)).date(),  # 10 years
    help="Recommended: 5-10 years for robust long-term analysis"
)

initial_investment = st.sidebar.number_input(
    "Initial Investment ($)",
    value=100000,
    min_value=10000,
    max_value=10000000,
    step=10000,
    help="Initial capital for backtesting"
)

# Analysis options
st.sidebar.subheader("📊 Analysis Options")
compare_strategies = st.sidebar.checkbox("Compare Long-term vs Short-term", value=True)
include_portfolio = st.sidebar.checkbox("Portfolio Analysis", value=True)
show_risk_metrics = st.sidebar.checkbox("Advanced Risk Metrics", value=True)

# Main analysis button
if st.sidebar.button("🚀 Run Long-Term Analysis", type="primary"):
    
    with st.spinner(f"🔍 Running comprehensive long-term analysis for {selected_symbol}..."):
        
        # Initialize long-term system
        lt_system = LongTermInvestmentSystem(
            symbol=selected_symbol,
            start_date=analysis_start.strftime('%Y-%m-%d'),
            initial_capital=initial_investment
        )
        
        # Run analysis
        results = lt_system.run_complete_analysis()
        
        if results:
            st.success(f"✅ Long-term analysis completed for {selected_symbol}")
            
            # Extract results
            signals = results['current_signals']
            backtest = results['backtest_results']
            strategy_perf = backtest['strategy_performance']
            buy_hold_perf = backtest['buy_hold_performance']
            
            # Main metrics display
            st.header(f"📊 Long-Term Investment Analysis: {selected_symbol}")
            
            # Key metrics row
            col1, col2, col3, col4 = st.columns(4)
            
            with col1:
                signal_color = "green" if signals['Overall']['signal'] == 'BUY' else "red" if signals['Overall']['signal'] == 'SELL' else "orange"
                st.metric(
                    "🎯 Investment Signal",
                    signals['Overall']['signal'],
                    f"{signals['Overall']['confidence']:.1%} confidence",
                    delta_color=signal_color
                )
            
            with col2:
                st.metric(
                    "📈 Strategy CAGR",
                    f"{strategy_perf['cagr']:.2f}%",
                    f"{strategy_perf['cagr'] - buy_hold_perf['cagr']:+.2f}% vs B&H"
                )
            
            with col3:
                st.metric(
                    "💎 Current Price",
                    f"${signals['Latest_Data']['Price']:.2f}",
                    f"{signals['Latest_Data']['Annual_Return']:+.1f}% (1Y)"
                )
            
            with col4:
                st.metric(
                    "🛡️ Sharpe Ratio",
                    f"{strategy_perf['sharpe_ratio']:.2f}",
                    f"{strategy_perf['max_drawdown']:.1f}% max drawdown"
                )
            
            # Performance comparison chart
            st.subheader("📈 Long-Term Performance Comparison")
            
            # Create performance comparison
            portfolio_history = pd.DataFrame(backtest['portfolio_history'])
            
            if not portfolio_history.empty:
                # Calculate buy-and-hold performance
                start_price = portfolio_history['Price'].iloc[0]
                portfolio_history['Buy_Hold_Value'] = (portfolio_history['Price'] / start_price) * initial_investment
                portfolio_history['Strategy_Return'] = (portfolio_history['Portfolio_Value'] / initial_investment - 1) * 100
                portfolio_history['BuyHold_Return'] = (portfolio_history['Buy_Hold_Value'] / initial_investment - 1) * 100
                
                # Create comparison chart
                fig = go.Figure()
                
                # Strategy performance
                fig.add_trace(go.Scatter(
                    x=portfolio_history['Date'],
                    y=portfolio_history['Strategy_Return'],
                    name='Long-Term Strategy',
                    line=dict(color='blue', width=2)
                ))
                
                # Buy and hold
                fig.add_trace(go.Scatter(
                    x=portfolio_history['Date'],
                    y=portfolio_history['BuyHold_Return'],
                    name='Buy & Hold',
                    line=dict(color='green', width=2, dash='dash')
                ))
                
                fig.update_layout(
                    title=f"Long-Term Performance Comparison: {selected_symbol}",
                    xaxis_title="Date",
                    yaxis_title="Cumulative Return (%)",
                    hovermode='x unified',
                    height=500
                )
                
                st.plotly_chart(fig, use_container_width=True)
            
            # Detailed analysis tabs
            tab1, tab2, tab3, tab4 = st.tabs([
                "📊 Strategy Details", 
                "📈 Risk Analysis", 
                "💰 Transactions", 
                "🎯 Recommendations"
            ])
            
            with tab1:
                st.subheader("📊 Strategy Performance Details")
                
                col1, col2 = st.columns(2)
                
                with col1:
                    st.write("**Long-Term Strategy Performance:**")
                    strategy_metrics = pd.DataFrame([
                        {"Metric": "Total Return", "Value": f"{strategy_perf['total_return']:.2f}%"},
                        {"Metric": "CAGR", "Value": f"{strategy_perf['cagr']:.2f}%"},
                        {"Metric": "Volatility", "Value": f"{strategy_perf['volatility']:.2f}%"},
                        {"Metric": "Sharpe Ratio", "Value": f"{strategy_perf['sharpe_ratio']:.2f}"},
                        {"Metric": "Max Drawdown", "Value": f"{strategy_perf['max_drawdown']:.2f}%"},
                        {"Metric": "Total Transactions", "Value": f"{strategy_perf['total_transactions']}"}
                    ])
                    st.dataframe(strategy_metrics, use_container_width=True)
                
                with col2:
                    st.write("**Buy & Hold Comparison:**")
                    buyhold_metrics = pd.DataFrame([
                        {"Metric": "Total Return", "Value": f"{buy_hold_perf['total_return']:.2f}%"},
                        {"Metric": "CAGR", "Value": f"{buy_hold_perf['cagr']:.2f}%"},
                        {"Metric": "Final Value", "Value": f"${buy_hold_perf['final_value']:,.0f}"},
                        {"Metric": "Strategy Excess Return", "Value": f"{backtest['outperformance']['excess_return']:+.2f}%"},
                        {"Metric": "Strategy Excess CAGR", "Value": f"{backtest['outperformance']['excess_cagr']:+.2f}%"},
                        {"Metric": "Transaction Cost Savings", "Value": "Significant"}
                    ])
                    st.dataframe(buyhold_metrics, use_container_width=True)
                
                # Long-term trends analysis
                st.subheader("🔄 Long-Term Trend Analysis")
                
                trend_data = signals['Trend']
                momentum_data = signals['Momentum']
                position_data = signals['Position']
                
                col1, col2, col3 = st.columns(3)
                
                with col1:
                    st.write("**🎯 Trend Signals:**")
                    for trend in trend_data:
                        st.write(f"• {trend}")
                    if not trend_data:
                        st.write("• No strong trends detected")
                
                with col2:
                    st.write("**⚡ Momentum Signals:**")
                    for momentum in momentum_data:
                        st.write(f"• {momentum}")
                    if not momentum_data:
                        st.write("• Neutral momentum")
                
                with col3:
                    st.write("**📍 Position Signals:**")
                    for position in position_data:
                        st.write(f"• {position}")
                    if not position_data:
                        st.write("• Neutral position")
            
            with tab2:
                st.subheader("📈 Risk Analysis")
                
                if show_risk_metrics:
                    # Risk metrics visualization
                    risk_metrics = pd.DataFrame([
                        {"Risk Metric": "Volatility", "Strategy": f"{strategy_perf['volatility']:.2f}%", "Interpretation": "Annual volatility"},
                        {"Risk Metric": "Sharpe Ratio", "Strategy": f"{strategy_perf['sharpe_ratio']:.2f}", "Interpretation": "Risk-adjusted returns"},
                        {"Risk Metric": "Max Drawdown", "Strategy": f"{strategy_perf['max_drawdown']:.2f}%", "Interpretation": "Worst peak-to-trough decline"},
                        {"Risk Metric": "Transaction Frequency", "Strategy": f"{strategy_perf['total_transactions']}", "Interpretation": "Low = Conservative approach"}
                    ])
                    
                    st.dataframe(risk_metrics, use_container_width=True)
                    
                    # Risk visualization
                    if not portfolio_history.empty:
                        # Calculate rolling volatility
                        portfolio_history['Rolling_Vol'] = portfolio_history['Strategy_Return'].rolling(window=30).std() * np.sqrt(252)
                        
                        fig = make_subplots(
                            rows=2, cols=1,
                            subplot_titles=('Cumulative Returns', 'Rolling 30-Day Volatility'),
                            vertical_spacing=0.1
                        )
                        
                        # Returns
                        fig.add_trace(
                            go.Scatter(x=portfolio_history['Date'], y=portfolio_history['Strategy_Return'], 
                                     name='Strategy Returns', line=dict(color='blue')),
                            row=1, col=1
                        )
                        
                        # Volatility
                        fig.add_trace(
                            go.Scatter(x=portfolio_history['Date'], y=portfolio_history['Rolling_Vol'], 
                                     name='Rolling Volatility', line=dict(color='red')),
                            row=2, col=1
                        )
                        
                        fig.update_layout(height=600, showlegend=False)
                        st.plotly_chart(fig, use_container_width=True)
                
                # Risk assessment
                st.subheader("🛡️ Risk Assessment")
                
                if strategy_perf['sharpe_ratio'] > 1.0:
                    st.success("🟢 **Excellent Risk-Adjusted Returns** - Sharpe ratio > 1.0")
                elif strategy_perf['sharpe_ratio'] > 0.5:
                    st.info("🟡 **Good Risk-Adjusted Returns** - Moderate Sharpe ratio")
                else:
                    st.warning("🔴 **Poor Risk-Adjusted Returns** - Low Sharpe ratio")
                
                if abs(strategy_perf['max_drawdown']) < 20:
                    st.success("🟢 **Low Drawdown Risk** - Max drawdown < 20%")
                elif abs(strategy_perf['max_drawdown']) < 40:
                    st.info("🟡 **Moderate Drawdown Risk** - Max drawdown 20-40%")
                else:
                    st.warning("🔴 **High Drawdown Risk** - Max drawdown > 40%")
            
            with tab3:
                st.subheader("💰 Transaction History")
                
                transactions = backtest['transactions']
                if transactions:
                    transactions_df = pd.DataFrame(transactions)
                    transactions_df['Date'] = pd.to_datetime(transactions_df['Date']).dt.date
                    
                    st.dataframe(transactions_df, use_container_width=True)
                    
                    # Transaction summary
                    buy_transactions = len([t for t in transactions if t['Action'] == 'BUY'])
                    sell_transactions = len([t for t in transactions if t['Action'] == 'SELL'])
                    
                    col1, col2, col3 = st.columns(3)
                    with col1:
                        st.metric("Total Transactions", len(transactions))
                    with col2:
                        st.metric("Buy Orders", buy_transactions)
                    with col3:
                        st.metric("Sell Orders", sell_transactions)
                    
                    st.info("💡 **Low transaction frequency indicates a disciplined, long-term approach**")
                else:
                    st.info("📊 No transactions executed - Pure buy & hold approach recommended")
            
            with tab4:
                st.subheader("🎯 Long-Term Investment Recommendations")
                
                # Current recommendation
                overall_signal = signals['Overall']['signal']
                confidence = signals['Overall']['confidence']
                
                if overall_signal == 'BUY' and confidence > 0.7:
                    st.success("🟢 **STRONG BUY RECOMMENDATION**")
                    st.write("✅ Excellent long-term opportunity with high confidence")
                    st.write("✅ Strong technical and fundamental indicators")
                    st.write("✅ Suitable for long-term portfolio allocation")
                elif overall_signal == 'BUY':
                    st.info("🟢 **BUY RECOMMENDATION**")
                    st.write("✅ Good long-term potential")
                    st.write("⚠️ Monitor for optimal entry point")
                elif overall_signal == 'SELL':
                    st.warning("🔴 **SELL RECOMMENDATION**")
                    st.write("⚠️ Consider reducing position size")
                    st.write("⚠️ Look for alternative investments")
                else:
                    st.info("🟡 **HOLD RECOMMENDATION**")
                    st.write("📊 Monitor current position")
                    st.write("⏰ Wait for better entry/exit signals")
                
                # Long-term strategy recommendations
                st.subheader("📋 Long-Term Strategy Recommendations")
                
                years_analyzed = (datetime.now() - datetime.strptime(analysis_start.strftime('%Y-%m-%d'), '%Y-%m-%d')).days / 365.25
                
                st.write(f"**Based on {years_analyzed:.1f} years of analysis:**")
                
                if strategy_perf['cagr'] > buy_hold_perf['cagr']:
                    st.write("🎯 **Active Strategy Recommended** - Outperforms buy & hold")
                    st.write(f"• Expected excess CAGR: +{strategy_perf['cagr'] - buy_hold_perf['cagr']:.2f}%")
                    st.write(f"• Transaction frequency: {strategy_perf['total_transactions']} over {years_analyzed:.1f} years")
                else:
                    st.write("🎯 **Buy & Hold Recommended** - Simpler approach with better returns")
                    st.write(f"• Buy & Hold CAGR: {buy_hold_perf['cagr']:.2f}%")
                    st.write("• Lower complexity and transaction costs")
                
                # Portfolio allocation suggestions
                st.subheader("💼 Portfolio Allocation Suggestions")
                
                if "ETF" in selected_symbol.upper() or selected_symbol in ["SPY", "QQQ", "VTI"]:
                    st.write("🏛️ **Core Holding Suitable** - Can be 20-40% of portfolio")
                elif signals['Latest_Data']['Annual_Return'] > 15:
                    st.write("🚀 **Growth Allocation** - Consider 10-20% allocation")
                else:
                    st.write("💎 **Satellite Holding** - Consider 5-15% allocation")
                
                st.write("**💡 Remember:**")
                st.write("• Diversify across asset classes and geographies")
                st.write("• Rebalance quarterly or semi-annually")
                st.write("• Focus on CAGR over short-term volatility")
                st.write("• Consider tax implications of trading frequency")
        
        else:
            st.error("❌ Analysis failed. Please check the symbol and try again.")

# Portfolio analysis section
if include_portfolio:
    st.header("💼 Multi-Asset Long-Term Portfolio Analysis")
    
    if st.button("🚀 Analyze Recommended Long-Term Portfolio"):
        
        # Recommended long-term portfolio
        lt_portfolio = {
            "US Large Cap": ["AAPL", "MSFT", "GOOGL"],
            "US Market ETF": ["SPY"],
            "Indian Large Cap": ["RELIANCE.NS", "TCS.NS"],
            "Indian ETF": ["NIFTYBEES.NS"]
        }
        
        portfolio_results = {}
        
        with st.spinner("🔍 Analyzing multi-asset long-term portfolio..."):
            
            progress_bar = st.progress(0)
            total_assets = sum(len(assets) for assets in lt_portfolio.values())
            current_asset = 0
            
            for category, symbols in lt_portfolio.items():
                for symbol in symbols:
                    try:
                        lt_system = LongTermInvestmentSystem(
                            symbol=symbol,
                            start_date=(datetime.now() - timedelta(days=1825)).strftime('%Y-%m-%d'),  # 5 years
                            initial_capital=20000  # Smaller allocation per asset
                        )
                        
                        results = lt_system.run_complete_analysis()
                        
                        if results:
                            portfolio_results[symbol] = {
                                'category': category,
                                'cagr': results['backtest_results']['strategy_performance']['cagr'],
                                'sharpe': results['backtest_results']['strategy_performance']['sharpe_ratio'],
                                'signal': results['current_signals']['Overall']['signal'],
                                'confidence': results['current_signals']['Overall']['confidence']
                            }
                    except:
                        pass
                    
                    current_asset += 1
                    progress_bar.progress(current_asset / total_assets)
        
        if portfolio_results:
            st.success(f"✅ Portfolio analysis completed for {len(portfolio_results)} assets")
            
            # Portfolio summary
            portfolio_df = pd.DataFrame.from_dict(portfolio_results, orient='index')
            portfolio_df.index.name = 'Symbol'
            portfolio_df = portfolio_df.reset_index()
            
            # Calculate portfolio metrics
            avg_cagr = portfolio_df['cagr'].mean()
            avg_sharpe = portfolio_df['sharpe'].mean()
            
            col1, col2, col3, col4 = st.columns(4)
            with col1:
                st.metric("Portfolio Avg CAGR", f"{avg_cagr:.2f}%")
            with col2:
                st.metric("Portfolio Avg Sharpe", f"{avg_sharpe:.2f}")
            with col3:
                buy_signals = len(portfolio_df[portfolio_df['signal'] == 'BUY'])
                st.metric("Buy Signals", f"{buy_signals}/{len(portfolio_df)}")
            with col4:
                high_confidence = len(portfolio_df[portfolio_df['confidence'] > 0.7])
                st.metric("High Confidence", f"{high_confidence}/{len(portfolio_df)}")
            
            # Portfolio visualization
            fig = px.scatter(
                portfolio_df, 
                x='cagr', 
                y='sharpe', 
                color='category',
                size='confidence',
                hover_data=['Symbol', 'signal'],
                title="Long-Term Portfolio Risk-Return Profile"
            )
            fig.update_layout(
                xaxis_title="CAGR (%)",
                yaxis_title="Sharpe Ratio",
                height=500
            )
            st.plotly_chart(fig, use_container_width=True)
            
            # Portfolio table
            st.subheader("📊 Portfolio Analysis Results")
            st.dataframe(portfolio_df, use_container_width=True)

# Footer
st.markdown("---")
st.markdown("**🎯 Long-Term Investment Dashboard** | Focus on CAGR, Quality, and Sustainable Growth")
st.markdown("💡 *Remember: Long-term investing requires patience, discipline, and focus on fundamentals over short-term noise.*") 