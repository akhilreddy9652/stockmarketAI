# 📈 Advanced Stock Predictor System

A comprehensive stock prediction and analysis system with advanced machine learning models, real-time data integration, and interactive dashboards for both US and Indian stock markets.

## 🚀 Features

### 🎯 Core Capabilities
- **Advanced ML Models**: LSTM, XGBoost, Random Forest, and ensemble methods
- **Dual Market Support**: US stocks ($) and Indian stocks (₹) with proper currency formatting
- **Real-time Data**: Live market data integration via Yahoo Finance
- **Future Forecasting**: Up to 2-year price predictions with confidence intervals
- **Backtesting Engine**: Walk-forward analysis with comprehensive performance metrics

### 📊 Analysis Features
- **Technical Indicators**: 40+ indicators including RSI, MACD, Bollinger Bands, etc.
- **Macroeconomic Integration**: Fed rates, inflation, GDP, and market sentiment
- **Trading Signals**: Multi-timeframe signals with confidence levels
- **Portfolio Analysis**: Risk-adjusted returns and portfolio optimization
- **Sentiment Analysis**: News sentiment and insider trading signals

### 🌐 Interactive Dashboards
- **Main Dashboard**: Comprehensive stock analysis and forecasting
- **Indian ETF Monitor**: Specialized dashboard for Indian ETF portfolio tracking
- **Real-time Monitoring**: Live price updates and trading signals

## 🛠️ Installation

### Prerequisites
- Python 3.8+
- pip package manager

### Setup
1. **Clone the repository**
   ```bash
   git clone https://github.com/akhilreddy9652/stockmarket.git
   cd stockmarket
   ```

2. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

3. **Run the application**
   ```bash
   # Main dashboard
   streamlit run streamlit_app.py --server.port 8501
   
   # Indian ETF dashboard
   streamlit run indian_etf_monitoring_dashboard.py --server.port 8502
   ```

## 📱 Usage

### Main Dashboard (Port 8501)
- **Stock Selection**: Choose from US stocks or 200+ Indian stocks
- **Analysis**: Get comprehensive technical and fundamental analysis
- **Forecasting**: Generate future price predictions up to 2 years
- **Backtesting**: Test strategies with historical data

### Indian ETF Dashboard (Port 8502)
- **Portfolio Monitoring**: Track Indian ETF performance
- **Trading Signals**: Real-time buy/sell recommendations
- **Risk Analysis**: Portfolio risk assessment and optimization

## 🏗️ System Architecture

### Core Components
```
├── streamlit_app.py                 # Main dashboard
├── indian_etf_monitoring_dashboard.py  # Indian ETF dashboard
├── data_ingestion.py               # Data fetching and processing
├── feature_engineering.py          # Technical indicators and features
├── future_forecasting.py           # ML-based forecasting
├── backtesting.py                  # Strategy backtesting
├── macro_indicators.py             # Macroeconomic data integration
└── train_enhanced_system.py        # Model training and optimization
```

### Data Pipeline
1. **Data Ingestion**: Real-time market data via Yahoo Finance API
2. **Feature Engineering**: 100+ technical and fundamental features
3. **Model Training**: Ensemble of LSTM, XGBoost, and Random Forest
4. **Prediction**: Multi-step ahead forecasting with confidence intervals
5. **Backtesting**: Performance validation with walk-forward analysis

## 📊 Supported Markets

### US Stocks
- **Major Indices**: S&P 500, NASDAQ, Dow Jones
- **Popular Stocks**: AAPL, MSFT, GOOGL, TSLA, AMZN, META, NVDA
- **Currency**: USD ($) with standard formatting

### Indian Stocks
- **Indices**: Nifty 50 (^NSEI), Bank Nifty, Sectoral indices
- **Stocks**: 200+ stocks including Nifty 50 and Next 50
- **ETFs**: NIFTYBEES, JUNIORBEES, BANKBEES, ITBEES
- **Currency**: INR (₹) with Indian number formatting (Lakhs/Crores)

## 🎯 Key Performance Metrics

### Model Accuracy
- **MAPE**: Mean Absolute Percentage Error < 2%
- **Directional Accuracy**: 85-90% for trend prediction
- **Sharpe Ratio**: Risk-adjusted returns > 1.0
- **Maximum Drawdown**: Controlled risk management

### Backtesting Results
- **NIFTYBEES**: 88.2% directional accuracy, 33,528% total return
- **Indian ETF Portfolio**: 88.7% average accuracy, 100% win rate
- **US Stocks**: Consistent performance across multiple timeframes

## 🔧 Advanced Features

### Machine Learning Models
- **Ultra-Enhanced LSTM**: Deep learning with attention mechanisms
- **XGBoost**: Gradient boosting with hyperparameter optimization
- **Random Forest**: Ensemble learning with feature importance
- **SVR**: Support Vector Regression for non-linear patterns

### Feature Engineering
- **Price Action**: OHLC patterns, gaps, and momentum
- **Volatility Models**: GARCH, realized volatility, VIX correlation
- **Volume Analysis**: Volume profile, money flow, accumulation
- **Market Regimes**: Bull/bear market identification
- **Seasonal Patterns**: Calendar effects and cyclical analysis

### Risk Management
- **Position Sizing**: Dynamic allocation based on volatility
- **Stop Losses**: Adaptive risk management
- **Portfolio Optimization**: Modern Portfolio Theory implementation
- **Stress Testing**: Scenario analysis and Monte Carlo simulation

## 📈 Trading Strategies

### Signal Generation
- **Technical Signals**: Multi-timeframe convergence
- **Momentum Strategies**: Trend following and mean reversion
- **Volatility Strategies**: Volatility breakout and contraction
- **Macro Signals**: Economic indicator-based signals

### Portfolio Management
- **Asset Allocation**: Sector and market cap diversification
- **Rebalancing**: Periodic portfolio optimization
- **Risk Parity**: Equal risk contribution across assets
- **Factor Investing**: Value, growth, momentum, and quality factors

## 🌍 Macroeconomic Integration

### Data Sources
- **Federal Reserve**: Interest rates, money supply, inflation
- **Economic Indicators**: GDP, employment, manufacturing PMI
- **Market Sentiment**: VIX, consumer confidence, yield curves
- **Commodity Prices**: Oil, gold, copper, and agricultural products

### Analysis Features
- **Correlation Analysis**: Stock-macro indicator relationships
- **Regime Detection**: Economic cycle identification
- **Stress Testing**: Economic scenario impact analysis
- **Policy Impact**: Central bank policy effect modeling

## 🔍 Technical Indicators

### Trend Indicators
- Simple/Exponential Moving Averages (SMA/EMA)
- MACD (Moving Average Convergence Divergence)
- ADX (Average Directional Index)
- Parabolic SAR

### Momentum Indicators
- RSI (Relative Strength Index)
- Stochastic Oscillator
- Williams %R
- Rate of Change (ROC)

### Volatility Indicators
- Bollinger Bands
- Average True Range (ATR)
- Volatility Index
- Keltner Channels

### Volume Indicators
- On-Balance Volume (OBV)
- Volume Price Trend (VPT)
- Accumulation/Distribution Line
- Money Flow Index (MFI)

## 📊 Data Sources

- **Yahoo Finance**: Primary data source for stock prices and fundamentals
- **FRED (Federal Reserve)**: Macroeconomic indicators
- **News APIs**: Sentiment analysis data
- **Insider Trading**: SEC filings and insider transactions

## 🔒 Security & Compliance

- **Data Privacy**: No personal financial data storage
- **API Rate Limits**: Respectful API usage with proper throttling
- **Disclaimer**: Educational and research purposes only
- **Risk Warning**: Past performance doesn't guarantee future results

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit your changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## ⚠️ Disclaimer

This software is for educational and research purposes only. It should not be considered as financial advice. Always consult with qualified financial advisors before making investment decisions. The developers are not responsible for any financial losses incurred from using this software.

## 📞 Support

For support, please open an issue on GitHub or contact the development team.

---

**Built with ❤️ for the trading and investment community**
