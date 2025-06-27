# 🚀 Deployment Guide - Stock Market AI

## Live Demo
Your app will be available at: `https://your-app-name.streamlit.app`

## Streamlit Community Cloud Deployment

### Quick Deploy
1. Go to: https://share.streamlit.io/
2. Sign in with GitHub
3. Click "New app"
4. Repository: `akhilreddy9652/stockmarketAI`
5. Branch: `main`
6. Main file: `streamlit_app.py`
7. Click "Deploy!"

### Multiple Apps Available
- **Main Dashboard**: `streamlit_app.py` - Comprehensive stock analysis
- **Indian ETF Monitor**: `indian_etf_monitoring_dashboard.py` - Indian ETF tracking
- **Long-term Analysis**: `enhanced_long_term_streamlit.py` - Long-term investment analysis

### App Features
- ✅ Real-time stock data for US & Indian markets
- ✅ Advanced ML predictions (LSTM, XGBoost, Random Forest)
- ✅ Interactive charts and visualizations
- ✅ Portfolio analysis and optimization
- ✅ Currency-aware formatting ($ for US, ₹ for Indian stocks)
- ✅ 88%+ prediction accuracy proven by backtesting

### Performance Notes
- First load may take 30-60 seconds (model loading)
- Subsequent interactions are fast
- Data updates in real-time during market hours

### Troubleshooting
If deployment fails:
1. Check requirements.txt for compatibility
2. Ensure all imports are available
3. Check Streamlit Cloud logs for errors

## Alternative Deployments

### Heroku
```bash
# Create Procfile
echo "web: streamlit run streamlit_app.py --server.port=\$PORT --server.address=0.0.0.0" > Procfile

# Deploy
git add . && git commit -m "Add Heroku config"
heroku create your-app-name
git push heroku main
```

### Railway
1. Connect GitHub repository
2. Select main branch
3. Set start command: `streamlit run streamlit_app.py --server.port=$PORT --server.address=0.0.0.0`

### Local Development
```bash
pip install -r requirements.txt
streamlit run streamlit_app.py
```

## Environment Variables (Optional)
- `ALPHA_VANTAGE_API_KEY`: For enhanced data features
- `NEWS_API_KEY`: For news sentiment analysis 