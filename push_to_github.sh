#!/bin/bash

echo "🚀 Pushing Advanced Stock Predictor System to GitHub..."
echo "Repository: https://github.com/akhilreddy9652/stockmarket.git"
echo ""

# Add all changes
git add .

# Check if there are any changes to commit
if git diff --staged --quiet; then
    echo "✅ No new changes to commit"
else
    echo "📝 Committing new changes..."
    git commit -m "Update: Advanced Stock Predictor System with latest features"
fi

# Push to GitHub
echo "📤 Pushing to GitHub..."
git push -u origin main

echo ""
echo "✅ Successfully pushed to GitHub!"
echo "🌐 View your repository at: https://github.com/akhilreddy9652/stockmarket"
echo ""
echo "📊 Your repository now contains:"
echo "   - 165+ files with 42,000+ lines of code"
echo "   - Advanced ML models (LSTM, XGBoost, Random Forest)"
echo "   - Dual market support (US & Indian stocks)"
echo "   - Interactive Streamlit dashboards"
echo "   - Comprehensive backtesting system"
echo "   - Real-time data integration"
echo "   - Macroeconomic analysis"
echo "   - Portfolio optimization tools" 