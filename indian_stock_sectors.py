#!/usr/bin/env python3
"""
Indian Stock Sectors and Categories
===================================
Comprehensive categorization of Indian stocks by sectors, market cap, and indices
"""

# Nifty 50 Index Stocks (Large Cap Blue Chips)
NIFTY_50 = {
    "^NSEI": {"name": "Nifty 50 Index", "sector": "Index", "market_cap": "Index"},
    "RELIANCE.NS": {"name": "Reliance Industries", "sector": "Oil & Gas", "market_cap": "Large Cap"},
    "TCS.NS": {"name": "Tata Consultancy Services", "sector": "IT Services", "market_cap": "Large Cap"},
    "HDFCBANK.NS": {"name": "HDFC Bank", "sector": "Private Bank", "market_cap": "Large Cap"},
    "INFY.NS": {"name": "Infosys", "sector": "IT Services", "market_cap": "Large Cap"},
    "ICICIBANK.NS": {"name": "ICICI Bank", "sector": "Private Bank", "market_cap": "Large Cap"},
    "HINDUNILVR.NS": {"name": "Hindustan Unilever", "sector": "FMCG", "market_cap": "Large Cap"},
    "ITC.NS": {"name": "ITC Limited", "sector": "FMCG", "market_cap": "Large Cap"},
    "SBIN.NS": {"name": "State Bank of India", "sector": "PSU Bank", "market_cap": "Large Cap"},
    "BHARTIARTL.NS": {"name": "Bharti Airtel", "sector": "Telecom", "market_cap": "Large Cap"},
    "AXISBANK.NS": {"name": "Axis Bank", "sector": "Private Bank", "market_cap": "Large Cap"},
    "KOTAKBANK.NS": {"name": "Kotak Mahindra Bank", "sector": "Private Bank", "market_cap": "Large Cap"},
    "ASIANPAINT.NS": {"name": "Asian Paints", "sector": "Paints", "market_cap": "Large Cap"},
    "MARUTI.NS": {"name": "Maruti Suzuki", "sector": "Auto", "market_cap": "Large Cap"},
    "SUNPHARMA.NS": {"name": "Sun Pharmaceutical", "sector": "Pharma", "market_cap": "Large Cap"},
    "TATAMOTORS.NS": {"name": "Tata Motors", "sector": "Auto", "market_cap": "Large Cap"},
    "WIPRO.NS": {"name": "Wipro", "sector": "IT Services", "market_cap": "Large Cap"},
    "ULTRACEMCO.NS": {"name": "UltraTech Cement", "sector": "Cement", "market_cap": "Large Cap"},
    "TITAN.NS": {"name": "Titan Company", "sector": "Jewellery", "market_cap": "Large Cap"},
    "BAJFINANCE.NS": {"name": "Bajaj Finance", "sector": "NBFC", "market_cap": "Large Cap"},
    "NESTLEIND.NS": {"name": "Nestle India", "sector": "FMCG", "market_cap": "Large Cap"},
    "POWERGRID.NS": {"name": "Power Grid Corp", "sector": "Power", "market_cap": "Large Cap"},
    "TECHM.NS": {"name": "Tech Mahindra", "sector": "IT Services", "market_cap": "Large Cap"},
    "BAJAJFINSV.NS": {"name": "Bajaj Finserv", "sector": "Financial Services", "market_cap": "Large Cap"},
    "NTPC.NS": {"name": "NTPC", "sector": "Power", "market_cap": "Large Cap"},
    "HCLTECH.NS": {"name": "HCL Technologies", "sector": "IT Services", "market_cap": "Large Cap"}
}

# Banking & Financial Services
BANKING_FINANCIAL = {
    # Private Banks
    "HDFCBANK.NS": {"name": "HDFC Bank", "sector": "Private Bank", "type": "Private Bank"},
    "ICICIBANK.NS": {"name": "ICICI Bank", "sector": "Private Bank", "type": "Private Bank"},
    "AXISBANK.NS": {"name": "Axis Bank", "sector": "Private Bank", "type": "Private Bank"},
    "KOTAKBANK.NS": {"name": "Kotak Mahindra Bank", "sector": "Private Bank", "type": "Private Bank"},
    "INDUSINDBK.NS": {"name": "IndusInd Bank", "sector": "Private Bank", "type": "Private Bank"},
    "FEDERALBNK.NS": {"name": "Federal Bank", "sector": "Private Bank", "type": "Private Bank"},
    "IDFCFIRSTB.NS": {"name": "IDFC First Bank", "sector": "Private Bank", "type": "Private Bank"},
    "BANDHANBNK.NS": {"name": "Bandhan Bank", "sector": "Private Bank", "type": "Private Bank"},
    "RBLBANK.NS": {"name": "RBL Bank", "sector": "Private Bank", "type": "Private Bank"},
    "YESBANK.NS": {"name": "Yes Bank", "sector": "Private Bank", "type": "Private Bank"},
    
    # PSU Banks
    "SBIN.NS": {"name": "State Bank of India", "sector": "PSU Bank", "type": "PSU Bank"},
    "PNB.NS": {"name": "Punjab National Bank", "sector": "PSU Bank", "type": "PSU Bank"},
    "CANBK.NS": {"name": "Canara Bank", "sector": "PSU Bank", "type": "PSU Bank"},
    "UNIONBANK.NS": {"name": "Union Bank of India", "sector": "PSU Bank", "type": "PSU Bank"},
    "BANKBARODA.NS": {"name": "Bank of Baroda", "sector": "PSU Bank", "type": "PSU Bank"},
    "IOB.NS": {"name": "Indian Overseas Bank", "sector": "PSU Bank", "type": "PSU Bank"},
    
    # NBFCs & Financial Services
    "BAJFINANCE.NS": {"name": "Bajaj Finance", "sector": "NBFC", "type": "NBFC"},
    "BAJAJFINSV.NS": {"name": "Bajaj Finserv", "sector": "Financial Services", "type": "Financial Services"},
    "CHOLAFIN.NS": {"name": "Cholamandalam Finance", "sector": "NBFC", "type": "NBFC"},
    "MUTHOOTFIN.NS": {"name": "Muthoot Finance", "sector": "NBFC", "type": "NBFC"},
    "HDFCAMC.NS": {"name": "HDFC AMC", "sector": "Asset Management", "type": "Asset Management"},
    
    # Insurance
    "SBILIFE.NS": {"name": "SBI Life Insurance", "sector": "Insurance", "type": "Insurance"},
    "HDFCLIFE.NS": {"name": "HDFC Life Insurance", "sector": "Insurance", "type": "Insurance"},
    "ICICIPRULI.NS": {"name": "ICICI Prudential Life", "sector": "Insurance", "type": "Insurance"}
}

# IT & Technology
IT_TECHNOLOGY = {
    "TCS.NS": {"name": "Tata Consultancy Services", "sector": "IT Services", "market_cap": "Large Cap"},
    "INFY.NS": {"name": "Infosys", "sector": "IT Services", "market_cap": "Large Cap"},
    "WIPRO.NS": {"name": "Wipro", "sector": "IT Services", "market_cap": "Large Cap"},
    "HCLTECH.NS": {"name": "HCL Technologies", "sector": "IT Services", "market_cap": "Large Cap"},
    "TECHM.NS": {"name": "Tech Mahindra", "sector": "IT Services", "market_cap": "Large Cap"},
    "MINDTREE.NS": {"name": "Mindtree", "sector": "IT Services", "market_cap": "Mid Cap"},
    "MPHASIS.NS": {"name": "Mphasis", "sector": "IT Services", "market_cap": "Mid Cap"},
    "PERSISTENT.NS": {"name": "Persistent Systems", "sector": "IT Services", "market_cap": "Mid Cap"},
    "COFORGE.NS": {"name": "Coforge", "sector": "IT Services", "market_cap": "Mid Cap"},
    "L&TI.NS": {"name": "Larsen & Toubro Infotech", "sector": "IT Services", "market_cap": "Mid Cap"}
}

# Auto & Manufacturing
AUTO_MANUFACTURING = {
    "TATAMOTORS.NS": {"name": "Tata Motors", "sector": "Auto", "type": "Passenger & Commercial Vehicles"},
    "MARUTI.NS": {"name": "Maruti Suzuki", "sector": "Auto", "type": "Passenger Cars"},
    "EICHERMOT.NS": {"name": "Eicher Motors", "sector": "Auto", "type": "Two Wheelers"},
    "HEROMOTOCO.NS": {"name": "Hero MotoCorp", "sector": "Auto", "type": "Two Wheelers"},
    "BAJAJ-AUTO.NS": {"name": "Bajaj Auto", "sector": "Auto", "type": "Two Wheelers"},
    "M&M.NS": {"name": "Mahindra & Mahindra", "sector": "Auto", "type": "SUVs & Tractors"},
    "ASHOKLEY.NS": {"name": "Ashok Leyland", "sector": "Auto", "type": "Commercial Vehicles"},
    "TVSMOTOR.NS": {"name": "TVS Motor", "sector": "Auto", "type": "Two Wheelers"},
    "ESCORTS.NS": {"name": "Escorts", "sector": "Auto", "type": "Tractors"},
    "MRF.NS": {"name": "MRF", "sector": "Auto", "type": "Tyres"}
}

# Pharma & Healthcare
PHARMA_HEALTHCARE = {
    "SUNPHARMA.NS": {"name": "Sun Pharmaceutical", "sector": "Pharma", "type": "Generics"},
    "DRREDDY.NS": {"name": "Dr. Reddy's Labs", "sector": "Pharma", "type": "Generics"},
    "CIPLA.NS": {"name": "Cipla", "sector": "Pharma", "type": "Generics"},
    "DIVISLAB.NS": {"name": "Divi's Laboratories", "sector": "Pharma", "type": "APIs"},
    "BIOCON.NS": {"name": "Biocon", "sector": "Pharma", "type": "Biologics"},
    "APOLLOHOSP.NS": {"name": "Apollo Hospitals", "sector": "Healthcare", "type": "Hospitals"},
    "FORTIS.NS": {"name": "Fortis Healthcare", "sector": "Healthcare", "type": "Hospitals"},
    "ALKEM.NS": {"name": "Alkem Laboratories", "sector": "Pharma", "type": "Generics"},
    "TORNTPHARM.NS": {"name": "Torrent Pharma", "sector": "Pharma", "type": "Generics"},
    "LUPIN.NS": {"name": "Lupin", "sector": "Pharma", "type": "Generics"}
}

# FMCG & Consumer Goods
FMCG_CONSUMER = {
    "HINDUNILVR.NS": {"name": "Hindustan Unilever", "sector": "FMCG", "type": "Personal Care"},
    "ITC.NS": {"name": "ITC Limited", "sector": "FMCG", "type": "Cigarettes & FMCG"},
    "NESTLEIND.NS": {"name": "Nestle India", "sector": "FMCG", "type": "Food & Beverages"},
    "BRITANNIA.NS": {"name": "Britannia Industries", "sector": "FMCG", "type": "Food"},
    "MARICO.NS": {"name": "Marico", "sector": "FMCG", "type": "Personal Care"},
    "DABUR.NS": {"name": "Dabur India", "sector": "FMCG", "type": "Personal Care"},
    "COLPAL.NS": {"name": "Colgate Palmolive", "sector": "FMCG", "type": "Personal Care"},
    "GODREJCP.NS": {"name": "Godrej Consumer", "sector": "FMCG", "type": "Personal Care"},
    "EMAMILTD.NS": {"name": "Emami", "sector": "FMCG", "type": "Personal Care"},
    "VBL.NS": {"name": "Varun Beverages", "sector": "FMCG", "type": "Beverages"}
}

# Energy & Oil
ENERGY_OIL = {
    "RELIANCE.NS": {"name": "Reliance Industries", "sector": "Oil & Gas", "type": "Integrated Oil"},
    "ONGC.NS": {"name": "Oil & Natural Gas Corp", "sector": "Oil & Gas", "type": "Exploration"},
    "COALINDIA.NS": {"name": "Coal India", "sector": "Mining", "type": "Coal Mining"},
    "NTPC.NS": {"name": "NTPC", "sector": "Power", "type": "Power Generation"},
    "POWERGRID.NS": {"name": "Power Grid Corp", "sector": "Power", "type": "Power Transmission"},
    "BPCL.NS": {"name": "Bharat Petroleum", "sector": "Oil & Gas", "type": "Oil Refining"},
    "IOC.NS": {"name": "Indian Oil Corp", "sector": "Oil & Gas", "type": "Oil Refining"},
    "HPCL.NS": {"name": "Hindustan Petroleum", "sector": "Oil & Gas", "type": "Oil Refining"},
    "ADANIGREEN.NS": {"name": "Adani Green Energy", "sector": "Renewable Energy", "type": "Solar & Wind"},
    "TATAPOWER.NS": {"name": "Tata Power", "sector": "Power", "type": "Power Generation"}
}

# Indian ETFs
INDIAN_ETFS = {
    "NIFTYBEES.NS": {"name": "Nippon India ETF Nifty BeES", "sector": "ETF", "tracks": "Nifty 50"},
    "BANKBEES.NS": {"name": "Nippon India ETF Bank BeES", "sector": "ETF", "tracks": "Nifty Bank"},
    "ITBEES.NS": {"name": "Nippon India ETF IT BeES", "sector": "ETF", "tracks": "Nifty IT"},
    "GOLDSHARE.NS": {"name": "Goldman Sachs Gold ETF", "sector": "ETF", "tracks": "Gold"},
    "LIQUIDBEES.NS": {"name": "Nippon India ETF Liquid BeES", "sector": "ETF", "tracks": "Liquid Fund"},
    "CPSE.NS": {"name": "CPSE ETF", "sector": "ETF", "tracks": "CPSE Index"},
    "JUNIORBEES.NS": {"name": "Nippon India ETF Junior BeES", "sector": "ETF", "tracks": "Nifty Next 50"},
    "PSUBNKBEES.NS": {"name": "Nippon India ETF PSU Bank BeES", "sector": "ETF", "tracks": "Nifty PSU Bank"}
}

# Sectoral categorization
INDIAN_SECTORS = {
    "Banking & Financial": BANKING_FINANCIAL,
    "IT & Technology": IT_TECHNOLOGY,
    "Auto & Manufacturing": AUTO_MANUFACTURING,
    "Pharma & Healthcare": PHARMA_HEALTHCARE,
    "FMCG & Consumer": FMCG_CONSUMER,
    "Energy & Oil": ENERGY_OIL,
    "Nifty 50 (Blue Chips)": NIFTY_50,
    "Indian ETFs": INDIAN_ETFS
}

# Popular Indian stocks by category
POPULAR_INDIAN_STOCKS = {
    "Top 20 Most Traded": [
        "RELIANCE.NS", "TCS.NS", "HDFCBANK.NS", "INFY.NS", "ICICIBANK.NS",
        "HINDUNILVR.NS", "ITC.NS", "SBIN.NS", "BHARTIARTL.NS", "AXISBANK.NS",
        "KOTAKBANK.NS", "ASIANPAINT.NS", "MARUTI.NS", "SUNPHARMA.NS", "TATAMOTORS.NS",
        "WIPRO.NS", "ULTRACEMCO.NS", "TITAN.NS", "BAJFINANCE.NS", "NESTLEIND.NS"
    ],
    "High Growth Stocks": [
        "BAJFINANCE.NS", "TITAN.NS", "ASIANPAINT.NS", "HDFCBANK.NS", "MARICO.NS",
        "PIDILITIND.NS", "DMART.NS", "ADANIPORTS.NS", "ADANIGREEN.NS", "ZOMATO.NS"
    ],
    "Dividend Aristocrats": [
        "ITC.NS", "COALINDIA.NS", "ONGC.NS", "VEDL.NS", "NMDC.NS",
        "HINDUNILVR.NS", "SBIN.NS", "POWERGRID.NS", "NTPC.NS", "IOC.NS"
    ],
    "Trending Stocks": [
        "NYKAA.NS", "ZOMATO.NS", "PAYTM.NS", "POLICYBZR.NS", "CARTRADE.NS",
        "EASEMYTRIP.NS", "DELHIVERY.NS", "MAPMYINDIA.NS", "WINDLAS.NS", "LATENTVIEW.NS"
    ]
}

def get_stock_info(symbol):
    """Get detailed information about an Indian stock"""
    for sector_name, stocks in INDIAN_SECTORS.items():
        if symbol in stocks:
            info = stocks[symbol].copy()
            info['sector_category'] = sector_name
            return info
    return {"name": "Unknown", "sector": "Unknown", "sector_category": "Unknown"}

def get_stocks_by_sector(sector_name):
    """Get all stocks in a specific sector"""
    return INDIAN_SECTORS.get(sector_name, {})

def search_indian_stocks(query):
    """Search for Indian stocks by name or symbol"""
    results = []
    query_upper = query.upper()
    
    for sector_name, stocks in INDIAN_SECTORS.items():
        for symbol, info in stocks.items():
            if (query_upper in symbol.upper() or 
                query_upper in info['name'].upper() or
                query_upper in info.get('sector', '').upper()):
                results.append({
                    'symbol': symbol,
                    'name': info['name'],
                    'sector': info.get('sector', 'Unknown'),
                    'sector_category': sector_name
                })
    
    return results

def get_sector_performance_stocks():
    """Get representative stocks for sector performance analysis"""
    return {
        "Banking": "HDFCBANK.NS",
        "IT": "TCS.NS", 
        "Auto": "MARUTI.NS",
        "Pharma": "SUNPHARMA.NS",
        "FMCG": "HINDUNILVR.NS",
        "Energy": "RELIANCE.NS",
        "Metals": "TATASTEEL.NS",
        "Telecom": "BHARTIARTL.NS"
    } 