#!/usr/bin/env python3
"""
Expanded Indian Stock Universe
============================
Comprehensive Indian stock selection for AI trading systems
"""

# NIFTY 50 - Large Cap Core Holdings
NIFTY_50 = [
    'RELIANCE.NS', 'TCS.NS', 'HDFCBANK.NS', 'INFY.NS', 'ICICIBANK.NS',
    'HINDUNILVR.NS', 'ITC.NS', 'SBIN.NS', 'BHARTIARTL.NS', 'KOTAKBANK.NS',
    'ASIANPAINT.NS', 'MARUTI.NS', 'SUNPHARMA.NS', 'TATAMOTORS.NS', 'WIPRO.NS',
    'ULTRACEMCO.NS', 'TITAN.NS', 'BAJFINANCE.NS', 'NESTLEIND.NS', 'POWERGRID.NS',
    'TECHM.NS', 'BAJAJFINSV.NS', 'NTPC.NS', 'HCLTECH.NS', 'ONGC.NS',
    'JSWSTEEL.NS', 'TATACONSUM.NS', 'ADANIENT.NS', 'COALINDIA.NS', 'HINDALCO.NS',
    'TATASTEEL.NS', 'BRITANNIA.NS', 'GRASIM.NS', 'INDUSINDBK.NS', 'M&M.NS',
    'BAJAJ-AUTO.NS', 'VEDL.NS', 'UPL.NS', 'BPCL.NS', 'SBILIFE.NS',
    'HDFCLIFE.NS', 'DIVISLAB.NS', 'CIPLA.NS', 'EICHERMOT.NS', 'HEROMOTOCO.NS',
    'SHREECEM.NS', 'ADANIPORTS.NS', 'DRREDDY.NS', 'APOLLOHOSP.NS', 'LT.NS'
]

# NIFTY NEXT 50 - Mid-Large Cap Growth
NIFTY_NEXT_50 = [
    'ACC.NS', 'AUBANK.NS', 'BERGEPAINT.NS', 'BIOCON.NS', 'BOSCHLTD.NS',
    'CADILAHC.NS', 'CHOLAFIN.NS', 'COLPAL.NS', 'CONCOR.NS', 'DABUR.NS',
    'DALBHARAT.NS', 'DEEPAKNTR.NS', 'DIVI.NS', 'GAIL.NS', 'GODREJCP.NS',
    'HAVELLS.NS', 'ICICIPRULI.NS', 'IDEA.NS', 'INDIGO.NS', 'IOC.NS',
    'JINDALSTEL.NS', 'LUPIN.NS', 'MARICO.NS', 'MOTHERSUMI.NS', 'MUTHOOTFIN.NS',
    'NMDC.NS', 'OBEROIRLTY.NS', 'OFSS.NS', 'PAGEIND.NS', 'PEL.NS',
    'PETRONET.NS', 'PIDILITIND.NS', 'PNB.NS', 'PGHH.NS', 'RBLBANK.NS',
    'RECLTD.NS', 'SAIL.NS', 'SBICARD.NS', 'SIEMENS.NS', 'SRF.NS',
    'TORNTPHARM.NS', 'TRENT.NS', 'TVSMOTOR.NS', 'VOLTAS.NS', 'WHIRLPOOL.NS',
    'YESBANK.NS', 'ZEEL.NS', 'PFC.NS', 'BANDHANBNK.NS', 'ALKEM.NS'
]

# HIGH-GROWTH SECTORS
FINTECH_BANKING = [
    'PAYTM.NS', 'POLICYBZR.NS', 'EASEMYTRIP.NS', 'CARTRADE.NS',
    'FEDERALBNK.NS', 'IDFCFIRSTB.NS', 'CANBK.NS', 'UNIONBANK.NS',
    'BANKBARODA.NS', 'IOB.NS', 'CENTRALBK.NS', 'ORIENTBANK.NS'
]

IT_TECHNOLOGY = [
    'MINDTREE.NS', 'MPHASIS.NS', 'PERSISTENT.NS', 'COFORGE.NS',
    'LTTS.NS', 'HEXAWARE.NS', 'CYIENT.NS', 'KPITTECH.NS',
    'SONATSOFTW.NS', 'RAMCOSYS.NS', 'INTELLECT.NS', 'TATAELXSI.NS',
    'ZENSARTECH.NS', 'NEWGEN.NS', 'FSL.NS', 'ROLTA.NS'
]

PHARMA_HEALTHCARE = [
    'AUROPHARMA.NS', 'GLENMARK.NS', 'NATCOPHARM.NS', 'AJANTPHARM.NS',
    'LAURUSLABS.NS', 'GRANULES.NS', 'IPCA.NS', 'PFIZER.NS',
    'SANOFI.NS', 'ABBOTINDIA.NS', 'GLAXO.NS', 'FORTIS.NS',
    'MAXHEALTHCARE.NS', 'METROPOLIS.NS', 'THYROCARE.NS', 'ASTER.NS'
]

CONSUMER_FMCG = [
    'EMAMILTD.NS', 'VBL.NS', 'UBL.NS', 'RADICO.NS',
    'GILLETTE.NS', 'HONAUT.NS', 'JYOTHYLAB.NS', 'RELAXO.NS',
    'VIPIND.NS', 'TASTYBITE.NS', 'HATSUN.NS', 'AVANTIFEED.NS'
]

MANUFACTURING_AUTO = [
    'ESCORTS.NS', 'MRF.NS', 'CEAT.NS', 'APOLLOTYRE.NS',
    'BALKRISIND.NS', 'AMARAJABAT.NS', 'EXIDEIND.NS', 'SUNDRMFAST.NS',
    'MOTHERSUMI.NS', 'BOSCHLTD.NS', 'BHEL.NS', 'CROMPTON.NS'
]

ENERGY_UTILITIES = [
    'ADANIGREEN.NS', 'TATAPOWER.NS', 'ADANITRANS.NS', 'ADANIGAS.NS',
    'HPCL.NS', 'INDIANB.NS', 'OIL.NS', 'GSPL.NS',
    'TORNTPOWER.NS', 'JSW.NS', 'SJVN.NS', 'NHPC.NS'
]

METALS_MINING = [
    'NATIONALUM.NS', 'WELCORP.NS', 'HINDZINC.NS', 'MOIL.NS',
    'RATNAMANI.NS', 'AMETEK.NS', 'GRAPHITE.NS', 'SANDUMA.NS'
]

REAL_ESTATE = [
    'DLF.NS', 'GODREJPROP.NS', 'PRESTIGE.NS', 'BRIGADE.NS',
    'OBEROIRLTY.NS', 'PHOENIXLTD.NS', 'SOBHA.NS', 'LODHA.NS',
    'MACROTECH.NS', 'MAHLIFE.NS', 'KOLTEPATIL.NS', 'SUNTECK.NS'
]

TELECOM_MEDIA = [
    'ZEEL.NS', 'PVR.NS', 'INOXLEISURE.NS', 'SUNTV.NS',
    'TV18BRDCST.NS', 'NETWORK18.NS', 'SAREGAMA.NS', 'EROS.NS'
]

# ETFs AND INDEX FUNDS
INDIAN_ETFS = [
    'NIFTYBEES.NS', 'JUNIORBEES.NS', 'BANKBEES.NS', 'ITBEES.NS',
    'PHARMBEES.NS', 'PSUBNKBEES.NS', 'LIQUIDBEES.NS', 'GOLDBEES.NS'
]

# EMERGING SECTORS
NEW_AGE_TECH = [
    'ZOMATO.NS', 'NYKAA.NS', 'DELHIVERY.NS', 'MAPMYINDIA.NS',
    'RATEGAIN.NS', 'BIKAJI.NS', 'DEVYANI.NS', 'SAPPHIRE.NS'
]

RENEWABLE_GREEN = [
    'SUZLON.NS', 'WEBSOL.NS', 'INOXWIND.NS', 'ORIENTGREEN.NS',
    'RPOWER.NS', 'JSWENERGY.NS', 'GREENPANEL.NS', 'CLEANTECH.NS'
]

# COMPREHENSIVE STOCK UNIVERSE
def get_comprehensive_indian_universe(category='ALL', size=50):
    """
    Get comprehensive Indian stock universe based on category and size
    
    Args:
        category: 'ALL', 'LARGE_CAP', 'MID_CAP', 'GROWTH', 'VALUE', 'SECTOR_SPECIFIC'
        size: Number of stocks to return
    """
    
    if category == 'ALL':
        universe = (NIFTY_50 + NIFTY_NEXT_50 + FINTECH_BANKING[:5] + 
                   IT_TECHNOLOGY[:10] + PHARMA_HEALTHCARE[:8] + 
                   CONSUMER_FMCG[:5] + MANUFACTURING_AUTO[:8] + 
                   ENERGY_UTILITIES[:6] + NEW_AGE_TECH[:4] + INDIAN_ETFS[:4])
    
    elif category == 'LARGE_CAP':
        universe = NIFTY_50 + NIFTY_NEXT_50[:25]
        
    elif category == 'MID_CAP':
        universe = (NIFTY_NEXT_50 + FINTECH_BANKING + IT_TECHNOLOGY[:8] + 
                   PHARMA_HEALTHCARE[:6] + REAL_ESTATE[:4])
        
    elif category == 'GROWTH':
        universe = (IT_TECHNOLOGY + NEW_AGE_TECH + PHARMA_HEALTHCARE[:10] + 
                   CONSUMER_FMCG[:6] + RENEWABLE_GREEN[:4])
        
    elif category == 'VALUE':
        universe = (ENERGY_UTILITIES + METALS_MINING + MANUFACTURING_AUTO + 
                   FINTECH_BANKING[:8] + REAL_ESTATE[:6])
        
    elif category == 'SECTOR_DIVERSE':
        # Balanced representation across sectors
        universe = (NIFTY_50[:20] + IT_TECHNOLOGY[:6] + PHARMA_HEALTHCARE[:6] + 
                   FINTECH_BANKING[:6] + MANUFACTURING_AUTO[:6] + 
                   ENERGY_UTILITIES[:4] + NEW_AGE_TECH[:2])
    
    else:
        universe = NIFTY_50
    
    # Remove duplicates and limit size
    universe = list(dict.fromkeys(universe))
    return universe[:size]

# SECTOR MAPPING
SECTOR_MAPPING = {
    'Banking & Financial': FINTECH_BANKING + ['HDFCBANK.NS', 'ICICIBANK.NS', 'SBIN.NS', 'KOTAKBANK.NS'],
    'Information Technology': IT_TECHNOLOGY + ['TCS.NS', 'INFY.NS', 'WIPRO.NS', 'HCLTECH.NS'],
    'Pharmaceuticals': PHARMA_HEALTHCARE + ['SUNPHARMA.NS', 'DRREDDY.NS', 'CIPLA.NS', 'DIVISLAB.NS'],
    'FMCG & Consumer': CONSUMER_FMCG + ['HINDUNILVR.NS', 'ITC.NS', 'NESTLEIND.NS', 'BRITANNIA.NS'],
    'Automotive': MANUFACTURING_AUTO + ['TATAMOTORS.NS', 'MARUTI.NS', 'EICHERMOT.NS', 'HEROMOTOCO.NS'],
    'Energy & Utilities': ENERGY_UTILITIES + ['RELIANCE.NS', 'ONGC.NS', 'NTPC.NS', 'POWERGRID.NS'],
    'Metals & Mining': METALS_MINING + ['TATASTEEL.NS', 'JSWSTEEL.NS', 'HINDALCO.NS', 'VEDL.NS'],
    'Real Estate': REAL_ESTATE,
    'New Age Tech': NEW_AGE_TECH,
    'Renewable Energy': RENEWABLE_GREEN
}

# RISK CATEGORIES
RISK_CATEGORIES = {
    'LOW_RISK': NIFTY_50[:20] + INDIAN_ETFS,
    'MEDIUM_RISK': NIFTY_50[20:] + NIFTY_NEXT_50[:30],
    'HIGH_RISK': NEW_AGE_TECH + RENEWABLE_GREEN + IT_TECHNOLOGY[:10],
    'ULTRA_HIGH_RISK': ['PAYTM.NS', 'ZOMATO.NS', 'NYKAA.NS'] + RENEWABLE_GREEN[:5]
}

# MARKET CAP CATEGORIES  
MARKET_CAP_CATEGORIES = {
    'MEGA_CAP': NIFTY_50[:10],  # Top 10 largest
    'LARGE_CAP': NIFTY_50[10:] + NIFTY_NEXT_50[:20],
    'MID_CAP': NIFTY_NEXT_50[20:] + IT_TECHNOLOGY[:15] + PHARMA_HEALTHCARE[:10],
    'SMALL_CAP': NEW_AGE_TECH + RENEWABLE_GREEN + REAL_ESTATE[:8]
}

def get_balanced_portfolio(size=30, risk_level='MEDIUM'):
    """
    Create a balanced portfolio across sectors and risk levels
    """
    if risk_level == 'CONSERVATIVE':
        weights = {'LOW_RISK': 0.6, 'MEDIUM_RISK': 0.3, 'HIGH_RISK': 0.1}
    elif risk_level == 'MODERATE':
        weights = {'LOW_RISK': 0.4, 'MEDIUM_RISK': 0.4, 'HIGH_RISK': 0.2}
    elif risk_level == 'AGGRESSIVE':
        weights = {'LOW_RISK': 0.2, 'MEDIUM_RISK': 0.3, 'HIGH_RISK': 0.5}
    else:
        weights = {'LOW_RISK': 0.4, 'MEDIUM_RISK': 0.4, 'HIGH_RISK': 0.2}
    
    portfolio = []
    for risk_cat, weight in weights.items():
        stocks_needed = int(size * weight)
        portfolio.extend(RISK_CATEGORIES[risk_cat][:stocks_needed])
    
    return list(dict.fromkeys(portfolio))[:size]

if __name__ == "__main__":
    print("🇮🇳 EXPANDED INDIAN STOCK UNIVERSE")
    print("=" * 50)
    
    # Demo different universes
    for category in ['ALL', 'LARGE_CAP', 'GROWTH', 'SECTOR_DIVERSE']:
        universe = get_comprehensive_indian_universe(category, 20)
        print(f"\n📊 {category} Universe ({len(universe)} stocks):")
        for i, stock in enumerate(universe[:10], 1):
            print(f"  {i:2d}. {stock}")
        if len(universe) > 10:
            print(f"     ... and {len(universe)-10} more")
    
    print(f"\n🎯 Balanced Portfolio (30 stocks):")
    balanced = get_balanced_portfolio(30, 'MODERATE')
    for i, stock in enumerate(balanced, 1):
        print(f"  {i:2d}. {stock}")
    
    print(f"\n📈 Total Available Stocks: {len(set(NIFTY_50 + NIFTY_NEXT_50 + FINTECH_BANKING + IT_TECHNOLOGY + PHARMA_HEALTHCARE + CONSUMER_FMCG + MANUFACTURING_AUTO + ENERGY_UTILITIES + NEW_AGE_TECH + INDIAN_ETFS))}") 