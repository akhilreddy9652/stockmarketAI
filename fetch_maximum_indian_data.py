"""
Fetch Maximum Historical Data for Major Indian Stocks
====================================================
This script fetches maximum available historical data from 2000 for all major Indian stocks.
Covers Nifty 50, Nifty Next 50, and other important Indian companies across all sectors.
"""

import yfinance as yf
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import os
import time
import warnings
from typing import Dict, List, Tuple
import concurrent.futures
from threading import Lock

warnings.filterwarnings('ignore')

class MaximalIndianDataFetcher:
    """Fetch maximum historical data for Indian stocks from 2000."""
    
    def __init__(self, start_date='2000-01-01'):
        self.start_date = start_date
        self.end_date = datetime.now().strftime('%Y-%m-%d')
        self.data_dir = 'data/max_historical_data'
        self.results_lock = Lock()
        self.fetch_results = {}
        
        # Create directory structure
        os.makedirs(self.data_dir, exist_ok=True)
        os.makedirs(f'{self.data_dir}/raw_data', exist_ok=True)
        os.makedirs(f'{self.data_dir}/processed_data', exist_ok=True)
        
        # Define comprehensive list of major Indian stocks
        self.major_indian_stocks = self._get_major_indian_stocks()
        
    def _get_major_indian_stocks(self) -> Dict[str, str]:
        """Get comprehensive list of major Indian stocks."""
        
        stocks = {
            # NIFTY 50 - Complete List (50 stocks)
            'RELIANCE.NS': 'Reliance Industries - Oil, Gas & Energy',
            'TCS.NS': 'Tata Consultancy Services - IT Services',
            'HDFCBANK.NS': 'HDFC Bank - Private Banking',
            'INFY.NS': 'Infosys - IT Services & Consulting',
            'ICICIBANK.NS': 'ICICI Bank - Private Banking',
            'HINDUNILVR.NS': 'Hindustan Unilever - FMCG',
            'ITC.NS': 'ITC Limited - FMCG & Cigarettes',
            'SBIN.NS': 'State Bank of India - Public Banking',
            'BHARTIARTL.NS': 'Bharti Airtel - Telecommunications',
            'AXISBANK.NS': 'Axis Bank - Private Banking',
            'KOTAKBANK.NS': 'Kotak Mahindra Bank - Private Banking',
            'ASIANPAINT.NS': 'Asian Paints - Paints & Coatings',
            'MARUTI.NS': 'Maruti Suzuki - Automotive',
            'SUNPHARMA.NS': 'Sun Pharmaceutical - Pharmaceuticals',
            'TATAMOTORS.NS': 'Tata Motors - Automotive',
            'WIPRO.NS': 'Wipro - IT Services',
            'ULTRACEMCO.NS': 'UltraTech Cement - Cement',
            'TITAN.NS': 'Titan Company - Jewelry & Watches',
            'BAJFINANCE.NS': 'Bajaj Finance - NBFC',
            'NESTLEIND.NS': 'Nestle India - FMCG',
            'POWERGRID.NS': 'Power Grid Corporation - Power',
            'TECHM.NS': 'Tech Mahindra - IT Services',
            'BAJAJFINSV.NS': 'Bajaj Finserv - Financial Services',
            'NTPC.NS': 'NTPC Limited - Power Generation',
            'HCLTECH.NS': 'HCL Technologies - IT Services',
            'ONGC.NS': 'Oil & Natural Gas Corporation - Oil & Gas',
            'JSWSTEEL.NS': 'JSW Steel - Steel',
            'TATACONSUM.NS': 'Tata Consumer Products - FMCG',
            'ADANIENT.NS': 'Adani Enterprises - Conglomerate',
            'COALINDIA.NS': 'Coal India - Mining',
            'HINDALCO.NS': 'Hindalco Industries - Metals',
            'TATASTEEL.NS': 'Tata Steel - Steel',
            'BRITANNIA.NS': 'Britannia Industries - FMCG',
            'GRASIM.NS': 'Grasim Industries - Diversified',
            'INDUSINDBK.NS': 'IndusInd Bank - Private Banking',
            'M&M.NS': 'Mahindra & Mahindra - Automotive',
            'BAJAJ-AUTO.NS': 'Bajaj Auto - Two Wheeler',
            'VEDL.NS': 'Vedanta Limited - Metals & Mining',
            'UPL.NS': 'UPL Limited - Agrochemicals',
            'BPCL.NS': 'Bharat Petroleum - Oil Refining',
            'SBILIFE.NS': 'SBI Life Insurance - Insurance',
            'HDFCLIFE.NS': 'HDFC Life Insurance - Insurance',
            'DIVISLAB.NS': 'Divi\'s Laboratories - Pharmaceuticals',
            'CIPLA.NS': 'Cipla - Pharmaceuticals',
            'EICHERMOT.NS': 'Eicher Motors - Automotive (Royal Enfield)',
            'HEROMOTOCO.NS': 'Hero MotoCorp - Two Wheeler',
            'SHREECEM.NS': 'Shree Cement - Cement',
            'ADANIPORTS.NS': 'Adani Ports - Port Operations',
            'DRREDDY.NS': 'Dr. Reddy\'s Laboratories - Pharmaceuticals',
            'APOLLOHOSP.NS': 'Apollo Hospitals - Healthcare',
            
            # NIFTY NEXT 50 - Key Stocks (20 additional)
            'LT.NS': 'Larsen & Toubro - Engineering & Construction',
            'GODREJCP.NS': 'Godrej Consumer Products - FMCG',
            'DABUR.NS': 'Dabur India - FMCG & Healthcare',
            'MARICO.NS': 'Marico Limited - FMCG',
            'PIDILITIND.NS': 'PID Industries - Adhesives',
            'SIEMENS.NS': 'Siemens Limited - Engineering',
            'BOSCHLTD.NS': 'Bosch Limited - Auto Components',
            'HAVELLS.NS': 'Havells India - Electrical Equipment',
            'BIOCON.NS': 'Biocon Limited - Biotechnology',
            'LUPIN.NS': 'Lupin Limited - Pharmaceuticals',
            'AUROPHARMA.NS': 'Aurobindo Pharma - Pharmaceuticals',
            'TORNTPHARM.NS': 'Torrent Pharmaceuticals - Pharmaceuticals',
            'CADILAHC.NS': 'Cadila Healthcare - Pharmaceuticals',
            'BANDHANBNK.NS': 'Bandhan Bank - Banking',
            'IDFCFIRSTB.NS': 'IDFC First Bank - Banking',
            'FEDERALBNK.NS': 'Federal Bank - Banking',
            'PNB.NS': 'Punjab National Bank - Public Banking',
            'CANBK.NS': 'Canara Bank - Public Banking',
            'UNIONBANK.NS': 'Union Bank of India - Public Banking',
            'BANKBARODA.NS': 'Bank of Baroda - Public Banking',
            
            # PSU & Government Companies (15 stocks)
            'IOC.NS': 'Indian Oil Corporation - Oil Refining',
            'HPCL.NS': 'Hindustan Petroleum - Oil Refining',
            'GAIL.NS': 'GAIL India - Natural Gas',
            'PETRONET.NS': 'Petronet LNG - LNG Terminal',
            'OIL.NS': 'Oil India Limited - Oil Exploration',
            'SAIL.NS': 'Steel Authority of India - Steel',
            'NMDC.NS': 'NMDC Limited - Iron Ore Mining',
            'HINDZINC.NS': 'Hindustan Zinc - Zinc Mining',
            'RECLTD.NS': 'REC Limited - Power Financing',
            'PFC.NS': 'Power Finance Corporation - Power Financing',
            'IRCTC.NS': 'Indian Railway Catering - Railway Services',
            'HAL.NS': 'Hindustan Aeronautics - Aerospace',
            'BEL.NS': 'Bharat Electronics - Defense Electronics',
            'BHEL.NS': 'Bharat Heavy Electricals - Heavy Engineering',
            'CONCOR.NS': 'Container Corporation - Logistics',
            
            # Auto & Auto Components (10 stocks)
            'ASHOKLEY.NS': 'Ashok Leyland - Commercial Vehicles',
            'TVSMOTOR.NS': 'TVS Motor Company - Two Wheeler',
            'ESCORTS.NS': 'Escorts Limited - Tractors',
            'MRF.NS': 'MRF Limited - Tyres',
            'APOLLOTYRE.NS': 'Apollo Tyres - Tyres',
            'BALKRISIND.NS': 'Balakrishna Industries - Tyres',
            'CEAT.NS': 'CEAT Limited - Tyres',
            'EXIDEIND.NS': 'Exide Industries - Batteries',
            'AMARA.NS': 'Amara Raja Batteries - Batteries',
            'MOTHERSUMI.NS': 'Motherson Sumi - Auto Components',
            
            # IT & Technology (10 stocks)
            'MINDTREE.NS': 'Mindtree Limited - IT Services',
            'LTI.NS': 'L&T Infotech - IT Services',
            'MPHASIS.NS': 'Mphasis Limited - IT Services',
            'PERSISTENT.NS': 'Persistent Systems - IT Services',
            'COFORGE.NS': 'Coforge Limited - IT Services',
            'LTTS.NS': 'L&T Technology Services - Engineering R&D',
            'HEXAWARE.NS': 'Hexaware Technologies - IT Services',
            'CYIENT.NS': 'Cyient Limited - Engineering Services',
            'TATAELXSI.NS': 'Tata Elxsi - Product Engineering',
            'ZENSARTECH.NS': 'Zensar Technologies - IT Services',
            
            # Real Estate & Construction (8 stocks)
            'DLF.NS': 'DLF Limited - Real Estate',
            'GODREJPROP.NS': 'Godrej Properties - Real Estate',
            'PRESTIGE.NS': 'Prestige Estates - Real Estate', 
            'BRIGADE.NS': 'Brigade Enterprises - Real Estate',
            'OBEROIRLTY.NS': 'Oberoi Realty - Real Estate',
            'PHOENIXLTD.NS': 'Phoenix Mills - Real Estate & Retail',
            'SOBHA.NS': 'Sobha Limited - Real Estate',
            'LODHA.NS': 'Macrotech Developers (Lodha) - Real Estate',
            
            # Cement & Infrastructure (8 stocks)
            'ACC.NS': 'ACC Limited - Cement',
            'AMBUJACEM.NS': 'Ambuja Cements - Cement',
            'RAMCOCEM.NS': 'Ramco Cements - Cement',
            'JKCEMENT.NS': 'JK Cement - Cement',
            'ORIENTCEM.NS': 'Orient Cement - Cement',
            'HEIDELBERG.NS': 'Heidelberg Cement - Cement',
            'JKLAKSHMI.NS': 'JK Lakshmi Cement - Cement',
            'BIRLACORPN.NS': 'Birla Corporation - Cement',
            
            # Chemicals & Fertilizers (10 stocks)
            'COROMANDEL.NS': 'Coromandel International - Fertilizers',
            'CHAMBLFERT.NS': 'Chambal Fertilizers - Fertilizers',
            'GSFC.NS': 'Gujarat State Fertilizers - Fertilizers',
            'RCF.NS': 'Rashtriya Chemicals - Fertilizers',
            'DEEPAKNTR.NS': 'Deepak Nitrite - Chemicals',
            'ALKYLAMINE.NS': 'Alkyl Amines Chemicals - Chemicals',
            'BALRAMCHIN.NS': 'Balrampur Chini Mills - Sugar & Chemicals',
            'TATACHEMICALS.NS': 'Tata Chemicals - Chemicals',
            'GNFC.NS': 'Gujarat Narmada Valley Fertilizers - Fertilizers',
            'KANSAINER.NS': 'Kansai Nerolac Paints - Paints',
            
            # Consumer & Retail (8 stocks)
            'COLPAL.NS': 'Colgate-Palmolive - Personal Care',
            'EMAMILTD.NS': 'Emami Limited - Personal Care',
            'VBL.NS': 'Varun Beverages - Beverages (PepsiCo)',
            'UBL.NS': 'United Breweries - Alcoholic Beverages',
            'RADICO.NS': 'Radico Khaitan - Alcoholic Beverages',
            'GILLETTE.NS': 'Gillette India - Personal Care',
            'PGHH.NS': 'Procter & Gamble Hygiene - Personal Care',
            'JUBLFOOD.NS': 'Jubilant FoodWorks - Food Services (Dominos)',
            
            # Textiles & Apparel (6 stocks)
            'RTNPOWER.NS': 'RattanIndia Power - Textiles & Power',
            'WELCORP.NS': 'Welspun Corp - Textiles',
            'TRIDENT.NS': 'Trident Limited - Textiles',
            'RAYMOND.NS': 'Raymond Limited - Textiles',
            'ARVIND.NS': 'Arvind Limited - Textiles',
            'PAGEIND.NS': 'Page Industries - Innerwear (Jockey)',
            
            # Media & Entertainment (6 stocks)
            'SUNTV.NS': 'Sun TV Network - Media',
            'ZEEL.NS': 'Zee Entertainment - Media',
            'SAREGAMA.NS': 'Saregama India - Music',
            'PVR.NS': 'PVR Limited - Cinema Multiplex',
            'INOXLEISURE.NS': 'INOX Leisure - Cinema Multiplex',
            'NETWORKIN.NS': 'Network18 Media - Media',
            
            # New Age & Digital (8 stocks) 
            'ZOMATO.NS': 'Zomato - Food Delivery',
            'NYKAA.NS': 'Nykaa (FSN E-Commerce) - Beauty E-commerce',
            'POLICYBZR.NS': 'PolicyBazaar - Insurance Broking',
            'PAYTM.NS': 'Paytm (One97 Communications) - Fintech',
            'EASEMYTRIP.NS': 'EaseMyTrip - Online Travel',
            'CARTRADE.NS': 'CarTrade Tech - Auto Marketplace',
            'DELHIVERY.NS': 'Delhivery - Logistics',
            'MAPMYINDIA.NS': 'MapmyIndia (CE Info Systems) - Mapping',
            
            # Aviation & Logistics (5 stocks)
            'INDIGO.NS': 'IndiGo (InterGlobe Aviation) - Airlines',
            'SPICEJET.NS': 'SpiceJet - Airlines',
            'BLUEDART.NS': 'Blue Dart Express - Logistics',
            'TCI.NS': 'Transport Corporation of India - Logistics',
            'GICRE.NS': 'General Insurance Corporation - Insurance',
            
            # Specialty & Others (10 stocks)
            'BATINDIA.NS': 'ITC (British American Tobacco) - Tobacco',
            'HONAUT.NS': 'Honeywell Automation - Industrial Automation',
            'THERMAX.NS': 'Thermax Limited - Energy & Environment',
            'CRISIL.NS': 'CRISIL Limited - Rating & Research',
            'MINDSPACE.NS': 'Mindspace Business Parks REIT - REITs',
            'BROOKFIELD.NS': 'Brookfield India Real Estate Trust - REITs',
            'EMBASSYOFC.NS': 'Embassy Office Parks REIT - REITs',
            'INDIGOPNTS.NS': 'Indigo Paints - Paints',
            'CLEAN.NS': 'Clean Science and Technology - Specialty Chemicals',
            'LICI.NS': 'Life Insurance Corporation - Insurance'
        }
        
        return stocks
    
    def fetch_single_stock(self, symbol: str, description: str) -> Tuple[str, bool, str, pd.DataFrame]:
        """Fetch maximum historical data for a single stock."""
        
        try:
            print(f"📊 Fetching {symbol} - {description}")
            
            # Use 'max' period to get maximum available data
            stock = yf.Ticker(symbol)
            df = stock.history(period='max', auto_adjust=True)
            
            if df.empty:
                return symbol, False, "No data available", pd.DataFrame()
            
            # Reset index to get Date as column
            df.reset_index(inplace=True)
            
            # Filter data from 2000 onwards
            df['Date'] = pd.to_datetime(df['Date'])
            df = df[df['Date'] >= self.start_date].copy()
            
            if df.empty:
                return symbol, False, f"No data available from {self.start_date}", pd.DataFrame()
            
            # Basic data validation
            if len(df) < 100:  # At least 100 trading days
                return symbol, False, f"Insufficient data: only {len(df)} records", df
            
            # Calculate basic statistics
            start_date = df['Date'].min().strftime('%Y-%m-%d')
            end_date = df['Date'].max().strftime('%Y-%m-%d') 
            years_of_data = (df['Date'].max() - df['Date'].min()).days / 365.25
            
            # Save raw data
            raw_file = f'{self.data_dir}/raw_data/{symbol.replace(".NS", "_NS")}_max_data.parquet'
            df.to_parquet(raw_file, index=False)
            
            success_msg = f"✅ {len(df)} records from {start_date} to {end_date} ({years_of_data:.1f} years)"
            print(f"   {success_msg}")
            
            return symbol, True, success_msg, df
            
        except Exception as e:
            error_msg = f"❌ Error fetching {symbol}: {str(e)}"
            print(f"   {error_msg}")
            return symbol, False, error_msg, pd.DataFrame()
    
    def fetch_all_stocks_parallel(self, max_workers: int = 10) -> Dict:
        """Fetch all stocks data in parallel."""
        
        print(f"🚀 FETCHING MAXIMUM HISTORICAL DATA FOR {len(self.major_indian_stocks)} INDIAN STOCKS")
        print("=" * 80)
        print(f"📅 Target Period: {self.start_date} to {self.end_date}")
        print(f"🔄 Parallel Workers: {max_workers}")
        print("=" * 80)
        
        start_time = time.time()
        results = {}
        
        with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
            # Submit all tasks
            future_to_symbol = {
                executor.submit(self.fetch_single_stock, symbol, description): symbol
                for symbol, description in self.major_indian_stocks.items()
            }
            
            # Collect results
            completed = 0
            total = len(future_to_symbol)
            
            for future in concurrent.futures.as_completed(future_to_symbol):
                symbol = future_to_symbol[future]
                
                try:
                    symbol, success, message, df = future.result()
                    
                    with self.results_lock:
                        results[symbol] = {
                            'success': success,
                            'message': message,
                            'records': len(df) if not df.empty else 0,
                            'data': df
                        }
                        
                        completed += 1
                        progress = (completed / total) * 100
                        print(f"📈 Progress: {completed}/{total} ({progress:.1f}%) - Latest: {symbol}")
                        
                except Exception as e:
                    with self.results_lock:
                        results[symbol] = {
                            'success': False,
                            'message': f"Exception: {str(e)}",
                            'records': 0,
                            'data': pd.DataFrame()
                        }
        
        elapsed_time = time.time() - start_time
        print(f"\n⏱️  Total fetch time: {elapsed_time:.1f} seconds")
        
        return results
    
    def analyze_results(self, results: Dict) -> pd.DataFrame:
        """Analyze and summarize the fetching results."""
        
        print("\n📊 ANALYZING FETCH RESULTS")
        print("=" * 50)
        
        successful = sum(1 for r in results.values() if r['success'])
        failed = len(results) - successful
        total_records = sum(r['records'] for r in results.values())
        
        print(f"✅ Successful: {successful}/{len(results)} ({successful/len(results)*100:.1f}%)")
        print(f"❌ Failed: {failed}/{len(results)} ({failed/len(results)*100:.1f}%)")
        print(f"📊 Total Records: {total_records:,}")
        print(f"📈 Average Records per Stock: {total_records/successful:.0f}" if successful > 0 else "N/A")
        
        # Create summary DataFrame
        summary_data = []
        
        for symbol, result in results.items():
            if result['success'] and not result['data'].empty:
                df = result['data']
                
                # Calculate statistics
                start_date = df['Date'].min()
                end_date = df['Date'].max()
                years_of_data = (end_date - start_date).days / 365.25
                
                # Price statistics
                latest_price = df['Close'].iloc[-1]
                first_price = df['Close'].iloc[0]
                total_return = ((latest_price / first_price) - 1) * 100
                
                # Volume statistics
                avg_volume = df['Volume'].mean()
                
                summary_data.append({
                    'Symbol': symbol,
                    'Description': self.major_indian_stocks.get(symbol, 'Unknown'),
                    'Records': len(df),
                    'Start_Date': start_date.strftime('%Y-%m-%d'),
                    'End_Date': end_date.strftime('%Y-%m-%d'),
                    'Years_of_Data': round(years_of_data, 1),
                    'Latest_Price': round(latest_price, 2),
                    'Total_Return_%': round(total_return, 2),
                    'Avg_Daily_Volume': int(avg_volume),
                    'Status': 'Success'
                })
            else:
                summary_data.append({
                    'Symbol': symbol,
                    'Description': self.major_indian_stocks.get(symbol, 'Unknown'),
                    'Records': 0,
                    'Start_Date': 'N/A',
                    'End_Date': 'N/A',
                    'Years_of_Data': 0,
                    'Latest_Price': 0,
                    'Total_Return_%': 0,
                    'Avg_Daily_Volume': 0,
                    'Status': 'Failed'
                })
        
        summary_df = pd.DataFrame(summary_data)
        
        # Save summary
        summary_file = f'{self.data_dir}/fetch_summary_{datetime.now().strftime("%Y%m%d_%H%M%S")}.csv'
        summary_df.to_csv(summary_file, index=False)
        print(f"💾 Summary saved to: {summary_file}")
        
        return summary_df
    
    def show_top_performers(self, summary_df: pd.DataFrame, top_n: int = 20):
        """Show top performing stocks by data availability and returns."""
        
        successful_stocks = summary_df[summary_df['Status'] == 'Success'].copy()
        
        if successful_stocks.empty:
            print("❌ No successful data fetches to analyze")
            return
        
        print(f"\n🏆 TOP {top_n} STOCKS BY DATA AVAILABILITY")
        print("=" * 60)
        
        # Sort by years of data and total return
        top_by_data = successful_stocks.nlargest(top_n, 'Years_of_Data')
        
        for idx, row in top_by_data.iterrows():
            symbol = row['Symbol']
            company = row['Description'][:50] + "..." if len(row['Description']) > 50 else row['Description']
            years = row['Years_of_Data']
            records = row['Records']
            total_return = row['Total_Return_%']
            
            print(f"{symbol:15} | {years:4.1f} years | {records:6,} records | {total_return:+8.1f}% return")
            print(f"                | {company}")
            print("-" * 80)
        
        # Show best performing stocks
        print(f"\n📈 TOP {top_n} BEST PERFORMING STOCKS (by Total Return)")
        print("=" * 60)
        
        top_by_return = successful_stocks.nlargest(top_n, 'Total_Return_%')
        
        for idx, row in top_by_return.iterrows():
            symbol = row['Symbol']
            company = row['Description'][:50] + "..." if len(row['Description']) > 50 else row['Description']
            years = row['Years_of_Data']
            total_return = row['Total_Return_%']
            latest_price = row['Latest_Price']
            
            print(f"{symbol:15} | {total_return:+8.1f}% return | ₹{latest_price:8,.2f} | {years:4.1f} years")
            print(f"                | {company}")
            print("-" * 80)
    
    def run_full_fetch(self, max_workers: int = 8):
        """Run the complete data fetching process."""
        
        print("🇮🇳 MAXIMUM HISTORICAL DATA FETCHER FOR INDIAN STOCKS")
        print("=" * 80)
        print(f"📊 Total Stocks to Fetch: {len(self.major_indian_stocks)}")
        print(f"📅 Target Date Range: {self.start_date} to {self.end_date}")
        print(f"💾 Data Directory: {self.data_dir}")
        print("=" * 80)
        
        # Fetch all data
        results = self.fetch_all_stocks_parallel(max_workers=max_workers)
        
        # Analyze results
        summary_df = self.analyze_results(results)
        
        # Show top performers
        self.show_top_performers(summary_df, top_n=15)
        
        # Save results to class attribute for later use
        self.fetch_results = results
        self.summary_df = summary_df
        
        print(f"\n✅ DATA FETCHING COMPLETED!")
        print(f"📁 All data saved to: {self.data_dir}")
        print("🚀 Ready for enhanced backtesting with maximum historical data!")
        
        return results, summary_df


def main():
    """Main execution function."""
    
    # Initialize fetcher (from 2000)
    fetcher = MaximalIndianDataFetcher(start_date='2000-01-01')
    
    # Run complete data fetch
    results, summary = fetcher.run_full_fetch(max_workers=8)
    
    print(f"\n🎯 FINAL SUMMARY:")
    print(f"✅ Successfully fetched: {len([r for r in results.values() if r['success']])} stocks")
    print(f"📊 Total trading records: {sum(r['records'] for r in results.values()):,}")
    print(f"💾 Data saved to: {fetcher.data_dir}")


if __name__ == "__main__":
    main() 