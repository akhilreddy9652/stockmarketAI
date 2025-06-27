
#!/usr/bin/env python3
"""Port Manager for Multiple Streamlit Apps"""

import subprocess
import time
import sys

def start_streamlit_apps():
    """Start all Streamlit apps on different ports"""
    apps = [
        {"script": "streamlit_app.py", "port": 8501, "name": "Main Dashboard"},
        {"script": "indian_etf_monitoring_dashboard.py", "port": 8502, "name": "ETF Monitor"},
        {"script": "comprehensive_dashboard.py", "port": 8503, "name": "Comprehensive Analysis"}
    ]
    
    for app in apps:
        try:
            print(f"🚀 Starting {app['name']} on port {app['port']}")
            subprocess.Popen([
                'streamlit', 'run', app['script'], 
                '--server.port', str(app['port']),
                '--server.headless', 'true'
            ])
            time.sleep(2)  # Wait between starts
        except Exception as e:
            print(f"❌ Failed to start {app['name']}: {e}")
    
    print("✅ All Streamlit apps started")

if __name__ == "__main__":
    start_streamlit_apps()
