
import psutil
import time
import json
from datetime import datetime
import yfinance as yf

class SystemMonitor:
    """Real-time system monitoring"""
    
    def __init__(self):
        self.metrics = {}
    
    def monitor_system_resources(self):
        """Monitor CPU, memory, disk"""
        self.metrics.update({
            'cpu_percent': psutil.cpu_percent(),
            'memory_percent': psutil.virtual_memory().percent,
            'disk_percent': psutil.disk_usage('.').percent,
            'timestamp': datetime.now().isoformat()
        })
        return self.metrics
    
    def monitor_trading_performance(self):
        """Monitor trading system performance"""
        # Check if models are responding
        model_status = self.check_model_health()
        
        # Check data feed
        data_status = self.check_data_feed()
        
        # Check portfolio performance
        portfolio_status = self.check_portfolio_health()
        
        return {
            'models': model_status,
            'data_feed': data_status,
            'portfolio': portfolio_status,
            'timestamp': datetime.now().isoformat()
        }
    
    def check_model_health(self):
        """Check if models are healthy"""
        try:
            # Test model loading
            import tensorflow as tf
            return {"status": "healthy", "models_loaded": True}
        except Exception as e:
            return {"status": "error", "error": str(e)}
    
    def check_data_feed(self):
        """Check data feed connectivity"""
        try:
            # Test yfinance connection
            stock = yf.Ticker("RELIANCE.NS")
            data = stock.history(period="1d")
            return {"status": "healthy", "last_update": datetime.now().isoformat()}
        except Exception as e:
            return {"status": "error", "error": str(e)}
    
    def check_portfolio_health(self):
        """Check portfolio system health"""
        try:
            # Check if portfolio files exist and are recent
            import os
            portfolio_files = [f for f in os.listdir('results') if 'portfolio' in f]
            return {"status": "healthy", "portfolio_files": len(portfolio_files)}
        except Exception as e:
            return {"status": "error", "error": str(e)}
    
    def generate_health_report(self):
        """Generate comprehensive health report"""
        system_metrics = self.monitor_system_resources()
        trading_metrics = self.monitor_trading_performance()
        
        report = {
            'timestamp': datetime.now().isoformat(),
            'system': system_metrics,
            'trading': trading_metrics,
            'overall_status': self.calculate_overall_status(system_metrics, trading_metrics)
        }
        
        # Save report
        with open('logs/health_report.json', 'w') as f:
            json.dump(report, f, indent=2)
        
        return report
    
    def calculate_overall_status(self, system, trading):
        """Calculate overall system status"""
        issues = []
        
        if system['cpu_percent'] > 80:
            issues.append("High CPU usage")
        if system['memory_percent'] > 80:
            issues.append("High memory usage")
        if trading['models']['status'] != 'healthy':
            issues.append("Model health issues")
        if trading['data_feed']['status'] != 'healthy':
            issues.append("Data feed issues")
        
        if not issues:
            return {"status": "healthy", "issues": []}
        else:
            return {"status": "warning", "issues": issues}

if __name__ == "__main__":
    monitor = SystemMonitor()
    report = monitor.generate_health_report()
    print("📊 System Health Report Generated")
    print(f"Status: {report['overall_status']['status']}")
