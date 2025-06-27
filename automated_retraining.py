
import schedule
import time
from datetime import datetime, timedelta
import os
import subprocess

class AutomatedRetraining:
    """Automated model retraining system"""
    
    def __init__(self):
        self.last_retrain = None
        self.performance_threshold = 0.85
    
    def check_performance_drift(self):
        """Check if model performance has drifted"""
        try:
            # Load recent performance metrics
            import json
            with open('logs/performance_metrics.json', 'r') as f:
                metrics = json.load(f)
            
            current_accuracy = metrics.get('accuracy', 0)
            return current_accuracy < self.performance_threshold
        except:
            return True  # Retrain if we can't check performance
    
    def retrain_models(self):
        """Retrain all models"""
        print(f"🔄 Starting automated retraining at {datetime.now()}")
        
        try:
            # Run model retraining
            result = subprocess.run([
                'python', 'train_enhanced_system.py'
            ], capture_output=True, text=True)
            
            if result.returncode == 0:
                print("✅ Model retraining successful")
                self.last_retrain = datetime.now()
                
                # Log retraining
                with open('logs/retraining_log.txt', 'a') as f:
                    f.write(f"{datetime.now()}: Retraining successful\n")
            else:
                print(f"❌ Model retraining failed: {result.stderr}")
                
        except Exception as e:
            print(f"❌ Retraining error: {e}")
    
    def schedule_retraining(self):
        """Schedule automatic retraining"""
        # Weekly retraining
        schedule.every().sunday.at("02:00").do(self.retrain_models)
        
        # Performance-based retraining check
        schedule.every().day.at("06:00").do(self.check_and_retrain)
        
        print("📅 Automated retraining scheduled")
        print("   - Weekly: Every Sunday at 2:00 AM")
        print("   - Performance check: Daily at 6:00 AM")
    
    def check_and_retrain(self):
        """Check performance and retrain if needed"""
        if self.check_performance_drift():
            print("⚠️ Performance drift detected, triggering retraining")
            self.retrain_models()
        else:
            print("✅ Model performance within acceptable range")
    
    def run_scheduler(self):
        """Run the retraining scheduler"""
        self.schedule_retraining()
        
        while True:
            schedule.run_pending()
            time.sleep(3600)  # Check every hour

if __name__ == "__main__":
    retrainer = AutomatedRetraining()
    retrainer.run_scheduler()
