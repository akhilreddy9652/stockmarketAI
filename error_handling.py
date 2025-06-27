
import logging
import traceback
from datetime import datetime
import json

class ProductionErrorHandler:
    """Comprehensive error handling for production"""
    
    def __init__(self):
        self.setup_logging()
    
    def setup_logging(self):
        """Setup production logging"""
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler('logs/production.log'),
                logging.StreamHandler()
            ]
        )
        self.logger = logging.getLogger(__name__)
    
    def handle_exception(self, func):
        """Decorator for handling exceptions"""
        def wrapper(*args, **kwargs):
            try:
                return func(*args, **kwargs)
            except Exception as e:
                self.log_error(func.__name__, e)
                return None
        return wrapper
    
    def log_error(self, function_name, error):
        """Log error with context"""
        error_info = {
            'timestamp': datetime.now().isoformat(),
            'function': function_name,
            'error': str(error),
            'traceback': traceback.format_exc()
        }
        
        self.logger.error(f"Error in {function_name}: {error}")
        
        # Save to error log file
        with open('logs/error_log.json', 'a') as f:
            f.write(json.dumps(error_info) + '\n')
    
    def create_alert(self, message, severity="INFO"):
        """Create system alert"""
        alert = {
            'timestamp': datetime.now().isoformat(),
            'message': message,
            'severity': severity
        }
        
        # Log alert
        if severity == "CRITICAL":
            self.logger.critical(message)
        elif severity == "ERROR":
            self.logger.error(message)
        else:
            self.logger.info(message)
        
        # Save alert
        with open('logs/alerts.json', 'a') as f:
            f.write(json.dumps(alert) + '\n')

# Global error handler instance
error_handler = ProductionErrorHandler()
