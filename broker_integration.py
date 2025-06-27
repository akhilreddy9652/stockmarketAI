#!/usr/bin/env python3
"""
Broker Integration System
=========================
Real trading integration with Indian brokers (Zerodha, Upstox, etc.)
"""

import os
import json
import time
from datetime import datetime, timedelta
from abc import ABC, abstractmethod
import hashlib
import hmac
import requests

class BrokerInterface(ABC):
    """Abstract base class for broker integrations"""
    
    @abstractmethod
    def authenticate(self):
        pass
    
    @abstractmethod
    def place_order(self, symbol, quantity, order_type, price=None):
        pass
    
    @abstractmethod
    def get_positions(self):
        pass
    
    @abstractmethod
    def get_balance(self):
        pass
    
    @abstractmethod
    def cancel_order(self, order_id):
        pass

class ZerodhaIntegration(BrokerInterface):
    """Zerodha Kite API Integration"""
    
    def __init__(self, api_key, api_secret):
        self.api_key = api_key
        self.api_secret = api_secret
        self.access_token = None
        self.base_url = "https://api.kite.trade"
        
    def authenticate(self, request_token=None):
        """Authenticate with Zerodha"""
        if request_token:
            # Generate session
            checksum = hashlib.sha256((self.api_key + request_token + self.api_secret).encode()).hexdigest()
            url = f"{self.base_url}/session/token"
            
            payload = {
                "api_key": self.api_key,
                "request_token": request_token,
                "checksum": checksum
            }
            
            response = requests.post(url, data=payload)
            if response.status_code == 200:
                data = response.json()
                self.access_token = data['data']['access_token']
                return True
        return False
    
    def place_order(self, symbol, quantity, order_type, price=None):
        """Place order through Zerodha"""
        if not self.access_token:
            return {"error": "Not authenticated"}
        
        headers = {
            "Authorization": f"token {self.api_key}:{self.access_token}",
            "Content-Type": "application/x-www-form-urlencoded"
        }
        
        payload = {
            "tradingsymbol": symbol,
            "exchange": "NSE",
            "transaction_type": order_type,  # BUY or SELL
            "quantity": quantity,
            "order_type": "LIMIT" if price else "MARKET",
            "product": "CNC",  # Cash and Carry
            "validity": "DAY"
        }
        
        if price:
            payload["price"] = price
        
        url = f"{self.base_url}/orders"
        response = requests.post(url, headers=headers, data=payload)
        
        return response.json()
    
    def get_positions(self):
        """Get current positions"""
        if not self.access_token:
            return {"error": "Not authenticated"}
        
        headers = {
            "Authorization": f"token {self.api_key}:{self.access_token}"
        }
        
        url = f"{self.base_url}/portfolio/positions"
        response = requests.get(url, headers=headers)
        
        return response.json()
    
    def get_balance(self):
        """Get account balance"""
        if not self.access_token:
            return {"error": "Not authenticated"}
        
        headers = {
            "Authorization": f"token {self.api_key}:{self.access_token}"
        }
        
        url = f"{self.base_url}/user/margins"
        response = requests.get(url, headers=headers)
        
        return response.json()
    
    def cancel_order(self, order_id):
        """Cancel order"""
        if not self.access_token:
            return {"error": "Not authenticated"}
        
        headers = {
            "Authorization": f"token {self.api_key}:{self.access_token}"
        }
        
        url = f"{self.base_url}/orders/{order_id}"
        response = requests.delete(url, headers=headers)
        
        return response.json()

class UpstoxIntegration(BrokerInterface):
    """Upstox API Integration"""
    
    def __init__(self, api_key, api_secret):
        self.api_key = api_key
        self.api_secret = api_secret
        self.access_token = None
        self.base_url = "https://api.upstox.com/v2"
    
    def authenticate(self, authorization_code=None):
        """Authenticate with Upstox"""
        # Implementation for Upstox authentication
        pass
    
    def place_order(self, symbol, quantity, order_type, price=None):
        """Place order through Upstox"""
        # Implementation for Upstox order placement
        pass
    
    def get_positions(self):
        """Get current positions"""
        pass
    
    def get_balance(self):
        """Get account balance"""
        pass
    
    def cancel_order(self, order_id):
        """Cancel order"""
        pass

class BrokerManager:
    """Manage multiple broker integrations"""
    
    def __init__(self):
        self.brokers = {}
        self.active_broker = None
        self.load_configuration()
    
    def load_configuration(self):
        """Load broker configuration"""
        config_file = 'broker_config.json'
        if os.path.exists(config_file):
            with open(config_file, 'r') as f:
                self.config = json.load(f)
        else:
            self.config = {
                "zerodha": {
                    "api_key": "",
                    "api_secret": "",
                    "enabled": False
                },
                "upstox": {
                    "api_key": "",
                    "api_secret": "",
                    "enabled": False
                },
                "paper_trading": {
                    "enabled": True,
                    "initial_balance": 1000000
                }
            }
            self.save_configuration()
    
    def save_configuration(self):
        """Save broker configuration"""
        with open('broker_config.json', 'w') as f:
            json.dump(self.config, f, indent=2)
    
    def add_broker(self, broker_name, broker_instance):
        """Add broker integration"""
        self.brokers[broker_name] = broker_instance
        
    def set_active_broker(self, broker_name):
        """Set active broker for trading"""
        if broker_name in self.brokers:
            self.active_broker = broker_name
            return True
        return False
    
    def execute_trade(self, symbol, quantity, action, price=None):
        """Execute trade through active broker"""
        if not self.active_broker:
            return self.paper_trade(symbol, quantity, action, price)
        
        broker = self.brokers[self.active_broker]
        
        try:
            result = broker.place_order(symbol, quantity, action, price)
            
            # Log trade
            self.log_trade({
                'timestamp': datetime.now().isoformat(),
                'broker': self.active_broker,
                'symbol': symbol,
                'quantity': quantity,
                'action': action,
                'price': price,
                'result': result
            })
            
            return result
            
        except Exception as e:
            return {"error": str(e)}
    
    def paper_trade(self, symbol, quantity, action, price=None):
        """Execute paper trade"""
        # Load paper trading state
        paper_file = 'paper_trading.json'
        if os.path.exists(paper_file):
            with open(paper_file, 'r') as f:
                paper_state = json.load(f)
        else:
            paper_state = {
                'balance': self.config['paper_trading']['initial_balance'],
                'positions': {},
                'orders': []
            }
        
        # Calculate trade value
        if price is None:
            # Get current market price (simplified)
            import yfinance as yf
            ticker = yf.Ticker(symbol)
            current_price = ticker.history(period="1d")['Close'].iloc[-1]
        else:
            current_price = price
        
        trade_value = quantity * current_price
        
        # Execute paper trade
        if action == "BUY":
            if paper_state['balance'] >= trade_value:
                paper_state['balance'] -= trade_value
                paper_state['positions'][symbol] = paper_state['positions'].get(symbol, 0) + quantity
                success = True
            else:
                return {"error": "Insufficient balance"}
        
        elif action == "SELL":
            current_position = paper_state['positions'].get(symbol, 0)
            if current_position >= quantity:
                paper_state['balance'] += trade_value
                paper_state['positions'][symbol] -= quantity
                success = True
            else:
                return {"error": "Insufficient position"}
        
        if success:
            # Record order
            order = {
                'order_id': f"PAPER_{int(time.time())}",
                'symbol': symbol,
                'quantity': quantity,
                'action': action,
                'price': current_price,
                'timestamp': datetime.now().isoformat(),
                'status': 'COMPLETE'
            }
            
            paper_state['orders'].append(order)
            
            # Save state
            with open(paper_file, 'w') as f:
                json.dump(paper_state, f, indent=2)
            
            return {"status": "success", "order_id": order['order_id']}
    
    def get_portfolio_status(self):
        """Get current portfolio status"""
        if self.active_broker and self.active_broker != 'paper':
            broker = self.brokers[self.active_broker]
            positions = broker.get_positions()
            balance = broker.get_balance()
            return {"positions": positions, "balance": balance}
        else:
            # Return paper trading status
            paper_file = 'paper_trading.json'
            if os.path.exists(paper_file):
                with open(paper_file, 'r') as f:
                    return json.load(f)
            return {"balance": 0, "positions": {}}
    
    def log_trade(self, trade_data):
        """Log trade execution"""
        log_file = 'trade_log.json'
        
        if os.path.exists(log_file):
            with open(log_file, 'r') as f:
                logs = json.load(f)
        else:
            logs = []
        
        logs.append(trade_data)
        
        with open(log_file, 'w') as f:
            json.dump(logs, f, indent=2)

class TradingExecutor:
    """Execute trades based on AI signals"""
    
    def __init__(self, broker_manager):
        self.broker_manager = broker_manager
        self.risk_manager = RiskManager()
        
    def execute_signal(self, signal_data):
        """Execute trading signal"""
        symbol = signal_data['symbol']
        signal = signal_data['signal']
        confidence = signal_data['confidence']
        price = signal_data.get('current_price')
        
        # Check risk constraints
        if not self.risk_manager.check_risk_constraints(symbol, signal_data):
            return {"status": "rejected", "reason": "Risk constraints violated"}
        
        # Calculate position size
        position_size = self.calculate_position_size(signal_data)
        
        if position_size <= 0:
            return {"status": "rejected", "reason": "Invalid position size"}
        
        # Execute trade
        if signal in ['BUY', 'STRONG_BUY']:
            result = self.broker_manager.execute_trade(
                symbol, position_size, 'BUY', price
            )
        elif signal in ['SELL', 'STRONG_SELL']:
            result = self.broker_manager.execute_trade(
                symbol, position_size, 'SELL', price
            )
        else:
            return {"status": "no_action", "reason": f"HOLD signal for {symbol}"}
        
        return result
    
    def calculate_position_size(self, signal_data):
        """Calculate position size based on signal strength and risk"""
        confidence = signal_data['confidence']
        price = signal_data.get('current_price', 100)  # Default price
        
        # Get available capital
        portfolio = self.broker_manager.get_portfolio_status()
        available_balance = portfolio.get('balance', 0)
        
        # Position sizing based on confidence and risk
        max_position_pct = 0.10  # Maximum 10% per position
        confidence_adjusted_pct = max_position_pct * confidence
        
        position_value = available_balance * confidence_adjusted_pct
        position_size = int(position_value / price)
        
        return position_size

class RiskManager:
    """Risk management for trading execution"""
    
    def __init__(self):
        self.load_risk_config()
    
    def load_risk_config(self):
        """Load risk management configuration"""
        self.risk_config = {
            'max_position_size': 0.15,  # 15% max per position
            'max_daily_loss': 0.05,     # 5% max daily loss
            'max_sector_exposure': 0.30, # 30% max per sector
            'min_confidence': 0.60       # 60% minimum confidence
        }
    
    def check_risk_constraints(self, symbol, signal_data):
        """Check if trade meets risk constraints"""
        confidence = signal_data['confidence']
        
        # Check minimum confidence
        if confidence < self.risk_config['min_confidence']:
            return False
        
        # Additional risk checks can be added here
        return True

# Example usage and testing
if __name__ == "__main__":
    print("🔗 BROKER INTEGRATION SYSTEM")
    print("=" * 50)
    
    # Initialize broker manager
    broker_manager = BrokerManager()
    
    # Setup paper trading (default)
    print("📝 Paper Trading Mode Enabled")
    
    # Initialize trading executor
    executor = TradingExecutor(broker_manager)
    
    # Example signal execution
    test_signal = {
        'symbol': 'RELIANCE.NS',
        'signal': 'BUY',
        'confidence': 0.75,
        'current_price': 1500.0
    }
    
    print(f"\n🎯 Testing signal execution...")
    result = executor.execute_signal(test_signal)
    print(f"Result: {result}")
    
    # Check portfolio status
    portfolio = broker_manager.get_portfolio_status()
    print(f"\n💼 Portfolio Status:")
    print(f"Balance: ₹{portfolio.get('balance', 0):,.2f}")
    print(f"Positions: {len(portfolio.get('positions', {}))}")
    
    print(f"\n✅ Broker integration system ready")
    print(f"📋 Configuration saved to broker_config.json")
    print(f"📊 Paper trading enabled by default") 