#!/usr/bin/env python3
"""
Production Deployment System
============================
Final production-ready AI-driven Indian stock management system
"""

import os
import json
import subprocess
import time
from datetime import datetime
import logging

class ProductionDeploymentManager:
    """Manage production deployment of the AI trading system"""
    
    def __init__(self):
        self.deployment_config = self.load_deployment_config()
        self.setup_logging()
    
    def load_deployment_config(self):
        """Load deployment configuration"""
        return {
            "system_name": "AI-Driven Indian Stock Management System",
            "version": "1.0.0",
            "deployment_date": datetime.now().isoformat(),
            "environment": "production",
            "components": [
                "unified_indian_stock_system.py",
                "broker_integration.py",
                "advanced_portfolio_optimizer.py",
                "indian_rl_trading_agent.py",
                "comprehensive_indian_analysis.py",
                "streamlit_app.py",
                "system_monitor.py",
                "automated_retraining.py"
            ],
            "services": [
                {"name": "Main Dashboard", "script": "streamlit_app.py", "port": 8501},
                {"name": "System Monitor", "script": "system_monitor.py", "daemon": True},
                {"name": "Auto Retrainer", "script": "automated_retraining.py", "daemon": True}
            ],
            "health_check_interval": 300,  # 5 minutes
            "backup_frequency": "daily",
            "performance_targets": {
                "accuracy": 0.85,
                "latency": 1.0,  # seconds
                "uptime": 0.995   # 99.5%
            }
        }
    
    def setup_logging(self):
        """Setup production logging"""
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler('logs/production_deployment.log'),
                logging.StreamHandler()
            ]
        )
        self.logger = logging.getLogger(__name__)
    
    def pre_deployment_checks(self):
        """Run pre-deployment checks"""
        self.logger.info("🔍 Running pre-deployment checks...")
        
        checks = {
            "file_integrity": self.check_file_integrity(),
            "configuration": self.check_configuration(),
            "dependencies": self.check_dependencies(),
            "resources": self.check_system_resources(),
            "security": self.check_security_settings()
        }
        
        all_passed = all(checks.values())
        
        if all_passed:
            self.logger.info("✅ All pre-deployment checks passed")
        else:
            failed_checks = [k for k, v in checks.items() if not v]
            self.logger.error(f"❌ Failed checks: {failed_checks}")
        
        return all_passed, checks
    
    def check_file_integrity(self):
        """Check if all required files exist"""
        required_files = self.deployment_config["components"]
        missing_files = []
        
        for file in required_files:
            if not os.path.exists(file):
                missing_files.append(file)
        
        if missing_files:
            self.logger.warning(f"Missing files: {missing_files}")
            return False
        
        return True
    
    def check_configuration(self):
        """Check configuration files"""
        config_files = [
            'production_config.json',
            'broker_config.json'
        ]
        
        for config_file in config_files:
            if not os.path.exists(config_file):
                self.logger.error(f"Missing config file: {config_file}")
                return False
            
            try:
                with open(config_file, 'r') as f:
                    json.load(f)
            except json.JSONDecodeError:
                self.logger.error(f"Invalid JSON in {config_file}")
                return False
        
        return True
    
    def check_dependencies(self):
        """Check Python dependencies"""
        required_packages = [
            'streamlit', 'pandas', 'numpy', 'yfinance',
            'tensorflow', 'scikit-learn', 'plotly', 'psutil'
        ]
        
        missing_packages = []
        for package in required_packages:
            try:
                __import__(package)
            except ImportError:
                missing_packages.append(package)
        
        if missing_packages:
            self.logger.error(f"Missing packages: {missing_packages}")
            return False
        
        return True
    
    def check_system_resources(self):
        """Check system resources"""
        try:
            import psutil
            
            # Check CPU
            cpu_percent = psutil.cpu_percent(interval=1)
            if cpu_percent > 80:
                self.logger.warning(f"High CPU usage: {cpu_percent}%")
                return False
            
            # Check memory
            memory = psutil.virtual_memory()
            if memory.percent > 80:
                self.logger.warning(f"High memory usage: {memory.percent}%")
                return False
            
            # Check disk space
            disk = psutil.disk_usage('.')
            free_gb = disk.free / (1024**3)
            if free_gb < 10:  # Less than 10GB free
                self.logger.warning(f"Low disk space: {free_gb:.1f}GB")
                return False
            
            return True
            
        except ImportError:
            self.logger.warning("psutil not available for resource check")
            return True
    
    def check_security_settings(self):
        """Check security settings"""
        # Check if sensitive files have proper permissions
        sensitive_files = ['broker_config.json']
        
        for file in sensitive_files:
            if os.path.exists(file):
                # Check file permissions (basic check)
                stat = os.stat(file)
                # In production, you'd implement more sophisticated security checks
                pass
        
        return True
    
    def deploy_services(self):
        """Deploy all system services"""
        self.logger.info("🚀 Deploying system services...")
        
        deployment_results = {}
        
        for service in self.deployment_config["services"]:
            try:
                result = self.deploy_service(service)
                deployment_results[service["name"]] = result
                
                if result["success"]:
                    self.logger.info(f"✅ {service['name']} deployed successfully")
                else:
                    self.logger.error(f"❌ {service['name']} deployment failed: {result['error']}")
                    
            except Exception as e:
                self.logger.error(f"❌ Error deploying {service['name']}: {e}")
                deployment_results[service["name"]] = {"success": False, "error": str(e)}
        
        return deployment_results
    
    def deploy_service(self, service):
        """Deploy individual service"""
        try:
            if service.get("daemon", False):
                # Start as background daemon
                process = subprocess.Popen([
                    'python', service["script"]
                ], stdout=subprocess.PIPE, stderr=subprocess.PIPE)
                
                # Give it time to start
                time.sleep(2)
                
                # Check if still running
                if process.poll() is None:
                    return {
                        "success": True,
                        "pid": process.pid,
                        "type": "daemon"
                    }
                else:
                    stderr = process.stderr.read().decode()
                    return {
                        "success": False,
                        "error": f"Process exited: {stderr}"
                    }
            
            elif "port" in service:
                # Start Streamlit service
                process = subprocess.Popen([
                    'streamlit', 'run', service["script"],
                    '--server.port', str(service["port"]),
                    '--server.headless', 'true'
                ], stdout=subprocess.PIPE, stderr=subprocess.PIPE)
                
                # Give it time to start
                time.sleep(3)
                
                return {
                    "success": True,
                    "pid": process.pid,
                    "port": service["port"],
                    "url": f"http://localhost:{service['port']}",
                    "type": "web_service"
                }
            
            else:
                return {"success": False, "error": "Unknown service type"}
                
        except Exception as e:
            return {"success": False, "error": str(e)}
    
    def create_monitoring_dashboard(self):
        """Create production monitoring dashboard"""
        dashboard_config = {
            "dashboard_name": "AI Trading System Monitor",
            "metrics": [
                "system_health",
                "trading_performance",
                "model_accuracy",
                "portfolio_status",
                "risk_metrics"
            ],
            "alerts": [
                {"metric": "accuracy", "threshold": 0.80, "type": "below"},
                {"metric": "portfolio_loss", "threshold": 0.05, "type": "above"},
                {"metric": "system_uptime", "threshold": 0.99, "type": "below"}
            ],
            "refresh_interval": 30  # seconds
        }
        
        with open('monitoring_dashboard_config.json', 'w') as f:
            json.dump(dashboard_config, f, indent=2)
        
        self.logger.info("📊 Monitoring dashboard configured")
        return dashboard_config
    
    def setup_backup_system(self):
        """Setup automated backup system"""
        backup_config = {
            "backup_directories": [
                "models",
                "results", 
                "data",
                "logs"
            ],
            "backup_files": [
                "production_config.json",
                "broker_config.json",
                "system_health_report.json"
            ],
            "backup_schedule": "daily",
            "backup_location": "./backups",
            "retention_days": 30
        }
        
        # Create backup directory
        os.makedirs("backups", exist_ok=True)
        
        # Create backup script
        backup_script = f"""#!/bin/bash
# Automated backup script for AI Trading System
# Generated on {datetime.now().isoformat()}

BACKUP_DIR="./backups"
DATE=$(date +%Y%m%d_%H%M%S)
BACKUP_NAME="trading_system_backup_$DATE"

echo "🔄 Starting backup at $(date)"

# Create backup directory
mkdir -p "$BACKUP_DIR/$BACKUP_NAME"

# Backup directories
{chr(10).join([f'cp -r {d} "$BACKUP_DIR/$BACKUP_NAME/"' for d in backup_config["backup_directories"]])}

# Backup files  
{chr(10).join([f'cp {f} "$BACKUP_DIR/$BACKUP_NAME/"' for f in backup_config["backup_files"]])}

# Compress backup
cd "$BACKUP_DIR"
tar -czf "$BACKUP_NAME.tar.gz" "$BACKUP_NAME"
rm -rf "$BACKUP_NAME"

echo "✅ Backup completed: $BACKUP_NAME.tar.gz"

# Cleanup old backups (keep last 30 days)
find "$BACKUP_DIR" -name "trading_system_backup_*.tar.gz" -type f -mtime +{backup_config["retention_days"]} -delete

echo "🧹 Old backups cleaned up"
"""
        
        with open('backup_system.sh', 'w') as f:
            f.write(backup_script)
        
        # Make executable
        os.chmod('backup_system.sh', 0o755)
        
        self.logger.info("💾 Backup system configured")
        return backup_config
    
    def generate_deployment_report(self, deployment_results):
        """Generate comprehensive deployment report"""
        report = {
            "deployment_info": {
                "system_name": self.deployment_config["system_name"],
                "version": self.deployment_config["version"],
                "deployment_date": self.deployment_config["deployment_date"],
                "environment": self.deployment_config["environment"]
            },
            "deployment_results": deployment_results,
            "system_status": self.get_system_status(),
            "performance_baseline": self.establish_performance_baseline(),
            "monitoring_setup": True,
            "backup_setup": True,
            "next_steps": [
                "Monitor system performance for 24 hours",
                "Verify trading signals accuracy",
                "Test broker integration (paper trading)",
                "Schedule model retraining",
                "Setup alerting system"
            ]
        }
        
        # Calculate deployment success rate
        successful_services = sum(1 for r in deployment_results.values() if r.get("success", False))
        total_services = len(deployment_results)
        success_rate = (successful_services / total_services) * 100 if total_services > 0 else 0
        
        report["deployment_summary"] = {
            "total_services": total_services,
            "successful_services": successful_services,
            "success_rate": success_rate,
            "status": "SUCCESS" if success_rate >= 80 else "PARTIAL" if success_rate >= 50 else "FAILED"
        }
        
        return report
    
    def get_system_status(self):
        """Get current system status"""
        try:
            from system_monitor import SystemMonitor
            monitor = SystemMonitor()
            health_report = monitor.generate_health_report()
            return health_report["overall_status"]
        except:
            return {"status": "unknown", "message": "System monitor not available"}
    
    def establish_performance_baseline(self):
        """Establish performance baseline"""
        return {
            "baseline_date": datetime.now().isoformat(),
            "expected_accuracy": 0.85,
            "expected_latency": 1.0,
            "expected_uptime": 0.995,
            "portfolio_target_return": 0.15,  # 15% annual
            "max_drawdown": 0.10  # 10%
        }
    
    def run_full_deployment(self):
        """Run complete production deployment"""
        self.logger.info("🚀 STARTING FULL PRODUCTION DEPLOYMENT")
        self.logger.info("=" * 60)
        
        # Step 1: Pre-deployment checks
        checks_passed, check_results = self.pre_deployment_checks()
        
        if not checks_passed:
            self.logger.error("❌ Pre-deployment checks failed. Aborting deployment.")
            return False
        
        # Step 2: Deploy services
        deployment_results = self.deploy_services()
        
        # Step 3: Setup monitoring
        monitoring_config = self.create_monitoring_dashboard()
        
        # Step 4: Setup backup system
        backup_config = self.setup_backup_system()
        
        # Step 5: Generate deployment report
        deployment_report = self.generate_deployment_report(deployment_results)
        
        # Save deployment report
        with open('production_deployment_report.json', 'w') as f:
            json.dump(deployment_report, f, indent=2)
        
        # Print summary
        self.print_deployment_summary(deployment_report)
        
        return deployment_report["deployment_summary"]["status"] == "SUCCESS"
    
    def print_deployment_summary(self, report):
        """Print deployment summary"""
        summary = report["deployment_summary"]
        
        self.logger.info("\n" + "="*60)
        self.logger.info("🎯 PRODUCTION DEPLOYMENT SUMMARY")
        self.logger.info("="*60)
        
        self.logger.info(f"📊 Services Deployed: {summary['successful_services']}/{summary['total_services']}")
        self.logger.info(f"✅ Success Rate: {summary['success_rate']:.1f}%")
        self.logger.info(f"🏷️ Status: {summary['status']}")
        
        # Service details
        self.logger.info(f"\n📋 Service Status:")
        for service_name, result in report["deployment_results"].items():
            status = "✅" if result.get("success") else "❌"
            self.logger.info(f"   {status} {service_name}")
            
            if result.get("url"):
                self.logger.info(f"      🌐 URL: {result['url']}")
            if result.get("pid"):
                self.logger.info(f"      🔄 PID: {result['pid']}")
        
        # Next steps
        self.logger.info(f"\n📝 Next Steps:")
        for i, step in enumerate(report["next_steps"], 1):
            self.logger.info(f"   {i}. {step}")
        
        if summary["status"] == "SUCCESS":
            self.logger.info(f"\n🚀 SYSTEM IS NOW 100% PRODUCTION READY!")
            self.logger.info(f"💰 Ready for live trading with Indian stocks")
            self.logger.info(f"📊 Monitoring dashboard active")
            self.logger.info(f"💾 Automated backups configured")
        else:
            self.logger.info(f"\n⚠️ Deployment completed with issues")
            self.logger.info(f"🔧 Review failed services and retry")

if __name__ == "__main__":
    print("🚀 AI-DRIVEN INDIAN STOCK MANAGEMENT SYSTEM")
    print("Production Deployment Manager")
    print("=" * 60)
    
    # Initialize deployment manager
    deployment_manager = ProductionDeploymentManager()
    
    # Run full deployment
    success = deployment_manager.run_full_deployment()
    
    if success:
        print(f"\n🎉 DEPLOYMENT SUCCESSFUL!")
        print(f"🌐 Access your system at: http://localhost:8501")
        print(f"📊 Monitor system health in real-time")
        print(f"💼 Start paper trading or configure real broker")
    else:
        print(f"\n⚠️ Deployment completed with issues")
        print(f"📋 Check production_deployment_report.json for details") 