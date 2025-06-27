#!/usr/bin/env python3
"""
GitHub Push Setup Script
This script helps push the stock predictor project to GitHub
"""

import subprocess
import os
import sys

def run_command(command, description=""):
    """Run a shell command and return the result"""
    print(f"🔄 {description}")
    try:
        result = subprocess.run(command, shell=True, capture_output=True, text=True)
        if result.returncode == 0:
            print(f"✅ {description} - Success")
            if result.stdout:
                print(result.stdout.strip())
            return True
        else:
            print(f"❌ {description} - Failed")
            if result.stderr:
                print(f"Error: {result.stderr.strip()}")
            return False
    except Exception as e:
        print(f"❌ {description} - Exception: {str(e)}")
        return False

def main():
    print("🚀 GitHub Push Setup for Advanced Stock Predictor System")
    print("=" * 60)
    
    # Check git status
    print("\n📊 Checking git status...")
    run_command("git status --porcelain", "Checking for uncommitted changes")
    
    # Check tracked files
    result = subprocess.run("git ls-files | wc -l", shell=True, capture_output=True, text=True)
    if result.returncode == 0:
        file_count = result.stdout.strip()
        print(f"📁 Files tracked by git: {file_count}")
    
    # Check remote
    print("\n🔗 Checking remote repository...")
    run_command("git remote -v", "Checking remote configuration")
    
    # Check branches
    print("\n🌿 Checking branches...")
    run_command("git branch -a", "Checking branches")
    
    # Try to push
    print("\n📤 Attempting to push to GitHub...")
    print("Note: You may need to authenticate with GitHub")
    print("If prompted, use your GitHub username and Personal Access Token")
    print("To create a token: GitHub Settings > Developer settings > Personal access tokens")
    
    # Set up authentication helper
    run_command("git config --global credential.helper store", "Setting up credential helper")
    
    # Try the push
    success = run_command("git push -u origin main", "Pushing to GitHub")
    
    if success:
        print("\n🎉 SUCCESS! Your repository has been pushed to GitHub!")
        print("🌐 View at: https://github.com/akhilreddy9652/stockmarket")
    else:
        print("\n⚠️  Push failed. Please check your authentication.")
        print("\n📋 Manual steps to try:")
        print("1. Create a Personal Access Token at: https://github.com/settings/tokens")
        print("2. Use your GitHub username and token when prompted")
        print("3. Run: git push -u origin main")
        
        # Show some sample files that should be pushed
        print("\n📁 Sample files that should be in your repository:")
        sample_files = [
            "streamlit_app.py",
            "indian_etf_monitoring_dashboard.py", 
            "README.md",
            "requirements.txt",
            "data_ingestion.py",
            "feature_engineering.py"
        ]
        
        for file in sample_files:
            if os.path.exists(file):
                size = os.path.getsize(file)
                print(f"   ✅ {file} ({size:,} bytes)")
            else:
                print(f"   ❌ {file} (missing)")

if __name__ == "__main__":
    main() 