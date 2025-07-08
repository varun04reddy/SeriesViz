#!/usr/bin/env python3
"""
SeriesViz Dashboard Launcher
A simple script to check dependencies and launch the Streamlit dashboard.
"""

import subprocess
import sys
import os

def check_python_version():
    """Check if Python version is compatible"""
    if sys.version_info < (3, 8):
        print("❌ Error: Python 3.8 or higher is required.")
        print(f"Current version: {sys.version}")
        return False
    print(f"✅ Python version: {sys.version.split()[0]}")
    return True

def check_dependencies():
    """Check if required packages are installed"""
    required_packages = [
        'streamlit',
        'pandas', 
        'plotly',
        'numpy',
        'scipy'
    ]
    
    missing_packages = []
    
    for package in required_packages:
        try:
            __import__(package)
            print(f"✅ {package} is installed")
        except ImportError:
            missing_packages.append(package)
            print(f"❌ {package} is missing")
    
    if missing_packages:
        print(f"\n📦 Installing missing packages: {', '.join(missing_packages)}")
        try:
            subprocess.check_call([
                sys.executable, '-m', 'pip', 'install', '-r', 'requirements.txt'
            ])
            print("✅ All dependencies installed successfully!")
            return True
        except subprocess.CalledProcessError:
            print("❌ Failed to install dependencies. Please run: pip install -r requirements.txt")
            return False
    
    return True

def launch_dashboard():
    """Launch the Streamlit dashboard"""
    print("\n🚀 Launching SeriesViz Dashboard...")
    print("📊 The dashboard will open in your default browser.")
    print("🌐 If it doesn't open automatically, go to: http://localhost:8501")
    print("\n" + "="*50)
    
    try:
        subprocess.run([
            sys.executable, '-m', 'streamlit', 'run', 'app.py',
            '--server.port', '8501',
            '--server.headless', 'false'
        ])
    except KeyboardInterrupt:
        print("\n👋 Dashboard stopped by user.")
    except Exception as e:
        print(f"❌ Error launching dashboard: {e}")

def main():
    """Main launcher function"""
    print("📊 SeriesViz Dashboard Launcher")
    print("="*50)
    
    # Check Python version
    if not check_python_version():
        sys.exit(1)
    
    # Check and install dependencies
    if not check_dependencies():
        sys.exit(1)
    
    # Launch dashboard
    launch_dashboard()

if __name__ == "__main__":
    main() 