#!/usr/bin/env python3
"""
Startup script for DocStudio
Launches the API server and optionally serves the UI
"""

import os
import sys
import subprocess
import threading
import time
import webbrowser
from pathlib import Path

def check_dependencies():
    """Check if required dependencies are installed."""
    try:
        import torch
        import transformers
        import fastapi
        import uvicorn
        print("✅ All dependencies are installed")
        return True
    except ImportError as e:
        print(f"❌ Missing dependency: {e}")
        print("Please install dependencies with: pip install -r requirements.txt")
        return False

def start_api_server():
    """Start the FastAPI server."""
    api_dir = Path(__file__).parent / "api"
    os.chdir(api_dir)
    
    print("🚀 Starting API server...")
    try:
        subprocess.run([
            sys.executable, "-m", "uvicorn", 
            "app:app", 
            "--host", "0.0.0.0", 
            "--port", "8000",
            "--reload"
        ], check=True)
    except KeyboardInterrupt:
        print("\n🛑 API server stopped")
    except Exception as e:
        print(f"❌ Error starting API server: {e}")

def start_ui_server():
    """Start a simple HTTP server for the UI."""
    ui_dir = Path(__file__).parent / "ui"
    os.chdir(ui_dir)
    
    print("🌐 Starting UI server...")
    try:
        subprocess.run([
            sys.executable, "-m", "http.server", "8080"
        ], check=True)
    except KeyboardInterrupt:
        print("\n🛑 UI server stopped")
    except Exception as e:
        print(f"❌ Error starting UI server: {e}")

def main():
    """Main startup function."""
    print("📄 DocStudio - Document Summarizer")
    print("=" * 50)
    
    # Check dependencies
    if not check_dependencies():
        return
    
    # Check if API server is already running
    try:
        import requests
        response = requests.get("http://localhost:8000/health", timeout=2)
        if response.status_code == 200:
            print("⚠️  API server is already running on port 8000")
            api_running = True
        else:
            api_running = False
    except:
        api_running = False
    
    if not api_running:
        # Start API server in a separate thread
        api_thread = threading.Thread(target=start_api_server, daemon=True)
        api_thread.start()
        
        # Wait for API server to start
        print("⏳ Waiting for API server to start...")
        time.sleep(3)
    
    # Start UI server in a separate thread
    ui_thread = threading.Thread(target=start_ui_server, daemon=True)
    ui_thread.start()
    
    # Wait a moment for servers to start
    time.sleep(2)
    
    print("\n🎉 DocStudio is ready!")
    print("=" * 50)
    print("📊 API Documentation: http://localhost:8000/docs")
    print("🌐 Web Interface: http://localhost:8080")
    print("🔍 Health Check: http://localhost:8000/health")
    print("\n💡 Press Ctrl+C to stop all servers")
    
    # Open web interface in browser
    try:
        webbrowser.open("http://localhost:8080")
    except:
        pass
    
    try:
        # Keep the main thread alive
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        print("\n🛑 Shutting down DocStudio...")
        print("✅ Goodbye!")

if __name__ == "__main__":
    main() 