#!/usr/bin/env python3
"""
Launcher script for the Research Publications Chat Agent
"""

import subprocess
import sys
import os
from pathlib import Path

def check_requirements():
    """Check if all requirements are met."""
    
    # Check if .env file exists
    if not Path('.env').exists():
        print("❌ .env file not found!")
        print("Please create a .env file with your Elasticsearch credentials:")
        print("ES_HOST=your-elasticsearch-host")
        print("ES_USER=your-username")
        print("ES_PASS=your-password")
        return False
    
    # Check if virtual environment is activated
    if not hasattr(sys, 'real_prefix') and not (hasattr(sys, 'base_prefix') and sys.base_prefix != sys.prefix):
        print("⚠️  Virtual environment not detected!")
        print("Please activate your virtual environment first:")
        print("source venv/bin/activate")
        return False
    
    # Check if streamlit is installed
    try:
        import streamlit
        print("✅ Streamlit found")
    except ImportError:
        print("❌ Streamlit not installed!")
        print("Please install requirements:")
        print("pip install -r requirements.txt")
        return False
    
    # Check if chat agent components are available
    try:
        from chat_parser import ChatParser
        from query_builder import QueryBuilder
        from response_formatter import ResponseFormatter
        import agent_tools
        print("✅ Chat agent components found")
    except ImportError as e:
        print(f"❌ Chat agent components not available: {e}")
        return False
    
    return True

def main():
    """Main launcher function."""
    print("🔍 Research Publications Chat Agent Launcher")
    print("=" * 50)
    
    if not check_requirements():
        print("\n❌ Requirements not met. Please fix the issues above.")
        sys.exit(1)
    
    print("✅ All requirements met!")
    print("\n🚀 Starting Streamlit application...")
    print("The app will open in your browser at http://localhost:8501")
    print("Press Ctrl+C to stop the application")
    
    try:
        # Run streamlit app
        subprocess.run([
            sys.executable, "-m", "streamlit", "run", "streamlit_app.py",
            "--server.headless", "false",
            "--server.port", "8501",
            "--server.address", "localhost"
        ])
    except KeyboardInterrupt:
        print("\n\n👋 Application stopped by user")
    except Exception as e:
        print(f"\n❌ Error running application: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()