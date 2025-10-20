#!/usr/bin/env python3
"""
Run the application with hot reload enabled
"""
import os
import sys
from pathlib import Path

# Set environment variable to enable hot reload
os.environ['ENABLE_HOT_RELOAD'] = '1'

# Add project root to Python path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

# Import and run the main application
if __name__ == "__main__":
    from src.main import main
    main()
