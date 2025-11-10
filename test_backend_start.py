#!/usr/bin/env python
"""Test script to start backend and show any errors"""
import sys
import os
from pathlib import Path

# Set environment variables
os.environ['FLASK_APP'] = 'backend/app.py'
os.environ['FLASK_ENV'] = 'development'
os.environ['DATABASE_URL'] = 'sqlite:///test.db'
os.environ['FRONTEND_URL'] = 'http://localhost:3000,http://localhost:3001'

# Add project root to path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

try:
    print("Importing Flask app...")
    from backend.app import app
    print("✓ App imported successfully")
    
    print("\nStarting Flask server on http://0.0.0.0:5000...")
    print("Press Ctrl+C to stop")
    print("-" * 50)
    
    app.run(host='0.0.0.0', port=5000, debug=True)
except Exception as e:
    print(f"\n✗ ERROR: {e}", file=sys.stderr)
    import traceback
    traceback.print_exc()
    sys.exit(1)

