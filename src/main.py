#!/usr/bin/env python3
"""
Main entry point for the Signal Analyzer application
"""

import sys
import os

# Add the current directory to Python path to fix import issues
if os.path.dirname(os.path.dirname(os.path.abspath(__file__))) not in sys.path:
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

def main():
    """Main application entry point with lazy imports"""
    try:
        # Lazy imports inside main for better startup performance
        from src.gui.app import SignalAnalyzerApp
        import tkinter as tk
        from src.utils.logger import app_logger
        
        # Create the main window
        root = tk.Tk()
        root.title("Signal Analyzer - Excel Learning Enhanced")
        root.geometry("1200x800")
        
        # Create the application
        app = SignalAnalyzerApp(root)
        
        # Set up window close handler
        root.protocol("WM_DELETE_WINDOW", app.on_closing)
        
        # Start the application
        app_logger.info("Starting Signal Analyzer application")
        root.mainloop()
        
    except Exception as e:
        from src.utils.logger import app_logger
        app_logger.critical(f"Application failed to start: {str(e)}")
        raise

if __name__ == "__main__":
    main()