#!/usr/bin/env python3
"""
Main entry point for the Signal Analyzer application
"""

import sys
import os

# Optimize path setup - only add if not already present
current_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if current_dir not in sys.path:
    sys.path.insert(0, current_dir)

# Import only what's needed at startup
import tkinter as tk

def main():
    """Main application entry point"""
    try:
        # Create the main window
        root = tk.Tk()
        root.title("Signal Analyzer - Excel Learning Enhanced")
        root.geometry("1200x800")
        
        # Import app class only when needed
        from src.gui.app import SignalAnalyzerApp
        
        # Create the application
        app = SignalAnalyzerApp(root)
        
        # Set up window close handler
        root.protocol("WM_DELETE_WINDOW", app.on_closing)
        
        # Import logger only when needed
        from src.utils.logger import app_logger
        
        # Start the application
        app_logger.info("Starting Signal Analyzer application")
        root.mainloop()
        
    except Exception as e:
        # Import logger for error handling if not already imported
        try:
            app_logger
        except NameError:
            from src.utils.logger import app_logger
        app_logger.critical(f"Application failed to start: {str(e)}")
        raise

if __name__ == "__main__":
    main()