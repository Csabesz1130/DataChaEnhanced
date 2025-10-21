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

# Try to import tkinterdnd2 for drag and drop functionality
DRAG_DROP_AVAILABLE = False
TkinterDnD = None

def _check_drag_drop_availability():
    """Check if tkinterdnd2 is available"""
    global DRAG_DROP_AVAILABLE, TkinterDnD
    try:
        from tkinterdnd2 import TkinterDnD
        DRAG_DROP_AVAILABLE = True
        return True
    except ImportError:
        DRAG_DROP_AVAILABLE = False
        return False

# Initial check
_check_drag_drop_availability()

def main():
    """Main application entry point"""
    try:
        # Create the main window with drag and drop support if available
        if _check_drag_drop_availability():
            root = TkinterDnD.Tk()
        else:
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
        app_logger.info("🔥 Hot reload is DISABLED by default - use the 'Hot Reload' button to enable if needed")
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