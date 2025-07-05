# src/main.py
import tkinter as tk
from src.gui.app import SignalAnalyzerApp  
from src.utils.logger import app_logger
from src.utils.auto_updater import AutoUpdater

def main():
    try:
        app_logger.info("Starting Signal Analyzer Application")
        root = tk.Tk()
        root.title("Signal Analyzer")
        
        # Set minimum window size
        root.minsize(1200, 800)
        
        # Start maximized
        root.state('zoomed')
        
        # Create and start application
        app = SignalAnalyzerApp(root)
        
        # Initialize and start auto-updater
        updater = AutoUpdater(root)
        updater.start_update_process()
        
        # Store updater reference in the app
        app.updater = updater
        
        # Start main event loop
        root.mainloop()
        
    except Exception as e:
        app_logger.critical(f"Application failed to start: {str(e)}", exc_info=True)
        raise

if __name__ == "__main__":
    main()