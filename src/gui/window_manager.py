import tkinter as tk
from tkinter import ttk, messagebox
from src.utils.logger import app_logger
from src.gui.plot_windows.baseline_window import BaselineCorrectionWindow
from src.gui.plot_windows.normalization_window import NormalizationWindow
from src.gui.plot_windows.integration_window import IntegrationWindow

class SignalWindowManager:
    def __init__(self, parent):
        self.parent = parent
        self.windows = {}
        self.data = None
        self.time_data = None
        self.processed_data = None
        
        # Create menu for window management
        self.setup_menu()
        
        # Initialize window counters for separate instances
        self.window_counters = {
            'baseline': 0,
            'normalization': 0,
            'integration': 0
        }
        
        app_logger.debug("Window manager initialized")

    def setup_menu(self):
        """Setup menu for window management"""
        menubar = tk.Menu(self.parent)
        self.parent.config(menu=menubar)
        
        # Create Windows menu
        window_menu = tk.Menu(menubar, tearoff=0)
        menubar.add_cascade(label="Windows", menu=window_menu)
        
        # Add window options
        window_menu.add_command(label="Baseline Correction",
                              command=self.open_baseline_window)
        window_menu.add_command(label="Normalization",
                              command=self.open_normalization_window)
        window_menu.add_command(label="Integration",
                              command=self.open_integration_window)
        window_menu.add_separator()
        window_menu.add_command(label="Close All Windows",
                              command=self.close_all_windows)

    def set_data(self, time_data, data):
        """Set data for processing"""
        self.time_data = time_data
        self.data = data
        self.processed_data = data.copy()

    def get_window_key(self, base_name, preserve_main=False):
        """Generate unique window key"""
        if preserve_main:
            self.window_counters[base_name] += 1
            return f"{base_name}_{self.window_counters[base_name]}"
        return base_name

    def open_baseline_window(self, preserve_main=False):
        """Open baseline correction window"""
        if self.data is None:
            app_logger.warning("No data loaded")
            messagebox.showwarning("Warning", "Please load data first")
            return
            
        try:
            # Create window with appropriate callback
            callback = None if preserve_main else self.on_baseline_update
            window = BaselineCorrectionWindow(self.parent, callback)
            window.set_data(self.time_data, self.data)
            
            # Store window reference
            window_key = self.get_window_key('baseline', preserve_main)
            self.windows[window_key] = window
            
            # Set window title to include instance number if preserved
            if preserve_main:
                window.title(f"Baseline Correction #{self.window_counters['baseline']}")
            
            app_logger.info(f"Baseline correction window opened (preserve_main={preserve_main})")
            
        except Exception as e:
            app_logger.error(f"Error opening baseline window: {str(e)}")
            raise

    def open_normalization_window(self, preserve_main=False):
        """Open normalization window"""
        if self.data is None:
            app_logger.warning("No data loaded")
            messagebox.showwarning("Warning", "Please load data first")
            return
            
        try:
            # Create window with appropriate callback
            callback = None if preserve_main else self.on_normalization_update
            window = NormalizationWindow(self.parent, callback)
            window.set_data(self.time_data, self.processed_data or self.data)
            
            # Store window reference
            window_key = self.get_window_key('normalization', preserve_main)
            self.windows[window_key] = window
            
            # Set window title to include instance number if preserved
            if preserve_main:
                window.title(f"Normalization #{self.window_counters['normalization']}")
            
            app_logger.info(f"Normalization window opened (preserve_main={preserve_main})")
            
        except Exception as e:
            app_logger.error(f"Error opening normalization window: {str(e)}")
            raise

    def open_integration_window(self, preserve_main=False):
        """Open integration window"""
        if self.data is None:
            app_logger.warning("No data loaded")
            messagebox.showwarning("Warning", "Please load data first")
            return
            
        try:
            # Create window with appropriate callback
            callback = None if preserve_main else self.on_integration_update
            window = IntegrationWindow(self.parent, callback)
            window.set_data(self.time_data, self.processed_data or self.data)
            
            # Store window reference
            window_key = self.get_window_key('integration', preserve_main)
            self.windows[window_key] = window
            
            # Set window title to include instance number if preserved
            if preserve_main:
                window.title(f"Integration #{self.window_counters['integration']}")
            
            app_logger.info(f"Integration window opened (preserve_main={preserve_main})")
            
        except Exception as e:
            app_logger.error(f"Error opening integration window: {str(e)}")
            raise

    def close_all_windows(self):
        """Close all open processing windows"""
        for window in self.windows.values():
            if window and window.winfo_exists():
                window.destroy()
        self.windows.clear()
        
        # Reset window counters
        for key in self.window_counters:
            self.window_counters[key] = 0
            
        app_logger.info("All processing windows closed")

    def on_baseline_update(self, processed_data):
        """Handle baseline correction updates"""
        self.processed_data = processed_data
        app_logger.info("Baseline correction applied")
        
        # Update other windows if open
        self._update_dependent_windows('baseline')

    def on_normalization_update(self, processed_data):
        """Handle normalization updates"""
        self.processed_data = processed_data
        app_logger.info("Normalization applied")
        
        # Update other windows if open
        self._update_dependent_windows('normalization')

    def on_integration_update(self, results):
        """Handle integration updates"""
        app_logger.info(f"Integration results: {results['integral_value']:.6f}")
        
        # Here you might want to store or display the results
        # in your main application

    def _update_dependent_windows(self, source_window):
        """Update other windows after changes"""
        try:
            for window_key, window in self.windows.items():
                # Skip source window and separate plot windows
                if '_' in window_key or window_key == source_window:
                    continue
                    
                if window and window.winfo_exists():
                    window.set_data(self.time_data, self.processed_data)
            
            app_logger.debug("Dependent windows updated")
            
        except Exception as e:
            app_logger.error(f"Error updating dependent windows: {str(e)}")
            raise

    def cleanup_closed_windows(self):
        """Remove references to closed windows"""
        closed_windows = []
        for key, window in self.windows.items():
            if not window.winfo_exists():
                closed_windows.append(key)
        
        for key in closed_windows:
            del self.windows[key]