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
        self.window_counters = {
            'baseline': 0,
            'normalization': 0,
            'integration': 0
        }
        
        app_logger.debug("Window manager initialized")

    def set_data(self, time_data, data):
        """Set data for processing"""
        self.time_data = time_data
        self.data = data
        self.processed_data = data.copy()
        
        # Update all existing windows with new data
        self.update_all_windows()
        
        app_logger.debug("Data updated in window manager")

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
            
            app_logger.info(f"Baseline window opened (preserve_main={preserve_main})")
            
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
            
            # Set data based on what's available
            if preserve_main:
                window.set_data(self.time_data, self.data)
            else:
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

    # In window_manager.py, update the open_integration_window method

    def open_integration_window(self, preserve_main=True):
        """Open integration window with consistent plot style"""
        if self.data is None:
            messagebox.showwarning("Warning", "No data loaded")
            return
            
        try:
            # Get current y-axis limits from main plot if possible
            ylim = None
            if hasattr(self.parent, 'ax'):
                ylim = self.parent.ax.get_ylim()
            
            # Create and setup integration window
            window = IntegrationWindow(self.parent, self.handle_integration_result)
            window.set_data(self.time_data, self.data, ylim=ylim)
            
            # Store window reference
            self.windows['integration'] = window
            
        except Exception as e:
            app_logger.error(f"Error opening integration window: {str(e)}")
            raise

    def update_all_windows(self):
        """Update all open windows with current data"""
        try:
            for window_key, window in self.windows.items():
                if window and window.winfo_exists():
                    if '_' in window_key:  # Separate window
                        window.set_data(self.time_data, self.data)
                    else:  # Main processing window
                        window.set_data(self.time_data, self.processed_data or self.data)
            
            app_logger.debug("All windows updated with new data")
            
        except Exception as e:
            app_logger.error(f"Error updating windows: {str(e)}")
            raise

    def close_all_windows(self):
        """Close all open processing windows"""
        try:
            for window in self.windows.values():
                if window and window.winfo_exists():
                    window.destroy()
                    
            self.windows.clear()
            
            # Reset window counters
            for key in self.window_counters:
                self.window_counters[key] = 0
                
            app_logger.info("All windows closed")
            
        except Exception as e:
            app_logger.error(f"Error closing windows: {str(e)}")
            raise

    def on_baseline_update(self, processed_data):
        """Handle baseline correction updates"""
        self.processed_data = processed_data
        app_logger.info("Baseline correction applied")
        self._update_dependent_windows('baseline')

    def on_normalization_update(self, processed_data):
        """Handle normalization updates"""
        self.processed_data = processed_data
        app_logger.info("Normalization applied")
        self._update_dependent_windows('normalization')

    def on_integration_update(self, results):
        """Handle integration updates"""
        if isinstance(results, dict) and 'integral_value' in results:
            app_logger.info(f"Integration results: {results['integral_value']:.6f}")

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
            if not window or not window.winfo_exists():
                closed_windows.append(key)
        
        for key in closed_windows:
            del self.windows[key]
        
        app_logger.debug(f"Cleaned up {len(closed_windows)} closed windows")