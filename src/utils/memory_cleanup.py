"""
Memory cleanup utilities for the Signal Analyzer application.
This module provides easy-to-use functions for cleaning up memory leaks.
"""

import gc
import matplotlib.pyplot as plt
from src.utils.logger import app_logger

def cleanup_memory(log=True):
    """
    Easy memory cleanup - call anywhere!
    
    Args:
        log (bool): Whether to log the cleanup action
    """
    try:
        # Close all matplotlib figures to prevent memory leaks
        plt.close('all')
        
        # Force garbage collection to free up memory immediately
        collected = gc.collect()
        
        if log:
            app_logger.debug(f"🧹 Memory cleaned up - collected {collected} objects")
            
    except Exception as e:
        if log:
            app_logger.warning(f"Memory cleanup warning: {str(e)}")

def cleanup_variables(*variables):
    """
    Clean up specific variables by deleting them.
    
    Args:
        *variables: Variable names to delete
    """
    import sys
    frame = sys._getframe(1)
    
    for var_name in variables:
        if var_name in frame.f_locals:
            del frame.f_locals[var_name]
    
    gc.collect()

def setup_auto_cleanup(master_widget, interval_ms=30000):
    """
    Set up automatic memory cleanup on a timer.
    
    Args:
        master_widget: The main tkinter widget (root or master)
        interval_ms (int): Cleanup interval in milliseconds (default: 30 seconds)
    """
    def auto_cleanup():
        try:
            cleanup_memory(log=False)  # Silent cleanup
            app_logger.debug("🕒 Auto-cleanup completed")
        except Exception as e:
            app_logger.warning(f"Auto-cleanup error: {str(e)}")
        finally:
            # Schedule next cleanup
            master_widget.after(interval_ms, auto_cleanup)
    
    # Start the cleanup timer
    master_widget.after(interval_ms, auto_cleanup)
    app_logger.info(f"Auto-cleanup timer started (every {interval_ms/1000} seconds)")

def with_cleanup(func):
    """
    Decorator that automatically cleans up memory after function execution.
    
    Usage:
        @with_cleanup
        def my_function():
            # Your code here
            pass
    """
    def wrapper(*args, **kwargs):
        try:
            result = func(*args, **kwargs)
            return result
        finally:
            cleanup_memory(log=False)
    return wrapper

def memory_cleanup_on_exit(func):
    """
    Decorator for cleanup when exiting/closing windows.
    """
    def wrapper(*args, **kwargs):
        try:
            return func(*args, **kwargs)
        finally:
            cleanup_memory()
            app_logger.info("Exit cleanup completed")
    return wrapper