# src/utils/__init__.py
# Empty file to make utils a package

# src/utils/logger.py
import logging
import sys
from datetime import datetime

class AppLogger:
    """Comprehensive terminal logger"""
    _instance = None
    _initialized = False

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super(AppLogger, cls).__new__(cls)
        return cls._instance

    def __init__(self):
        if not self._initialized:
            self._initialized = True
            self._handler_created = False
            self._setup_logger()

    def _setup_logger(self):
        """Setup optimized logging configuration with lazy handler creation"""
        self.logger = logging.getLogger('SignalAnalysisApp')
        self.logger.setLevel(logging.INFO)  # Use INFO level for better startup performance

    def _ensure_handler(self):
        """Create handler on first use to improve startup time"""
        if not self._handler_created:
            # Create console handler with simplified formatting
            console_handler = logging.StreamHandler(sys.stdout)
            console_handler.setLevel(logging.INFO)
            
            # Simpler formatter for better performance
            formatter = logging.Formatter(
                '%(asctime)s | %(levelname)s | %(message)s',
                datefmt='%H:%M:%S'
            )
            console_handler.setFormatter(formatter)
            
            self.logger.addHandler(console_handler)
            self._handler_created = True
    
    def get_logger(self):
        """Get the configured logger"""
        # Ensure handler is created on first log
        if not self._handler_created:
            self._ensure_handler()
        return self.logger

# Create logger instance
logger_instance = AppLogger()
app_logger = logger_instance.get_logger()

# Export the logger
__all__ = ['app_logger']