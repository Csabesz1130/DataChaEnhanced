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
            self._setup_logger()

    def _setup_logger(self):
        """Setup detailed logging configuration"""
        self.logger = logging.getLogger('SignalAnalysisApp')
        # Use INFO level by default for production (reduces startup overhead)
        default_level = logging.INFO
        self.logger.setLevel(default_level)

        # Lazy handler initialization - only create when first log is written
        self._handler_created = False
        self._pending_level = default_level
        
    def _ensure_handler(self):
        """Create handler on first use to improve startup time"""
        if not self._handler_created:
            # Create console handler with simpler formatting for production
            console_handler = logging.StreamHandler(sys.stdout)
            console_handler.setLevel(self._pending_level)
            
            # Use simpler format for better performance
            formatter = logging.Formatter(
                '%(asctime)s | %(levelname)s | %(message)s',
                datefmt='%H:%M:%S'
            )
            console_handler.setFormatter(formatter)
            
            self.logger.addHandler(console_handler)
            self._handler_created = True

    def get_logger(self):
        """Get the configured logger with lazy initialization wrapper"""
        # Return a wrapper that ensures handler is created on first use
        return LazyLoggerWrapper(self)

class LazyLoggerWrapper:
    """Wrapper that ensures handler is created on first log call"""
    def __init__(self, app_logger_instance):
        self.app_logger_instance = app_logger_instance
        self.logger = app_logger_instance.logger
        
    def __getattr__(self, name):
        # Ensure handler is created before any logging method is called
        if name in ['debug', 'info', 'warning', 'error', 'critical', 'exception', 'log']:
            self.app_logger_instance._ensure_handler()
        return getattr(self.logger, name)

# Create logger instance
logger_instance = AppLogger()
app_logger = logger_instance.get_logger()

# Export the logger
__all__ = ['app_logger']