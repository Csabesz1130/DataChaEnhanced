"""
Hot Reload Development System for DataChaEnhanced
===============================================
Location: src/utils/hot_reload.py

This module provides automatic code reloading during development.
Usage: Initialize once in your main application.
"""

import os
import sys
import time
import importlib
import threading
import logging
from pathlib import Path
from typing import Dict, Set, Callable, Optional
from watchdog.observers import Observer
from watchdog.events import FileSystemEventHandler, FileModifiedEvent

logger = logging.getLogger(__name__)

class HotReloadHandler(FileSystemEventHandler):
    """Handles file system events for hot reloading."""
    
    def __init__(self, reload_manager):
        self.reload_manager = reload_manager
        self.last_reload_time = {}
        self.reload_delay = 0.5  # Prevent multiple reloads within 0.5 seconds
    
    def on_modified(self, event):
        if isinstance(event, FileModifiedEvent):
            if event.src_path.endswith('.py'):
                current_time = time.time()
                last_time = self.last_reload_time.get(event.src_path, 0)
                
                if current_time - last_time > self.reload_delay:
                    self.last_reload_time[event.src_path] = current_time
                    self.reload_manager.queue_reload(event.src_path)

class HotReloadManager:
    """Manages automatic module reloading during development."""
    
    def __init__(self, project_root: str, callback: Optional[Callable] = None):
        self.project_root = Path(project_root)
        self.callback = callback  # Called after successful reload
        self.watched_modules: Dict[str, float] = {}
        self.reload_queue = set()
        self.observer = None
        self.reload_thread = None
        self.running = False
        self.enabled = True
        
        # Modules to exclude from reloading (critical system modules)
        self.exclude_modules = {
            'tkinter', 'matplotlib', 'numpy', 'scipy', 'pandas',
            '__main__', 'logging', 'threading', 'queue'
        }
        
        logger.info(f"HotReloadManager initialized for {self.project_root}")
    
    def start(self):
        """Start the hot reload system."""
        if not self.enabled or self.running:
            return
        
        try:
            # Start file watcher
            self.observer = Observer()
            handler = HotReloadHandler(self)
            
            # Watch the src directory
            src_path = self.project_root / 'src'
            if src_path.exists():
                self.observer.schedule(handler, str(src_path), recursive=True)
                logger.info(f"Watching {src_path} for changes")
            
            self.observer.start()
            
            # Start reload processor thread
            self.running = True
            self.reload_thread = threading.Thread(target=self._process_reloads, daemon=True)
            self.reload_thread.start()
            
            logger.info("Hot reload system started")
            
        except Exception as e:
            logger.error(f"Failed to start hot reload system: {e}")
            self.enabled = False
    
    def stop(self):
        """Stop the hot reload system."""
        self.running = False
        
        if self.observer:
            self.observer.stop()
            self.observer.join()
        
        logger.info("Hot reload system stopped")
    
    def queue_reload(self, file_path: str):
        """Queue a file for reloading."""
        if not self.enabled:
            return
            
        self.reload_queue.add(file_path)
        logger.debug(f"Queued for reload: {file_path}")
    
    def _process_reloads(self):
        """Process the reload queue in a separate thread."""
        while self.running:
            if self.reload_queue:
                # Process all queued reloads
                files_to_reload = list(self.reload_queue)
                self.reload_queue.clear()
                
                for file_path in files_to_reload:
                    try:
                        self._reload_module(file_path)
                    except Exception as e:
                        logger.error(f"Failed to reload {file_path}: {e}")
                
                # Call callback if provided
                if self.callback:
                    try:
                        self.callback()
                    except Exception as e:
                        logger.error(f"Reload callback failed: {e}")
            
            time.sleep(0.1)  # Check queue every 100ms
    
    def _reload_module(self, file_path: str):
        """Reload a specific module."""
        try:
            # Convert file path to module name
            rel_path = Path(file_path).relative_to(self.project_root)
            if not str(rel_path).startswith('src'):
                return
            
            # Remove src/ prefix and .py suffix
            module_path = str(rel_path)[4:]  # Remove 'src/'
            if module_path.endswith('.py'):
                module_path = module_path[:-3]
            
            module_name = module_path.replace(os.sep, '.')
            
            # Skip if module should be excluded
            for exclude in self.exclude_modules:
                if module_name.startswith(exclude):
                    logger.debug(f"Skipping excluded module: {module_name}")
                    return
            
            # Find the module in sys.modules
            full_module_name = None
            for name in sys.modules:
                if name.endswith(module_name) or name == module_name:
                    full_module_name = name
                    break
            
            if not full_module_name:
                logger.debug(f"Module {module_name} not found in sys.modules")
                return
            
            # Reload the module
            module = sys.modules[full_module_name]
            importlib.reload(module)
            
            logger.info(f"✅ Reloaded: {module_name}")
            
        except Exception as e:
            logger.error(f"Error reloading {file_path}: {e}")
    
    def disable(self):
        """Disable hot reloading temporarily."""
        self.enabled = False
        logger.info("Hot reload disabled")
    
    def enable(self):
        """Re-enable hot reloading."""
        self.enabled = True
        if not self.running:
            self.start()
        logger.info("Hot reload enabled")

# Global instance
_reload_manager: Optional[HotReloadManager] = None

def initialize_hot_reload(project_root: str, callback: Optional[Callable] = None):
    """Initialize the global hot reload manager."""
    global _reload_manager
    
    try:
        _reload_manager = HotReloadManager(project_root, callback)
        _reload_manager.start()
        return True
    except Exception as e:
        logger.error(f"Failed to initialize hot reload: {e}")
        return False

def stop_hot_reload():
    """Stop the global hot reload manager."""
    global _reload_manager
    if _reload_manager:
        _reload_manager.stop()
        _reload_manager = None

def get_reload_manager() -> Optional[HotReloadManager]:
    """Get the global reload manager instance."""
    return _reload_manager