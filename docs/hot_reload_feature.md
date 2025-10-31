# Hot Reload Feature Documentation

## Overview

The hot reload feature in DataChaEnhanced provides automatic code reloading during development, allowing developers to modify Python code and see changes immediately without restarting the application. This feature is built using the `watchdog` library for file system monitoring and includes comprehensive state management, debouncing, and integration with drag and drop operations.

## System Architecture

### Core Components

1. **HotReloadManager** (`src/utils/hot_reload.py`)
   - Main hot reload system manager
   - Handles file system monitoring and module reloading
   - Manages reload queue and threading

2. **HotReloadHandler** (`src/utils/hot_reload.py`)
   - File system event handler
   - Monitors file changes and queues reloads
   - Implements debouncing to prevent rapid reloads

3. **Application Integration** (`src/gui/app.py`)
   - Hot reload UI controls and state management
   - Component refresh logic
   - Integration with drag and drop operations

### Dependencies

#### Required Libraries:
```python
# Core hot reload functionality
from watchdog.observers import Observer
from watchdog.events import FileSystemEventHandler, FileModifiedEvent

# Standard libraries
import os
import sys
import time
import importlib
import threading
import logging
from pathlib import Path
from typing import Dict, Set, Callable, Optional
```

#### Installation:
```bash
pip install watchdog
```

## Implementation Details

### 1. Hot Reload Manager

#### Core Manager Class:
```python
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
```

#### Start Method:
```python
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
```

### 2. File System Event Handler

#### Event Handler Class:
```python
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
```

### 3. Module Reloading Process

#### Reload Queue Processing:
```python
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
                    # Don't let callback errors crash the hot reload system
        
        time.sleep(0.1)  # Check queue every 100ms
```

#### Module Reloading Logic:
```python
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
        
        logger.info(f"Reloaded: {module_name}")
        
    except Exception as e:
        logger.error(f"Error reloading {file_path}: {e}")
```

### 4. Application Integration

#### Hot Reload Setup:
```python
def setup_hot_reload(self):
    """Setup hot reload system for development."""
    try:
        import os

        # Get project root directory (repository root that contains `src/`)
        # __file__ -> src/gui/app.py; we need three levels up -> project root
        project_root = os.path.dirname(
            os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        )

        # Initialize hot reload with callback to refresh components
        success = initialize_hot_reload(
            project_root=project_root, callback=self.on_code_reloaded
        )

        if success:
            app_logger.info("Hot reload enabled - modify code without restarting!")
        else:
            app_logger.warning("Hot reload failed to initialize")

    except Exception as e:
        app_logger.error(f"Hot reload setup failed: {e}")
```

#### Code Reload Callback:
```python
def on_code_reloaded(self):
    """Called after code is hot reloaded - refresh components if needed."""
    try:
        import time

        # Check if hot reload is enabled
        if not self._hot_reload_enabled:
            app_logger.debug("Hot reload disabled")
            return

        # Don't reload during drag and drop operations
        if hasattr(self, '_drag_drop_active') and self._drag_drop_active:
            app_logger.info("🖱️ Hot reload skipped - drag and drop operation in progress")
            return

        # Debounce rapid reloads
        current_time = time.time()
        if current_time - self._last_reload_time < self._reload_debounce_delay:
            app_logger.debug("Hot reload debounced - too frequent")
            return

        self._last_reload_time = current_time
        app_logger.info("Hot reload triggered - smart refresh...")

        # 1. Refresh analysis processors (most important for code changes)
        try:
            self._refresh_analysis_processors()
        except Exception as e:
            app_logger.error(f"Error refreshing analysis processors: {e}")

        # 2. Refresh UI components without tab switching (only if needed)
        try:
            self._refresh_ui_components_smart()
        except Exception as e:
            app_logger.error(f"Error refreshing UI components: {e}")

        # 3. Refresh plot and data if loaded (preserves UI state) - only if data exists
        if hasattr(self, "data") and self.data is not None:
            try:
                self._refresh_plot_and_data()
            except Exception as e:
                app_logger.error(f"Error refreshing plot and data: {e}")

        # 4. Refresh curve fitting if active
        try:
            self._refresh_curve_fitting()
        except Exception as e:
            app_logger.error(f"Error refreshing curve fitting: {e}")

        app_logger.info("Hot reload complete - all components refreshed")

    except Exception as e:
        app_logger.error(f"Error in reload callback: {e}")
        # Don't let hot reload errors crash the application
```

## Common Issues and Solutions

### 1. Module Import Errors

#### Problem: Module not found during reload
```python
# Error: ModuleNotFoundError: No module named 'src.analysis.action_potential'
```

**Solution:**
```python
def _reload_module(self, file_path: str):
    """Reload a specific module with proper error handling."""
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
        
        logger.info(f"Reloaded: {module_name}")
        
    except Exception as e:
        logger.error(f"Error reloading {file_path}: {e}")
```

### 2. Rapid Reload Issues

#### Problem: Multiple rapid reloads causing instability
```python
# Error: Too many reloads in short time
```

**Solution:**
```python
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
```

### 3. UI State Loss

#### Problem: UI state lost during hot reload
```python
# Error: Tab selection lost, plot state reset
```

**Solution:**
```python
def _refresh_ui_components_smart(self):
    """Refresh UI components without causing tab switching."""
    try:
        # Remember current tab selection
        current_tab_index = 0
        if hasattr(self, "notebook") and self.notebook.tabs():
            try:
                current_tab_index = self.notebook.index(self.notebook.select())
            except:
                current_tab_index = 0

        # Refresh action potential tab
        if hasattr(self, "action_potential_tab") and self.action_potential_tab:
            # Re-import the tab module
            import importlib
            import src.gui.action_potential_tab as apt_module

            importlib.reload(apt_module)

            # Recreate the tab with updated code
            self.action_potential_tab = apt_module.ActionPotentialTab(
                self.notebook, self.on_action_potential_analysis
            )

            # Replace in notebook while preserving tab order
            self._replace_tab_smart(
                "Action Potential", self.action_potential_tab.frame
            )
            self.tabs["action_potential"] = self.action_potential_tab
            app_logger.info("Action potential tab refreshed")

        # Restore current tab selection
        if hasattr(self, "notebook") and self.notebook.tabs():
            try:
                self.notebook.select(current_tab_index)
            except:
                pass

    except Exception as e:
        app_logger.error(f"Error refreshing UI components: {e}")
```

### 4. Drag and Drop Conflicts

#### Problem: Hot reload triggers during drag and drop
```python
# Error: Hot reload interferes with drag operations
```

**Solution:**
```python
def on_code_reloaded(self):
    """Called after code is hot reloaded - refresh components if needed."""
    try:
        # Check if hot reload is enabled
        if not self._hot_reload_enabled:
            app_logger.debug("Hot reload disabled")
            return

        # Don't reload during drag and drop operations
        if hasattr(self, '_drag_drop_active') and self._drag_drop_active:
            app_logger.info("🖱️ Hot reload skipped - drag and drop operation in progress")
            return

        # ... rest of reload logic ...
        
    except Exception as e:
        app_logger.error(f"Error in reload callback: {e}")
```

### 5. Memory Leaks

#### Problem: Memory leaks from repeated reloads
```python
# Error: Memory usage increases with each reload
```

**Solution:**
```python
def _refresh_analysis_processors(self):
    """Refresh analysis processors and related components."""
    try:
        # Re-import and refresh action potential processor
        if (
            hasattr(self, "action_potential_processor")
            and self.action_potential_processor
        ):
            # Re-import the module to get updated classes
            import importlib
            import src.analysis.action_potential as ap_module

            importlib.reload(ap_module)

            # Create new processor instance with updated code
            self.action_potential_processor = ap_module.ActionPotentialProcessor()
            app_logger.info("Action potential processor refreshed")

        # Refresh other analysis modules
        analysis_modules = [
            "src.analysis.linear_fit_subtractor",
            "src.analysis.curve_fitting",
            "src.analysis.signal_processing",
            "src.analysis.baseline_correction",
            "src.analysis.normalization",
            "src.analysis.spike_detection",
            "src.analysis.peak_detection",
        ]

        for module_name in analysis_modules:
            try:
                import importlib

                module = importlib.import_module(module_name)
                importlib.reload(module)
                app_logger.info(f"{module_name} refreshed")
            except Exception as e:
                app_logger.debug(f"{module_name} refresh skipped: {e}")

    except Exception as e:
        app_logger.error(f"Error refreshing analysis processors: {e}")
```

## What to Avoid

### 1. Reloading Critical System Modules

**❌ DON'T:**
```python
# Don't reload critical system modules
exclude_modules = set()  # This will cause crashes
```

**✅ DO:**
```python
# Always exclude critical system modules
exclude_modules = {
    'tkinter', 'matplotlib', 'numpy', 'scipy', 'pandas',
    '__main__', 'logging', 'threading', 'queue'
}
```

### 2. Ignoring Debouncing

**❌ DON'T:**
```python
# Don't ignore debouncing
def on_modified(self, event):
    if event.src_path.endswith('.py'):
        self.reload_manager.queue_reload(event.src_path)  # No debouncing
```

**✅ DO:**
```python
# Always implement debouncing
def on_modified(self, event):
    if isinstance(event, FileModifiedEvent):
        if event.src_path.endswith('.py'):
            current_time = time.time()
            last_time = self.last_reload_time.get(event.src_path, 0)
            
            if current_time - last_time > self.reload_delay:
                self.last_reload_time[event.src_path] = current_time
                self.reload_manager.queue_reload(event.src_path)
```

### 3. Not Preserving UI State

**❌ DON'T:**
```python
# Don't recreate UI without preserving state
def _refresh_ui_components(self):
    # Recreate all tabs without preserving selection
    self.setup_tabs()  # This will lose current tab
```

**✅ DO:**
```python
# Always preserve UI state
def _refresh_ui_components_smart(self):
    # Remember current tab selection
    current_tab_index = 0
    if hasattr(self, "notebook") and self.notebook.tabs():
        try:
            current_tab_index = self.notebook.index(self.notebook.select())
        except:
            current_tab_index = 0
    
    # ... refresh logic ...
    
    # Restore current tab selection
    if hasattr(self, "notebook") and self.notebook.tabs():
        try:
            self.notebook.select(current_tab_index)
        except:
            pass
```

### 4. Ignoring Error Handling

**❌ DON'T:**
```python
# Don't ignore errors in reload callback
def on_code_reloaded(self):
    # No error handling
    self._refresh_analysis_processors()
    self._refresh_ui_components()
```

**✅ DO:**
```python
# Always handle errors gracefully
def on_code_reloaded(self):
    try:
        # 1. Refresh analysis processors
        try:
            self._refresh_analysis_processors()
        except Exception as e:
            app_logger.error(f"Error refreshing analysis processors: {e}")

        # 2. Refresh UI components
        try:
            self._refresh_ui_components_smart()
        except Exception as e:
            app_logger.error(f"Error refreshing UI components: {e}")

    except Exception as e:
        app_logger.error(f"Error in reload callback: {e}")
        # Don't let hot reload errors crash the application
```

### 5. Not Coordinating with Other Systems

**❌ DON'T:**
```python
# Don't ignore other system states
def on_code_reloaded(self):
    # Always reload regardless of other operations
    self._refresh_all_components()
```

**✅ DO:**
```python
# Always check other system states
def on_code_reloaded(self):
    # Don't reload during drag and drop operations
    if hasattr(self, '_drag_drop_active') and self._drag_drop_active:
        app_logger.info("🖱️ Hot reload skipped - drag and drop operation in progress")
        return

    # Debounce rapid reloads
    current_time = time.time()
    if current_time - self._last_reload_time < self._reload_debounce_delay:
        app_logger.debug("Hot reload debounced - too frequent")
        return

    # ... proceed with reload ...
```

## Best Practices

### 1. Proper State Management

```python
class SignalAnalyzerApp:
    def __init__(self, master):
        # Hot reload state management
        self._hot_reload_enabled = False  # DISABLED BY DEFAULT
        self._last_reload_time = 0
        self._reload_debounce_delay = 1.0  # 1 second debounce
        
        # Drag and drop coordination
        self._drag_drop_active = False
```

### 2. Smart Component Refresh

```python
def _refresh_ui_components_smart(self):
    """Refresh UI components without causing tab switching."""
    try:
        # Remember current tab selection
        current_tab_index = 0
        if hasattr(self, "notebook") and self.notebook.tabs():
            try:
                current_tab_index = self.notebook.index(self.notebook.select())
            except:
                current_tab_index = 0

        # Refresh components
        self._refresh_tab_smart("filter", "Filters")
        self._refresh_tab_smart("analysis", "Analysis")
        self._refresh_tab_smart("view", "View")

        # Restore current tab selection
        if hasattr(self, "notebook") and self.notebook.tabs():
            try:
                self.notebook.select(current_tab_index)
            except:
                pass

    except Exception as e:
        app_logger.error(f"Error refreshing UI components: {e}")
```

### 3. Comprehensive Error Handling

```python
def on_code_reloaded(self):
    """Called after code is hot reloaded - refresh components if needed."""
    try:
        # Check if hot reload is enabled
        if not self._hot_reload_enabled:
            app_logger.debug("Hot reload disabled")
            return

        # Don't reload during drag and drop operations
        if hasattr(self, '_drag_drop_active') and self._drag_drop_active:
            app_logger.info("🖱️ Hot reload skipped - drag and drop operation in progress")
            return

        # Debounce rapid reloads
        current_time = time.time()
        if current_time - self._last_reload_time < self._reload_debounce_delay:
            app_logger.debug("Hot reload debounced - too frequent")
            return

        self._last_reload_time = current_time
        app_logger.info("Hot reload triggered - smart refresh...")

        # Refresh components with individual error handling
        try:
            self._refresh_analysis_processors()
        except Exception as e:
            app_logger.error(f"Error refreshing analysis processors: {e}")

        try:
            self._refresh_ui_components_smart()
        except Exception as e:
            app_logger.error(f"Error refreshing UI components: {e}")

        if hasattr(self, "data") and self.data is not None:
            try:
                self._refresh_plot_and_data()
            except Exception as e:
                app_logger.error(f"Error refreshing plot and data: {e}")

        try:
            self._refresh_curve_fitting()
        except Exception as e:
            app_logger.error(f"Error refreshing curve fitting: {e}")

        app_logger.info("Hot reload complete - all components refreshed")

    except Exception as e:
        app_logger.error(f"Error in reload callback: {e}")
        # Don't let hot reload errors crash the application
```

### 4. User Control

```python
def toggle_hot_reload_ui(self):
    """Toggle hot reload with UI update."""
    enabled = self.toggle_hot_reload()
    button_text = "Hot Reload: ON" if enabled else "Hot Reload: OFF"
    self.hot_reload_button.config(text=button_text)
    
    # Show user feedback
    if enabled:
        app_logger.info("🔥 Hot reload ENABLED - app will auto-reload on code changes")
        # Show a brief message to user
        self.master.after(100, lambda: app_logger.info("⚠️  Hot reload is now active - changes to code will trigger automatic reloads"))
    else:
        app_logger.info("❌ Hot reload DISABLED - app will not auto-reload")
```

### 5. Thread Safety

```python
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
                    # Don't let callback errors crash the hot reload system
        
        time.sleep(0.1)  # Check queue every 100ms
```

## Testing and Debugging

### 1. Debug Logging

```python
def _reload_module(self, file_path: str):
    """Reload a specific module with comprehensive logging."""
    try:
        # Convert file path to module name
        rel_path = Path(file_path).relative_to(self.project_root)
        if not str(rel_path).startswith('src'):
            logger.debug(f"Skipping non-src file: {file_path}")
            return
        
        # Remove src/ prefix and .py suffix
        module_path = str(rel_path)[4:]  # Remove 'src/'
        if module_path.endswith('.py'):
            module_path = module_path[:-3]
        
        module_name = module_path.replace(os.sep, '.')
        logger.debug(f"Processing module: {module_name}")
        
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
        
        logger.info(f"Reloaded: {module_name}")
        
    except Exception as e:
        logger.error(f"Error reloading {file_path}: {e}")
```

### 2. Testing Hot Reload

```python
def test_hot_reload(self):
    """Test method to manually trigger hot reload."""
    try:
        app_logger.info("🧪 Manual hot reload test triggered")
        self.on_code_reloaded()
    except Exception as e:
        app_logger.error(f"Manual hot reload test failed: {e}")
```

## Performance Considerations

### 1. Debouncing

```python
class HotReloadHandler(FileSystemEventHandler):
    def __init__(self, reload_manager):
        self.reload_manager = reload_manager
        self.last_reload_time = {}
        self.reload_delay = 0.5  # Prevent multiple reloads within 0.5 seconds
```

### 2. Module Exclusion

```python
# Modules to exclude from reloading (critical system modules)
exclude_modules = {
    'tkinter', 'matplotlib', 'numpy', 'scipy', 'pandas',
    '__main__', 'logging', 'threading', 'queue'
}
```

### 3. Thread Management

```python
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
```

## Conclusion

The hot reload feature in DataChaEnhanced provides a powerful development tool that allows for rapid iteration and testing. Key aspects include:

- **File System Monitoring**: Uses watchdog to monitor file changes
- **Module Reloading**: Automatically reloads Python modules
- **State Preservation**: Maintains UI state during reloads
- **Error Handling**: Comprehensive error handling and recovery
- **User Control**: Toggle functionality for enabling/disabling
- **Thread Safety**: Proper threading for file monitoring and reloading
- **Performance**: Debouncing and module exclusion for efficiency

The implementation follows best practices for hot reloading in Python applications and provides a solid foundation for development productivity.
