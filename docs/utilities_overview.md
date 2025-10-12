# Utilities Overview

## Overview

The utilities module provides essential supporting functionality for the DataChaEnhanced application. These utilities handle logging, hot reloading, analysis history management, and other core services that support the main application functionality.

**Main Components**:
- **Logger** (`logger.py`) - Comprehensive logging system
- **Hot Reload** (`hot_reload.py`) - Development hot reload system
- **Analysis History Manager** (`analysis_history_manager.py`) - Analysis history tracking
- **Memory Management** - Memory optimization and cleanup
- **Performance Monitoring** - Performance tracking and optimization

## Logger

### Overview
The logger provides comprehensive logging functionality for the application with configurable output levels and formatting.

**File**: `src/utils/logger.py`  
**Main Class**: `AppLogger`

### Key Features
- **Singleton Pattern**: Single logger instance across the application
- **Configurable Levels**: DEBUG, INFO, WARNING, ERROR, CRITICAL
- **Detailed Formatting**: Timestamp, level, and message formatting
- **Console Output**: Real-time logging to console
- **Thread Safe**: Safe for multi-threaded applications

### Implementation
```python
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
        self.logger.setLevel(logging.DEBUG)

        # Create console handler with detailed formatting
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setLevel(logging.DEBUG)
        
        # Create a detailed format for terminal output
        formatter = logging.Formatter(
            '%(asctime)s | %(levelname)-8s | %(message)s',
            datefmt='%Y-%m-%d %H:%M:%S'
        )
        console_handler.setFormatter(formatter)
        
        self.logger.addHandler(console_handler)
```

### Usage
```python
from src.utils.logger import app_logger

# Log different levels
app_logger.debug("Debug message")
app_logger.info("Information message")
app_logger.warning("Warning message")
app_logger.error("Error message")
app_logger.critical("Critical message")

# Log with context
app_logger.info(f"Processing file: {filename}")
app_logger.error(f"Error processing {filename}: {str(e)}")
```

## Hot Reload System

### Overview
The hot reload system provides automatic code reloading during development, allowing developers to see changes without restarting the application.

**File**: `src/utils/hot_reload.py`  
**Main Class**: `HotReloadManager`

### Key Features
- **File Watching**: Monitors Python files for changes
- **Automatic Reloading**: Reloads modules when files change
- **Thread Safe**: Safe reloading in background thread
- **Exclusion Support**: Excludes critical system modules
- **Callback Support**: Custom callbacks after reload

### Implementation
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
```

### File Watching
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

### Reload Processing
```python
def _process_reloads(self):
    """Process reload queue in background thread."""
    while self.running:
        if self.reload_queue:
            file_path = self.reload_queue.pop()
            self._reload_module(file_path)
        
        time.sleep(0.1)  # Check every 100ms

def _reload_module(self, file_path: str):
    """Reload a specific module."""
    try:
        # Convert file path to module name
        module_name = self._file_path_to_module_name(file_path)
        
        if module_name in self.exclude_modules:
            return
        
        # Reload the module
        if module_name in sys.modules:
            importlib.reload(sys.modules[module_name])
            logger.info(f"Reloaded module: {module_name}")
            
            # Call callback if provided
            if self.callback:
                self.callback(module_name)
        
    except Exception as e:
        logger.error(f"Failed to reload {file_path}: {e}")
```

### Usage
```python
from src.utils.hot_reload import HotReloadManager

# Initialize hot reload manager
reload_manager = HotReloadManager(
    project_root=".",
    callback=lambda module: print(f"Reloaded: {module}")
)

# Start hot reloading
reload_manager.start()

# Stop hot reloading
reload_manager.stop()
```

## Analysis History Manager

### Overview
The analysis history manager tracks and displays analysis results with timestamps, providing a comprehensive history of all analyses performed.

**File**: `src/utils/analysis_history_manager.py`  
**Main Class**: `AnalysisHistoryManager`

### Key Features
- **History Tracking**: Tracks all analysis results
- **Timestamp Management**: Records analysis timestamps
- **Duplicate Prevention**: Prevents duplicate entries
- **GUI Integration**: Provides history display dialog
- **Data Export**: Export history to various formats

### Implementation
```python
class AnalysisHistoryManager:
    """
    Manages the history of analyses performed in the application.
    Stores and displays analysis results with timestamps.
    """
    def __init__(self, parent):
        self.parent = parent
        self.history_entries = []
        self.history_window = None
        
    def add_entry(self, filename, results, analysis_type="manual"):
        """
        Add a new entry to the analysis history.
        
        Args:
            filename (str): The name of the analyzed file
            results (dict): The analysis results including integral values
            analysis_type (str): Type of analysis ("manual" or "auto")
        """
        try:
            # Create timestamp
            timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            
            # Extract basename if full path is provided
            basename = os.path.basename(filename) if filename else "Unknown"
            
            # Extract relevant data from results
            integral_value = results.get('integral_value', 'N/A')
            if isinstance(integral_value, str) and ',' in integral_value:
                integral_value = integral_value.split(',')[0].strip()
                
            hyperpol_area = results.get('hyperpol_area', 'N/A')
            depol_area = results.get('depol_area', 'N/A')
            capacitance = results.get('capacitance_nF', 'N/A')
            
            # Get V2 value
            v2_voltage = "N/A"
            if hasattr(self.parent, 'action_potential_processor'):
                processor = self.parent.action_potential_processor
                if processor and hasattr(processor, 'params'):
                    v2_voltage = f"{processor.params.get('V2', 'N/A')} mV"
            
            # Create the entry
            entry = {
                'timestamp': timestamp,
                'filename': basename,
                'integral_value': integral_value,
                'hyperpol_area': hyperpol_area,
                'depol_area': depol_area,
                'capacitance_nF': capacitance,
                'v2_voltage': v2_voltage,
                'analysis_type': analysis_type
            }
            
            # Check for duplicates within the last 2 seconds
            duplicate = False
            if self.history_entries:
                last_entry = self.history_entries[-1]
                if (last_entry['filename'] == basename and 
                    abs(datetime.datetime.strptime(last_entry['timestamp'], "%Y-%m-%d %H:%M:%S") - 
                        datetime.datetime.strptime(timestamp, "%Y-%m-%d %H:%M:%S")).total_seconds() < 5):
                    # If this is a manual analysis that follows an auto one, replace the entry
                    if analysis_type == "manual" and last_entry.get('analysis_type') == "auto":
                        self.history_entries[-1] = entry
                        app_logger.info(f"Updated existing entry for {basename} (converted auto to manual)")
                        duplicate = True
            
            # Add new entry if not a duplicate
            if not duplicate:
                self.history_entries.append(entry)
                app_logger.info(f"Added history entry: {basename}, {integral_value} ({analysis_type})")
            
        except Exception as e:
            app_logger.error(f"Error adding history entry: {str(e)}")
```

### History Display
```python
def show_history_dialog(self):
    """Show dialog with analysis history."""
    if not self.history_entries:
        tk.messagebox.showinfo("History", "No analysis history available.")
        return
        
    if self.history_window and self.history_window.winfo_exists():
        self.history_window.focus_force()
        self.refresh_history_display()
        return
        
    try:
        # Create new window
        self.history_window = tk.Toplevel(self.parent.master)
        self.history_window.title("Analysis History")
        self.history_window.geometry("800x400")
        self.history_window.transient(self.parent.master)
        self.history_window.grab_set()
        
        # Create main frame
        main_frame = ttk.Frame(self.history_window)
        main_frame.pack(fill='both', expand=True, padx=10, pady=10)
        
        # Create treeview for history display
        columns = ('Timestamp', 'Filename', 'Integral Value', 'Hyperpol Area', 'Depol Area', 'Capacitance', 'V2 Voltage', 'Type')
        self.history_tree = ttk.Treeview(main_frame, columns=columns, show='headings', height=15)
        
        # Configure columns
        for col in columns:
            self.history_tree.heading(col, text=col)
            self.history_tree.column(col, width=100, anchor='center')
        
        # Add scrollbar
        scrollbar = ttk.Scrollbar(main_frame, orient='vertical', command=self.history_tree.yview)
        self.history_tree.configure(yscrollcommand=scrollbar.set)
        
        # Pack widgets
        self.history_tree.pack(side='left', fill='both', expand=True)
        scrollbar.pack(side='right', fill='y')
        
        # Populate history
        self.refresh_history_display()
        
        # Add buttons
        button_frame = ttk.Frame(main_frame)
        button_frame.pack(fill='x', pady=(10, 0))
        
        ttk.Button(button_frame, text="Export to CSV", 
                  command=self.export_history_csv).pack(side='left', padx=5)
        ttk.Button(button_frame, text="Clear History", 
                  command=self.clear_history).pack(side='left', padx=5)
        ttk.Button(button_frame, text="Close", 
                  command=self.history_window.destroy).pack(side='right', padx=5)
        
    except Exception as e:
        app_logger.error(f"Error creating history dialog: {str(e)}")
        tk.messagebox.showerror("Error", f"Failed to create history dialog: {str(e)}")
```

### History Export
```python
def export_history_csv(self):
    """Export history to CSV file."""
    if not self.history_entries:
        tk.messagebox.showwarning("No Data", "No history entries to export.")
        return
    
    try:
        # Ask for save location
        filename = tk.filedialog.asksaveasfilename(
            defaultextension=".csv",
            filetypes=[("CSV files", "*.csv"), ("All files", "*.*")],
            title="Save History as CSV"
        )
        
        if not filename:
            return
        
        # Write CSV file
        with open(filename, 'w', newline='', encoding='utf-8') as csvfile:
            fieldnames = ['timestamp', 'filename', 'integral_value', 'hyperpol_area', 
                         'depol_area', 'capacitance_nF', 'v2_voltage', 'analysis_type']
            writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
            
            writer.writeheader()
            for entry in self.history_entries:
                writer.writerow(entry)
        
        tk.messagebox.showinfo("Success", f"History exported to {filename}")
        
    except Exception as e:
        app_logger.error(f"Error exporting history: {str(e)}")
        tk.messagebox.showerror("Error", f"Failed to export history: {str(e)}")
```

## Memory Management

### Overview
Memory management utilities provide memory optimization and cleanup functionality to prevent memory leaks and improve application performance.

### Key Features
- **Memory Monitoring**: Track memory usage
- **Garbage Collection**: Automatic cleanup
- **Weak References**: Prevent circular references
- **Memory Optimization**: Optimize data structures

### Implementation
```python
import gc
import psutil
import weakref
from typing import Any, Dict, List

class MemoryManager:
    """Memory management utilities"""
    
    def __init__(self):
        self.memory_usage = {}
        self.weak_refs = weakref.WeakSet()
    
    def get_memory_usage(self):
        """Get current memory usage"""
        process = psutil.Process()
        memory_info = process.memory_info()
        return {
            'rss': memory_info.rss / 1024 / 1024,  # MB
            'vms': memory_info.vms / 1024 / 1024,  # MB
            'percent': process.memory_percent()
        }
    
    def cleanup_memory(self):
        """Force garbage collection"""
        collected = gc.collect()
        return collected
    
    def add_weak_ref(self, obj):
        """Add weak reference to object"""
        self.weak_refs.add(obj)
    
    def get_weak_ref_count(self):
        """Get count of weak references"""
        return len(self.weak_refs)
```

## Performance Monitoring

### Overview
Performance monitoring utilities track application performance and provide optimization insights.

### Key Features
- **Performance Metrics**: Track execution times
- **Resource Usage**: Monitor CPU and memory usage
- **Bottleneck Detection**: Identify performance bottlenecks
- **Optimization Suggestions**: Provide optimization recommendations

### Implementation
```python
import time
import threading
from typing import Dict, List, Callable

class PerformanceMonitor:
    """Performance monitoring utilities"""
    
    def __init__(self):
        self.metrics = {}
        self.start_times = {}
        self.thread_local = threading.local()
    
    def start_timer(self, name: str):
        """Start timing an operation"""
        self.start_times[name] = time.time()
    
    def end_timer(self, name: str) -> float:
        """End timing an operation and return duration"""
        if name in self.start_times:
            duration = time.time() - self.start_times[name]
            del self.start_times[name]
            
            if name not in self.metrics:
                self.metrics[name] = []
            self.metrics[name].append(duration)
            
            return duration
        return 0.0
    
    def get_average_time(self, name: str) -> float:
        """Get average time for an operation"""
        if name in self.metrics and self.metrics[name]:
            return sum(self.metrics[name]) / len(self.metrics[name])
        return 0.0
    
    def get_performance_report(self) -> Dict[str, Dict]:
        """Get comprehensive performance report"""
        report = {}
        for name, times in self.metrics.items():
            report[name] = {
                'count': len(times),
                'total_time': sum(times),
                'average_time': sum(times) / len(times),
                'min_time': min(times),
                'max_time': max(times)
            }
        return report
```

## Configuration Management

### Overview
Configuration management utilities handle application configuration, settings, and preferences.

### Key Features
- **Settings Storage**: Store application settings
- **Configuration Validation**: Validate configuration values
- **Default Values**: Provide default configurations
- **Settings Persistence**: Save and load settings

### Implementation
```python
import json
import os
from typing import Dict, Any, Optional

class ConfigManager:
    """Configuration management utilities"""
    
    def __init__(self, config_file: str = "config.json"):
        self.config_file = config_file
        self.config = self.load_config()
    
    def load_config(self) -> Dict[str, Any]:
        """Load configuration from file"""
        if os.path.exists(self.config_file):
            try:
                with open(self.config_file, 'r') as f:
                    return json.load(f)
            except Exception as e:
                app_logger.error(f"Error loading config: {e}")
                return self.get_default_config()
        return self.get_default_config()
    
    def save_config(self):
        """Save configuration to file"""
        try:
            with open(self.config_file, 'w') as f:
                json.dump(self.config, f, indent=2)
        except Exception as e:
            app_logger.error(f"Error saving config: {e}")
    
    def get(self, key: str, default: Any = None) -> Any:
        """Get configuration value"""
        return self.config.get(key, default)
    
    def set(self, key: str, value: Any):
        """Set configuration value"""
        self.config[key] = value
        self.save_config()
    
    def get_default_config(self) -> Dict[str, Any]:
        """Get default configuration"""
        return {
            'logging_level': 'INFO',
            'auto_save': True,
            'theme': 'default',
            'window_size': [800, 600],
            'recent_files': [],
            'export_format': 'xlsx'
        }
```

## Error Handling

### Overview
Error handling utilities provide comprehensive error management and recovery mechanisms.

### Key Features
- **Error Logging**: Log errors with context
- **Error Recovery**: Attempt error recovery
- **User Notifications**: Notify users of errors
- **Error Reporting**: Generate error reports

### Implementation
```python
import traceback
from typing import Optional, Callable

class ErrorHandler:
    """Error handling utilities"""
    
    def __init__(self):
        self.error_callbacks = []
    
    def add_error_callback(self, callback: Callable):
        """Add error callback"""
        self.error_callbacks.append(callback)
    
    def handle_error(self, error: Exception, context: str = ""):
        """Handle error with logging and callbacks"""
        error_msg = f"Error in {context}: {str(error)}"
        app_logger.error(error_msg)
        app_logger.error(traceback.format_exc())
        
        # Call error callbacks
        for callback in self.error_callbacks:
            try:
                callback(error, context)
            except Exception as e:
                app_logger.error(f"Error in error callback: {e}")
    
    def safe_execute(self, func: Callable, *args, **kwargs):
        """Safely execute function with error handling"""
        try:
            return func(*args, **kwargs)
        except Exception as e:
            self.handle_error(e, f"Function {func.__name__}")
            return None
```

## Integration with Main Application

### Logger Integration
```python
# In main application
from src.utils.logger import app_logger

class SignalAnalyzerApp:
    def __init__(self, master):
        app_logger.info("Initializing Signal Analyzer Application")
        # ... rest of initialization
```

### Hot Reload Integration
```python
# In main application
from src.utils.hot_reload import HotReloadManager

class SignalAnalyzerApp:
    def __init__(self, master):
        # ... other initialization
        
        # Setup hot reload for development
        if self.is_development_mode():
            self.reload_manager = HotReloadManager(
                project_root=".",
                callback=self.on_module_reloaded
            )
            self.reload_manager.start()
    
    def on_module_reloaded(self, module_name):
        """Handle module reload"""
        app_logger.info(f"Module reloaded: {module_name}")
        # Refresh UI or reinitialize components
```

### History Manager Integration
```python
# In main application
from src.utils.analysis_history_manager import AnalysisHistoryManager

class SignalAnalyzerApp:
    def __init__(self, master):
        # ... other initialization
        
        # Setup history manager
        self.history_manager = AnalysisHistoryManager(self)
    
    def on_analysis_complete(self, results):
        """Handle analysis completion"""
        # Add to history
        self.history_manager.add_entry(
            filename=self.current_filename,
            results=results,
            analysis_type="manual"
        )
```

## Troubleshooting

### Common Issues

#### 1. Logger Issues
**Symptoms**: Logs not appearing or incorrect formatting
**Solutions**:
- Check logger configuration
- Verify log level settings
- Check console output redirection
- Enable debug logging

#### 2. Hot Reload Issues
**Symptoms**: Files not reloading or application crashes
**Solutions**:
- Check file permissions
- Verify module dependencies
- Check for circular imports
- Disable hot reload for problematic modules

#### 3. History Manager Issues
**Symptoms**: History not saving or displaying incorrectly
**Solutions**:
- Check file permissions
- Verify data format
- Check for memory issues
- Clear corrupted history

### Debugging Tools

#### 1. Logger Debugging
```python
def debug_logger():
    """Debug logger configuration"""
    print(f"Logger level: {app_logger.level}")
    print(f"Handlers: {app_logger.handlers}")
    print(f"Effective level: {app_logger.getEffectiveLevel()}")
```

#### 2. Memory Debugging
```python
def debug_memory():
    """Debug memory usage"""
    memory_manager = MemoryManager()
    usage = memory_manager.get_memory_usage()
    print(f"Memory usage: {usage}")
    print(f"Weak references: {memory_manager.get_weak_ref_count()}")
```

## API Reference

### AppLogger Methods
- `get_logger()`: Get the configured logger
- `debug(message)`: Log debug message
- `info(message)`: Log info message
- `warning(message)`: Log warning message
- `error(message)`: Log error message
- `critical(message)`: Log critical message

### HotReloadManager Methods
- `__init__(project_root, callback)`: Initialize hot reload manager
- `start()`: Start hot reload system
- `stop()`: Stop hot reload system
- `queue_reload(file_path)`: Queue file for reload
- `_reload_module(file_path)`: Reload specific module

### AnalysisHistoryManager Methods
- `__init__(parent)`: Initialize history manager
- `add_entry(filename, results, analysis_type)`: Add history entry
- `show_history_dialog()`: Show history dialog
- `export_history_csv()`: Export history to CSV
- `clear_history()`: Clear all history entries

### Key Attributes
- `logger`: Logger instance
- `history_entries`: List of history entries
- `watched_modules`: Modules being watched
- `config`: Configuration dictionary
- `memory_usage`: Memory usage tracking
