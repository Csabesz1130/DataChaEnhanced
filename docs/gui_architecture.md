# GUI Architecture

## Overview

The GUI architecture of DataChaEnhanced is built around a main application class that coordinates multiple specialized components for signal analysis, visualization, and data processing. The architecture follows a modular design with clear separation of concerns between data management, user interface, and analysis components.

**Main Application**: `src/gui/app.py`  
**Main Class**: `SignalAnalyzerApp`

## Architecture Components

### 1. Main Application (`SignalAnalyzerApp`)
- **Central Controller**: Coordinates all GUI components and data flow
- **Memory Management**: Optimized memory handling for large datasets
- **Data Management**: Property-based data access with automatic cleanup
- **Component Integration**: Integrates tabs, windows, and analysis tools

### 2. Window Manager (`SignalWindowManager`)
- **Window Management**: Manages specialized analysis windows
- **Data Synchronization**: Keeps windows synchronized with main data
- **Window Lifecycle**: Handles window creation, updates, and cleanup

### 3. Tab System
- **Filter Tab**: Signal filtering and preprocessing
- **Analysis Tab**: Data analysis and processing
- **View Tab**: Data visualization and inspection
- **Action Potential Tab**: Action potential analysis
- **AI Analysis Tab**: AI-powered analysis features
- **Excel Learning Tab**: Excel integration and learning

### 4. Plot System
- **Main Plot**: Central matplotlib-based plotting system
- **Interactive Features**: Zoom, pan, and selection capabilities
- **Memory Optimization**: Efficient handling of large datasets
- **Plot Windows**: Specialized windows for different analysis types

## Main Application Structure

### Initialization
```python
from src.gui.app import SignalAnalyzerApp
import tkinter as tk

# Create main window
root = tk.Tk()

# Initialize application
app = SignalAnalyzerApp(root)

# Start main loop
root.mainloop()
```

### Key Components
```python
class SignalAnalyzerApp:
    def __init__(self, master):
        # Data management
        self._data = None
        self._time_data = None
        self._filtered_data = None
        self._processed_data = None
        
        # Component managers
        self.window_manager = SignalWindowManager(self.master)
        self.history_manager = AnalysisHistoryManager(self)
        
        # UI components
        self.setup_main_layout()
        self.setup_menubar()
        self.setup_toolbar()
        self.setup_plot()
        self.setup_tabs()
        
        # Development features
        self.setup_hot_reload()
        self.setup_memory_management()
```

## Data Management

### Property-Based Data Access
```python
# Data properties with automatic cleanup
@property
def data(self):
    """Property getter for data"""
    return self._data

@data.setter
def data(self, value):
    """Property setter for data with automatic cleanup"""
    # Clean up old data
    if self._data is not None:
        old_size = self._data.nbytes if hasattr(self._data, "nbytes") else 0
        del self._data
        self._memory_usage -= old_size
    
    # Store new data
    if value is not None:
        self._data = np.asarray(value, dtype=np.float64)
        new_size = self._data.nbytes
        self._memory_usage += new_size
    
    # Force garbage collection
    gc.collect()
```

### Memory Management
```python
# Memory optimization features
def setup_memory_management(self):
    """Setup memory management systems"""
    self._memory_usage = 0
    self.active_figures = weakref.WeakSet()
    
    # Enable memory monitoring
    self.monitor_memory_usage()
    
    # Setup automatic cleanup
    self.setup_automatic_cleanup()

def monitor_memory_usage(self):
    """Monitor and log memory usage"""
    memory_mb = self._memory_usage / 1024 / 1024
    app_logger.debug(f"Current memory usage: {memory_mb:.1f} MB")
```

## Window Management System

### Window Manager Initialization
```python
from src.gui.window_manager import SignalWindowManager

# Initialize window manager
window_manager = SignalWindowManager(parent)

# Set data for processing
window_manager.set_data(time_data, data)
```

### Specialized Windows
```python
# Open baseline correction window
window_manager.open_baseline_window()

# Open normalization window
window_manager.open_normalization_window()

# Open integration window
window_manager.open_integration_window()

# Open multiple instances
window_manager.open_baseline_window(preserve_main=True)
```

### Window Data Synchronization
```python
# Update all windows with new data
window_manager.update_all_windows()

# Set data for specific window type
window_manager.set_data(time_data, data)

# Get window references
baseline_window = window_manager.windows.get('baseline')
normalization_window = window_manager.windows.get('normalization')
```

## Tab System Architecture

### Tab Initialization
```python
def setup_tabs(self):
    """Setup the tab system"""
    # Create notebook for tabs
    self.notebook = ttk.Notebook(self.main_frame)
    
    # Initialize tabs
    self.filter_tab = FilterTab(self.notebook, self)
    self.analysis_tab = AnalysisTab(self.notebook, self)
    self.view_tab = ViewTab(self.notebook, self)
    self.action_potential_tab = ActionPotentialTab(self.notebook, self)
    
    # Add tabs to notebook
    self.notebook.add(self.filter_tab, text="Filter")
    self.notebook.add(self.analysis_tab, text="Analysis")
    self.notebook.add(self.view_tab, text="View")
    self.notebook.add(self.action_potential_tab, text="Action Potential")
```

### Tab Communication
```python
# Tab-to-tab communication
def update_tabs(self, tab_name=None):
    """Update all tabs or specific tab"""
    if tab_name is None:
        # Update all tabs
        self.filter_tab.update_display()
        self.analysis_tab.update_display()
        self.view_tab.update_display()
        self.action_potential_tab.update_display()
    else:
        # Update specific tab
        getattr(self, f"{tab_name}_tab").update_display()
```

## Plot System

### Main Plot Setup
```python
def setup_plot(self):
    """Setup the main plotting system"""
    # Create matplotlib figure
    self.figure = Figure(figsize=(12, 8), dpi=100)
    self.ax = self.figure.add_subplot(111)
    
    # Create canvas
    self.canvas = FigureCanvasTkAgg(self.figure, self.plot_frame)
    self.canvas.draw()
    
    # Add navigation toolbar
    self.toolbar = NavigationToolbar2Tk(self.canvas, self.plot_frame)
    
    # Setup plot interaction
    self.setup_plot_interaction()
```

### Interactive Features
```python
def setup_plot_interaction(self):
    """Setup interactive plot features"""
    # Connect mouse events
    self.canvas.mpl_connect('button_press_event', self.on_click)
    self.canvas.mpl_connect('motion_notify_event', self.on_motion)
    self.canvas.mpl_connect('button_release_event', self.on_release)
    
    # Setup zoom and pan
    self.setup_zoom_pan()
    
    # Setup selection tools
    self.setup_selection_tools()
```

### Plot Data Management
```python
def update_plot(self, data=None, time_data=None):
    """Update the main plot"""
    if data is not None:
        self.ax.clear()
        self.ax.plot(time_data, data)
        self.ax.set_xlabel('Time (s)')
        self.ax.set_ylabel('Current (pA)')
        self.ax.grid(True)
        self.canvas.draw()
```

## Menu and Toolbar System

### Menu Bar Setup
```python
def setup_menubar(self):
    """Setup the menu bar"""
    menubar = tk.Menu(self.master)
    self.master.config(menu=menubar)
    
    # File menu
    file_menu = tk.Menu(menubar, tearoff=0)
    menubar.add_cascade(label="File", menu=file_menu)
    file_menu.add_command(label="Open", command=self.open_file)
    file_menu.add_command(label="Save", command=self.save_file)
    file_menu.add_separator()
    file_menu.add_command(label="Exit", command=self.master.quit)
    
    # Analysis menu
    analysis_menu = tk.Menu(menubar, tearoff=0)
    menubar.add_cascade(label="Analysis", menu=analysis_menu)
    analysis_menu.add_command(label="Process", command=self.process_data)
    analysis_menu.add_command(label="Export", command=self.export_data)
```

### Toolbar Setup
```python
def setup_toolbar(self):
    """Setup the toolbar"""
    toolbar_frame = ttk.Frame(self.master)
    toolbar_frame.pack(side=tk.TOP, fill=tk.X)
    
    # Add toolbar buttons
    ttk.Button(toolbar_frame, text="Open", command=self.open_file).pack(side=tk.LEFT, padx=2)
    ttk.Button(toolbar_frame, text="Save", command=self.save_file).pack(side=tk.LEFT, padx=2)
    ttk.Button(toolbar_frame, text="Process", command=self.process_data).pack(side=tk.LEFT, padx=2)
    ttk.Button(toolbar_frame, text="Export", command=self.export_data).pack(side=tk.LEFT, padx=2)
```

## Development Features

### Hot Reload System
```python
def setup_hot_reload(self):
    """Setup hot reload for development"""
    if self._hot_reload_enabled:
        try:
            initialize_hot_reload(self.master, self.on_hot_reload)
            app_logger.info("Hot reload enabled")
        except Exception as e:
            app_logger.warning(f"Hot reload setup failed: {e}")
            self._hot_reload_enabled = False

def on_hot_reload(self, file_path):
    """Handle hot reload events"""
    current_time = time.time()
    if current_time - self._last_reload_time < self._reload_debounce_delay:
        return
    
    self._last_reload_time = current_time
    app_logger.info(f"Hot reload triggered for: {file_path}")
    
    # Reload specific components
    self.reload_component(file_path)
```

### Memory Monitoring
```python
def monitor_memory_usage(self):
    """Monitor memory usage and cleanup"""
    import psutil
    process = psutil.Process()
    memory_mb = process.memory_info().rss / 1024 / 1024
    
    if memory_mb > 1000:  # 1GB threshold
        app_logger.warning(f"High memory usage: {memory_mb:.1f} MB")
        self.cleanup_memory()
```

## Integration Examples

### Complete Application Setup
```python
import tkinter as tk
from src.gui.app import SignalAnalyzerApp

def main():
    """Main application entry point"""
    # Create main window
    root = tk.Tk()
    root.title("DataChaEnhanced - Signal Analyzer")
    root.geometry("1400x900")
    
    # Initialize application
    app = SignalAnalyzerApp(root)
    
    # Configure window
    root.protocol("WM_DELETE_WINDOW", app.on_closing)
    
    # Start main loop
    root.mainloop()

if __name__ == "__main__":
    main()
```

### Custom Application Extension
```python
class CustomSignalAnalyzerApp(SignalAnalyzerApp):
    """Extended application with custom features"""
    
    def __init__(self, master):
        super().__init__(master)
        self.setup_custom_features()
    
    def setup_custom_features(self):
        """Setup custom application features"""
        # Add custom menu items
        self.add_custom_menu_items()
        
        # Add custom toolbar buttons
        self.add_custom_toolbar_buttons()
        
        # Setup custom data processing
        self.setup_custom_processing()
    
    def add_custom_menu_items(self):
        """Add custom menu items"""
        # Add custom menu to menubar
        custom_menu = tk.Menu(self.master.menubar, tearoff=0)
        self.master.menubar.add_cascade(label="Custom", menu=custom_menu)
        custom_menu.add_command(label="Custom Analysis", command=self.custom_analysis)
    
    def custom_analysis(self):
        """Perform custom analysis"""
        # Custom analysis implementation
        pass
```

### Data Processing Integration
```python
# Integrate with analysis components
def process_data_with_gui(self):
    """Process data with GUI integration"""
    # Load data
    data = self.load_data()
    time_data = self.load_time_data()
    
    # Process with action potential processor
    processor = ActionPotentialProcessor(data, time_data, self.params)
    processor.process()
    
    # Update GUI with results
    self.update_plot(processor.processed_data, time_data)
    self.update_tabs()
    
    # Store results
    self.action_potential_processor = processor
```

## Configuration Options

### Application Configuration
```python
# Configure application settings
app_config = {
    'window_size': (1400, 900),
    'memory_limit': 1000,  # MB
    'hot_reload': True,
    'auto_save': True,
    'theme': 'default'
}

app.configure(app_config)
```

### Memory Management Configuration
```python
# Configure memory management
memory_config = {
    'auto_cleanup': True,
    'cleanup_interval': 30,  # seconds
    'memory_threshold': 800,  # MB
    'gc_frequency': 10  # garbage collection frequency
}

app.configure_memory_management(memory_config)
```

### Hot Reload Configuration
```python
# Configure hot reload
hot_reload_config = {
    'enabled': True,
    'debounce_delay': 2.0,  # seconds
    'watch_directories': ['src/gui', 'src/analysis'],
    'exclude_patterns': ['*.pyc', '__pycache__']
}

app.configure_hot_reload(hot_reload_config)
```

## Performance Optimization

### Memory Optimization
```python
# Optimize memory usage
def optimize_memory_usage(self):
    """Optimize memory usage"""
    # Clean up unused data
    self.cleanup_unused_data()
    
    # Optimize plot memory
    self.optimize_plot_memory()
    
    # Force garbage collection
    gc.collect()
    
    # Monitor memory usage
    self.monitor_memory_usage()
```

### Plot Optimization
```python
# Optimize plot performance
def optimize_plot_performance(self):
    """Optimize plot performance"""
    # Use efficient plotting methods
    self.ax.set_autoscale_on(False)
    
    # Limit data points for display
    if len(self.data) > 10000:
        step = len(self.data) // 10000
        display_data = self.data[::step]
        display_time = self.time_data[::step]
    else:
        display_data = self.data
        display_time = self.time_data
    
    # Update plot
    self.ax.clear()
    self.ax.plot(display_time, display_data)
    self.canvas.draw()
```

## Troubleshooting

### Common Issues

#### 1. Memory Issues
**Symptoms**: High memory usage, slow performance
**Solutions**:
- Enable automatic memory cleanup
- Reduce data size for display
- Use efficient data types
- Monitor memory usage

#### 2. Plot Performance Issues
**Symptoms**: Slow plot updates, freezing
**Solutions**:
- Limit displayed data points
- Use efficient plotting methods
- Optimize plot settings
- Enable plot caching

#### 3. Hot Reload Issues
**Symptoms**: Hot reload not working, errors
**Solutions**:
- Check file permissions
- Verify watch directories
- Adjust debounce delay
- Check error logs

### Debugging Tools

#### 1. Memory Monitoring
```python
# Monitor memory usage
def debug_memory_usage(self):
    """Debug memory usage"""
    import psutil
    process = psutil.Process()
    
    print(f"Memory usage: {process.memory_info().rss / 1024 / 1024:.1f} MB")
    print(f"Data size: {self._memory_usage / 1024 / 1024:.1f} MB")
    print(f"Active figures: {len(self.active_figures)}")
```

#### 2. Performance Profiling
```python
# Profile application performance
def profile_performance(self):
    """Profile application performance"""
    import cProfile
    import pstats
    
    profiler = cProfile.Profile()
    profiler.enable()
    
    # Run application operations
    self.process_data()
    
    profiler.disable()
    stats = pstats.Stats(profiler)
    stats.sort_stats('cumulative')
    stats.print_stats()
```

#### 3. Component Testing
```python
# Test individual components
def test_components(self):
    """Test individual components"""
    # Test data loading
    self.test_data_loading()
    
    # Test plot updates
    self.test_plot_updates()
    
    # Test tab functionality
    self.test_tab_functionality()
    
    # Test window management
    self.test_window_management()
```

## API Reference

### SignalAnalyzerApp Methods
- `__init__(master)`: Initialize application
- `setup_main_layout()`: Setup main layout
- `setup_menubar()`: Setup menu bar
- `setup_toolbar()`: Setup toolbar
- `setup_plot()`: Setup plotting system
- `setup_tabs()`: Setup tab system
- `process_data()`: Process data
- `update_plot()`: Update main plot
- `update_tabs()`: Update tabs
- `on_closing()`: Handle application closing

### Key Attributes
- `data`: Main data array
- `time_data`: Time data array
- `filtered_data`: Filtered data
- `processed_data`: Processed data
- `window_manager`: Window manager instance
- `history_manager`: History manager instance
- `notebook`: Tab notebook widget
- `figure`: Matplotlib figure
- `ax`: Matplotlib axes
- `canvas`: Matplotlib canvas

### Configuration Parameters
- `window_size`: Application window size
- `memory_limit`: Memory usage limit
- `hot_reload`: Enable hot reload
- `auto_save`: Enable auto-save
- `theme`: Application theme
