# GUI Plot Window System

## Overview

The GUI plot window system provides specialized analysis windows for different aspects of signal processing in DataChaEnhanced. These windows offer focused interfaces for specific analysis tasks while maintaining integration with the main application.

**Main Components**:
- **PlotWindowBase** (`plot_window_base.py`) - Base class for all plot windows
- **BaselineCorrectionWindow** (`baseline_window.py`) - Baseline correction analysis
- **IntegrationWindow** (`integration_window.py`) - Signal integration analysis
- **NormalizationWindow** (`normalization_window.py`) - Signal normalization analysis
- **SignalWindowManager** (`window_manager.py`) - Window lifecycle management

## PlotWindowBase

### Overview
The base class for all plot windows, providing common functionality and structure.

**File**: `src/gui/plot_windows/plot_window_base.py`  
**Main Class**: `PlotWindowBase`

### Key Features
- **Common Layout**: Standardized window layout with controls and plot
- **Data Management**: Shared data handling and updates
- **Plot Integration**: Matplotlib integration with Tkinter
- **Event Handling**: Mouse and keyboard event handling
- **Auto-update**: Automatic plot updates on parameter changes

### Base Structure
```python
class PlotWindowBase:
    def __init__(self, parent, title="Plot Window"):
        """Initialize base plot window"""
        self.parent = parent
        self.title = title
        
        # Create main window
        self.window = tk.Toplevel(parent)
        self.window.title(title)
        self.window.geometry("800x600")
        
        # Create main frame
        self.main_frame = ttk.Frame(self.window)
        self.main_frame.pack(fill='both', expand=True, padx=5, pady=5)
        
        # Create control frame
        self.control_frame = ttk.Frame(self.main_frame, width=200)
        self.control_frame.pack(side='left', fill='y', padx=(0, 5))
        
        # Create plot frame
        self.plot_frame = ttk.Frame(self.main_frame)
        self.plot_frame.pack(side='right', fill='both', expand=True)
        
        # Initialize plot
        self.setup_plot()
        
        # Data storage
        self.time_data = None
        self.data = None
        self.processed_data = None
```

### Common Methods
```python
def set_data(self, time_data, data):
    """Set data for the window"""
    self.time_data = np.array(time_data)
    self.data = np.array(data)
    self.update_plot()

def update_plot(self):
    """Update the plot (to be overridden by subclasses)"""
    if self.time_data is not None and self.data is not None:
        self.plot_data()

def plot_data(self):
    """Plot the data (to be overridden by subclasses)"""
    pass

def on_parameter_change(self, *args):
    """Handle parameter changes"""
    if self.auto_update.get():
        self.update_plot()

def close_window(self):
    """Close the window"""
    self.window.destroy()
```

## BaselineCorrectionWindow

### Overview
Specialized window for baseline correction analysis with multiple correction methods and real-time preview.

**File**: `src/gui/plot_windows/baseline_window.py`  
**Main Class**: `BaselineCorrectionWindow`

### Key Features
- **Multiple Methods**: Window average, constant value, and polynomial fitting
- **Real-time Preview**: Live preview of baseline correction
- **Interactive Controls**: Sliders and entry fields for parameter adjustment
- **Statistics Display**: Real-time statistics of correction results

### Baseline Methods

#### 1. Window Average Method
```python
def apply_window_baseline(self, data, time_data):
    """Apply window average baseline correction"""
    start_idx = int(self.window_start.get() * len(data))
    end_idx = int(self.window_end.get() * len(data))
    window_size = self.window_size.get()
    
    # Calculate moving average baseline
    baseline = np.convolve(data[start_idx:end_idx], 
                          np.ones(window_size)/window_size, 
                          mode='same')
    
    # Extend baseline to full data length
    full_baseline = np.zeros_like(data)
    full_baseline[start_idx:end_idx] = baseline
    
    # Corrected data
    corrected_data = data - full_baseline
    
    return corrected_data, full_baseline
```

#### 2. Constant Value Method
```python
def apply_constant_baseline(self, data):
    """Apply constant value baseline correction"""
    constant_value = self.constant_value.get()
    baseline = np.full_like(data, constant_value)
    corrected_data = data - baseline
    
    return corrected_data, baseline
```

#### 3. Polynomial Fit Method
```python
def apply_polynomial_baseline(self, data, time_data):
    """Apply polynomial fit baseline correction"""
    start_idx = int(self.window_start.get() * len(data))
    end_idx = int(self.window_end.get() * len(data))
    order = self.polynomial_order.get()
    
    # Fit polynomial to selected window
    window_data = data[start_idx:end_idx]
    window_time = time_data[start_idx:end_idx]
    
    poly_coeffs = np.polyfit(window_time, window_data, order)
    baseline = np.polyval(poly_coeffs, time_data)
    
    corrected_data = data - baseline
    
    return corrected_data, baseline
```

### Control Interface
```python
def setup_baseline_controls(self):
    """Setup baseline correction specific controls"""
    # Method selection
    method_frame = ttk.LabelFrame(self.control_frame, text="Baseline Method")
    method_frame.pack(fill='x', padx=5, pady=5)
    
    ttk.Radiobutton(method_frame, text="Window Average",
                   variable=self.baseline_method, value="window",
                   command=self.update_plot).pack(pady=2)
    
    ttk.Radiobutton(method_frame, text="Constant Value",
                   variable=self.baseline_method, value="constant",
                   command=self.update_plot).pack(pady=2)
    
    ttk.Radiobutton(method_frame, text="Polynomial Fit",
                   variable=self.baseline_method, value="polynomial",
                   command=self.update_plot).pack(pady=2)
    
    # Window controls
    window_frame = ttk.LabelFrame(self.control_frame, text="Window Settings")
    window_frame.pack(fill='x', padx=5, pady=5)
    
    # Start time control
    start_frame = ttk.Frame(window_frame)
    start_frame.pack(fill='x', padx=5, pady=2)
    ttk.Label(start_frame, text="Start Time (s):").pack(side='left')
    ttk.Entry(start_frame, textvariable=self.window_start,
             width=10).pack(side='right')
    ttk.Scale(window_frame, from_=0, to=1,
             variable=self.window_start,
             orient='horizontal').pack(fill='x')
    
    # End time control
    end_frame = ttk.Frame(window_frame)
    end_frame.pack(fill='x', padx=5, pady=2)
    ttk.Label(end_frame, text="End Time (s):").pack(side='left')
    ttk.Entry(end_frame, textvariable=self.window_end,
             width=10).pack(side='right')
    ttk.Scale(window_frame, from_=0, to=1,
             variable=self.window_end,
             orient='horizontal').pack(fill='x')
```

### Plot Visualization
```python
def plot_data(self):
    """Plot baseline correction results"""
    if self.time_data is None or self.data is None:
        return
    
    # Clear previous plot
    self.ax.clear()
    
    # Plot original data
    self.ax.plot(self.time_data, self.data, 'b-', label='Original', alpha=0.7)
    
    # Apply baseline correction
    corrected_data, baseline = self.apply_baseline_correction()
    
    # Plot corrected data
    self.ax.plot(self.time_data, corrected_data, 'r-', label='Corrected', linewidth=2)
    
    # Plot baseline
    if self.show_baseline.get():
        self.ax.plot(self.time_data, baseline, 'g--', label='Baseline', alpha=0.8)
    
    # Plot window
    if self.show_window.get() and self.baseline_method.get() != "constant":
        start_time = self.window_start.get()
        end_time = self.window_end.get()
        self.ax.axvspan(start_time, end_time, alpha=0.2, color='yellow', label='Window')
    
    # Formatting
    self.ax.set_xlabel('Time (s)')
    self.ax.set_ylabel('Amplitude')
    self.ax.set_title('Baseline Correction')
    self.ax.legend()
    self.ax.grid(True, alpha=0.3)
    
    # Update statistics
    self.update_statistics(corrected_data)
    
    # Refresh plot
    self.canvas.draw()
```

## IntegrationWindow

### Overview
Specialized window for signal integration analysis with multiple integration methods and range selection.

**File**: `src/gui/plot_windows/integration_window.py`  
**Main Class**: `IntegrationWindow`

### Key Features
- **Multiple Methods**: Simpson's rule and trapezoidal rule
- **Range Selection**: Custom integration range selection
- **Visual Feedback**: Filled area and cumulative plot display
- **Real-time Calculation**: Live integration value updates

### Integration Methods

#### 1. Simpson's Rule
```python
def apply_simps_integration(self, data, time_data):
    """Apply Simpson's rule integration"""
    if self.use_range.get():
        start_idx = int(self.range_start.get() * len(data))
        end_idx = int(self.range_end.get() * len(data))
        data = data[start_idx:end_idx]
        time_data = time_data[start_idx:end_idx]
    
    integral_value = simpson(data, time_data)
    return integral_value, data, time_data
```

#### 2. Trapezoidal Rule
```python
def apply_trapz_integration(self, data, time_data):
    """Apply trapezoidal rule integration"""
    if self.use_range.get():
        start_idx = int(self.range_start.get() * len(data))
        end_idx = int(self.range_end.get() * len(data))
        data = data[start_idx:end_idx]
        time_data = time_data[start_idx:end_idx]
    
    integral_value = np.trapz(data, time_data)
    return integral_value, data, time_data
```

### Control Interface
```python
def setup_integration_controls(self):
    """Setup integration specific controls"""
    # Method selection
    method_frame = ttk.LabelFrame(self.control_frame, text="Integration Method")
    method_frame.pack(fill='x', padx=5, pady=5)
    
    ttk.Radiobutton(method_frame, text="Simpson's Rule",
                   variable=self.integration_method, value="simps",
                   command=self.update_plot).pack(pady=2)
    
    ttk.Radiobutton(method_frame, text="Trapezoidal Rule",
                   variable=self.integration_method, value="trapz",
                   command=self.update_plot).pack(pady=2)
    
    # Range selection
    range_frame = ttk.LabelFrame(self.control_frame, text="Integration Range")
    range_frame.pack(fill='x', padx=5, pady=5)
    
    ttk.Checkbutton(range_frame, text="Use Custom Range",
                   variable=self.use_range,
                   command=self.update_plot).pack(pady=2)
    
    # Start time control
    start_frame = ttk.Frame(range_frame)
    start_frame.pack(fill='x', padx=5, pady=2)
    ttk.Label(start_frame, text="Start:").pack(side='left')
    ttk.Entry(start_frame, textvariable=self.range_start,
             width=10).pack(side='right')
    ttk.Scale(range_frame, from_=0, to=1,
             variable=self.range_start,
             orient='horizontal',
             command=lambda *args: self.update_plot()).pack(fill='x')
    
    # End time control
    end_frame = ttk.Frame(range_frame)
    end_frame.pack(fill='x', padx=5, pady=2)
    ttk.Label(end_frame, text="End:").pack(side='left')
    ttk.Entry(end_frame, textvariable=self.range_end,
             width=10).pack(side='right')
    ttk.Scale(range_frame, from_=0, to=1,
             variable=self.range_end,
             orient='horizontal',
             command=lambda *args: self.update_plot()).pack(fill='x')
```

### Plot Visualization
```python
def plot_data(self):
    """Plot integration results"""
    if self.time_data is None or self.data is None:
        return
    
    # Clear previous plot
    self.ax.clear()
    
    # Plot original data
    self.ax.plot(self.time_data, self.data, 'b-', label='Original', alpha=0.7)
    
    # Apply integration
    integral_value, int_data, int_time = self.apply_integration()
    
    # Plot integration range
    if self.use_range.get():
        start_time = self.range_start.get()
        end_time = self.range_end.get()
        self.ax.axvspan(start_time, end_time, alpha=0.2, color='yellow', label='Integration Range')
    
    # Plot filled area
    if self.show_filled.get():
        self.ax.fill_between(int_time, int_data, alpha=0.3, color='red', label='Integral Area')
    
    # Plot cumulative integral
    if self.show_cumulative.get():
        cumulative = np.cumsum(int_data) * np.diff(int_time)[0]
        self.ax2 = self.ax.twinx()
        self.ax2.plot(int_time, cumulative, 'g-', label='Cumulative', linewidth=2)
        self.ax2.set_ylabel('Cumulative Integral')
    
    # Update integral value display
    self.integral_value.set(f"Integral Value: {integral_value:.6f}")
    
    # Formatting
    self.ax.set_xlabel('Time (s)')
    self.ax.set_ylabel('Amplitude')
    self.ax.set_title('Signal Integration')
    self.ax.legend()
    self.ax.grid(True, alpha=0.3)
    
    # Refresh plot
    self.canvas.draw()
```

## NormalizationWindow

### Overview
Specialized window for signal normalization analysis with multiple normalization methods and voltage range controls.

**File**: `src/gui/plot_windows/normalization_window.py`  
**Main Class**: `NormalizationWindow`

### Key Features
- **Multiple Methods**: Min-max, voltage range, and custom range normalization
- **Voltage Controls**: V0 and V1 voltage parameter controls
- **Custom Range**: User-defined normalization range
- **Reference Display**: Visual reference lines and statistics

### Normalization Methods

#### 1. Min-Max Normalization
```python
def apply_minmax_normalization(self, data):
    """Apply min-max normalization"""
    data_min = np.min(data)
    data_max = np.max(data)
    
    if data_max != data_min:
        normalized_data = (data - data_min) / (data_max - data_min)
    else:
        normalized_data = np.zeros_like(data)
    
    return normalized_data, data_min, data_max
```

#### 2. Voltage Range Normalization
```python
def apply_voltage_normalization(self, data):
    """Apply voltage range normalization"""
    v0 = self.v0.get()
    v1 = self.v1.get()
    
    # Normalize to voltage range
    normalized_data = (data - v0) / (v1 - v0)
    
    return normalized_data, v0, v1
```

#### 3. Custom Range Normalization
```python
def apply_custom_normalization(self, data):
    """Apply custom range normalization"""
    custom_min = self.custom_min.get()
    custom_max = self.custom_max.get()
    
    # Normalize to custom range
    normalized_data = (data - custom_min) / (custom_max - custom_min)
    
    return normalized_data, custom_min, custom_max
```

### Control Interface
```python
def setup_normalization_controls(self):
    """Setup normalization specific controls"""
    # Method selection
    method_frame = ttk.LabelFrame(self.control_frame, text="Normalization Method")
    method_frame.pack(fill='x', padx=5, pady=5)
    
    ttk.Radiobutton(method_frame, text="Min-Max",
                   variable=self.norm_method, value="minmax",
                   command=self.update_plot).pack(pady=2)
    
    ttk.Radiobutton(method_frame, text="Voltage Range",
                   variable=self.norm_method, value="voltage",
                   command=self.update_plot).pack(pady=2)
    
    ttk.Radiobutton(method_frame, text="Custom Range",
                   variable=self.norm_method, value="custom",
                   command=self.update_plot).pack(pady=2)
    
    # Voltage range controls
    voltage_frame = ttk.LabelFrame(self.control_frame, text="Voltage Range")
    voltage_frame.pack(fill='x', padx=5, pady=5)
    
    # V0 controls
    v0_frame = ttk.Frame(voltage_frame)
    v0_frame.pack(fill='x', padx=5, pady=2)
    ttk.Label(v0_frame, text="V0 (mV):").pack(side='left')
    ttk.Entry(v0_frame, textvariable=self.v0, width=10).pack(side='right')
    ttk.Scale(voltage_frame, from_=-100, to=0,
             variable=self.v0, orient='horizontal').pack(fill='x')
    
    # V1 controls
    v1_frame = ttk.Frame(voltage_frame)
    v1_frame.pack(fill='x', padx=5, pady=2)
    ttk.Label(v1_frame, text="V1 (mV):").pack(side='left')
    ttk.Entry(v1_frame, textvariable=self.v1, width=10).pack(side='right')
    ttk.Scale(voltage_frame, from_=0, to=100,
             variable=self.v1, orient='horizontal').pack(fill='x')
```

### Plot Visualization
```python
def plot_data(self):
    """Plot normalization results"""
    if self.time_data is None or self.data is None:
        return
    
    # Clear previous plot
    self.ax.clear()
    
    # Plot original data
    self.ax.plot(self.time_data, self.data, 'b-', label='Original', alpha=0.7)
    
    # Apply normalization
    normalized_data, ref_min, ref_max = self.apply_normalization()
    
    # Plot normalized data
    self.ax.plot(self.time_data, normalized_data, 'r-', label='Normalized', linewidth=2)
    
    # Plot reference lines
    if self.show_reference.get():
        self.ax.axhline(y=ref_min, color='g', linestyle='--', alpha=0.7, label=f'Ref Min: {ref_min:.2f}')
        self.ax.axhline(y=ref_max, color='g', linestyle='--', alpha=0.7, label=f'Ref Max: {ref_max:.2f}')
    
    # Update statistics
    if self.show_stats.get():
        self.update_statistics(normalized_data)
    
    # Formatting
    self.ax.set_xlabel('Time (s)')
    self.ax.set_ylabel('Amplitude')
    self.ax.set_title('Signal Normalization')
    self.ax.legend()
    self.ax.grid(True, alpha=0.3)
    
    # Refresh plot
    self.canvas.draw()
```

## SignalWindowManager

### Overview
Manages the lifecycle of plot windows and ensures data consistency across all windows.

**File**: `src/gui/window_manager.py`  
**Main Class**: `SignalWindowManager`

### Key Features
- **Window Lifecycle**: Creation, management, and cleanup of plot windows
- **Data Consistency**: Ensures all windows have consistent data
- **Window Tracking**: Tracks active windows and their states
- **Memory Management**: Prevents memory leaks from orphaned windows

### Window Management
```python
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

    def create_baseline_window(self):
        """Create a new baseline correction window"""
        window_id = f"baseline_{self.window_counters['baseline']}"
        self.window_counters['baseline'] += 1
        
        window = BaselineCorrectionWindow(self.parent, self.on_window_close)
        window.set_data(self.time_data, self.data)
        
        self.windows[window_id] = window
        return window

    def create_integration_window(self):
        """Create a new integration window"""
        window_id = f"integration_{self.window_counters['integration']}"
        self.window_counters['integration'] += 1
        
        window = IntegrationWindow(self.parent, self.on_window_close)
        window.set_data(self.time_data, self.data)
        
        self.windows[window_id] = window
        return window

    def create_normalization_window(self):
        """Create a new normalization window"""
        window_id = f"normalization_{self.window_counters['normalization']}"
        self.window_counters['normalization'] += 1
        
        window = NormalizationWindow(self.parent, self.on_window_close)
        window.set_data(self.time_data, self.data)
        
        self.windows[window_id] = window
        return window
```

### Data Management
```python
def update_data(self, time_data, data, processed_data=None):
    """Update data for all windows"""
    self.time_data = time_data
    self.data = data
    self.processed_data = processed_data
    
    # Update all active windows
    for window in self.windows.values():
        if hasattr(window, 'set_data'):
            window.set_data(time_data, data)

def on_window_close(self, window):
    """Handle window close event"""
    # Remove window from tracking
    for window_id, tracked_window in list(self.windows.items()):
        if tracked_window == window:
            del self.windows[window_id]
            break

def close_all_windows(self):
    """Close all active windows"""
    for window in list(self.windows.values()):
        if hasattr(window, 'close_window'):
            window.close_window()
    self.windows.clear()
```

## Window Integration

### Main Application Integration
```python
def setup_plot_windows(self):
    """Setup plot window system in main app"""
    # Create window manager
    self.window_manager = SignalWindowManager(self.master)
    
    # Add window creation buttons to toolbar
    self.toolbar.add_separator()
    self.toolbar.add_button("Baseline", self.open_baseline_window)
    self.toolbar.add_button("Integration", self.open_integration_window)
    self.toolbar.add_button("Normalization", self.open_normalization_window)

def open_baseline_window(self):
    """Open baseline correction window"""
    if self.data is not None and self.time_data is not None:
        self.window_manager.create_baseline_window()
    else:
        messagebox.showwarning("No Data", "Please load data first")

def open_integration_window(self):
    """Open integration window"""
    if self.data is not None and self.time_data is not None:
        self.window_manager.create_integration_window()
    else:
        messagebox.showwarning("No Data", "Please load data first")

def open_normalization_window(self):
    """Open normalization window"""
    if self.data is not None and self.time_data is not None:
        self.window_manager.create_normalization_window()
    else:
        messagebox.showwarning("No Data", "Please load data first")
```

### Data Flow
```python
def update_plot_windows(self):
    """Update all plot windows with current data"""
    if hasattr(self, 'window_manager'):
        self.window_manager.update_data(
            self.time_data, 
            self.data, 
            self.processed_data
        )
```

## Configuration Options

### Window Configuration
```python
# Configure window behavior
window_config = {
    'auto_update': True,
    'update_interval': 100,  # milliseconds
    'enable_hotkeys': True,
    'enable_tooltips': True,
    'default_size': (800, 600),
    'resizable': True
}

# Apply configuration to all windows
for window in self.window_manager.windows.values():
    window.configure(window_config)
```

### Individual Window Configuration
```python
# Configure specific window types
baseline_config = {
    'default_method': 'window',
    'auto_apply': True,
    'show_statistics': True
}

integration_config = {
    'default_method': 'simps',
    'show_filled_area': True,
    'show_cumulative': True
}

normalization_config = {
    'default_method': 'minmax',
    'show_reference_lines': True,
    'show_statistics': True
}
```

## Troubleshooting

### Common Issues

#### 1. Window Not Updating
**Symptoms**: Plot windows not updating with new data
**Solutions**:
- Check data flow from main application
- Verify window manager data updates
- Check for errors in window update methods
- Ensure proper window initialization

#### 2. Memory Issues
**Symptoms**: Memory usage increasing with multiple windows
**Solutions**:
- Check window cleanup on close
- Verify weak references for data
- Monitor window lifecycle
- Use window manager cleanup

#### 3. Plot Display Issues
**Symptoms**: Plots not displaying correctly
**Solutions**:
- Check matplotlib backend
- Verify data format compatibility
- Check plot update methods
- Enable plot debugging

### Debugging Tools

#### 1. Window State Debugging
```python
def debug_window_states(self):
    """Debug window states"""
    print(f"Active windows: {len(self.window_manager.windows)}")
    for window_id, window in self.window_manager.windows.items():
        print(f"  {window_id}: {type(window).__name__}")
        print(f"    Data available: {hasattr(window, 'data') and window.data is not None}")
        print(f"    Time data available: {hasattr(window, 'time_data') and window.time_data is not None}")
```

#### 2. Data Flow Debugging
```python
def debug_data_flow(self):
    """Debug data flow to windows"""
    print("Data flow debug:")
    print(f"  Main app data: {self.data is not None}")
    print(f"  Main app time data: {self.time_data is not None}")
    print(f"  Window manager data: {self.window_manager.data is not None}")
    print(f"  Window manager time data: {self.window_manager.time_data is not None}")
```

## API Reference

### PlotWindowBase Methods
- `__init__(parent, title)`: Initialize base plot window
- `set_data(time_data, data)`: Set data for the window
- `update_plot()`: Update the plot
- `plot_data()`: Plot the data (override in subclasses)
- `on_parameter_change()`: Handle parameter changes
- `close_window()`: Close the window

### BaselineCorrectionWindow Methods
- `apply_window_baseline(data, time_data)`: Apply window average baseline
- `apply_constant_baseline(data)`: Apply constant value baseline
- `apply_polynomial_baseline(data, time_data)`: Apply polynomial fit baseline
- `update_statistics(corrected_data)`: Update correction statistics

### IntegrationWindow Methods
- `apply_simps_integration(data, time_data)`: Apply Simpson's rule integration
- `apply_trapz_integration(data, time_data)`: Apply trapezoidal rule integration
- `calculate_integral()`: Calculate and display integral value

### NormalizationWindow Methods
- `apply_minmax_normalization(data)`: Apply min-max normalization
- `apply_voltage_normalization(data)`: Apply voltage range normalization
- `apply_custom_normalization(data)`: Apply custom range normalization
- `update_statistics(normalized_data)`: Update normalization statistics

### SignalWindowManager Methods
- `create_baseline_window()`: Create baseline correction window
- `create_integration_window()`: Create integration window
- `create_normalization_window()`: Create normalization window
- `update_data(time_data, data, processed_data)`: Update data for all windows
- `on_window_close(window)`: Handle window close event
- `close_all_windows()`: Close all active windows

### Key Attributes
- `window`: Main Tkinter window
- `control_frame`: Control panel frame
- `plot_frame`: Plot display frame
- `time_data`: Time data array
- `data`: Data array
- `processed_data`: Processed data array
- `auto_update`: Auto-update flag
