# GUI Tab System

## Overview

The GUI tab system provides specialized interfaces for different aspects of signal analysis in DataChaEnhanced. Each tab focuses on specific functionality while maintaining integration with the main application and shared data.

**Main Components**:
- **Filter Tab** (`filter_tab.py`) - Signal filtering and preprocessing
- **Analysis Tab** (`analysis_tab.py`) - Data analysis and statistics
- **View Tab** (`view_tab.py`) - Data visualization and inspection
- **Action Potential Tab** (`action_potential_tab.py`) - Action potential analysis
- **AI Analysis Tab** (`ai_analysis_tab.py`) - AI-powered analysis features
- **Excel Learning Tab** (`excel_learning_tab.py`) - Excel integration and learning

## Filter Tab

### Overview
The Filter Tab provides comprehensive signal filtering capabilities with multiple filter types and real-time parameter adjustment.

**File**: `src/gui/filter_tab.py`  
**Main Class**: `FilterTab`

### Key Features
- **Savitzky-Golay Filter**: Smoothing with polynomial fitting
- **Wavelet Filter**: Multi-scale wavelet denoising
- **Butterworth Filter**: Frequency domain filtering
- **Extract-Add Filter**: Peak extraction and removal
- **Real-time Preview**: Live filter preview with parameter adjustment

### Filter Types

#### 1. Savitzky-Golay Filter
```python
# Savitzky-Golay filter controls
savgol_frame = ttk.LabelFrame(self.frame, text="Savitzky-Golay Filter")

# Enable checkbox
ttk.Checkbutton(savgol_frame, text="Enable", 
               variable=self.use_savgol,
               command=self.on_filter_change).pack(pady=2)

# Window length control (5-101, odd numbers only)
ttk.Scale(window_frame, from_=5, to=101, variable=self.savgol_window,
         orient='horizontal', command=lambda _: self.on_filter_change())

# Polynomial order control (2-5)
ttk.Scale(order_frame, from_=2, to=5, variable=self.savgol_order,
         orient='horizontal', command=lambda _: self.on_filter_change())
```

#### 2. Wavelet Filter
```python
# Wavelet filter controls
wavelet_frame = ttk.LabelFrame(self.frame, text="Wavelet Filter")

# Enable checkbox
ttk.Checkbutton(wavelet_frame, text="Enable",
               variable=self.use_wavelet,
               command=self.on_filter_change).pack(pady=2)

# Wavelet level control (1-8)
ttk.Scale(level_frame, from_=1, to=8, variable=self.wavelet_level,
         orient='horizontal', command=lambda _: self.on_filter_change())
```

#### 3. Butterworth Filter
```python
# Butterworth filter controls
butterworth_frame = ttk.LabelFrame(self.frame, text="Butterworth Filter")

# Enable checkbox
ttk.Checkbutton(butterworth_frame, text="Enable",
               variable=self.use_butterworth,
               command=self.on_filter_change).pack(pady=2)

# Cutoff frequency control (0.01-0.5)
ttk.Scale(cutoff_frame, from_=0.01, to=0.5, variable=self.butter_cutoff,
         orient='horizontal', command=lambda _: self.on_filter_change())

# Filter order control (1-10)
ttk.Scale(order_frame, from_=1, to=10, variable=self.butter_order,
         orient='horizontal', command=lambda _: self.on_filter_change())
```

#### 4. Extract-Add Filter
```python
# Extract-Add filter controls
extract_frame = ttk.LabelFrame(self.frame, text="Extract-Add Filter")

# Enable checkbox
ttk.Checkbutton(extract_frame, text="Enable",
               variable=self.use_extract_add,
               command=self.on_filter_change).pack(pady=2)

# Prominence control (0-1000)
ttk.Scale(prom_frame, from_=0, to=1000, variable=self.extract_prominence,
         orient='horizontal', command=lambda _: self.on_filter_change())

# Width range controls
ttk.Scale(width_min_frame, from_=1, to=100, variable=self.extract_width_min,
         orient='horizontal', command=lambda _: self.on_filter_change())
ttk.Scale(width_max_frame, from_=1, to=100, variable=self.extract_width_max,
         orient='horizontal', command=lambda _: self.on_filter_change())
```

### Filter Application
```python
def apply_filters(self, data):
    """Apply all enabled filters to data"""
    filtered_data = data.copy()
    
    # Apply Savitzky-Golay filter
    if self.use_savgol.get():
        filtered_data = self.apply_savgol_filter(filtered_data)
    
    # Apply wavelet filter
    if self.use_wavelet.get():
        filtered_data = self.apply_wavelet_filter(filtered_data)
    
    # Apply Butterworth filter
    if self.use_butterworth.get():
        filtered_data = self.apply_butterworth_filter(filtered_data)
    
    # Apply extract-add filter
    if self.use_extract_add.get():
        filtered_data = self.apply_extract_add_filter(filtered_data)
    
    return filtered_data
```

## Analysis Tab

### Overview
The Analysis Tab provides statistical analysis, peak detection, and event analysis capabilities for processed signals.

**File**: `src/gui/analysis_tab.py`  
**Main Class**: `AnalysisTab`

### Key Features
- **Signal Statistics**: Comprehensive statistical analysis
- **Peak Detection**: Advanced peak detection with configurable parameters
- **Event Analysis**: Event detection and analysis
- **Real-time Updates**: Live statistics and analysis updates

### Statistics Display
```python
def setup_statistics_display(self):
    """Setup the statistics display section"""
    stats_frame = ttk.LabelFrame(self.frame, text="Signal Statistics")
    
    # Basic statistics display
    basic_stats = ttk.Label(stats_frame, textvariable=self.stats_text,
                          wraplength=300, justify='left')
    basic_stats.pack(fill='x', padx=5, pady=5)
    
    # Refresh button
    ttk.Button(stats_frame, text="Refresh Statistics",
              command=self.update_statistics).pack(pady=5)

def update_statistics(self):
    """Update signal statistics display"""
    if self.data is not None:
        stats = self.calculate_statistics(self.data)
        stats_text = f"""
        Mean: {stats['mean']:.3f}
        Std Dev: {stats['std']:.3f}
        Min: {stats['min']:.3f}
        Max: {stats['max']:.3f}
        Range: {stats['range']:.3f}
        RMS: {stats['rms']:.3f}
        """
        self.stats_text.set(stats_text)
```

### Peak Detection
```python
def setup_peak_detection(self):
    """Setup peak detection controls"""
    peak_frame = ttk.LabelFrame(self.frame, text="Peak Detection")
    
    # Enable peak detection
    ttk.Checkbutton(peak_frame, text="Enable Peak Detection",
                   variable=self.detect_peaks,
                   command=self.on_peak_settings_change).pack(pady=2)
    
    # Peak prominence control (0-1000)
    ttk.Scale(prom_frame, from_=0, to=1000, variable=self.peak_prominence,
             orient='horizontal', command=self.on_peak_settings_change)
    
    # Peak distance control (1-200)
    ttk.Scale(dist_frame, from_=1, to=200, variable=self.peak_distance,
             orient='horizontal', command=self.on_peak_settings_change)
    
    # Peak width control (1-100)
    ttk.Scale(width_frame, from_=1, to=100, variable=self.peak_width,
             orient='horizontal', command=self.on_peak_settings_change)

def detect_peaks(self, data):
    """Detect peaks in data"""
    if not self.detect_peaks.get():
        return []
    
    peaks, properties = find_peaks(
        data,
        prominence=self.peak_prominence.get(),
        distance=self.peak_distance.get(),
        width=self.peak_width.get(),
        height=self.peak_height.get()
    )
    
    return peaks, properties
```

### Event Analysis
```python
def setup_event_analysis(self):
    """Setup event analysis controls"""
    event_frame = ttk.LabelFrame(self.frame, text="Event Analysis")
    
    # Enable event analysis
    ttk.Checkbutton(event_frame, text="Enable Event Analysis",
                   variable=self.analyze_events,
                   command=self.on_event_settings_change).pack(pady=2)
    
    # Event threshold control (0-1000)
    ttk.Scale(threshold_frame, from_=0, to=1000, variable=self.event_threshold,
             orient='horizontal', command=self.on_event_settings_change)
    
    # Minimum event duration (0.01-1.0 seconds)
    ttk.Scale(duration_frame, from_=0.01, to=1.0, variable=self.min_event_duration,
             orient='horizontal', command=self.on_event_settings_change)

def analyze_events(self, data, time_data):
    """Analyze events in data"""
    if not self.analyze_events.get():
        return []
    
    events = []
    threshold = self.event_threshold.get()
    min_duration = self.min_event_duration.get()
    
    # Find events above threshold
    above_threshold = data > threshold
    
    # Find event start and end points
    event_starts = []
    event_ends = []
    
    in_event = False
    for i, above in enumerate(above_threshold):
        if above and not in_event:
            event_starts.append(i)
            in_event = True
        elif not above and in_event:
            event_ends.append(i)
            in_event = False
    
    # Filter events by duration
    for start, end in zip(event_starts, event_ends):
        duration = time_data[end] - time_data[start]
        if duration >= min_duration:
            events.append({
                'start': start,
                'end': end,
                'duration': duration,
                'amplitude': np.max(data[start:end])
            })
    
    return events
```

## View Tab

### Overview
The View Tab provides data visualization and inspection capabilities with interactive plotting and data exploration tools.

**File**: `src/gui/view_tab.py`  
**Main Class**: `ViewTab`

### Key Features
- **Interactive Plotting**: Zoom, pan, and selection capabilities
- **Data Inspection**: Point-by-point data inspection
- **Multiple Views**: Different visualization modes
- **Export Options**: Save plots and data

### Interactive Plotting
```python
def setup_plot_controls(self):
    """Setup interactive plot controls"""
    plot_frame = ttk.LabelFrame(self.frame, text="Plot Controls")
    
    # Plot type selection
    ttk.Label(plot_frame, text="Plot Type:").pack(side='left')
    plot_type = ttk.Combobox(plot_frame, values=['Line', 'Scatter', 'Histogram'])
    plot_type.pack(side='left', padx=5)
    
    # Zoom controls
    zoom_frame = ttk.Frame(plot_frame)
    zoom_frame.pack(fill='x', pady=5)
    
    ttk.Button(zoom_frame, text="Zoom In", command=self.zoom_in).pack(side='left')
    ttk.Button(zoom_frame, text="Zoom Out", command=self.zoom_out).pack(side='left')
    ttk.Button(zoom_frame, text="Reset View", command=self.reset_view).pack(side='left')
    
    # Selection tools
    selection_frame = ttk.Frame(plot_frame)
    selection_frame.pack(fill='x', pady=5)
    
    ttk.Button(selection_frame, text="Select Region", command=self.select_region).pack(side='left')
    ttk.Button(selection_frame, text="Clear Selection", command=self.clear_selection).pack(side='left')
```

### Data Inspection
```python
def setup_data_inspection(self):
    """Setup data inspection tools"""
    inspect_frame = ttk.LabelFrame(self.frame, text="Data Inspection")
    
    # Point selection
    ttk.Label(inspect_frame, text="Point Index:").pack(side='left')
    self.point_index = tk.IntVar()
    point_spinbox = ttk.Spinbox(inspect_frame, from_=0, to=10000, 
                               textvariable=self.point_index,
                               command=self.on_point_selection)
    point_spinbox.pack(side='left', padx=5)
    
    # Point information display
    self.point_info = ttk.Label(inspect_frame, text="No point selected")
    self.point_info.pack(fill='x', padx=5, pady=5)
    
    # Data range selection
    range_frame = ttk.Frame(inspect_frame)
    range_frame.pack(fill='x', pady=5)
    
    ttk.Label(range_frame, text="Start:").pack(side='left')
    self.range_start = tk.IntVar()
    ttk.Spinbox(range_frame, from_=0, to=10000, textvariable=self.range_start).pack(side='left')
    
    ttk.Label(range_frame, text="End:").pack(side='left')
    self.range_end = tk.IntVar()
    ttk.Spinbox(range_frame, from_=0, to=10000, textvariable=self.range_end).pack(side='left')
    
    ttk.Button(range_frame, text="Analyze Range", command=self.analyze_range).pack(side='left', padx=5)
```

## Action Potential Tab

### Overview
The Action Potential Tab provides comprehensive action potential analysis capabilities with parameter configuration, curve fitting, and integration tools.

**File**: `src/gui/action_potential_tab.py`  
**Main Class**: `ActionPotentialTab`

### Key Features
- **Parameter Configuration**: Voltage protocol and analysis parameters
- **Normalization Points**: Interactive normalization point selection
- **Integration Ranges**: Configurable integration range selection
- **Curve Fitting**: Interactive curve fitting tools
- **Spike Removal**: Periodic spike removal capabilities
- **Results Display**: Comprehensive analysis results

### Parameter Controls
```python
def setup_parameter_controls(self):
    """Setup analysis parameter controls"""
    param_frame = ttk.LabelFrame(self.scrollable_frame, text="Analysis Parameters")
    param_frame.pack(fill='x', padx=5, pady=5)
    
    # Voltage protocol parameters
    protocol_frame = ttk.Frame(param_frame)
    protocol_frame.pack(fill='x', pady=2)
    
    # Number of cycles
    ttk.Label(protocol_frame, text="Cycles:").pack(side='left')
    self.n_cycles = tk.IntVar(value=2)
    ttk.Spinbox(protocol_frame, from_=1, to=10, textvariable=self.n_cycles).pack(side='left', padx=5)
    
    # Time parameters
    time_frame = ttk.Frame(param_frame)
    time_frame.pack(fill='x', pady=2)
    
    ttk.Label(time_frame, text="t0 (ms):").pack(side='left')
    self.t0 = tk.IntVar(value=20)
    ttk.Spinbox(time_frame, from_=10, to=100, textvariable=self.t0).pack(side='left', padx=5)
    
    ttk.Label(time_frame, text="t1 (ms):").pack(side='left')
    self.t1 = tk.IntVar(value=100)
    ttk.Spinbox(time_frame, from_=50, to=500, textvariable=self.t1).pack(side='left', padx=5)
    
    # Voltage parameters
    voltage_frame = ttk.Frame(param_frame)
    voltage_frame.pack(fill='x', pady=2)
    
    ttk.Label(voltage_frame, text="V0 (mV):").pack(side='left')
    self.V0 = tk.IntVar(value=-80)
    ttk.Spinbox(voltage_frame, from_=-120, to=-40, textvariable=self.V0).pack(side='left', padx=5)
    
    ttk.Label(voltage_frame, text="V1 (mV):").pack(side='left')
    self.V1 = tk.IntVar(value=-100)
    ttk.Spinbox(voltage_frame, from_=-150, to=-50, textvariable=self.V1).pack(side='left', padx=5)
```

### Normalization Points
```python
def setup_normalization_points(self):
    """Setup normalization point controls"""
    norm_frame = ttk.LabelFrame(self.scrollable_frame, text="Normalization Points")
    norm_frame.pack(fill='x', padx=5, pady=5)
    
    # Segment 1 normalization points
    seg1_frame = ttk.Frame(norm_frame)
    seg1_frame.pack(fill='x', pady=2)
    
    ttk.Label(seg1_frame, text="Segment 1 Start:").pack(side='left')
    self.seg1_start = tk.IntVar(value=35)
    ttk.Spinbox(seg1_frame, from_=1, to=1000, textvariable=self.seg1_start).pack(side='left', padx=5)
    
    ttk.Label(seg1_frame, text="End:").pack(side='left')
    self.seg1_end = tk.IntVar(value=234)
    ttk.Spinbox(seg1_frame, from_=1, to=1000, textvariable=self.seg1_end).pack(side='left', padx=5)
    
    # Interactive selection button
    ttk.Button(seg1_frame, text="Select on Plot", 
              command=self.select_normalization_points).pack(side='left', padx=10)
```

### Integration Ranges
```python
def setup_integration_range_controls(self):
    """Setup integration range controls"""
    int_frame = ttk.LabelFrame(self.scrollable_frame, text="Integration Ranges")
    int_frame.pack(fill='x', padx=5, pady=5)
    
    # Hyperpolarization integration range
    hyperpol_frame = ttk.Frame(int_frame)
    hyperpol_frame.pack(fill='x', pady=2)
    
    ttk.Label(hyperpol_frame, text="Hyperpol Start:").pack(side='left')
    self.hyperpol_start = tk.IntVar(value=11)
    ttk.Spinbox(hyperpol_frame, from_=1, to=1000, textvariable=self.hyperpol_start).pack(side='left', padx=5)
    
    ttk.Label(hyperpol_frame, text="End:").pack(side='left')
    self.hyperpol_end = tk.IntVar(value=210)
    ttk.Spinbox(hyperpol_frame, from_=1, to=1000, textvariable=self.hyperpol_end).pack(side='left', padx=5)
    
    # Depolarization integration range
    depol_frame = ttk.Frame(int_frame)
    depol_frame.pack(fill='x', pady=2)
    
    ttk.Label(depol_frame, text="Depol Start:").pack(side='left')
    self.depol_start = tk.IntVar(value=211)
    ttk.Spinbox(depol_frame, from_=1, to=1000, textvariable=self.depol_start).pack(side='left', padx=5)
    
    ttk.Label(depol_frame, text="End:").pack(side='left')
    self.depol_end = tk.IntVar(value=410)
    ttk.Spinbox(depol_frame, from_=1, to=1000, textvariable=self.depol_end).pack(side='left', padx=5)
```

### Curve Fitting Integration
```python
def setup_curve_fitting(self):
    """Setup curve fitting controls"""
    fitting_frame = ttk.LabelFrame(self.scrollable_frame, text="Curve Fitting")
    fitting_frame.pack(fill='x', padx=5, pady=5)
    
    # Curve fitting panel
    self.curve_fitting_panel = CurveFittingPanel(fitting_frame, self)
    self.curve_fitting_panel.pack(fill='x', pady=5)
    
    # Fitting controls
    controls_frame = ttk.Frame(fitting_frame)
    controls_frame.pack(fill='x', pady=2)
    
    ttk.Button(controls_frame, text="Start Linear Fitting", 
              command=self.start_linear_fitting).pack(side='left', padx=5)
    ttk.Button(controls_frame, text="Start Exponential Fitting", 
              command=self.start_exponential_fitting).pack(side='left', padx=5)
    ttk.Button(controls_frame, text="Clear Fits", 
              command=self.clear_fits).pack(side='left', padx=5)
```

### Results Display
```python
def setup_results_display(self):
    """Setup analysis results display"""
    results_frame = ttk.LabelFrame(self.scrollable_frame, text="Analysis Results")
    results_frame.pack(fill='x', padx=5, pady=5)
    
    # Results text widget
    self.results_text = tk.Text(results_frame, height=10, width=50, wrap=tk.WORD)
    results_scrollbar = ttk.Scrollbar(results_frame, orient="vertical", command=self.results_text.yview)
    self.results_text.configure(yscrollcommand=results_scrollbar.set)
    
    self.results_text.pack(side="left", fill="both", expand=True)
    results_scrollbar.pack(side="right", fill="y")
    
    # Clear results button
    ttk.Button(results_frame, text="Clear Results", 
              command=self.clear_results).pack(pady=5)

def update_results_display(self, results):
    """Update the results display with analysis results"""
    self.results_text.delete(1.0, tk.END)
    
    if results:
        results_text = f"""
Analysis Results:
================

Hyperpolarization Integral: {results.get('hyperpol_integral', 'N/A'):.6f}
Depolarization Integral: {results.get('depol_integral', 'N/A'):.6f}
Total Charge: {results.get('total_charge', 'N/A'):.6f}

Quality Metrics:
- Overall Quality: {results.get('quality', 'N/A'):.3f}
- Signal-to-Noise Ratio: {results.get('snr', 'N/A'):.2f} dB
- Baseline Stability: {results.get('baseline_stability', 'N/A'):.3f}

Processing Time: {results.get('processing_time', 'N/A'):.3f} seconds
        """
        self.results_text.insert(1.0, results_text)
```

## AI Analysis Tab

### Overview
The AI Analysis Tab provides AI-powered analysis features including automated parameter detection, intelligent curve fitting, and machine learning-based analysis.

**File**: `src/gui/ai_analysis_tab.py`  
**Main Class**: `AIAnalysisTab`

### Key Features
- **AI Parameter Detection**: Automated detection of analysis parameters
- **Intelligent Curve Fitting**: AI-assisted curve fitting
- **Confidence Scoring**: Quality assessment and confidence metrics
- **Learning Integration**: Integration with machine learning systems

### AI Parameter Detection
```python
def setup_ai_parameter_detection(self):
    """Setup AI parameter detection controls"""
    ai_frame = ttk.LabelFrame(self.frame, text="AI Parameter Detection")
    ai_frame.pack(fill='x', padx=5, pady=5)
    
    # Enable AI detection
    ttk.Checkbutton(ai_frame, text="Enable AI Parameter Detection",
                   variable=self.enable_ai_detection,
                   command=self.on_ai_settings_change).pack(pady=2)
    
    # AI confidence threshold
    confidence_frame = ttk.Frame(ai_frame)
    confidence_frame.pack(fill='x', pady=2)
    
    ttk.Label(confidence_frame, text="Confidence Threshold:").pack(side='left')
    ttk.Scale(confidence_frame, from_=0.0, to=1.0, variable=self.ai_confidence_threshold,
             orient='horizontal', command=self.on_ai_settings_change).pack(side='left', fill='x', expand=True)
    ttk.Label(confidence_frame, textvariable=self.ai_confidence_threshold).pack(side='right')
    
    # Detect parameters button
    ttk.Button(ai_frame, text="Detect Parameters", 
              command=self.detect_parameters).pack(pady=5)
```

### AI Curve Fitting
```python
def setup_ai_curve_fitting(self):
    """Setup AI curve fitting controls"""
    fitting_frame = ttk.LabelFrame(self.frame, text="AI Curve Fitting")
    fitting_frame.pack(fill='x', padx=5, pady=5)
    
    # Enable AI fitting
    ttk.Checkbutton(fitting_frame, text="Enable AI Curve Fitting",
                   variable=self.enable_ai_fitting,
                   command=self.on_ai_settings_change).pack(pady=2)
    
    # Fitting method selection
    method_frame = ttk.Frame(fitting_frame)
    method_frame.pack(fill='x', pady=2)
    
    ttk.Label(method_frame, text="Fitting Method:").pack(side='left')
    self.ai_fitting_method = ttk.Combobox(method_frame, 
                                        values=['Auto', 'Linear', 'Exponential', 'Polynomial'])
    self.ai_fitting_method.pack(side='left', padx=5)
    
    # Start AI fitting button
    ttk.Button(fitting_frame, text="Start AI Fitting", 
              command=self.start_ai_fitting).pack(pady=5)
```

## Excel Learning Tab

### Overview
The Excel Learning Tab provides integration with Excel analysis workflows and machine learning capabilities for automated analysis.

**File**: `src/gui/excel_learning_tab.py`  
**Main Class**: `ExcelLearningTab`

### Key Features
- **Excel Integration**: Direct integration with Excel analysis workflows
- **Learning System**: Machine learning from Excel data
- **Data Collection**: Automated data collection from Excel files
- **Model Training**: Training and validation of AI models

### Excel Integration
```python
def setup_excel_integration(self):
    """Setup Excel integration controls"""
    excel_frame = ttk.LabelFrame(self.frame, text="Excel Integration")
    excel_frame.pack(fill='x', padx=5, pady=5)
    
    # Excel file selection
    file_frame = ttk.Frame(excel_frame)
    file_frame.pack(fill='x', pady=2)
    
    ttk.Button(file_frame, text="Select Excel File", 
              command=self.select_excel_file).pack(side='left', padx=5)
    
    self.excel_file_path = tk.StringVar()
    ttk.Label(file_frame, textvariable=self.excel_file_path).pack(side='left', padx=5)
    
    # Load Excel data button
    ttk.Button(excel_frame, text="Load Excel Data", 
              command=self.load_excel_data).pack(pady=5)
```

### Learning System
```python
def setup_learning_system(self):
    """Setup learning system controls"""
    learning_frame = ttk.LabelFrame(self.frame, text="Learning System")
    learning_frame.pack(fill='x', padx=5, pady=5)
    
    # Training controls
    training_frame = ttk.Frame(learning_frame)
    training_frame.pack(fill='x', pady=2)
    
    ttk.Button(training_frame, text="Start Training", 
              command=self.start_training).pack(side='left', padx=5)
    ttk.Button(training_frame, text="Validate Model", 
              command=self.validate_model).pack(side='left', padx=5)
    
    # Learning progress
    progress_frame = ttk.Frame(learning_frame)
    progress_frame.pack(fill='x', pady=2)
    
    ttk.Label(progress_frame, text="Training Progress:").pack(side='left')
    self.training_progress = ttk.Progressbar(progress_frame, mode='determinate')
    self.training_progress.pack(side='left', fill='x', expand=True, padx=5)
```

## Tab Integration

### Tab Communication
```python
def update_tabs(self, tab_name=None):
    """Update all tabs or specific tab"""
    if tab_name is None:
        # Update all tabs
        self.filter_tab.update_display()
        self.analysis_tab.update_display()
        self.view_tab.update_display()
        self.action_potential_tab.update_display()
        self.ai_analysis_tab.update_display()
        self.excel_learning_tab.update_display()
    else:
        # Update specific tab
        getattr(self, f"{tab_name}_tab").update_display()
```

### Data Sharing
```python
def share_data_between_tabs(self, data_type, data):
    """Share data between tabs"""
    # Update all tabs with new data
    for tab in [self.filter_tab, self.analysis_tab, self.view_tab, 
                self.action_potential_tab, self.ai_analysis_tab, self.excel_learning_tab]:
        if hasattr(tab, 'update_data'):
            tab.update_data(data_type, data)
```

## Configuration Options

### Tab Configuration
```python
# Configure tab behavior
tab_config = {
    'auto_update': True,
    'update_interval': 100,  # milliseconds
    'enable_hotkeys': True,
    'enable_tooltips': True
}

# Apply configuration to all tabs
for tab in [self.filter_tab, self.analysis_tab, self.view_tab, 
            self.action_potential_tab, self.ai_analysis_tab, self.excel_learning_tab]:
    tab.configure(tab_config)
```

### Individual Tab Configuration
```python
# Configure specific tab
filter_config = {
    'default_filters': ['savgol', 'wavelet'],
    'auto_apply': True,
    'preview_mode': True
}
self.filter_tab.configure(filter_config)

analysis_config = {
    'auto_detect_peaks': True,
    'peak_threshold': 200,
    'event_threshold': 100
}
self.analysis_tab.configure(analysis_config)
```

## Troubleshooting

### Common Issues

#### 1. Tab Update Issues
**Symptoms**: Tabs not updating with new data
**Solutions**:
- Check tab update callbacks
- Verify data sharing mechanism
- Ensure proper tab initialization
- Check for errors in tab update methods

#### 2. Filter Application Issues
**Symptoms**: Filters not applying correctly
**Solutions**:
- Check filter parameter validation
- Verify filter implementation
- Check data format compatibility
- Enable filter debugging

#### 3. Analysis Tab Issues
**Symptoms**: Statistics or peak detection not working
**Solutions**:
- Check data availability
- Verify analysis parameters
- Check for numerical errors
- Enable analysis debugging

### Debugging Tools

#### 1. Tab State Debugging
```python
def debug_tab_states(self):
    """Debug tab states"""
    for tab_name in ['filter', 'analysis', 'view', 'action_potential', 'ai_analysis', 'excel_learning']:
        tab = getattr(self, f"{tab_name}_tab")
        print(f"{tab_name} tab state:")
        print(f"  Data available: {hasattr(tab, 'data') and tab.data is not None}")
        print(f"  Update callback: {hasattr(tab, 'update_callback') and tab.update_callback is not None}")
        print(f"  Initialized: {hasattr(tab, 'initialized') and tab.initialized}")
```

#### 2. Filter Debugging
```python
def debug_filters(self):
    """Debug filter state"""
    print("Filter state:")
    print(f"  Savitzky-Golay: {self.filter_tab.use_savgol.get()}")
    print(f"  Wavelet: {self.filter_tab.use_wavelet.get()}")
    print(f"  Butterworth: {self.filter_tab.use_butterworth.get()}")
    print(f"  Extract-Add: {self.filter_tab.use_extract_add.get()}")
```

## API Reference

### FilterTab Methods
- `__init__(parent, callback)`: Initialize filter tab
- `apply_filters(data)`: Apply all enabled filters
- `on_filter_change()`: Handle filter parameter changes
- `update_display()`: Update filter display

### AnalysisTab Methods
- `__init__(parent, callback)`: Initialize analysis tab
- `update_statistics()`: Update signal statistics
- `detect_peaks(data)`: Detect peaks in data
- `analyze_events(data, time_data)`: Analyze events in data

### ActionPotentialTab Methods
- `__init__(parent, callback)`: Initialize action potential tab
- `setup_parameter_controls()`: Setup parameter controls
- `setup_normalization_points()`: Setup normalization points
- `setup_integration_range_controls()`: Setup integration ranges
- `update_results_display(results)`: Update results display

### Key Attributes
- `frame`: Main tab frame
- `update_callback`: Callback function for updates
- `data`: Current data
- `time_data`: Current time data
- `filtered_data`: Filtered data
- `processed_data`: Processed data
