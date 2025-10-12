# GUI Export Tools

## Overview

The GUI export tools provide comprehensive data export and batch processing capabilities for DataChaEnhanced. These tools enable users to export analysis results, process multiple files, and generate reports in various formats.

**Main Components**:
- **BatchSetExporter** (`batch_set_exporter.py`) - Batch processing and export of ATF files
- **SimplifiedSetExporter** (`simplified_set_exporter.py`) - Simplified export interface
- **MultiFileAnalyzer** (`multi_file_analysis.py`) - Multi-file analysis and comparison
- **Excel Export Integration** - Excel export with charts and formatting

## BatchSetExporter

### Overview
The main batch processing and export tool for processing multiple ATF files and exporting results to Excel format.

**File**: `src/gui/batch_set_exporter.py`  
**Main Class**: `BatchSetExporter`

### Key Features
- **Batch Processing**: Process multiple ATF files in sets
- **Excel Export**: Export results to formatted Excel files
- **Integration Debugging**: Multiple integration range testing
- **Set Organization**: Automatic organization of files by sets
- **Comprehensive Analysis**: Full action potential analysis pipeline

### Integration Methods
```python
def __init__(self, parent_app):
    self.parent_app = parent_app
    
    # Try different integration ranges to find the correct one
    self.integration_methods = [
        {'name': 'Default_0_200', 'hyperpol_start': 0, 'hyperpol_end': 200, 'depol_start': 0, 'depol_end': 200},
        {'name': 'First_100', 'hyperpol_start': 0, 'hyperpol_end': 100, 'depol_start': 0, 'depol_end': 100},
        {'name': 'Middle_50_150', 'hyperpol_start': 50, 'hyperpol_end': 150, 'depol_start': 50, 'depol_end': 150},
        {'name': 'Custom_Auto', 'hyperpol_start': 'auto', 'hyperpol_end': 'auto', 'depol_start': 'auto', 'depol_end': 'auto'},
    ]
```

### Batch Export Process
```python
def export_folder_by_sets(self):
    """Export ATF files organized by sets to Excel files"""
    try:
        # Select folder
        folder_path = filedialog.askdirectory(
            title="Select folder containing ATF files"
        )
        
        if not folder_path:
            return False
        
        folder_path = Path(folder_path)
        
        # Find all ATF files
        atf_files = list(folder_path.glob("*.atf"))
        
        if not atf_files:
            messagebox.showwarning("No Files", "No ATF files found")
            return False
        
        # Organize by sets
        file_sets = self._organize_files_by_sets(atf_files)
        
        if not file_sets:
            messagebox.showerror("Error", "Could not parse file names")
            return False
        
        # Select output directory
        output_dir = filedialog.askdirectory(
            title="Select output directory"
        )
        
        if not output_dir:
            return False
        
        output_dir = Path(output_dir) / "ExportedSets"
        output_dir.mkdir(exist_ok=True)
        
        # Process each set
        results = []
        for set_number, files_info in file_sets.items():
            success = self._process_set(set_number, files_info, output_dir)
            if success:
                results.append(f"✓ Set {set_number}: Success")
            else:
                results.append(f"✗ Set {set_number}: Failed")
        
        # Show results
        messagebox.showinfo("Export Complete", "\n".join(results))
        return True
        
    except Exception as e:
        app_logger.error(f"Error in batch export: {str(e)}")
        messagebox.showerror("Error", f"Export failed: {str(e)}")
        return False
```

### File Organization
```python
def _organize_files_by_sets(self, atf_files):
    """Organize ATF files by sets based on filename patterns"""
    file_sets = defaultdict(list)
    
    for file_path in atf_files:
        filename = file_path.stem
        
        # Try different patterns to extract set number
        patterns = [
            r'set(\d+)',           # set1, set2, etc.
            r'(\d+)_',             # 1_, 2_, etc.
            r'_(\d+)_',            # _1_, _2_, etc.
            r'(\d+)\.',            # 1., 2., etc.
        ]
        
        set_number = None
        for pattern in patterns:
            match = re.search(pattern, filename, re.IGNORECASE)
            if match:
                set_number = int(match.group(1))
                break
        
        if set_number is not None:
            file_sets[set_number].append({
                'path': file_path,
                'filename': filename
            })
    
    return dict(file_sets)
```

### Set Processing
```python
def _process_set(self, set_number, files_info, output_dir):
    """Process a single set of files"""
    try:
        app_logger.info(f"Processing set {set_number} with {len(files_info)} files")
        
        # Create workbook for this set
        wb = Workbook()
        ws = wb.active
        ws.title = f"Set {set_number}"
        
        # Setup headers
        self._setup_excel_headers(ws)
        
        # Process each file in the set
        row = 2
        for file_info in files_info:
            file_path = file_info['path']
            filename = file_info['filename']
            
            app_logger.info(f"Processing file: {filename}")
            
            # Load and process file
            results = self._process_single_file(file_path)
            
            if results:
                # Write results to Excel
                self._write_file_results(ws, row, filename, results)
                row += 1
        
        # Save workbook
        output_file = output_dir / f"Set_{set_number}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.xlsx"
        wb.save(output_file)
        
        app_logger.info(f"Set {set_number} exported to {output_file}")
        return True
        
    except Exception as e:
        app_logger.error(f"Error processing set {set_number}: {str(e)}")
        return False
```

### Single File Processing
```python
def _process_single_file(self, file_path):
    """Process a single ATF file and return results"""
    try:
        # Load ATF file
        atf_handler = ATFHandler(file_path)
        atf_handler.load_atf()
        
        time_data = atf_handler.get_column("Time")
        raw_data = atf_handler.get_column("#1")
        
        if time_data is None or raw_data is None:
            raise ValueError("Could not load time or current data")
        
        # Apply filters
        filter_params = {
            'savgol_params': {'window_length': 101, 'polyorder': 3},
            'butter_params': {'cutoff': 2000, 'fs': 10000, 'order': 2}
        }
        filtered_data = combined_filter(raw_data, **filter_params)
        
        # Create processor
        processor = ActionPotentialProcessor(
            filtered_data, 
            time_data,
            params={
                'n_cycles': 2,
                't0': 20,
                't1': 100,
                't2': 100,
                't3': 1000,
                'V0': -80,
                'V1': -100,
                'V2': -20,
                'cell_area_cm2': 1e-4
            }
        )
        
        # Process data
        processor.process_data()
        
        # Get results
        results = processor.get_results()
        
        # Try different integration methods
        integration_results = {}
        for method in self.integration_methods:
            try:
                if method['hyperpol_start'] == 'auto':
                    # Use automatic range detection
                    hyperpol_start, hyperpol_end = processor.get_hyperpol_range()
                    depol_start, depol_end = processor.get_depol_range()
                else:
                    hyperpol_start = method['hyperpol_start']
                    hyperpol_end = method['hyperpol_end']
                    depol_start = method['depol_start']
                    depol_end = method['depol_end']
                
                # Calculate integrals
                hyperpol_integral = processor.calculate_integral(
                    'hyperpol', hyperpol_start, hyperpol_end
                )
                depol_integral = processor.calculate_integral(
                    'depol', depol_start, depol_end
                )
                
                integration_results[method['name']] = {
                    'hyperpol_integral': hyperpol_integral,
                    'depol_integral': depol_integral,
                    'total_charge': hyperpol_integral + depol_integral,
                    'hyperpol_range': (hyperpol_start, hyperpol_end),
                    'depol_range': (depol_start, depol_end)
                }
                
            except Exception as e:
                app_logger.warning(f"Integration method {method['name']} failed: {str(e)}")
                continue
        
        # Combine results
        final_results = {
            'filename': Path(file_path).name,
            'filepath': str(file_path),
            'processing_time': results.get('processing_time', 0),
            'quality_score': results.get('quality_score', 0),
            'integration_results': integration_results,
            'raw_data_stats': {
                'mean': np.mean(raw_data),
                'std': np.std(raw_data),
                'min': np.min(raw_data),
                'max': np.max(raw_data)
            },
            'filtered_data_stats': {
                'mean': np.mean(filtered_data),
                'std': np.std(filtered_data),
                'min': np.min(filtered_data),
                'max': np.max(filtered_data)
            }
        }
        
        return final_results
        
    except Exception as e:
        app_logger.error(f"Error processing file {file_path}: {str(e)}")
        return None
```

### Excel Export
```python
def _setup_excel_headers(self, ws):
    """Setup Excel worksheet headers"""
    headers = [
        'Filename', 'File Path', 'Processing Time (s)', 'Quality Score',
        'Raw Data Mean', 'Raw Data Std', 'Raw Data Min', 'Raw Data Max',
        'Filtered Data Mean', 'Filtered Data Std', 'Filtered Data Min', 'Filtered Data Max'
    ]
    
    # Add integration method headers
    for method in self.integration_methods:
        headers.extend([
            f"{method['name']}_Hyperpol_Integral",
            f"{method['name']}_Depol_Integral",
            f"{method['name']}_Total_Charge",
            f"{method['name']}_Hyperpol_Range",
            f"{method['name']}_Depol_Range"
        ])
    
    # Write headers
    for col, header in enumerate(headers, 1):
        cell = ws.cell(row=1, column=col, value=header)
        cell.font = Font(bold=True)
        cell.fill = PatternFill(start_color="CCCCCC", end_color="CCCCCC", fill_type="solid")
    
    # Auto-adjust column widths
    for col in range(1, len(headers) + 1):
        ws.column_dimensions[get_column_letter(col)].width = 15

def _write_file_results(self, ws, row, filename, results):
    """Write file results to Excel worksheet"""
    col = 1
    
    # Basic information
    ws.cell(row=row, column=col, value=results['filename']); col += 1
    ws.cell(row=row, column=col, value=results['filepath']); col += 1
    ws.cell(row=row, column=col, value=results['processing_time']); col += 1
    ws.cell(row=row, column=col, value=results['quality_score']); col += 1
    
    # Raw data statistics
    raw_stats = results['raw_data_stats']
    ws.cell(row=row, column=col, value=raw_stats['mean']); col += 1
    ws.cell(row=row, column=col, value=raw_stats['std']); col += 1
    ws.cell(row=row, column=col, value=raw_stats['min']); col += 1
    ws.cell(row=row, column=col, value=raw_stats['max']); col += 1
    
    # Filtered data statistics
    filtered_stats = results['filtered_data_stats']
    ws.cell(row=row, column=col, value=filtered_stats['mean']); col += 1
    ws.cell(row=row, column=col, value=filtered_stats['std']); col += 1
    ws.cell(row=row, column=col, value=filtered_stats['min']); col += 1
    ws.cell(row=row, column=col, value=filtered_stats['max']); col += 1
    
    # Integration results
    integration_results = results['integration_results']
    for method in self.integration_methods:
        method_name = method['name']
        if method_name in integration_results:
            method_results = integration_results[method_name]
            ws.cell(row=row, column=col, value=method_results['hyperpol_integral']); col += 1
            ws.cell(row=row, column=col, value=method_results['depol_integral']); col += 1
            ws.cell(row=row, column=col, value=method_results['total_charge']); col += 1
            ws.cell(row=row, column=col, value=str(method_results['hyperpol_range'])); col += 1
            ws.cell(row=row, column=col, value=str(method_results['depol_range'])); col += 1
        else:
            # Fill with empty values
            for _ in range(5):
                ws.cell(row=row, column=col, value="N/A"); col += 1
```

## SimplifiedSetExporter

### Overview
A simplified interface for the batch set exporter, providing backward compatibility and easier access to export functionality.

**File**: `src/gui/simplified_set_exporter.py`  
**Main Class**: `SimplifiedSetExporter`

### Key Features
- **Backward Compatibility**: Maintains compatibility with existing code
- **Simplified Interface**: Easier access to export functionality
- **Alias Support**: Provides aliases for main exporter classes

### Implementation
```python
"""
Simplified Set Exporter - Redirects to batch_set_exporter
Location: src/gui/simplified_set_exporter.py
This file exists for backward compatibility
"""

# Import everything from the batch set exporter
from .batch_set_exporter import (
    BatchSetExporter,
    BatchSetExportButton,
    add_set_export_to_toolbar
)

# For backward compatibility, create aliases
SimplifiedSetExporter = BatchSetExporter
SimplifiedSetExportButton = BatchSetExportButton

__all__ = [
    'BatchSetExporter',
    'BatchSetExportButton', 
    'add_set_export_to_toolbar',
    'SimplifiedSetExporter',
    'SimplifiedSetExportButton'
]
```

## MultiFileAnalyzer

### Overview
A comprehensive multi-file analysis tool that allows users to load, process, and compare multiple ATF files simultaneously.

**File**: `src/gui/multi_file_analysis.py`  
**Main Class**: `MultiFileAnalyzer`

### Key Features
- **Multi-File Loading**: Load and process multiple ATF files
- **Side-by-Side Comparison**: Compare analysis results across files
- **Full Processing Pipeline**: Complete action potential analysis
- **Memory Efficient**: Optimized for handling multiple files
- **Interactive Interface**: User-friendly file management

### FileSlot Class
```python
class FileSlot:
    """Represents a single file slot in the multi-file analyzer"""
    
    def __init__(self, slot_number):
        self.slot_number = slot_number
        self.filename = None
        self.filepath = None
        self.time_data = None
        self.raw_data = None
        self.filtered_data = None
        self.processor = None
        self.results = {}
        self.is_loaded = False
        self.is_processed = False
        
        # Store all processing curves
        self.processed_data = None
        self.orange_curve = None
        self.orange_curve_times = None
        self.normalized_curve = None
        self.normalized_curve_times = None
        self.average_curve = None
        self.average_curve_times = None
        self.modified_hyperpol = None
        self.modified_hyperpol_times = None
        self.modified_depol = None
        self.modified_depol_times = None
```

### File Loading
```python
def load_file(self, filepath):
    """Load ATF file into this slot"""
    try:
        self.filepath = filepath
        self.filename = Path(filepath).name
        
        # Load ATF file
        atf_handler = ATFHandler(filepath)
        atf_handler.load_atf()
        
        self.time_data = atf_handler.get_column("Time")
        self.raw_data = atf_handler.get_column("#1")
        
        if self.time_data is None or self.raw_data is None:
            raise ValueError("Could not load time or current data")
        
        self.is_loaded = True
        self.is_processed = False
        
        app_logger.info(f"Loaded file {self.filename} into slot {self.slot_number}")
        return True
        
    except Exception as e:
        app_logger.error(f"Error loading file into slot {self.slot_number}: {str(e)}")
        self.clear()
        return False
```

### File Processing
```python
def process_file(self, voltage=-50, parent_history_manager=None):
    """Process the loaded file with FULL action potential analysis pipeline"""
    if not self.is_loaded:
        return False
        
    try:
        # Get sampling rate
        sampling_interval = 1e-5  # Default 100 µs
        fs_hz = 1.0 / sampling_interval
        
        # Apply filters (same as main app)
        filter_params = {
            'savgol_params': {'window_length': 101, 'polyorder': 3},
            'butter_params': {'cutoff': 2000, 'fs': fs_hz, 'order': 2}
        }
        self.filtered_data = combined_filter(self.raw_data, **filter_params)
        
        # Create processor with same parameters as main app
        self.processor = ActionPotentialProcessor(
            self.filtered_data, 
            self.time_data,
            params={
                'n_cycles': 2,
                't0': 20,
                't1': 100,
                't2': 100,
                't3': 1000,
                'V0': -80,
                'V1': -100,
                'V2': -20,
                'cell_area_cm2': 1e-4
            }
        )
        
        # Process data
        self.processor.process_data()
        
        # Get all processed curves
        self.processed_data = self.processor.get_processed_data()
        self.orange_curve = self.processor.get_orange_curve()
        self.orange_curve_times = self.processor.get_orange_curve_times()
        self.normalized_curve = self.processor.get_normalized_curve()
        self.normalized_curve_times = self.processor.get_normalized_curve_times()
        self.average_curve = self.processor.get_average_curve()
        self.average_curve_times = self.processor.get_average_curve_times()
        self.modified_hyperpol = self.processor.get_modified_hyperpol()
        self.modified_hyperpol_times = self.processor.get_modified_hyperpol_times()
        self.modified_depol = self.processor.get_modified_depol()
        self.modified_depol_times = self.processor.get_modified_depol_times()
        
        # Get results
        self.results = self.processor.get_results()
        
        # Add to history if available
        if parent_history_manager:
            parent_history_manager.add_analysis(
                self.filename,
                self.results,
                self.processor
            )
        
        self.is_processed = True
        app_logger.info(f"Processed file {self.filename} in slot {self.slot_number}")
        return True
        
    except Exception as e:
        app_logger.error(f"Error processing file in slot {self.slot_number}: {str(e)}")
        return False
```

### Multi-File Interface
```python
class MultiFileAnalyzer:
    """Multi-file analysis interface"""
    
    def __init__(self, parent):
        self.parent = parent
        self.slots = {}
        self.history_manager = None
        
        # Create main window
        self.window = tk.Toplevel(parent)
        self.window.title("Multi-File Analyzer")
        self.window.geometry("1200x800")
        
        # Create interface
        self.setup_interface()
        
        # Initialize slots
        self.setup_slots()
    
    def setup_interface(self):
        """Setup the main interface"""
        # Create main frame
        main_frame = ttk.Frame(self.window)
        main_frame.pack(fill='both', expand=True, padx=5, pady=5)
        
        # Create control frame
        control_frame = ttk.Frame(main_frame)
        control_frame.pack(fill='x', pady=(0, 5))
        
        # Add control buttons
        ttk.Button(control_frame, text="Load Files", 
                  command=self.load_files).pack(side='left', padx=5)
        ttk.Button(control_frame, text="Process All", 
                  command=self.process_all).pack(side='left', padx=5)
        ttk.Button(control_frame, text="Export Results", 
                  command=self.export_results).pack(side='left', padx=5)
        
        # Create slots frame
        self.slots_frame = ttk.Frame(main_frame)
        self.slots_frame.pack(fill='both', expand=True)
    
    def setup_slots(self):
        """Setup file slots"""
        for i in range(4):  # 4 slots
            slot = FileSlot(i + 1)
            self.slots[i + 1] = slot
            
            # Create slot frame
            slot_frame = ttk.LabelFrame(self.slots_frame, text=f"Slot {i + 1}")
            slot_frame.pack(fill='x', padx=5, pady=5)
            
            # Create slot controls
            self.create_slot_controls(slot_frame, slot)
    
    def create_slot_controls(self, parent, slot):
        """Create controls for a file slot"""
        # File selection
        file_frame = ttk.Frame(parent)
        file_frame.pack(fill='x', padx=5, pady=2)
        
        ttk.Button(file_frame, text="Load File", 
                  command=lambda: self.load_slot_file(slot)).pack(side='left', padx=5)
        
        self.filename_var = tk.StringVar()
        ttk.Label(file_frame, textvariable=self.filename_var).pack(side='left', padx=5)
        
        # Process button
        ttk.Button(file_frame, text="Process", 
                  command=lambda: self.process_slot_file(slot)).pack(side='left', padx=5)
        
        # Results display
        results_frame = ttk.Frame(parent)
        results_frame.pack(fill='x', padx=5, pady=2)
        
        self.results_text = tk.Text(results_frame, height=3, width=50)
        self.results_text.pack(fill='x')
```

## Excel Export Integration

### Overview
Integration with Excel export functionality for creating formatted reports and charts.

**Files**: 
- `src/excel_export/excel_export.py`
- `src/excel_charted/enhanced_excel_export_with_charts.py`

### Key Features
- **Formatted Reports**: Professional Excel formatting
- **Chart Generation**: Automatic chart creation
- **Data Validation**: Input validation and error handling
- **Multiple Formats**: Support for various Excel formats

### Excel Export Process
```python
def export_to_excel(self, data, filename):
    """Export data to Excel with formatting"""
    try:
        # Create workbook
        wb = Workbook()
        ws = wb.active
        ws.title = "Analysis Results"
        
        # Setup formatting
        self.setup_excel_formatting(ws)
        
        # Write data
        self.write_data_to_excel(ws, data)
        
        # Create charts
        self.create_excel_charts(ws, data)
        
        # Save file
        wb.save(filename)
        
        return True
        
    except Exception as e:
        app_logger.error(f"Error exporting to Excel: {str(e)}")
        return False
```

## Integration with Main Application

### Toolbar Integration
```python
def add_export_tools_to_toolbar(self, toolbar):
    """Add export tools to main application toolbar"""
    # Add batch export button
    toolbar.add_separator()
    toolbar.add_button("Batch Export", self.open_batch_exporter)
    toolbar.add_button("Multi-File Analysis", self.open_multi_file_analyzer)
    toolbar.add_button("Export Results", self.export_current_results)

def open_batch_exporter(self):
    """Open batch exporter"""
    if not hasattr(self, 'batch_exporter'):
        self.batch_exporter = BatchSetExporter(self)
    
    self.batch_exporter.export_folder_by_sets()

def open_multi_file_analyzer(self):
    """Open multi-file analyzer"""
    if not hasattr(self, 'multi_file_analyzer'):
        self.multi_file_analyzer = MultiFileAnalyzer(self.master)
    
    self.multi_file_analyzer.window.lift()

def export_current_results(self):
    """Export current analysis results"""
    if self.processor and self.processor.get_results():
        # Get current results
        results = self.processor.get_results()
        
        # Create export dialog
        export_dialog = ExportDialog(self.master, results)
        export_dialog.show()
    else:
        messagebox.showwarning("No Data", "No analysis results to export")
```

## Configuration Options

### Export Configuration
```python
# Configure export behavior
export_config = {
    'default_format': 'xlsx',
    'include_charts': True,
    'include_raw_data': True,
    'include_statistics': True,
    'auto_format': True,
    'compression': True
}

# Apply configuration to exporters
self.batch_exporter.configure(export_config)
self.multi_file_analyzer.configure(export_config)
```

### Batch Processing Configuration
```python
# Configure batch processing
batch_config = {
    'max_files_per_set': 100,
    'parallel_processing': True,
    'memory_limit': '2GB',
    'progress_callback': self.update_progress,
    'error_handling': 'continue'
}
```

## Troubleshooting

### Common Issues

#### 1. Export Failures
**Symptoms**: Excel export fails or produces corrupted files
**Solutions**:
- Check file permissions
- Verify Excel installation
- Check available disk space
- Validate data format

#### 2. Memory Issues
**Symptoms**: Out of memory errors during batch processing
**Solutions**:
- Reduce batch size
- Enable memory optimization
- Check available RAM
- Use streaming processing

#### 3. File Loading Issues
**Symptoms**: ATF files fail to load
**Solutions**:
- Check file format compatibility
- Verify file integrity
- Check file permissions
- Enable detailed logging

### Debugging Tools

#### 1. Export Debugging
```python
def debug_export_process(self):
    """Debug export process"""
    print("Export debugging:")
    print(f"  Available memory: {psutil.virtual_memory().available / 1024**3:.2f} GB")
    print(f"  Excel available: {self.check_excel_availability()}")
    print(f"  Output directory writable: {self.check_output_permissions()}")
```

#### 2. Batch Processing Debugging
```python
def debug_batch_processing(self):
    """Debug batch processing"""
    print("Batch processing debug:")
    print(f"  Files found: {len(self.atf_files)}")
    print(f"  Sets organized: {len(self.file_sets)}")
    print(f"  Processing method: {self.integration_method}")
```

## API Reference

### BatchSetExporter Methods
- `__init__(parent_app)`: Initialize batch exporter
- `export_folder_by_sets()`: Export folder by sets
- `_organize_files_by_sets(atf_files)`: Organize files by sets
- `_process_set(set_number, files_info, output_dir)`: Process single set
- `_process_single_file(file_path)`: Process single file
- `_setup_excel_headers(ws)`: Setup Excel headers
- `_write_file_results(ws, row, filename, results)`: Write file results

### MultiFileAnalyzer Methods
- `__init__(parent)`: Initialize multi-file analyzer
- `setup_interface()`: Setup main interface
- `setup_slots()`: Setup file slots
- `load_slot_file(slot)`: Load file into slot
- `process_slot_file(slot)`: Process file in slot
- `process_all()`: Process all loaded files
- `export_results()`: Export analysis results

### FileSlot Methods
- `__init__(slot_number)`: Initialize file slot
- `load_file(filepath)`: Load ATF file
- `process_file(voltage, parent_history_manager)`: Process file
- `clear()`: Clear slot data
- `get_results()`: Get analysis results

### Key Attributes
- `integration_methods`: Available integration methods
- `file_sets`: Organized file sets
- `slots`: File slots for multi-file analysis
- `results`: Analysis results
- `export_config`: Export configuration
