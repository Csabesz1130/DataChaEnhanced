# Purple Curves Excel Export System

## Overview

The Purple Curves Excel Export system is a comprehensive data export functionality that creates enhanced Excel files containing processed action potential data with automatic chart generation, manual curve fitting frameworks, and comprehensive analysis tools. This system is designed to export the "purple curves" - the modified/processed data from action potential analysis.

## System Architecture

### Core Components

1. **ActionPotentialProcessor** (`src/analysis/action_potential.py`)
   - Generates the purple curve data through signal processing
   - Contains methods: `modified_hyperpol`, `modified_depol`, `modified_hyperpol_times`, `modified_depol_times`
   - Provides `export_all_curves()` method for basic CSV export

2. **Enhanced Excel Export System** (`src/excel_charted/`)
   - **enhanced_excel_export_with_charts.py**: Main export implementation with dual curves support
   - **excel_export_enhanced_integration.py**: Integration layer for purple curves only
   - **dual_curves_export_integration.py**: Integration for both purple and red curves

3. **GUI Integration** (`src/gui/app.py`)
   - `on_export_purple_curves()`: Main export handler
   - Export buttons and status labels
   - File dialog integration

### Data Flow

```
Action Potential Analysis → Purple Curves Generation → Excel Export → Enhanced Excel File
```

## Features and Capabilities

### 1. Enhanced Excel Export with Charts

**File**: `src/excel_charted/enhanced_excel_export_with_charts.py`

#### Key Features:
- **Automatic Chart Generation**: Creates interactive scatter plots with smooth curves
- **Multiple Worksheets**: 6 comprehensive worksheets per export
- **Professional Formatting**: Color-coded headers, proper number formatting
- **Data Validation**: Ensures all required data is available before export

#### Worksheets Created:

1. **Purple_Curve_Data**
   - Raw data export with time and current values
   - Hyperpolarization and depolarization sections
   - Properly formatted with headers and metadata

2. **Interactive_Charts**
   - Automatic scatter plot generation
   - Smooth curve interpolation
   - Professional chart styling with legends and axis labels
   - Multiple chart types for different analysis needs

3. **Hyperpol_Analysis**
   - Statistical analysis of hyperpolarization curves
   - Manual fitting framework for exponential decay analysis
   - Point selection tools for linear fitting
   - Parameter extraction capabilities

4. **Depol_Analysis**
   - Statistical analysis of depolarization curves
   - Manual fitting framework for exponential rise analysis
   - Point selection tools for linear fitting
   - Parameter extraction capabilities

5. **Manual_Fitting_Tools**
   - Step-by-step analysis workflow
   - Training data preparation tools
   - Solver helpers for curve fitting
   - Documentation framework

6. **Instructions**
   - Complete workflow guide
   - Analysis methodology
   - Best practices and tips

### 2. Integration Layers

#### Purple Curves Only Export
**File**: `src/excel_charted/excel_export_enhanced_integration.py`

- **Function**: `enhanced_export_purple_curves(app)`
- **Purpose**: Export only purple curves with enhanced features
- **Features**:
  - Automatic chart generation
  - Manual curve fitting framework
  - Point selection tools
  - Exponential parameter extraction
  - Step-by-step analysis workflow
  - Training data preparation tools

#### Dual Curves Export
**File**: `src/excel_charted/dual_curves_export_integration.py`

- **Function**: `export_both_curves_with_charts(processor, app, filename)`
- **Purpose**: Export both purple (modified) and red (filtered) curves for comparison
- **Features**:
  - Side-by-side curve comparison
  - Statistical comparison tables
  - Processing effect analysis
  - Validation and method development tools

### 3. Data Processing Pipeline

#### Purple Curves Generation
The purple curves are generated through the following process:

1. **Signal Processing**:
   ```python
   # Baseline correction and normalization
   self.baseline_correction_initial()
   self.advanced_baseline_normalization()
   
   # Generate normalized curves
   self.normalized_curve, self.normalized_curve_times = self.calculate_normalized_curve()
   self.average_curve, self.average_curve_times = self.calculate_segment_average()
   
   # Generate purple curves
   (self.modified_hyperpol, 
    self.modified_hyperpol_times,
    self.modified_depol,
    self.modified_depol_times) = self.apply_average_to_peaks()
   ```

2. **Data Validation**:
   ```python
   def _validate_purple_curve_data(processor):
       required_attrs = [
           'modified_hyperpol', 'modified_depol',
           'modified_hyperpol_times', 'modified_depol_times'
       ]
       # Validation logic...
   ```

#### Excel Export Process

1. **Workbook Creation**:
   ```python
   self.workbook = xlsxwriter.Workbook(filename)
   ```

2. **Worksheet Generation**:
   - Data worksheet with formatted purple curve data
   - Charts worksheet with interactive visualizations
   - Analysis worksheets with statistical comparisons
   - Manual fitting worksheets with analysis tools
   - Instructions worksheet with complete workflow

3. **Chart Generation**:
   ```python
   # Create scatter plot with smooth curves
   chart = self.workbook.add_chart({'type': 'scatter', 'subtype': 'smooth'})
   
   # Add data series
   chart.add_series({
       'name': 'Purple Hyperpol (Modified)',
       'categories': ['Purple_Curve_Data', 6, 0, 300, 0],
       'values': ['Purple_Curve_Data', 6, 1, 300, 1],
       'line': {'color': 'purple', 'width': 2},
       'marker': {'type': 'none'}
   })
   ```

## Usage Workflow

### 1. Prerequisites

- **Action Potential Analysis**: Must be completed to generate purple curves
- **Data Requirements**: 
  - `modified_hyperpol` and `modified_depol` arrays
  - `modified_hyperpol_times` and `modified_depol_times` arrays
  - Integral results from analysis

### 2. Export Process

#### GUI Workflow:
1. **Run Analysis**: Complete action potential analysis to generate purple curves
2. **Access Export**: Click "Export Purple Curves to Excel (Enhanced)" button
3. **File Selection**: Choose save location and filename
4. **Processing**: System validates data and creates enhanced Excel file
5. **Completion**: Receive confirmation with feature summary

#### Programmatic Workflow:
```python
# Get processor with purple curve data
processor = app.action_potential_processor

# Validate data availability
if not _validate_purple_curve_data(processor):
    raise ValueError("Purple curve data not available")

# Perform enhanced export
result_filename = export_purple_curves_with_charts(processor, filename)
```

### 3. Export Options

#### Enhanced Export (Recommended)
- **Requirements**: `xlsxwriter` library installed
- **Features**: Automatic charts, manual fitting framework, comprehensive analysis
- **File Size**: Larger due to charts and formatting
- **Use Case**: Research, publication, detailed analysis

#### Basic Export (Fallback)
- **Requirements**: `pandas` and `openpyxl` libraries
- **Features**: Data export only, basic formatting
- **File Size**: Smaller, data-focused
- **Use Case**: Quick data export, compatibility

## Technical Implementation

### Dependencies

#### Required Libraries:
```python
import numpy as np
import pandas as pd
import xlsxwriter  # For enhanced features
from datetime import datetime
import os
```

#### Optional Libraries:
```python
import openpyxl  # For basic export fallback
```

### Data Format

#### Purple Curve Data Structure:
```python
# Hyperpolarization data
modified_hyperpol: np.ndarray  # Current values in pA
modified_hyperpol_times: np.ndarray  # Time values in seconds

# Depolarization data  
modified_depol: np.ndarray  # Current values in pA
modified_depol_times: np.ndarray  # Time values in seconds
```

#### Excel Output Format:
- **Time Units**: Converted to milliseconds (×1000)
- **Precision**: 7 decimal places for current values
- **Formatting**: Professional headers, color coding, number formatting

### Error Handling

#### Data Validation:
```python
def _validate_purple_curve_data(processor):
    """Validate that purple curve data is available for export."""
    required_attrs = [
        'modified_hyperpol', 'modified_depol',
        'modified_hyperpol_times', 'modified_depol_times'
    ]
    
    for attr in required_attrs:
        if not hasattr(processor, attr):
            return False
        if getattr(processor, attr) is None:
            return False
        if len(getattr(processor, attr)) == 0:
            return False
    
    return True
```

#### Export Error Handling:
```python
try:
    result_filename = export_purple_curves_with_charts(processor, filename)
    app_logger.info(f"Enhanced export completed: {result_filename}")
except Exception as e:
    app_logger.error(f"Error in enhanced export: {str(e)}")
    # Fallback to basic export or show error message
```

## Advanced Features

### 1. Manual Curve Fitting Framework

The Excel export includes comprehensive tools for manual curve fitting:

- **Point Selection Tools**: Interactive point selection for fitting ranges
- **Linear Fitting**: Tools for linear regression analysis
- **Exponential Fitting**: Framework for exponential decay/rise analysis
- **Parameter Extraction**: Automatic calculation of fitting parameters
- **Validation Tools**: Methods to validate fitting quality

### 2. Statistical Analysis

Built-in statistical analysis capabilities:

- **Descriptive Statistics**: Mean, standard deviation, min/max, peak-to-peak
- **Comparison Tools**: Side-by-side statistical comparison
- **Trend Analysis**: Identification of trends and patterns
- **Quality Metrics**: Assessment of data quality and processing effects

### 3. Training Data Preparation

Tools for preparing data for machine learning:

- **Data Formatting**: Consistent data structure for ML algorithms
- **Feature Extraction**: Automated extraction of relevant features
- **Label Generation**: Tools for creating training labels
- **Data Validation**: Quality checks for training data

## File Structure

### Generated Excel File Structure:
```
purple_curves_analysis_YYYYMMDD_HHMMSS.xlsx
├── Purple_Curve_Data (Worksheet 1)
│   ├── Headers and metadata
│   ├── Hyperpolarization data (time, current)
│   └── Depolarization data (time, current)
├── Interactive_Charts (Worksheet 2)
│   ├── Hyperpolarization chart
│   └── Depolarization chart
├── Hyperpol_Analysis (Worksheet 3)
│   ├── Statistical analysis
│   └── Manual fitting framework
├── Depol_Analysis (Worksheet 4)
│   ├── Statistical analysis
│   └── Manual fitting framework
├── Manual_Fitting_Tools (Worksheet 5)
│   ├── Analysis workflow
│   └── Training data preparation
└── Instructions (Worksheet 6)
    ├── Complete workflow guide
    └── Best practices
```

## Performance Considerations

### Memory Usage:
- **Data Size**: Depends on signal length and sampling rate
- **Excel File Size**: Typically 1-10 MB depending on data size and charts
- **Processing Time**: 1-5 seconds for typical datasets

### Optimization:
- **Lazy Loading**: Data is processed on-demand
- **Efficient Formatting**: Minimal memory overhead for formatting
- **Streaming**: Large datasets are handled efficiently

## Troubleshooting

### Common Issues:

1. **Missing Data Error**:
   - **Cause**: Action potential analysis not completed
   - **Solution**: Run complete analysis before export

2. **Import Error**:
   - **Cause**: Missing `xlsxwriter` library
   - **Solution**: Install with `pip install xlsxwriter`

3. **Memory Error**:
   - **Cause**: Very large datasets
   - **Solution**: Use basic export or reduce data size

4. **File Access Error**:
   - **Cause**: File is open in Excel or permission issues
   - **Solution**: Close Excel file or check permissions

### Debug Information:
```python
# Enable debug logging
app_logger.setLevel(logging.DEBUG)

# Check export capabilities
capabilities = get_export_capabilities()
print(f"Enhanced export: {capabilities['enhanced_export_available']}")
print(f"Charts supported: {capabilities['charts_supported']}")
```

## Future Enhancements

### Planned Features:
- **3D Visualization**: 3D charts for complex data analysis
- **Machine Learning Integration**: Direct ML model training from Excel data
- **Real-time Collaboration**: Multi-user editing capabilities
- **Advanced Analytics**: More sophisticated statistical analysis tools

### Extensibility:
The system is designed for easy extension:
- **Custom Worksheets**: Add new analysis worksheets
- **Custom Charts**: Implement new chart types
- **Custom Analysis**: Add specialized analysis tools
- **Integration**: Connect with external analysis tools

## Conclusion

The Purple Curves Excel Export system provides a comprehensive solution for exporting and analyzing action potential data. With its automatic chart generation, manual fitting frameworks, and extensive analysis tools, it serves as both a data export mechanism and a complete analysis platform. The system's modular architecture ensures maintainability and extensibility for future enhancements.

The combination of automated features and manual analysis tools makes it suitable for both routine data export and advanced research applications, providing researchers with the tools they need to extract maximum value from their action potential data.
