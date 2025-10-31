# Linear Fit Subtraction Feature

## Overview

The Linear Fit Subtraction feature allows you to subtract linearly fitted curves from hyperpol and depol purple curves separately, then reload the whole plot. This is useful for removing linear drift or baseline shifts from your data.

## Features

- **Separate Processing**: Subtract linear fits from hyperpol and depol curves independently
- **Real-time Updates**: Automatically updates the plot after subtraction
- **Reset Functionality**: Restore original data when needed
- **Integration**: Seamlessly integrates with existing curve fitting workflow
- **GUI Controls**: Easy-to-use interface for all operations

## Modules

### Core Modules

1. **`src/analysis/linear_fit_subtractor.py`**
   - Core subtraction logic
   - Data management for original and subtracted curves
   - Linear fit parameter handling

2. **`src/gui/linear_fit_subtraction_gui.py`**
   - GUI controls and display
   - Status monitoring
   - Results visualization

3. **`src/gui/linear_fit_subtraction_integration.py`**
   - Integration with main application
   - Plot reloading functionality
   - Processor data management

### Integration Module

4. **`src/gui/linear_fit_subtraction_example.py`**
   - Example integration code
   - Connection to curve fitting manager
   - Usage patterns

### Examples

5. **`examples/linear_fit_subtraction_usage.py`**
   - Complete usage examples
   - Command-line and GUI demos
   - Visualization examples

## Usage

### Basic Integration

```python
from src.gui.linear_fit_subtraction_example import integrate_linear_fit_subtraction

# Integrate with main application
integration = integrate_linear_fit_subtraction(main_app)

# Get the GUI panel
panel = integration.get_panel()
```

### Manual Usage

```python
from src.analysis.linear_fit_subtractor import LinearFitSubtractor

# Create subtractor
subtractor = LinearFitSubtractor()

# Set original data
subtractor.set_original_data('hyperpol', hyperpol_data, hyperpol_times)
subtractor.set_original_data('depol', depol_data, depol_times)

# Set linear fit parameters (from curve fitting manager)
subtractor.set_fitted_curves('hyperpol', linear_params, linear_curve, r_squared)

# Perform subtraction
subtracted_data, times = subtractor.subtract_linear_fit('hyperpol')
```

## GUI Controls

The GUI panel provides the following controls:

- **Individual Curve Controls**:
  - Subtract Linear Fit (Hyperpol)
  - Subtract Linear Fit (Depol)
  - Reset buttons for each curve

- **Both Curves Controls**:
  - Subtract Both Curves
  - Reset All

- **Plot Management**:
  - Reload Plot button

- **Status Display**:
  - Real-time status updates
  - Fit information display
  - Subtraction status indicators

## Workflow

1. **Load Data**: Ensure hyperpol and depol purple curves are loaded
2. **Perform Linear Fitting**: Use the curve fitting manager to fit linear curves
3. **Subtract**: Use the subtraction panel to subtract linear fits
4. **Visualize**: The plot automatically reloads with subtracted data
5. **Reset**: Use reset buttons to restore original data if needed

## Integration Points

The feature integrates with:

- **Curve Fitting Manager**: Automatically receives linear fit parameters
- **Action Potential Processor**: Updates modified curve data
- **Main Plot**: Reloads with updated data
- **Status System**: Provides user feedback

## Error Handling

The feature includes comprehensive error handling:

- Data validation before operations
- Graceful failure with user feedback
- Automatic fallback to original data
- Detailed logging for debugging

## Example Output

After subtraction, you'll see:

- Original purple curves with linear drift
- Subtracted curves with drift removed
- Status indicators showing completion
- Fit information display
- Updated plot with corrected data

## Requirements

- PyQt6 for GUI components
- NumPy for data processing
- Matplotlib for plotting
- Existing curve fitting infrastructure

## Notes

- The feature preserves original data for easy restoration
- Linear fits must be performed before subtraction
- Plot reloading uses multiple fallback methods for compatibility
- All operations are logged for debugging purposes
