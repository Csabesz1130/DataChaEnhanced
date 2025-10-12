# Curve Fitting and Management

## Overview

The curve fitting and management system provides interactive tools for manual curve fitting, automated fitting algorithms, and comprehensive curve analysis. This system enables users to perform precise curve fitting operations on purple curves (hyperpolarization and depolarization segments) with both manual and automated approaches.

## Components

### 1. Curve Fitting Manager (`curve_fitting_manager.py`)

**Purpose**: Interactive curve fitting tools for manual point selection and curve fitting operations.

#### Key Class: `CurveFittingManager`

```python
from src.analysis.curve_fitting_manager import CurveFittingManager

# Initialize with matplotlib figure and axes
manager = CurveFittingManager(figure, ax)
```

**Features**:
- **Interactive Point Selection**: Click-to-select points for fitting
- **Multiple Fitting Types**: Linear and exponential fitting
- **Real-time Visualization**: Live preview of fitted curves
- **Quality Metrics**: R-squared and statistical validation
- **Integration Points**: Separate fitting for integration ranges

### 2. Linear Fit Subtractor (`linear_fit_subtractor.py`)

**Purpose**: Subtraction of linear fits from purple curves for drift correction.

#### Key Class: `LinearFitSubtractor`

```python
from src.analysis.linear_fit_subtractor import LinearFitSubtractor

# Initialize subtractor
subtractor = LinearFitSubtractor()
```

**Features**:
- **Separate Processing**: Independent handling of hyperpol and depol curves
- **Linear Component Removal**: Subtraction of fitted linear trends
- **Data Preservation**: Maintains original data for comparison
- **Quality Assessment**: Validation of subtraction results

## Curve Fitting Manager

### Initialization

```python
import matplotlib.pyplot as plt
from src.analysis.curve_fitting_manager import CurveFittingManager

# Create figure and axes
fig, ax = plt.subplots()

# Initialize manager
manager = CurveFittingManager(fig, ax)

# Set curve data
manager.set_curve_data('hyperpol', hyperpol_data, hyperpol_times)
manager.set_curve_data('depol', depol_data, depol_times)
```

### Interactive Fitting Operations

#### 1. Linear Fitting

##### Start Linear Selection
```python
# Start selecting points for linear fitting
success = manager.start_linear_selection('hyperpol')

if success:
    print("Click two points on the hyperpolarization curve for linear fitting")
    # User clicks two points on the plot
```

##### Complete Linear Fitting
```python
# After point selection, complete the fitting
linear_result = manager.complete_linear_fitting('hyperpol')

if linear_result['success']:
    print(f"Linear fit completed: R² = {linear_result['r_squared']:.4f}")
    print(f"Equation: y = {linear_result['slope']:.6f}x + {linear_result['intercept']:.6f}")
```

#### 2. Exponential Fitting

##### Start Exponential Selection
```python
# Start selecting points for exponential fitting
success = manager.start_exponential_selection('depol')

if success:
    print("Click multiple points on the depolarization curve for exponential fitting")
    # User clicks multiple points on the plot
```

##### Complete Exponential Fitting
```python
# After point selection, complete the fitting
exp_result = manager.complete_exponential_fitting('depol')

if exp_result['success']:
    print(f"Exponential fit completed: R² = {exp_result['r_squared']:.4f}")
    print(f"Parameters: A={exp_result['amplitude']:.6f}, τ={exp_result['tau']:.6f}")
```

#### 3. Integration Range Selection

##### Start Integration Selection
```python
# Start selecting integration range
success = manager.start_integration_selection('hyperpol')

if success:
    print("Click start and end points for integration range")
    # User clicks start and end points
```

##### Complete Integration Selection
```python
# Complete integration range selection
integration_result = manager.complete_integration_selection('hyperpol')

if integration_result['success']:
    start_idx = integration_result['start_index']
    end_idx = integration_result['end_index']
    print(f"Integration range: points {start_idx} to {end_idx}")
```

### Fitting Methods

#### Linear Fitting
Uses least squares linear regression to fit a straight line to selected points.

**Equation**: `y = mx + b`

**Parameters**:
- `slope` (m): Slope of the line
- `intercept` (b): Y-intercept
- `r_squared`: Coefficient of determination

**Usage**:
```python
# Linear fitting for hyperpolarization
linear_result = manager.fit_linear('hyperpol', selected_points)

# Access results
slope = linear_result['slope']
intercept = linear_result['intercept']
r_squared = linear_result['r_squared']
```

#### Exponential Fitting
Fits exponential decay/growth curves to selected points.

**Equation**: `y = A * exp(-x/τ) + C`

**Parameters**:
- `amplitude` (A): Initial amplitude
- `tau` (τ): Time constant
- `offset` (C): Vertical offset
- `r_squared`: Goodness of fit

**Usage**:
```python
# Exponential fitting for depolarization
exp_result = manager.fit_exponential('depol', selected_points)

# Access results
amplitude = exp_result['amplitude']
tau = exp_result['tau']
offset = exp_result['offset']
r_squared = exp_result['r_squared']
```

### Event Handling

#### Click Event Management
```python
# Enable click events
manager.enable_click_events()

# Disable click events
manager.disable_click_events()

# Check if events are active
if manager.is_active:
    print("Click events are active")
```

#### Callback Functions
```python
# Set callback for fit completion
def on_fit_complete(curve_type, fit_type, result):
    print(f"Fit completed for {curve_type} {fit_type}")
    print(f"R² = {result['r_squared']:.4f}")

manager.set_fit_complete_callback(on_fit_complete)
```

### Visualization

#### Plot Fitted Curves
```python
# Plot linear fit
manager.plot_linear_fit('hyperpol')

# Plot exponential fit
manager.plot_exponential_fit('depol')

# Plot all fits
manager.plot_all_fits()
```

#### Clear Plots
```python
# Clear specific fit
manager.clear_linear_fit('hyperpol')

# Clear all fits
manager.clear_all_fits()

# Clear selected points
manager.clear_selected_points()
```

## Linear Fit Subtractor

### Initialization

```python
from src.analysis.linear_fit_subtractor import LinearFitSubtractor

# Initialize subtractor
subtractor = LinearFitSubtractor()
```

### Setting Fitted Curves

#### Set Linear Fit Parameters
```python
# Set linear fit for hyperpolarization
linear_params = {
    'slope': -0.001234,
    'intercept': 0.045678,
    'start_idx': 35,
    'end_idx': 234
}

linear_curve = {
    'times': hyperpol_times[35:235],
    'data': fitted_hyperpol_data
}

subtractor.set_fitted_curves(
    'hyperpol', 
    linear_params, 
    linear_curve, 
    r_squared=0.9856
)
```

#### Set Original Data
```python
# Set original data for subtraction
subtractor.set_original_data(
    'hyperpol', 
    original_hyperpol_data, 
    original_hyperpol_times
)
```

### Performing Subtraction

#### Subtract Linear Fit
```python
# Subtract linear fit from hyperpolarization curve
subtracted_data, subtracted_times = subtractor.subtract_linear_fit('hyperpol')

print(f"Subtracted {len(subtracted_data)} points")
print(f"Original range: {original_hyperpol_data.min():.6f} to {original_hyperpol_data.max():.6f}")
print(f"Subtracted range: {subtracted_data.min():.6f} to {subtracted_data.max():.6f}")
```

#### Batch Subtraction
```python
# Subtract from both curve types
hyperpol_subtracted = subtractor.subtract_linear_fit('hyperpol')
depol_subtracted = subtractor.subtract_linear_fit('depol')

# Get results
hyperpol_data, hyperpol_times = hyperpol_subtracted
depol_data, depol_times = depol_subtracted
```

### Quality Assessment

#### Validate Subtraction
```python
# Validate subtraction quality
validation = subtractor.validate_subtraction('hyperpol')

if validation['valid']:
    print("Subtraction validation passed")
    print(f"Residual RMS: {validation['residual_rms']:.6f}")
    print(f"Correlation: {validation['correlation']:.4f}")
else:
    print(f"Subtraction validation failed: {validation['error']}")
```

#### Get Subtraction Statistics
```python
# Get detailed statistics
stats = subtractor.get_subtraction_statistics('hyperpol')

print(f"Original mean: {stats['original_mean']:.6f}")
print(f"Subtracted mean: {stats['subtracted_mean']:.6f}")
print(f"Mean change: {stats['mean_change']:.6f}")
print(f"Variance reduction: {stats['variance_reduction']:.4f}")
```

## Advanced Features

### Custom Fitting Functions

#### Define Custom Function
```python
def custom_exponential(x, a, b, c):
    """Custom exponential function: y = a * exp(-b * x) + c"""
    return a * np.exp(-b * x) + c

# Use custom function
result = manager.fit_custom_function('depol', selected_points, custom_exponential)
```

#### Weighted Fitting
```python
# Define weights for points
weights = np.array([1.0, 1.0, 0.5, 0.5, 1.0, 1.0])  # Lower weight for middle points

# Perform weighted fitting
result = manager.fit_linear_weighted('hyperpol', selected_points, weights)
```

### Batch Processing

#### Process Multiple Curves
```python
# Process multiple curve types
curve_types = ['hyperpol', 'depol']
results = {}

for curve_type in curve_types:
    # Set data
    manager.set_curve_data(curve_type, data[curve_type], times[curve_type])
    
    # Perform linear fitting
    linear_result = manager.fit_linear_auto(curve_type)
    
    # Perform exponential fitting
    exp_result = manager.fit_exponential_auto(curve_type)
    
    results[curve_type] = {
        'linear': linear_result,
        'exponential': exp_result
    }
```

#### Automated Fitting
```python
# Automated fitting with default parameters
auto_results = manager.fit_all_curves_automatically()

# Automated fitting with custom parameters
custom_params = {
    'linear_points': 10,
    'exponential_points': 20,
    'min_r_squared': 0.8
}
auto_results = manager.fit_all_curves_automatically(custom_params)
```

### Data Export

#### Export Fitting Results
```python
# Export to dictionary
results_dict = manager.export_fitting_results()

# Export to JSON
manager.export_to_json('fitting_results.json')

# Export to CSV
manager.export_to_csv('fitting_results.csv')
```

#### Export Subtracted Data
```python
# Export subtracted data
subtractor.export_subtracted_data('subtracted_curves.json')

# Export with metadata
subtractor.export_with_metadata('subtracted_with_metadata.json')
```

## Integration with Main Application

### GUI Integration
```python
# Connect to main application GUI
def setup_curve_fitting_gui(main_app):
    # Get plot widget from main app
    plot_widget = main_app.get_plot_widget()
    figure, ax = plot_widget.get_figure_and_axes()
    
    # Initialize curve fitting manager
    manager = CurveFittingManager(figure, ax)
    
    # Connect to GUI controls
    manager.connect_to_gui_controls(main_app.get_curve_fitting_controls())
    
    return manager
```

### Event Integration
```python
# Connect to application events
def on_curve_data_loaded(data, times):
    # Update curve fitting manager with new data
    manager.set_curve_data('hyperpol', data['hyperpol'], times['hyperpol'])
    manager.set_curve_data('depol', data['depol'], times['depol'])
    
    # Enable fitting controls
    manager.enable_fitting_controls()

def on_fit_completed(curve_type, fit_type, result):
    # Update GUI with results
    main_app.update_fitting_results(curve_type, fit_type, result)
    
    # Update plot
    main_app.refresh_plot()
```

## Performance Optimization

### Memory Management
```python
# Clear unused data
manager.clear_unused_data()

# Optimize memory usage
manager.optimize_memory_usage()

# Monitor memory usage
memory_usage = manager.get_memory_usage()
print(f"Memory usage: {memory_usage:.2f} MB")
```

### Processing Speed
```python
# Enable fast mode
manager.enable_fast_mode()

# Use optimized algorithms
manager.use_optimized_algorithms()

# Parallel processing
manager.enable_parallel_processing()
```

## Troubleshooting

### Common Issues

#### 1. Poor Fitting Quality
**Symptoms**: Low R-squared values, poor visual fit
**Solutions**:
- Select better points (avoid outliers)
- Try different fitting methods
- Check data quality
- Use weighted fitting

#### 2. Point Selection Issues
**Symptoms**: Incorrect points selected, selection not working
**Solutions**:
- Check click event handling
- Verify data is loaded
- Clear previous selections
- Check coordinate conversion

#### 3. Memory Issues
**Symptoms**: Slow performance, memory errors
**Solutions**:
- Clear unused data
- Use smaller datasets
- Enable memory optimization
- Process in batches

### Debugging Tools

#### Verbose Logging
```python
import logging
logging.basicConfig(level=logging.DEBUG)

# Enable debug mode
manager.enable_debug_mode()
```

#### Visualization Tools
```python
# Plot selection process
manager.plot_selection_process()

# Plot fitting diagnostics
manager.plot_fitting_diagnostics()

# Plot residual analysis
manager.plot_residual_analysis()
```

## Examples

### Complete Fitting Workflow
```python
import matplotlib.pyplot as plt
from src.analysis.curve_fitting_manager import CurveFittingManager
from src.analysis.linear_fit_subtractor import LinearFitSubtractor

# Load data
hyperpol_data = np.load('hyperpol_curve.npy')
hyperpol_times = np.load('hyperpol_times.npy')
depol_data = np.load('depol_curve.npy')
depol_times = np.load('depol_times.npy')

# Create figure
fig, ax = plt.subplots(figsize=(12, 8))

# Initialize manager
manager = CurveFittingManager(fig, ax)

# Set data
manager.set_curve_data('hyperpol', hyperpol_data, hyperpol_times)
manager.set_curve_data('depol', depol_data, depol_times)

# Plot data
ax.plot(hyperpol_times, hyperpol_data, 'b-', label='Hyperpol', alpha=0.7)
ax.plot(depol_times, depol_data, 'r-', label='Depol', alpha=0.7)
ax.legend()
ax.grid(True)

# Interactive fitting
print("Click two points on hyperpolarization curve for linear fitting")
manager.start_linear_selection('hyperpol')

# Wait for user interaction, then complete fitting
# (In real application, this would be handled by GUI events)

# Complete linear fitting
linear_result = manager.complete_linear_fitting('hyperpol')
print(f"Linear fit: R² = {linear_result['r_squared']:.4f}")

# Plot linear fit
manager.plot_linear_fit('hyperpol')

# Perform exponential fitting on depolarization
print("Click multiple points on depolarization curve for exponential fitting")
manager.start_exponential_selection('depol')

# Complete exponential fitting
exp_result = manager.complete_exponential_fitting('depol')
print(f"Exponential fit: R² = {exp_result['r_squared']:.4f}")

# Plot exponential fit
manager.plot_exponential_fit('depol')

# Initialize subtractor
subtractor = LinearFitSubtractor()

# Set fitted curves
subtractor.set_fitted_curves('hyperpol', 
                           linear_result['params'], 
                           linear_result['curve'], 
                           linear_result['r_squared'])

# Set original data
subtractor.set_original_data('hyperpol', hyperpol_data, hyperpol_times)

# Perform subtraction
subtracted_data, subtracted_times = subtractor.subtract_linear_fit('hyperpol')

# Plot subtracted data
ax.plot(subtracted_times, subtracted_data, 'g--', label='Subtracted Hyperpol', linewidth=2)

# Show plot
plt.show()
```

### Automated Batch Processing
```python
# Automated processing of multiple curves
def process_curves_automatically(data_dict, times_dict):
    results = {}
    
    for curve_name, curve_data in data_dict.items():
        curve_times = times_dict[curve_name]
        
        # Set data
        manager.set_curve_data(curve_name, curve_data, curve_times)
        
        # Automated linear fitting
        linear_result = manager.fit_linear_auto(curve_name)
        
        # Automated exponential fitting
        exp_result = manager.fit_exponential_auto(curve_name)
        
        # Store results
        results[curve_name] = {
            'linear': linear_result,
            'exponential': exp_result,
            'best_fit': 'linear' if linear_result['r_squared'] > exp_result['r_squared'] else 'exponential'
        }
    
    return results

# Process curves
curve_data = {
    'hyperpol': hyperpol_data,
    'depol': depol_data
}

curve_times = {
    'hyperpol': hyperpol_times,
    'depol': depol_times
}

results = process_curves_automatically(curve_data, curve_times)

# Print results
for curve_name, result in results.items():
    print(f"{curve_name}:")
    print(f"  Linear R²: {result['linear']['r_squared']:.4f}")
    print(f"  Exponential R²: {result['exponential']['r_squared']:.4f}")
    print(f"  Best fit: {result['best_fit']}")
```

## API Reference

### CurveFittingManager Methods
- `set_curve_data()`: Set curve data for fitting
- `start_linear_selection()`: Start linear point selection
- `complete_linear_fitting()`: Complete linear fitting
- `start_exponential_selection()`: Start exponential point selection
- `complete_exponential_fitting()`: Complete exponential fitting
- `plot_linear_fit()`: Plot linear fit
- `plot_exponential_fit()`: Plot exponential fit
- `clear_all_fits()`: Clear all fits
- `export_fitting_results()`: Export results

### LinearFitSubtractor Methods
- `set_fitted_curves()`: Set fitted curve parameters
- `set_original_data()`: Set original data
- `subtract_linear_fit()`: Perform linear subtraction
- `validate_subtraction()`: Validate subtraction quality
- `get_subtraction_statistics()`: Get detailed statistics
- `export_subtracted_data()`: Export subtracted data
