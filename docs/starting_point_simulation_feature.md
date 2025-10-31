# Starting Point Simulation Feature

## Overview

The Starting Point Simulation feature is a new tool that helps users find the optimal starting point for ActionPotentialTab analysis. This feature addresses the common problem where poor starting point selection leads to outstanding spikes in purple curves, particularly at the beginning of depolarization and end/beginning of hyperpolarization segments.

## Problem Description

When analyzing action potential data, the starting point (n) determines:
- **Normalized curve segments**: `n` to `n+199`, `n+200` to `n+399`, etc.
- **Purple curve generation**: Uses segments at `n+800` to `n+999` (depol) and `n+1000` to `n+1199` (hyperpol)
- **Curve smoothness**: Poor starting points can cause outstanding spikes in purple curves

The default starting point is 35, but this may not be optimal for all datasets, leading to:
- Highly outstanding points in purple curves
- Irregular spikes at segment boundaries
- Poor curve smoothness and analysis quality

## Solution

The simulation feature automatically tests different starting points and evaluates curve quality to recommend the optimal value that produces smooth curves without outstanding points.

## Features

### 1. Automated Testing
- Tests a range of starting points (configurable, default: 10-100)
- Evaluates curve smoothness and outstanding points
- Provides statistical analysis of curve quality

### 2. Quality Metrics
- **Smoothness Score**: Based on derivative variance and curve regularity (0-1, higher is better)
- **Outstanding Points**: Count of spikes and irregularities detected
- **Overall Quality**: Categorical assessment (excellent/good/fair/poor)

### 3. User Interface
- Integrated button in ActionPotentialTab: "Find Optimal Point"
- Simulation dialog with progress tracking
- Detailed results display
- One-click application of recommended starting point

### 4. Analysis Methods
- **Statistical Analysis**: Variance, skewness, kurtosis of curves
- **Derivative Analysis**: First and second derivative smoothness
- **Outlier Detection**: Multiple methods for spike detection
- **Comparative Analysis**: Performance vs. default starting point

## Usage

### In ActionPotentialTab

1. Load your signal data
2. Go to the Action Potential tab
3. In the "Normalization Point" section, click "Find Optimal Point"
4. Configure simulation parameters (optional):
   - Starting point range (default: 10-100)
   - Step size (default: 5)
5. Click "Run Simulation"
6. Review results and click "Apply Recommended Starting Point"
7. Run analysis with the optimized starting point

### Programmatic Usage

```python
from src.analysis.starting_point_simulator import StartingPointSimulator

# Create simulator
simulator = StartingPointSimulator(data, time_data, params)

# Configure simulation
simulator.start_point_range = (10, 100)
simulator.step_size = 5

# Run simulation
results = simulator.run_simulation()

# Get optimal starting point
optimal_point = results['optimal_starting_point']
```

## Implementation Details

### Core Classes

#### `StartingPointSimulator`
- Main simulation engine
- Tests different starting points
- Evaluates curve quality
- Provides recommendations

#### `StartingPointSimulationGUI`
- User interface for simulation
- Progress tracking
- Results display
- Integration with ActionPotentialTab

### Key Methods

#### `run_simulation(progress_callback=None)`
Runs the complete simulation across all starting points.

#### `_test_starting_point(start_point)`
Tests a specific starting point and evaluates curve quality.

#### `_evaluate_curve_quality(hyperpol_data, depol_data, ...)`
Evaluates the quality of purple curves based on smoothness and outstanding points.

#### `_analyze_curve_smoothness(data, times, curve_type)`
Analyzes the smoothness of a single curve using statistical methods.

### Quality Assessment

The simulation uses multiple criteria to assess curve quality:

1. **Smoothness Analysis**
   - First derivative variance (lower = smoother)
   - Second derivative variance (lower = smoother)
   - Local variance analysis

2. **Outstanding Point Detection**
   - Statistical outlier detection (3-sigma rule)
   - Derivative-based spike detection
   - Local variance outliers

3. **Overall Quality Rating**
   - Excellent: Smoothness ≥ 0.8, Outstanding points ≤ 2
   - Good: Smoothness ≥ 0.6, Outstanding points ≤ 5
   - Fair: Smoothness ≥ 0.4, Outstanding points ≤ 10
   - Poor: Below fair criteria

## Configuration

### Simulation Parameters

```python
# Range of starting points to test
simulator.start_point_range = (10, 100)  # (min, max)

# Step size between tests
simulator.step_size = 5

# Analysis parameters
params = {
    'n_cycles': 2,
    't0': 20.0,
    't1': 100.0,
    't2': 100.0,
    'V0': -80.0,
    'V1': -100.0,
    'V2': -20.0,
    'use_alternative_method': False
}
```

## Results Interpretation

### Smoothness Score (0-1)
- **0.8-1.0**: Excellent smoothness, minimal irregularities
- **0.6-0.8**: Good smoothness, few outstanding points
- **0.4-0.6**: Fair smoothness, some irregularities
- **0.0-0.4**: Poor smoothness, many outstanding points

### Outstanding Points
- **0-2**: Excellent, very smooth curves
- **3-5**: Good, minor irregularities
- **6-10**: Fair, some noticeable spikes
- **>10**: Poor, many outstanding points

### Confidence Levels
- **High**: Clear optimal point with excellent metrics
- **Medium**: Good optimal point with solid improvement
- **Low**: Marginal improvement or unclear results

## Benefits

1. **Automated Optimization**: No manual trial-and-error needed
2. **Quantitative Analysis**: Objective metrics for curve quality
3. **Time Saving**: Quickly finds optimal parameters
4. **Improved Results**: Better curve smoothness and analysis quality
5. **User-Friendly**: Simple interface integrated into existing workflow

## Technical Notes

### Performance
- Simulation time depends on data size and range tested
- Typical simulation: 10-30 seconds for 1000-point range
- Progress tracking provides real-time feedback

### Memory Usage
- Each test creates a new ActionPotentialProcessor instance
- Memory usage scales with number of tests
- Results are stored for detailed analysis

### Error Handling
- Graceful handling of failed tests
- Detailed error logging
- Fallback to default starting point if simulation fails

## Future Enhancements

1. **Parallel Processing**: Multi-threaded simulation for faster results
2. **Advanced Metrics**: Additional curve quality measures
3. **Visualization**: Plot comparison of different starting points
4. **Batch Processing**: Test multiple datasets simultaneously
5. **Machine Learning**: Predictive models for optimal starting points

## Troubleshooting

### Common Issues

1. **"No Data" Error**
   - Ensure data is loaded before running simulation
   - Check that filtered_data and time_data are available

2. **Simulation Fails**
   - Check data quality and parameters
   - Verify starting point range is reasonable
   - Review error logs for specific issues

3. **Poor Results**
   - Try different starting point ranges
   - Check data preprocessing
   - Verify analysis parameters

### Debug Information

Enable debug logging to see detailed simulation progress:

```python
import logging
logging.getLogger('src.analysis.starting_point_simulator').setLevel(logging.DEBUG)
```

## Examples

See `test_starting_point_simulation.py` for a complete working example demonstrating the simulation feature with synthetic data.
