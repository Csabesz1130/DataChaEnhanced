# Specialized Processors and Utilities

## Overview

The specialized processors and utilities module provides targeted analysis tools for specific aspects of patch clamp electrophysiology data. These components handle specialized tasks such as negative control processing, starting point simulation, and integration with external analysis tools.

## Components

### 1. Negative Control Processor (`negative_control_processor.py`)

**Purpose**: Handles negative control traces, averaging, baseline subtraction, and ohmic component removal as implemented in the original ChaMa VB application.

#### Key Class: `NegativeControlProcessor`

```python
from src.analysis.negative_control_processor import NegativeControlProcessor

# Initialize processor
processor = NegativeControlProcessor()
```

**Features**:
- **Baseline Subtraction**: Automatic baseline correction
- **Negative Control Averaging**: Averaging of multiple control traces
- **Ohmic Component Removal**: Removal of linear ohmic components
- **Charge Movement Calculation**: Calculation of charge movement parameters

#### Protocol Parameters

```python
from src.analysis.negative_control_processor import ProtocolParameters

# Set protocol parameters
params = ProtocolParameters(
    baseline_duration=20.0,           # Baseline duration in ms
    neg_control_on_duration=100.0,   # Negative control ON duration
    neg_control_off_duration=100.0,  # Negative control OFF duration
    num_control_traces=5,            # Number of control traces
    test_pulse_v1=-80.0,            # Test pulse voltage 1
    test_pulse_v2=-100.0,           # Test pulse voltage 2
    test_pulse_v3=-20.0,            # Test pulse voltage 3
    test_on_duration=100.0,         # Test pulse ON duration
    test_off_duration=100.0,        # Test pulse OFF duration
    sampling_interval=0.1,          # Sampling interval in ms
    neg_control_v1=-80.0,           # Negative control voltage 1
    neg_control_v2=-100.0           # Negative control voltage 2
)

processor.set_protocol_parameters(params)
```

#### Core Methods

##### Average Control Pulses
```python
# Average negative control pulses and subtract baseline
averaged_control = processor.average_control_pulses(
    current_trace, 
    time_data
)

print(f"Averaged {processor.protocol_params.num_control_traces} control traces")
print(f"Control trace length: {len(averaged_control)} points")
```

##### Remove Ohmic Component
```python
# Remove ohmic component from current trace
corrected_current = processor.remove_ohmic_component(
    current_trace,
    time_data
)

print(f"Ohmic component removed")
print(f"Original RMS: {np.sqrt(np.mean(current_trace**2)):.6f}")
print(f"Corrected RMS: {np.sqrt(np.mean(corrected_current**2)):.6f}")
```

##### Calculate Charge Movement
```python
# Calculate charge movement parameters
charge_movement = processor.calculate_charge_movement(
    current_trace,
    time_data
)

print(f"Charge movement: {charge_movement['total_charge']:.6f} pC")
print(f"Peak current: {charge_movement['peak_current']:.6f} pA")
print(f"Time to peak: {charge_movement['time_to_peak']:.3f} ms")
```

### 2. Starting Point Simulator (`starting_point_simulator.py`)

**Purpose**: Simulates different starting points for action potential analysis and evaluates which produces the smoothest purple curves.

#### Key Class: `StartingPointSimulator`

```python
from src.analysis.starting_point_simulator import StartingPointSimulator

# Initialize simulator
simulator = StartingPointSimulator(data, time_data, params)
```

**Features**:
- **Starting Point Testing**: Tests various starting points (n values)
- **Curve Smoothness Analysis**: Evaluates curve quality and smoothness
- **Optimal Point Recommendation**: Recommends best starting point
- **Batch Processing**: Efficient processing of multiple starting points

#### Simulation Parameters

```python
# Configure simulation parameters
simulator.set_simulation_parameters(
    start_point_range=(10, 100),  # Range of starting points to test
    step_size=5,                  # Step size between tests
    quality_threshold=0.8,        # Minimum quality threshold
    smoothness_weight=0.7         # Weight for smoothness in scoring
)
```

#### Core Methods

##### Run Simulation
```python
# Run complete simulation
def progress_callback(progress, message):
    print(f"Progress: {progress:.1f}% - {message}")

results = simulator.run_simulation(progress_callback)

print(f"Simulation completed")
print(f"Best starting point: {results['optimal_starting_point']}")
print(f"Quality score: {results['optimal_quality_score']:.3f}")
```

##### Test Single Starting Point
```python
# Test a specific starting point
result = simulator.test_starting_point(35)

print(f"Starting point 35:")
print(f"  Smoothness score: {result['smoothness_score']:.3f}")
print(f"  Outstanding points: {result['outstanding_points']}")
print(f"  Quality level: {result['quality_level']}")
```

##### Analyze Results
```python
# Analyze simulation results
analysis = simulator.analyze_results(results)

print(f"Analysis results:")
print(f"  Best starting point: {analysis['best_point']}")
print(f"  Quality improvement: {analysis['quality_improvement']:.3f}")
print(f"  Recommended parameters: {analysis['recommended_params']}")
```

### 3. Action Potential Integration (`action_potential_integration.py`)

**Purpose**: Provides integration utilities for connecting action potential analysis with other components.

#### Key Functions

##### Integration with GUI
```python
from src.analysis.action_potential_integration import integrate_with_gui

# Integrate with main GUI
integration = integrate_with_gui(main_app)

# Connect to GUI events
integration.connect_events()

# Update GUI with results
integration.update_display(results)
```

##### Data Export Integration
```python
from src.analysis.action_potential_integration import export_analysis_results

# Export results in multiple formats
export_analysis_results(
    results,
    output_dir='results',
    formats=['excel', 'csv', 'json']
)
```

### 4. AI Workflow Validator (`ai_workflow_validator.py`)

**Purpose**: Validates AI workflow and ensures proper integration between AI components.

#### Key Class: `AIWorkflowValidator`

```python
from src.analysis.ai_workflow_validator import AIWorkflowValidator

# Initialize validator
validator = AIWorkflowValidator()
```

**Features**:
- **Workflow Validation**: Validates AI workflow integrity
- **Component Integration**: Ensures proper component integration
- **Performance Monitoring**: Monitors AI performance
- **Error Detection**: Detects and reports workflow errors

#### Validation Methods

##### Validate Workflow
```python
# Validate complete AI workflow
validation_result = validator.validate_workflow(ai_components)

if validation_result['valid']:
    print("AI workflow validation passed")
else:
    print(f"Validation failed: {validation_result['errors']}")
```

##### Monitor Performance
```python
# Monitor AI performance
performance = validator.monitor_performance(ai_system)

print(f"Prediction accuracy: {performance['accuracy']:.3f}")
print(f"Processing time: {performance['processing_time']:.3f} s")
print(f"Memory usage: {performance['memory_usage']:.2f} MB")
```

### 5. AI Visualization Explain (`ai_visualization_explain.py`)

**Purpose**: Provides visualization and explanation tools for AI analysis results.

#### Key Class: `AIVisualizationExplainer`

```python
from src.analysis.ai_visualization_explain import AIVisualizationExplainer

# Initialize explainer
explainer = AIVisualizationExplainer()
```

**Features**:
- **Prediction Visualization**: Visualizes AI predictions
- **Feature Importance**: Shows feature importance
- **Confidence Visualization**: Displays confidence scores
- **Explanation Generation**: Generates human-readable explanations

#### Visualization Methods

##### Plot Predictions
```python
# Plot AI predictions
explainer.plot_predictions(
    curve_data,
    predictions,
    confidence_scores
)
```

##### Show Feature Importance
```python
# Display feature importance
explainer.plot_feature_importance(
    feature_names,
    importance_scores
)
```

##### Generate Explanation
```python
# Generate explanation for prediction
explanation = explainer.generate_explanation(
    prediction,
    feature_values,
    model_info
)

print(f"Explanation: {explanation}")
```

## Integration Examples

### Complete Analysis Workflow
```python
from src.analysis.negative_control_processor import NegativeControlProcessor
from src.analysis.starting_point_simulator import StartingPointSimulator
from src.analysis.action_potential import ActionPotentialProcessor

# Load data
data = np.load('patch_clamp_data.npy')
time_data = np.load('time_data.npy')

# Step 1: Process negative controls
neg_control_processor = NegativeControlProcessor()
neg_control_processor.set_protocol_parameters({
    'baseline_duration': 20.0,
    'num_control_traces': 5,
    'sampling_interval': 0.1
})

averaged_control = neg_control_processor.average_control_pulses(data, time_data)
corrected_data = neg_control_processor.remove_ohmic_component(data, time_data)

# Step 2: Find optimal starting point
params = {
    'n_cycles': 2,
    't0': 20, 't1': 100, 't2': 100, 't3': 1000,
    'V0': -80, 'V1': -100, 'V2': -20,
    'cell_area_cm2': 1e-4
}

simulator = StartingPointSimulator(corrected_data, time_data, params)
simulation_results = simulator.run_simulation()

optimal_start_point = simulation_results['optimal_starting_point']
print(f"Optimal starting point: {optimal_start_point}")

# Step 3: Process with optimal parameters
params['normalization_points'] = {'seg1': (optimal_start_point, optimal_start_point + 199)}
processor = ActionPotentialProcessor(corrected_data, time_data, params)
processor.process()

# Get results
curves = processor.get_curves()
quality = processor.assess_quality()

print(f"Analysis completed with quality score: {quality['overall_quality']:.3f}")
```

### AI-Enhanced Analysis
```python
from src.analysis.ai_workflow_validator import AIWorkflowValidator
from src.analysis.ai_visualization_explain import AIVisualizationExplainer
from src.analysis.ai_realtime_integration import AIIntegrationManager

# Initialize AI components
ai_manager = AIIntegrationManager(main_app)
validator = AIWorkflowValidator()
explainer = AIVisualizationExplainer()

# Validate AI workflow
validation = validator.validate_workflow(ai_manager.get_components())
if not validation['valid']:
    print(f"AI workflow validation failed: {validation['errors']}")
    return

# Process data with AI
ai_results = ai_manager.process_data(data, time_data)

# Explain results
explanation = explainer.generate_explanation(
    ai_results['predictions'],
    ai_results['features'],
    ai_results['model_info']
)

print(f"AI Analysis Results:")
print(f"  Predictions: {ai_results['predictions']}")
print(f"  Confidence: {ai_results['confidence']:.3f}")
print(f"  Explanation: {explanation}")

# Visualize results
explainer.plot_predictions(data, ai_results['predictions'], ai_results['confidence'])
```

### Batch Processing
```python
# Process multiple experiments
experiments = [
    {'data': data1, 'time': time1, 'params': params1},
    {'data': data2, 'time': time2, 'params': params2},
    {'data': data3, 'time': time3, 'params': params3}
]

results = []

for i, exp in enumerate(experiments):
    print(f"Processing experiment {i+1}/{len(experiments)}")
    
    # Process negative controls
    neg_processor = NegativeControlProcessor()
    neg_processor.set_protocol_parameters(exp['params'])
    corrected_data = neg_processor.remove_ohmic_component(exp['data'], exp['time'])
    
    # Find optimal starting point
    simulator = StartingPointSimulator(corrected_data, exp['time'], exp['params'])
    sim_results = simulator.run_simulation()
    
    # Process with optimal parameters
    exp['params']['normalization_points'] = {
        'seg1': (sim_results['optimal_starting_point'], sim_results['optimal_starting_point'] + 199)
    }
    
    processor = ActionPotentialProcessor(corrected_data, exp['time'], exp['params'])
    processor.process()
    
    # Store results
    results.append({
        'experiment': i+1,
        'optimal_start_point': sim_results['optimal_starting_point'],
        'curves': processor.get_curves(),
        'quality': processor.assess_quality()
    })

# Analyze batch results
print(f"Batch processing completed: {len(results)} experiments")
for result in results:
    print(f"Experiment {result['experiment']}: "
          f"Start point {result['optimal_start_point']}, "
          f"Quality {result['quality']['overall_quality']:.3f}")
```

## Configuration

### Negative Control Processor Configuration
```python
# Protocol configuration
protocol_config = {
    'baseline_duration': 20.0,
    'neg_control_on_duration': 100.0,
    'neg_control_off_duration': 100.0,
    'num_control_traces': 5,
    'test_pulse_v1': -80.0,
    'test_pulse_v2': -100.0,
    'test_pulse_v3': -20.0,
    'test_on_duration': 100.0,
    'test_off_duration': 100.0,
    'sampling_interval': 0.1,
    'neg_control_v1': -80.0,
    'neg_control_v2': -100.0
}

processor.set_protocol_parameters(protocol_config)
```

### Starting Point Simulator Configuration
```python
# Simulation configuration
simulation_config = {
    'start_point_range': (10, 100),
    'step_size': 5,
    'quality_threshold': 0.8,
    'smoothness_weight': 0.7,
    'outstanding_threshold': 0.1,
    'max_iterations': 1000
}

simulator.configure(simulation_config)
```

### AI Workflow Configuration
```python
# AI workflow configuration
ai_config = {
    'enable_validation': True,
    'performance_monitoring': True,
    'error_detection': True,
    'auto_correction': False,
    'logging_level': 'INFO'
}

validator.configure(ai_config)
```

## Troubleshooting

### Common Issues

#### 1. Negative Control Processing Issues
**Symptoms**: Poor baseline correction, incorrect averaging
**Solutions**:
- Check protocol parameters
- Verify data quality
- Adjust baseline window
- Check sampling interval

#### 2. Starting Point Simulation Issues
**Symptoms**: No optimal point found, poor curve quality
**Solutions**:
- Expand starting point range
- Adjust quality thresholds
- Check data preprocessing
- Verify parameter ranges

#### 3. AI Workflow Issues
**Symptoms**: Validation failures, poor performance
**Solutions**:
- Check component integration
- Verify AI model loading
- Adjust performance thresholds
- Check data compatibility

### Debugging Tools

#### Verbose Logging
```python
# Enable detailed logging
import logging
logging.basicConfig(level=logging.DEBUG)

# Enable component-specific logging
neg_processor.enable_debug_logging()
simulator.enable_debug_logging()
validator.enable_debug_logging()
```

#### Performance Monitoring
```python
# Monitor performance
performance = validator.monitor_performance(ai_system)
print(f"Performance metrics: {performance}")

# Profile specific components
profile_results = validator.profile_component(component_name)
print(f"Profile results: {profile_results}")
```

## API Reference

### NegativeControlProcessor Methods
- `set_protocol_parameters()`: Set protocol parameters
- `average_control_pulses()`: Average control pulses
- `remove_ohmic_component()`: Remove ohmic component
- `calculate_charge_movement()`: Calculate charge movement
- `get_processing_results()`: Get processing results

### StartingPointSimulator Methods
- `set_simulation_parameters()`: Set simulation parameters
- `run_simulation()`: Run complete simulation
- `test_starting_point()`: Test single starting point
- `analyze_results()`: Analyze simulation results
- `get_optimal_parameters()`: Get optimal parameters

### AIWorkflowValidator Methods
- `validate_workflow()`: Validate AI workflow
- `monitor_performance()`: Monitor performance
- `detect_errors()`: Detect workflow errors
- `generate_report()`: Generate validation report

### AIVisualizationExplainer Methods
- `plot_predictions()`: Plot AI predictions
- `plot_feature_importance()`: Plot feature importance
- `generate_explanation()`: Generate explanation
- `visualize_confidence()`: Visualize confidence scores
