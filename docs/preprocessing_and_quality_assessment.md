# Preprocessing and Quality Assessment

## Overview

The preprocessing and quality assessment module provides advanced signal preprocessing techniques and comprehensive quality evaluation for patch clamp electrophysiology data. This module implements sophisticated algorithms for noise reduction, artifact detection, baseline correction, and signal quality assessment.

**File**: `src/analysis/enhanced_preprocessing.py`  
**Main Classes**: `SignalQualityAssessor`, `EnhancedPreprocessor`

## Key Features

### 1. Advanced Signal Preprocessing
- **Multi-stage Baseline Correction**: Multiple baseline correction algorithms
- **Adaptive Noise Reduction**: Context-aware noise filtering
- **Artifact Detection**: Automatic identification and removal of artifacts
- **Signal Quality Assessment**: Comprehensive quality metrics

### 2. Quality Assessment
- **Signal-to-Noise Ratio**: SNR calculation and monitoring
- **Noise Level Estimation**: Automatic noise level detection
- **Artifact Detection**: Spike and discontinuity identification
- **Overall Quality Scoring**: Composite quality assessment

### 3. Preprocessing Methods
- **Baseline Correction**: Mean, median, linear, polynomial, adaptive
- **Noise Reduction**: Gaussian, median, bilateral, Savitzky-Golay, wavelet
- **Artifact Removal**: Spike detection and correction
- **Signal Enhancement**: Adaptive filtering and smoothing

## Signal Quality Assessor

### Initialization

```python
from src.analysis.enhanced_preprocessing import SignalQualityAssessor

# Initialize quality assessor
assessor = SignalQualityAssessor()
```

### Quality Assessment

#### Basic Quality Assessment
```python
# Assess signal quality
quality_metrics = assessor.assess_quality(data, sampling_rate=10000)

print(f"Signal-to-Noise Ratio: {quality_metrics['snr']:.2f}")
print(f"Noise Level: {quality_metrics['noise_level']:.6f}")
print(f"Overall Quality: {quality_metrics['overall_quality']:.3f}")
```

#### Comprehensive Quality Analysis
```python
# Detailed quality analysis
detailed_metrics = assessor.assess_quality_detailed(
    data, 
    sampling_rate=10000,
    include_frequency_analysis=True,
    include_artifact_detection=True
)

# Access specific metrics
print(f"Dynamic Range: {detailed_metrics['dynamic_range']:.6f}")
print(f"RMS Value: {detailed_metrics['rms']:.6f}")
print(f"Spike Count: {detailed_metrics['spike_count']}")
print(f"Drift Measure: {detailed_metrics['drift_measure']:.6f}")
```

### Quality Metrics

#### Signal-to-Noise Ratio
```python
snr = assessor.calculate_snr(data)
print(f"SNR: {snr:.2f} dB")
```

#### Noise Level Estimation
```python
noise_level = assessor.estimate_noise_level(data)
print(f"Noise Level: {noise_level:.6f}")
```

#### Artifact Detection
```python
artifacts = assessor.detect_artifacts(data)
print(f"Detected {len(artifacts)} artifacts")
for i, artifact in enumerate(artifacts):
    print(f"Artifact {i+1}: position {artifact['position']}, type {artifact['type']}")
```

## Enhanced Preprocessor

### Initialization

```python
from src.analysis.enhanced_preprocessing import EnhancedPreprocessor

# Initialize preprocessor
preprocessor = EnhancedPreprocessor()
```

### Preprocessing Methods

#### 1. Baseline Correction

##### Mean Baseline Correction
```python
# Simple mean baseline correction
corrected_data = preprocessor.correct_baseline(
    data, 
    method='mean',
    baseline_window=100  # Use first 100 points for baseline
)
```

##### Linear Baseline Correction
```python
# Linear baseline correction
corrected_data = preprocessor.correct_baseline(
    data, 
    method='linear',
    baseline_points=[0, 100, 200]  # Specific baseline points
)
```

##### Polynomial Baseline Correction
```python
# Polynomial baseline correction
corrected_data = preprocessor.correct_baseline(
    data, 
    method='polynomial',
    degree=2,  # Quadratic polynomial
    baseline_points=[0, 100, 200, 300]
)
```

##### Adaptive Baseline Correction
```python
# Adaptive baseline correction
corrected_data = preprocessor.correct_baseline(
    data, 
    method='adaptive',
    window_size=50,
    threshold=0.1
)
```

#### 2. Noise Reduction

##### Gaussian Filtering
```python
# Gaussian noise reduction
filtered_data = preprocessor.reduce_noise(
    data, 
    method='gaussian',
    sigma=1.0  # Standard deviation
)
```

##### Median Filtering
```python
# Median filtering
filtered_data = preprocessor.reduce_noise(
    data, 
    method='median',
    kernel_size=5
)
```

##### Savitzky-Golay Filtering
```python
# Savitzky-Golay filtering
filtered_data = preprocessor.reduce_noise(
    data, 
    method='savitzky_golay',
    window_length=11,
    polyorder=3
)
```

##### Wavelet Denoising
```python
# Wavelet denoising
filtered_data = preprocessor.reduce_noise(
    data, 
    method='wavelet',
    wavelet='db4',
    threshold_mode='soft'
)
```

##### Bilateral Filtering
```python
# Bilateral filtering
filtered_data = preprocessor.reduce_noise(
    data, 
    method='bilateral',
    sigma_color=0.1,
    sigma_space=1.0
)
```

#### 3. Artifact Removal

##### Spike Detection and Removal
```python
# Detect and remove spikes
cleaned_data = preprocessor.remove_artifacts(
    data, 
    artifact_types=['spikes'],
    threshold=3.0  # Z-score threshold
)
```

##### Discontinuity Correction
```python
# Detect and correct discontinuities
cleaned_data = preprocessor.remove_artifacts(
    data, 
    artifact_types=['discontinuities'],
    max_gap=5  # Maximum gap size
)
```

##### Comprehensive Artifact Removal
```python
# Remove all types of artifacts
cleaned_data = preprocessor.remove_artifacts(
    data, 
    artifact_types=['spikes', 'discontinuities', 'outliers'],
    spike_threshold=3.0,
    discontinuity_threshold=0.5,
    outlier_threshold=2.5
)
```

### Advanced Preprocessing

#### Multi-Stage Processing
```python
# Multi-stage preprocessing pipeline
pipeline = [
    ('baseline_correction', {'method': 'adaptive'}),
    ('noise_reduction', {'method': 'savitzky_golay', 'window_length': 11}),
    ('artifact_removal', {'artifact_types': ['spikes'], 'threshold': 3.0}),
    ('final_smoothing', {'method': 'gaussian', 'sigma': 0.5})
]

processed_data = preprocessor.process_pipeline(data, pipeline)
```

#### Adaptive Processing
```python
# Adaptive processing based on signal characteristics
processed_data = preprocessor.process_adaptive(
    data, 
    quality_threshold=0.7,
    noise_level_threshold=0.01
)
```

### Quality Assessment Integration

#### Preprocessing with Quality Monitoring
```python
# Process with quality monitoring
result = preprocessor.process_with_quality_monitoring(
    data, 
    target_quality=0.8,
    max_iterations=5
)

print(f"Final quality: {result['quality_score']:.3f}")
print(f"Iterations used: {result['iterations']}")
print(f"Processing log: {result['processing_log']}")
```

#### Quality-Based Method Selection
```python
# Select best preprocessing method based on quality
best_method = preprocessor.select_best_method(
    data, 
    methods=['gaussian', 'median', 'savitzky_golay', 'wavelet'],
    quality_metric='snr'
)

print(f"Best method: {best_method['method']}")
print(f"Quality improvement: {best_method['improvement']:.3f}")
```

## Preprocessing Results

### Results Data Structure
```python
@dataclass
class PreprocessingResults:
    original_data: np.ndarray
    processed_data: np.ndarray
    baseline: Optional[np.ndarray] = None
    noise_estimate: Optional[float] = None
    quality_score: Optional[float] = None
    artifacts_detected: Optional[List[Tuple[int, int]]] = None
    processing_log: Optional[List[str]] = None
```

### Accessing Results
```python
# Get preprocessing results
results = preprocessor.get_results()

# Access specific components
original = results.original_data
processed = results.processed_data
baseline = results.baseline
quality = results.quality_score
artifacts = results.artifacts_detected
log = results.processing_log
```

## Configuration

### Preprocessing Configuration
```python
# Configure preprocessing parameters
config = {
    'baseline_correction': {
        'method': 'adaptive',
        'window_size': 50,
        'threshold': 0.1
    },
    'noise_reduction': {
        'method': 'savitzky_golay',
        'window_length': 11,
        'polyorder': 3
    },
    'artifact_removal': {
        'spike_threshold': 3.0,
        'discontinuity_threshold': 0.5,
        'outlier_threshold': 2.5
    },
    'quality_assessment': {
        'min_snr': 10.0,
        'max_noise_level': 0.01,
        'min_quality_score': 0.7
    }
}

preprocessor.configure(config)
```

### Method-Specific Parameters
```python
# Gaussian filter parameters
gaussian_params = {
    'sigma': 1.0,
    'truncate': 4.0
}

# Savitzky-Golay parameters
savgol_params = {
    'window_length': 11,
    'polyorder': 3,
    'deriv': 0,
    'delta': 1.0
}

# Wavelet parameters
wavelet_params = {
    'wavelet': 'db4',
    'mode': 'soft',
    'threshold': 0.1
}
```

## Performance Optimization

### Memory Management
```python
# Optimize memory usage
preprocessor.optimize_memory_usage()

# Process in chunks for large datasets
chunked_results = preprocessor.process_chunked(
    data, 
    chunk_size=1000,
    overlap=100
)
```

### Processing Speed
```python
# Enable fast mode
preprocessor.enable_fast_mode()

# Use optimized algorithms
preprocessor.use_optimized_algorithms()

# Parallel processing
preprocessor.enable_parallel_processing()
```

## Integration Examples

### Basic Preprocessing Workflow
```python
from src.analysis.enhanced_preprocessing import EnhancedPreprocessor, SignalQualityAssessor

# Load data
data = np.load('patch_clamp_data.npy')
time_data = np.load('time_data.npy')

# Initialize components
preprocessor = EnhancedPreprocessor()
assessor = SignalQualityAssessor()

# Assess original quality
original_quality = assessor.assess_quality(data)
print(f"Original quality: {original_quality['overall_quality']:.3f}")

# Preprocess data
processed_data = preprocessor.process(
    data,
    baseline_method='adaptive',
    noise_reduction_method='savitzky_golay',
    artifact_removal=True
)

# Assess processed quality
processed_quality = assessor.assess_quality(processed_data)
print(f"Processed quality: {processed_quality['overall_quality']:.3f}")

# Compare results
improvement = processed_quality['overall_quality'] - original_quality['overall_quality']
print(f"Quality improvement: {improvement:.3f}")
```

### Advanced Preprocessing with Quality Monitoring
```python
# Advanced preprocessing with iterative quality improvement
def preprocess_with_quality_control(data, target_quality=0.8):
    preprocessor = EnhancedPreprocessor()
    assessor = SignalQualityAssessor()
    
    # Initial quality assessment
    current_quality = assessor.assess_quality(data)
    print(f"Initial quality: {current_quality['overall_quality']:.3f}")
    
    # Iterative preprocessing
    max_iterations = 5
    current_data = data.copy()
    
    for iteration in range(max_iterations):
        if current_quality['overall_quality'] >= target_quality:
            break
            
        # Select preprocessing method based on quality issues
        if current_quality['snr'] < 10:
            # Low SNR - apply noise reduction
            current_data = preprocessor.reduce_noise(
                current_data, 
                method='savitzky_golay'
            )
        elif current_quality['spike_count'] > 5:
            # High spike count - remove artifacts
            current_data = preprocessor.remove_artifacts(
                current_data, 
                artifact_types=['spikes']
            )
        else:
            # Apply baseline correction
            current_data = preprocessor.correct_baseline(
                current_data, 
                method='adaptive'
            )
        
        # Reassess quality
        current_quality = assessor.assess_quality(current_data)
        print(f"Iteration {iteration + 1}: Quality = {current_quality['overall_quality']:.3f}")
    
    return current_data, current_quality

# Use the function
processed_data, final_quality = preprocess_with_quality_control(data)
```

### Integration with Action Potential Processor
```python
from src.analysis.action_potential import ActionPotentialProcessor
from src.analysis.enhanced_preprocessing import EnhancedPreprocessor

# Load raw data
raw_data = np.load('raw_patch_clamp_data.npy')
time_data = np.load('time_data.npy')

# Preprocess data before analysis
preprocessor = EnhancedPreprocessor()
processed_data = preprocessor.process(
    raw_data,
    baseline_method='adaptive',
    noise_reduction_method='savitzky_golay',
    artifact_removal=True
)

# Process with ActionPotentialProcessor
processor = ActionPotentialProcessor(processed_data, time_data, params)
processor.process()

# Get results
curves = processor.get_curves()
quality = processor.assess_quality()

print(f"Analysis completed with quality score: {quality['overall_quality']:.3f}")
```

## Troubleshooting

### Common Issues

#### 1. Poor Quality After Preprocessing
**Symptoms**: Quality score decreases after preprocessing
**Solutions**:
- Check preprocessing parameters
- Try different methods
- Verify data integrity
- Adjust quality thresholds

#### 2. Over-filtering
**Symptoms**: Signal appears too smooth, loss of important features
**Solutions**:
- Reduce filter strength
- Use gentler methods
- Adjust parameters
- Check quality metrics

#### 3. Under-filtering
**Symptoms**: High noise level, poor signal quality
**Solutions**:
- Increase filter strength
- Use stronger methods
- Apply multiple stages
- Check noise characteristics

### Debugging Tools

#### Quality Visualization
```python
# Plot quality metrics
assessor.plot_quality_metrics(data, processed_data)

# Plot preprocessing steps
preprocessor.plot_preprocessing_steps(data)

# Plot artifact detection
preprocessor.plot_artifact_detection(data)
```

#### Detailed Logging
```python
# Enable detailed logging
import logging
logging.basicConfig(level=logging.DEBUG)

# Process with logging
preprocessor.enable_verbose_logging()
processed_data = preprocessor.process(data)
```

## API Reference

### SignalQualityAssessor Methods
- `assess_quality()`: Basic quality assessment
- `assess_quality_detailed()`: Comprehensive quality analysis
- `calculate_snr()`: Calculate signal-to-noise ratio
- `estimate_noise_level()`: Estimate noise level
- `detect_artifacts()`: Detect artifacts in signal
- `plot_quality_metrics()`: Visualize quality metrics

### EnhancedPreprocessor Methods
- `process()`: Main preprocessing method
- `correct_baseline()`: Apply baseline correction
- `reduce_noise()`: Apply noise reduction
- `remove_artifacts()`: Remove artifacts
- `process_pipeline()`: Multi-stage processing
- `process_adaptive()`: Adaptive processing
- `select_best_method()`: Method selection
- `get_results()`: Get preprocessing results
