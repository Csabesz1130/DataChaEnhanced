# Enhanced AI Features

## Overview

The Enhanced AI Features system provides advanced feature engineering and signal processing techniques for improved AI analysis. This system extends basic feature extraction with sophisticated signal processing methods to capture nuanced curve characteristics that experts implicitly use in their analysis.

**File**: `src/analysis/enhanced_ai_features.py`  
**Main Class**: `EnhancedFeatureExtractor`

## Key Features

### 1. Advanced Feature Engineering
- **Comprehensive Feature Set**: Extensive feature extraction covering multiple signal domains
- **Multi-scale Analysis**: Wavelet-based multi-scale signal analysis
- **Frequency Domain Features**: Spectral analysis and frequency characteristics
- **Shape Complexity Analysis**: Advanced curve characterization and complexity metrics

### 2. Wavelet Analysis
- **Multi-scale Decomposition**: Wavelet decomposition for multi-scale analysis
- **Energy Distribution**: Energy distribution analysis across scales
- **Wavelet Entropy**: Entropy-based wavelet analysis
- **Scale-specific Features**: Features extracted from different wavelet scales

### 3. Frequency Domain Analysis
- **Spectral Analysis**: FFT-based frequency domain analysis
- **Dominant Frequency**: Identification of dominant frequency components
- **Spectral Centroid**: Spectral centroid calculation
- **Bandwidth Analysis**: Frequency bandwidth characteristics

### 4. Shape Complexity Analysis
- **Curve Complexity**: Advanced curve complexity metrics
- **Inflection Points**: Detection and analysis of inflection points
- **Curvature Analysis**: Curvature-based shape analysis
- **Symmetry Analysis**: Symmetry and asymmetry metrics

## Initialization

### Basic Initialization
```python
from src.analysis.enhanced_ai_features import EnhancedFeatureExtractor

# Initialize enhanced feature extractor
feature_extractor = EnhancedFeatureExtractor()

# Extract comprehensive features
features = feature_extractor.extract_all_features(time_data, current_data)
```

### Advanced Initialization
```python
# Initialize with custom configuration
config = {
    'wavelet_analysis': True,
    'frequency_analysis': True,
    'shape_analysis': True,
    'complexity_analysis': True,
    'wavelet_type': 'db4',
    'max_wavelet_level': 5
}

feature_extractor = EnhancedFeatureExtractor(config)
```

## Feature Extraction Methods

### 1. Basic Statistical Features
```python
# Extract basic statistical features
basic_features = feature_extractor._extract_basic_features(time_data, current_data)

print(f"Mean: {basic_features['mean']:.6f}")
print(f"Standard deviation: {basic_features['std']:.6f}")
print(f"Median: {basic_features['median']:.6f}")
print(f"Range: {basic_features['range']:.6f}")
print(f"Energy: {basic_features['energy']:.6f}")
print(f"RMS: {basic_features['rms']:.6f}")
```

### 2. Wavelet Analysis Features
```python
# Extract wavelet-based features
wavelet_features = feature_extractor._extract_wavelet_features(current_data)

print(f"Wavelet entropy: {wavelet_features['wavelet_entropy']:.6f}")
print(f"Approximation energy: {wavelet_features['wavelet_approx_energy']:.6f}")
print(f"Detail energy: {wavelet_features['wavelet_detail_1_energy']:.6f}")
print(f"Energy ratio: {wavelet_features['wavelet_approx_energy_ratio']:.6f}")
```

### 3. Frequency Domain Features
```python
# Extract frequency domain features
freq_features = feature_extractor._extract_frequency_features(time_data, current_data)

print(f"Dominant frequency: {freq_features['dominant_frequency']:.3f} Hz")
print(f"Spectral centroid: {freq_features['spectral_centroid']:.3f} Hz")
print(f"Bandwidth: {freq_features['bandwidth']:.3f} Hz")
print(f"Peak frequency: {freq_features['peak_frequency']:.3f} Hz")
```

### 4. Shape Complexity Features
```python
# Extract shape complexity features
shape_features = feature_extractor._extract_shape_complexity_features(time_data, current_data)

print(f"Shape complexity: {shape_features['shape_complexity']:.6f}")
print(f"Curvature variance: {shape_features['curvature_variance']:.6f}")
print(f"Inflection points: {shape_features['inflection_points']}")
print(f"Symmetry index: {shape_features['symmetry_index']:.6f}")
```

### 5. Critical Points Features
```python
# Extract critical points features
critical_features = feature_extractor._extract_critical_points_features(time_data, current_data)

print(f"Peak amplitude: {critical_features['peak_amplitude']:.6f}")
print(f"Time to peak: {critical_features['time_to_peak']:.6f}")
print(f"Half decay time: {critical_features['half_decay_time']:.6f}")
print(f"Rise time: {critical_features['rise_time']:.6f}")
print(f"Fall time: {critical_features['fall_time']:.6f}")
```

### 6. Phase Space Features
```python
# Extract phase space features
phase_features = feature_extractor._extract_phase_space_features(time_data, current_data)

print(f"Phase space area: {phase_features['phase_space_area']:.6f}")
print(f"Trajectory length: {phase_features['trajectory_length']:.6f}")
print(f"Phase velocity: {phase_features['phase_velocity']:.6f}")
print(f"Phase acceleration: {phase_features['phase_acceleration']:.6f}")
```

## Advanced Analysis Methods

### 1. Comprehensive Feature Extraction
```python
# Extract all features at once
all_features = feature_extractor.extract_all_features(time_data, current_data)

print(f"Total features extracted: {len(all_features)}")
print(f"Feature names: {feature_extractor.feature_names}")

# Access specific feature categories
basic_features = {k: v for k, v in all_features.items() if k in ['mean', 'std', 'median']}
wavelet_features = {k: v for k, v in all_features.items() if k.startswith('wavelet_')}
freq_features = {k: v for k, v in all_features.items() if k.startswith('freq_')}
```

### 2. Feature Selection
```python
# Select most important features
important_features = feature_extractor.select_important_features(
    all_features,
    method='variance_threshold',
    threshold=0.01
)

print(f"Important features: {important_features}")

# Select features by category
wavelet_features = feature_extractor.select_features_by_category(
    all_features,
    category='wavelet'
)
```

### 3. Feature Normalization
```python
# Normalize features
normalized_features = feature_extractor.normalize_features(all_features)

print(f"Normalized features: {normalized_features}")

# Standardize features
standardized_features = feature_extractor.standardize_features(all_features)

print(f"Standardized features: {standardized_features}")
```

## Wavelet Analysis

### 1. Wavelet Decomposition
```python
# Perform wavelet decomposition
wavelet_result = feature_extractor.perform_wavelet_decomposition(
    current_data,
    wavelet='db4',
    max_level=5
)

print(f"Wavelet coefficients: {len(wavelet_result['coefficients'])}")
print(f"Energy distribution: {wavelet_result['energy_distribution']}")
print(f"Entropy: {wavelet_result['entropy']:.6f}")
```

### 2. Multi-scale Analysis
```python
# Analyze different scales
scale_analysis = feature_extractor.analyze_scales(
    current_data,
    scales=[1, 2, 3, 4, 5]
)

for scale, features in scale_analysis.items():
    print(f"Scale {scale}: Energy={features['energy']:.6f}, "
          f"Variance={features['variance']:.6f}")
```

### 3. Wavelet Denoising
```python
# Apply wavelet denoising
denoised_data = feature_extractor.wavelet_denoise(
    current_data,
    wavelet='db4',
    threshold='soft'
)

print(f"Original data: {len(current_data)} points")
print(f"Denoised data: {len(denoised_data)} points")
print(f"SNR improvement: {feature_extractor.calculate_snr_improvement(current_data, denoised_data):.2f} dB")
```

## Frequency Domain Analysis

### 1. Spectral Analysis
```python
# Perform spectral analysis
spectral_result = feature_extractor.perform_spectral_analysis(
    time_data,
    current_data
)

print(f"Dominant frequency: {spectral_result['dominant_frequency']:.3f} Hz")
print(f"Spectral centroid: {spectral_result['spectral_centroid']:.3f} Hz")
print(f"Bandwidth: {spectral_result['bandwidth']:.3f} Hz")
print(f"Total power: {spectral_result['total_power']:.6f}")
```

### 2. Frequency Band Analysis
```python
# Analyze frequency bands
band_analysis = feature_extractor.analyze_frequency_bands(
    time_data,
    current_data,
    bands={
        'low': (0, 10),      # 0-10 Hz
        'medium': (10, 50),  # 10-50 Hz
        'high': (50, 100)    # 50-100 Hz
    }
)

for band, features in band_analysis.items():
    print(f"{band} band: Power={features['power']:.6f}, "
          f"Peak={features['peak_frequency']:.3f} Hz")
```

### 3. Spectral Features
```python
# Extract spectral features
spectral_features = feature_extractor.extract_spectral_features(
    time_data,
    current_data
)

print(f"Spectral rolloff: {spectral_features['spectral_rolloff']:.3f} Hz")
print(f"Spectral flux: {spectral_features['spectral_flux']:.6f}")
print(f"Spectral contrast: {spectral_features['spectral_contrast']:.6f}")
print(f"Spectral flatness: {spectral_features['spectral_flatness']:.6f}")
```

## Shape Complexity Analysis

### 1. Curve Complexity
```python
# Analyze curve complexity
complexity_result = feature_extractor.analyze_curve_complexity(
    time_data,
    current_data
)

print(f"Shape complexity: {complexity_result['complexity']:.6f}")
print(f"Fractal dimension: {complexity_result['fractal_dimension']:.6f}")
print(f"Information content: {complexity_result['information_content']:.6f}")
print(f"Regularity index: {complexity_result['regularity_index']:.6f}")
```

### 2. Inflection Point Analysis
```python
# Analyze inflection points
inflection_result = feature_extractor.analyze_inflection_points(
    time_data,
    current_data
)

print(f"Inflection points: {inflection_result['count']}")
print(f"Inflection strength: {inflection_result['strength']:.6f}")
print(f"Inflection distribution: {inflection_result['distribution']:.6f}")
```

### 3. Curvature Analysis
```python
# Analyze curvature
curvature_result = feature_extractor.analyze_curvature(
    time_data,
    current_data
)

print(f"Mean curvature: {curvature_result['mean_curvature']:.6f}")
print(f"Curvature variance: {curvature_result['curvature_variance']:.6f}")
print(f"Max curvature: {curvature_result['max_curvature']:.6f}")
print(f"Curvature range: {curvature_result['curvature_range']:.6f}")
```

## Integration Examples

### Complete Feature Extraction Workflow
```python
from src.analysis.enhanced_ai_features import EnhancedFeatureExtractor
import numpy as np

# Load data
time_data = np.load('time_data.npy')
current_data = np.load('current_data.npy')

# Initialize feature extractor
feature_extractor = EnhancedFeatureExtractor()

# Extract comprehensive features
all_features = feature_extractor.extract_all_features(time_data, current_data)

# Analyze feature categories
basic_features = {k: v for k, v in all_features.items() 
                  if k in ['mean', 'std', 'median', 'range', 'energy']}
wavelet_features = {k: v for k, v in all_features.items() 
                   if k.startswith('wavelet_')}
freq_features = {k: v for k, v in all_features.items() 
                if k.startswith('freq_')}
shape_features = {k: v for k, v in all_features.items() 
                 if k.startswith('shape_')}

print(f"Feature extraction completed:")
print(f"  Basic features: {len(basic_features)}")
print(f"  Wavelet features: {len(wavelet_features)}")
print(f"  Frequency features: {len(freq_features)}")
print(f"  Shape features: {len(shape_features)}")
print(f"  Total features: {len(all_features)}")
```

### Feature Selection and Analysis
```python
# Select important features
important_features = feature_extractor.select_important_features(
    all_features,
    method='variance_threshold',
    threshold=0.01
)

# Normalize features
normalized_features = feature_extractor.normalize_features(important_features)

# Analyze feature importance
feature_importance = feature_extractor.analyze_feature_importance(
    normalized_features
)

print(f"Feature importance analysis:")
for feature, importance in feature_importance.items():
    print(f"  {feature}: {importance:.6f}")
```

### Integration with AI Systems
```python
from src.analysis.ai_curve_learning import CurveFittingAI

# Initialize AI system
ai_system = CurveFittingAI()

# Extract features
features = feature_extractor.extract_all_features(time_data, current_data)

# Train AI system with enhanced features
training_data = {
    'features': features,
    'labels': expert_labels,
    'metadata': experiment_metadata
}

ai_system.train_with_enhanced_features(training_data)

# Make predictions
predictions = ai_system.predict_with_enhanced_features(features)

print(f"AI predictions: {predictions}")
```

## Configuration Options

### 1. Feature Extraction Configuration
```python
# Configure feature extraction
extraction_config = {
    'wavelet_analysis': True,
    'frequency_analysis': True,
    'shape_analysis': True,
    'complexity_analysis': True,
    'phase_space_analysis': True,
    'wavelet_type': 'db4',
    'max_wavelet_level': 5,
    'frequency_bands': {
        'low': (0, 10),
        'medium': (10, 50),
        'high': (50, 100)
    }
}

feature_extractor.configure(extraction_config)
```

### 2. Wavelet Analysis Configuration
```python
# Configure wavelet analysis
wavelet_config = {
    'wavelet_type': 'db4',
    'max_level': 5,
    'threshold_mode': 'soft',
    'denoising': True,
    'energy_analysis': True,
    'entropy_analysis': True
}

feature_extractor.configure_wavelet_analysis(wavelet_config)
```

### 3. Frequency Analysis Configuration
```python
# Configure frequency analysis
freq_config = {
    'fft_size': 1024,
    'window_function': 'hann',
    'overlap': 0.5,
    'frequency_bands': {
        'low': (0, 10),
        'medium': (10, 50),
        'high': (50, 100)
    },
    'spectral_features': True
}

feature_extractor.configure_frequency_analysis(freq_config)
```

## Performance Optimization

### 1. Feature Caching
```python
# Enable feature caching
feature_extractor.enable_caching()

# Extract features with caching
features = feature_extractor.extract_all_features(time_data, current_data)

# Check cache status
cache_status = feature_extractor.get_cache_status()
print(f"Cache hits: {cache_status['hits']}")
print(f"Cache misses: {cache_status['misses']}")
print(f"Cache efficiency: {cache_status['efficiency']:.3f}")
```

### 2. Parallel Processing
```python
# Enable parallel processing
feature_extractor.enable_parallel_processing(max_workers=4)

# Extract features in parallel
features = feature_extractor.extract_all_features_parallel(
    time_data,
    current_data
)
```

### 3. Memory Optimization
```python
# Optimize memory usage
feature_extractor.optimize_memory_usage()

# Extract features with memory optimization
features = feature_extractor.extract_all_features(
    time_data,
    current_data,
    memory_optimized=True
)
```

## Troubleshooting

### Common Issues

#### 1. Feature Extraction Errors
**Symptoms**: Errors during feature extraction
**Solutions**:
- Check data quality and format
- Verify input parameters
- Adjust feature extraction configuration
- Check system resources

#### 2. Performance Issues
**Symptoms**: Slow feature extraction
**Solutions**:
- Enable feature caching
- Use parallel processing
- Optimize memory usage
- Adjust feature selection

#### 3. Memory Issues
**Symptoms**: High memory usage
**Solutions**:
- Enable memory optimization
- Reduce feature set
- Use chunked processing
- Check system resources

### Debugging Tools

#### 1. Feature Extraction Profiling
```python
# Profile feature extraction
profile_results = feature_extractor.profile_extraction(time_data, current_data)

print(f"Feature extraction timing:")
for feature_type, time in profile_results['timing'].items():
    print(f"  {feature_type}: {time:.3f} seconds")
```

#### 2. Detailed Logging
```python
# Enable detailed logging
import logging
logging.basicConfig(level=logging.DEBUG)

# Enable feature extractor logging
feature_extractor.enable_debug_logging()
```

#### 3. Feature Validation
```python
# Validate features
validation_results = feature_extractor.validate_features(features)

print(f"Feature validation:")
print(f"  Valid features: {validation_results['valid_count']}")
print(f"  Invalid features: {validation_results['invalid_count']}")
print(f"  Issues: {validation_results['issues']}")
```

## API Reference

### EnhancedFeatureExtractor Methods
- `extract_all_features()`: Extract comprehensive features
- `extract_basic_features()`: Extract basic statistical features
- `extract_wavelet_features()`: Extract wavelet-based features
- `extract_frequency_features()`: Extract frequency domain features
- `extract_shape_features()`: Extract shape complexity features
- `select_important_features()`: Select important features
- `normalize_features()`: Normalize features
- `configure()`: Configure parameters

### Key Attributes
- `feature_names`: List of feature names
- `config`: Configuration parameters
- `cache`: Feature caching system
- `performance_metrics`: Performance tracking

### Configuration Parameters
- `wavelet_analysis`: Enable wavelet analysis
- `frequency_analysis`: Enable frequency analysis
- `shape_analysis`: Enable shape analysis
- `complexity_analysis`: Enable complexity analysis
- `wavelet_type`: Wavelet type for analysis
- `max_wavelet_level`: Maximum wavelet decomposition level
