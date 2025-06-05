# src/config/ai_config.py

"""
AI Analysis Configuration
Contains default parameters and settings for AI-powered integral calculation.
Based on manual Excel analysis patterns.
"""

import numpy as np
from typing import Dict, Tuple, List, Optional, Union

class AIConfig:
    """
    Configuration class for AI analysis parameters.
    
    This class contains all the default parameters based on the Excel analysis patterns:
    - Point ranges: Hyperpol (11-210), Depol (211-410) 
    - Sampling rate: 0.5 ms per point
    - Integration method: Trapezoidal with /2 correction
    - Outlier handling: Set to 0 ("kiugró értékeket 0-ra tenni")
    """
    
    # Excel-based default ranges (0-based indexing)
    DEFAULT_RANGES = {
        'hyperpol_range': (10, 210),    # Excel points 11-210 (1-based)
        'depol_range': (210, 410),      # Excel points 211-410 (1-based)
        'segment_length': 200,           # Standard segment length from Excel
        'overlap_points': 10,            # Overlap for smooth transitions
        'excel_point_offset': 1          # Conversion factor from Excel 1-based to Python 0-based
    }
    
    # Signal processing parameters based on Excel analysis
    SIGNAL_PROCESSING = {
        'sampling_interval_ms': 0.5,     # 0.5 ms per point (from Excel analysis)
        'sampling_frequency_hz': 2000,   # 2 kHz (1/0.0005s)
        'baseline_window': 50,           # Points for baseline calculation
        'smooth_window': 5,              # Smoothing window size
        'outlier_threshold': 3.0,        # Z-score threshold for outliers (Excel: set to 0)
        'noise_threshold': 0.1,          # Minimum signal-to-noise ratio
        'excel_division_factor': 2.0     # Division by 2 as in Excel calculations
    }
    
    # AI algorithm parameters
    AI_PARAMETERS = {
        'outlier_detection_method': 'zscore',      # 'zscore', 'iqr', 'isolation'
        'outlier_correction_method': 'set_zero',   # 'set_zero' (Excel), 'interpolate', 'median'
        'baseline_correction_method': 'linear',    # 'linear', 'polynomial', 'median'
        'integration_method': 'trapezoidal',       # 'trapezoidal' (Excel), 'simpson', 'cumulative'
        'excel_compatible_mode': True,             # Use Excel-compatible calculations
        'quality_threshold': 0.7,                  # Minimum quality score
        'confidence_levels': {
            'high': 0.8,      # High confidence threshold
            'medium': 0.5,    # Medium confidence threshold
            'low': 0.0        # Low confidence threshold
        },
        'auto_range_detection': True,              # Enable automatic range detection
        'feature_detection_prominence': 0.5       # Prominence factor for peak detection
    }
    
    # Validation parameters
    VALIDATION = {
        'default_tolerance': 0.15,       # 15% tolerance (reasonable for biological signals)
        'strict_tolerance': 0.10,        # 10% for strict validation
        'loose_tolerance': 0.25,         # 25% for loose validation
        'min_correlation': 0.8,          # Minimum correlation coefficient
        'max_iterations': 10,            # Maximum optimization iterations
        'validation_methods': ['relative_error', 'correlation', 'absolute_difference']
    }
    
    # Physical parameter ranges (based on Excel data analysis)
    PHYSICAL_LIMITS = {
        'hyperpol_integral_range': (-50.0, -0.1),   # pC range (negative values expected)
        'depol_integral_range': (0.1, 50.0),        # pC range (positive values expected)
        'integral_ratio_range': (0.1, 10.0),        # Reasonable ratio range
        'current_range': (-5000, 5000),             # pA range
        'time_range': (0, 1000),                    # ms range
        'data_length_range': (100, 10000),          # Number of data points
        'quality_score_range': (0.0, 1.0)           # Quality score range
    }
    
    # Export settings
    EXPORT = {
        'decimal_precision': 3,          # Decimal places for integral results
        'time_precision': 1,             # Decimal places for time values
        'current_precision': 2,          # Decimal places for current values
        'time_units': 'ms',             # Time units in export
        'current_units': 'pA',          # Current units
        'integral_units': 'pC',         # Integral units (picocoulombs)
        'include_metadata': True,        # Include processing metadata
        'include_validation': True,      # Include validation results
        'include_quality_metrics': True, # Include quality assessment
        'excel_compatibility': True,     # Ensure Excel-compatible format
        'timestamp_format': '%Y-%m-%d %H:%M:%S'
    }
    
    # Visualization parameters
    VISUALIZATION = {
        'colors': {
            'hyperpol': '#4287f5',       # Blue for hyperpolarization
            'depol': '#f54242',          # Red for depolarization
            'ai_result': '#28a745',      # Green for AI results
            'manual_result': '#fd7e14',  # Orange for manual results
            'error_pass': '#28a745',     # Green for passed validation
            'error_fail': '#dc3545',     # Red for failed validation
            'confidence_high': '#28a745', # Green for high confidence
            'confidence_medium': '#ffc107', # Yellow for medium confidence
            'confidence_low': '#dc3545',  # Red for low confidence
            'quality_excellent': '#28a745', # Green for excellent quality
            'quality_good': '#17a2b8',   # Blue for good quality
            'quality_poor': '#dc3545'    # Red for poor quality
        },
        'alpha_values': {
            'range_highlight': 0.3,      # Transparency for range highlighting
            'confidence_band': 0.2,      # Transparency for confidence bands
            'overlay': 0.7,              # Transparency for overlays
            'background': 0.1            # Transparency for backgrounds
        },
        'line_styles': {
            'ai': '-',                   # Solid line for AI results
            'manual': '--',              # Dashed line for manual results
            'validation': ':',           # Dotted line for validation
            'threshold': '-.'            # Dash-dot line for thresholds
        },
        'line_widths': {
            'signal': 2,                 # Width for signal lines
            'range_boundary': 3,         # Width for range boundaries
            'threshold': 2,              # Width for threshold lines
            'highlight': 4               # Width for highlighted features
        }
    }
    
    # Performance settings
    PERFORMANCE = {
        'processing_time_target': 2.0,   # Target processing time (seconds)
        'memory_usage_target': 100.0,    # Target memory usage (MB)
        'max_data_points': 20000,        # Maximum data points to process
        'chunk_size': 1000,              # Chunk size for large data processing
        'parallel_processing': False,     # Enable parallel processing (if available)
        'cache_results': True,           # Cache intermediate results
        'optimization_level': 'standard' # 'fast', 'standard', 'accurate'
    }
    
    # Error handling and logging
    ERROR_HANDLING = {
        'log_level': 'INFO',             # Logging level
        'retry_attempts': 3,             # Number of retry attempts
        'fallback_to_defaults': True,    # Use defaults on parameter errors
        'validate_inputs': True,         # Validate input parameters
        'detailed_error_messages': True, # Provide detailed error information
        'error_recovery': True           # Attempt error recovery
    }
    
    @classmethod
    def get_default_params(cls) -> Dict:
        """Get default parameters for AI analysis."""
        return {
            **cls.DEFAULT_RANGES,
            **cls.SIGNAL_PROCESSING,
            **cls.AI_PARAMETERS
        }
    
    @classmethod
    def get_excel_compatible_params(cls) -> Dict:
        """Get Excel-compatible parameters for exact replication."""
        params = cls.get_default_params()
        params.update({
            'excel_compatible_mode': True,
            'outlier_correction_method': 'set_zero',
            'integration_method': 'trapezoidal',
            'baseline_correction_method': 'linear'
        })
        return params
    
    @classmethod
    def get_validation_params(cls) -> Dict:
        """Get validation parameters."""
        return cls.VALIDATION.copy()
    
    @classmethod
    def get_physical_limits(cls) -> Dict:
        """Get physical parameter limits."""
        return cls.PHYSICAL_LIMITS.copy()
    
    @classmethod
    def get_visualization_config(cls) -> Dict:
        """Get visualization configuration."""
        return cls.VISUALIZATION.copy()
    
    @classmethod
    def validate_integral_result(cls, hyperpol_integral: float, depol_integral: float) -> Dict:
        """
        Validate integral results against physical limits.
        
        Args:
            hyperpol_integral: Hyperpolarization integral value
            depol_integral: Depolarization integral value
            
        Returns:
            dict: Validation results with detailed analysis
        """
        limits = cls.PHYSICAL_LIMITS
        
        # Check hyperpol range (should be negative)
        hyperpol_valid = (limits['hyperpol_integral_range'][0] <= 
                         hyperpol_integral <= 
                         limits['hyperpol_integral_range'][1])
        
        # Check depol range (should be positive)
        depol_valid = (limits['depol_integral_range'][0] <= 
                      depol_integral <= 
                      limits['depol_integral_range'][1])
        
        # Check ratio
        if hyperpol_integral != 0:
            ratio = abs(depol_integral / hyperpol_integral)
            ratio_valid = (limits['integral_ratio_range'][0] <= 
                          ratio <= 
                          limits['integral_ratio_range'][1])
        else:
            ratio_valid = False
            ratio = 0
        
        # Check signs (hyperpol should be negative, depol positive)
        sign_hyperpol_correct = hyperpol_integral < 0
        sign_depol_correct = depol_integral > 0
        
        # Overall validation
        overall_valid = (hyperpol_valid and depol_valid and ratio_valid and 
                        sign_hyperpol_correct and sign_depol_correct)
        
        return {
            'hyperpol_valid': hyperpol_valid,
            'depol_valid': depol_valid,
            'ratio_valid': ratio_valid,
            'sign_hyperpol_correct': sign_hyperpol_correct,
            'sign_depol_correct': sign_depol_correct,
            'overall_valid': overall_valid,
            'ratio': ratio,
            'validation_details': {
                'hyperpol_range': limits['hyperpol_integral_range'],
                'depol_range': limits['depol_integral_range'],
                'ratio_range': limits['integral_ratio_range'],
                'expected_signs': {'hyperpol': 'negative', 'depol': 'positive'}
            }
        }
    
    @classmethod
    def get_quality_thresholds(cls) -> Dict:
        """Get quality assessment thresholds."""
        return {
            'signal_to_noise_min': 2.0,         # Minimum SNR
            'correlation_min': 0.7,             # Minimum correlation
            'outlier_percentage_max': 0.1,      # 10% max outliers
            'baseline_stability_max': 0.05,     # 5% max baseline drift
            'data_completeness_min': 0.8,       # 80% minimum data completeness
            'processing_time_max': 5.0,         # 5 seconds max processing time
            'memory_usage_max': 200.0,          # 200 MB max memory usage
            'confidence_thresholds': cls.AI_PARAMETERS['confidence_levels']
        }
    
    @classmethod
    def adapt_ranges_for_data_length(cls, data_length: int) -> Dict:
        """
        Adapt integration ranges based on actual data length.
        
        Args:
            data_length: Length of available data
            
        Returns:
            dict: Adapted ranges that fit within data bounds
        """
        default_ranges = cls.DEFAULT_RANGES.copy()
        
        if data_length < 410:
            # Adapt ranges for shorter data
            mid_point = data_length // 2
            
            # Calculate proportional ranges
            original_hyperpol_length = default_ranges['hyperpol_range'][1] - default_ranges['hyperpol_range'][0]
            original_depol_length = default_ranges['depol_range'][1] - default_ranges['depol_range'][0]
            
            # Scale ranges proportionally
            scale_factor = data_length / 410  # 410 is the original expected length
            
            new_hyperpol_length = int(original_hyperpol_length * scale_factor)
            new_depol_length = int(original_depol_length * scale_factor)
            
            # Ensure ranges don't overlap and fit within data
            hyperpol_start = min(10, data_length // 10)
            hyperpol_end = min(hyperpol_start + new_hyperpol_length, mid_point)
            
            depol_start = max(hyperpol_end + 5, mid_point)
            depol_end = min(depol_start + new_depol_length, data_length - 1)
            
            adapted_ranges = {
                'hyperpol_range': (hyperpol_start, hyperpol_end),
                'depol_range': (depol_start, depol_end),
                'segment_length': min(200, data_length // 4),
                'overlap_points': min(10, data_length // 40),
                'adaptation_method': 'proportional_scaling',
                'scale_factor': scale_factor,
                'original_data_length': 410,
                'actual_data_length': data_length
            }
            
            return adapted_ranges
        
        # Return default ranges if data is long enough
        default_ranges['adaptation_method'] = 'no_adaptation_needed'
        default_ranges['actual_data_length'] = data_length
        return default_ranges
    
    @classmethod
    def get_excel_ranges_description(cls) -> Dict:
        """Get description of Excel ranges for documentation."""
        return {
            'hyperpol_excel_points': '11-210 (1-based indexing)',
            'hyperpol_python_points': '10-210 (0-based indexing)',
            'depol_excel_points': '211-410 (1-based indexing)',
            'depol_python_points': '210-410 (0-based indexing)',
            'total_points_per_segment': 200,
            'time_per_point': '0.5 ms',
            'segment_duration': '100 ms',
            'total_analysis_duration': '200 ms',
            'conversion_note': 'Excel uses 1-based indexing, Python uses 0-based'
        }
    
    @classmethod
    def create_custom_config(cls, **overrides) -> Dict:
        """
        Create a custom configuration with specified overrides.
        
        Args:
            **overrides: Parameters to override from defaults
            
        Returns:
            dict: Custom configuration
        """
        config = cls.get_default_params()
        config.update(overrides)
        
        # Validate the custom configuration
        validation_result = cls._validate_config(config)
        if not validation_result['valid']:
            raise ValueError(f"Invalid configuration: {validation_result['errors']}")
        
        config['custom_config'] = True
        config['overrides_applied'] = list(overrides.keys())
        
        return config
    
    @classmethod
    def _validate_config(cls, config: Dict) -> Dict:
        """
        Validate a configuration dictionary.
        
        Args:
            config: Configuration to validate
            
        Returns:
            dict: Validation result
        """
        errors = []
        warnings = []
        
        # Check required parameters
        required_params = ['hyperpol_range', 'depol_range', 'sampling_interval_ms']
        for param in required_params:
            if param not in config:
                errors.append(f"Missing required parameter: {param}")
        
        # Validate ranges
        if 'hyperpol_range' in config:
            h_range = config['hyperpol_range']
            if not isinstance(h_range, (tuple, list)) or len(h_range) != 2:
                errors.append("hyperpol_range must be a tuple/list of 2 values")
            elif h_range[0] >= h_range[1]:
                errors.append("hyperpol_range start must be less than end")
        
        if 'depol_range' in config:
            d_range = config['depol_range']
            if not isinstance(d_range, (tuple, list)) or len(d_range) != 2:
                errors.append("depol_range must be a tuple/list of 2 values")
            elif d_range[0] >= d_range[1]:
                errors.append("depol_range start must be less than end")
        
        # Check for overlapping ranges
        if ('hyperpol_range' in config and 'depol_range' in config):
            h_range = config['hyperpol_range']
            d_range = config['depol_range']
            if h_range[1] > d_range[0]:
                warnings.append("Hyperpol and depol ranges overlap")
        
        # Validate numeric parameters
        numeric_params = {
            'sampling_interval_ms': (0.1, 10.0),
            'outlier_threshold': (1.0, 10.0),
            'baseline_window': (5, 200)
        }
        
        for param, (min_val, max_val) in numeric_params.items():
            if param in config:
                value = config[param]
                if not isinstance(value, (int, float)):
                    errors.append(f"{param} must be numeric")
                elif not (min_val <= value <= max_val):
                    errors.append(f"{param} must be between {min_val} and {max_val}")
        
        return {
            'valid': len(errors) == 0,
            'errors': errors,
            'warnings': warnings
        }
    
    @classmethod
    def get_performance_config(cls, optimization_level: str = 'standard') -> Dict:
        """
        Get performance configuration for different optimization levels.
        
        Args:
            optimization_level: 'fast', 'standard', or 'accurate'
            
        Returns:
            dict: Performance configuration
        """
        base_config = cls.PERFORMANCE.copy()
        
        if optimization_level == 'fast':
            base_config.update({
                'processing_time_target': 1.0,
                'chunk_size': 2000,
                'parallel_processing': True,
                'cache_results': True,
                'detailed_validation': False
            })
        elif optimization_level == 'accurate':
            base_config.update({
                'processing_time_target': 5.0,
                'chunk_size': 500,
                'parallel_processing': False,
                'cache_results': False,
                'detailed_validation': True
            })
        # 'standard' uses base configuration
        
        base_config['optimization_level'] = optimization_level
        return base_config


class AIErrorMessages:
    """Standard error messages and warnings for AI analysis."""
    
    ERRORS = {
        'no_data': "No data available for analysis. Please load data first.",
        'no_processor': "Action potential processor not available. Please run processing first.",
        'invalid_range': "Invalid integration range specified. Check range boundaries.",
        'calculation_failed': "AI calculation failed. Check data quality and parameters.",
        'validation_failed': "Validation failed. AI results may be unreliable.",
        'export_failed': "Export operation failed. Check file permissions and path.",
        'import_failed': "Import operation failed. Check file format and content.",
        'insufficient_data': "Insufficient data for reliable analysis.",
        'outlier_threshold_invalid': "Outlier threshold must be between 1.0 and 10.0.",
        'range_overlap': "Integration ranges overlap. Please adjust range boundaries.",
        'excel_compatibility_error': "Excel compatibility mode requires specific parameters."
    }
    
    WARNINGS = {
        'low_quality': "Low signal quality detected. Results may be less reliable.",
        'outliers_detected': "Outliers detected and corrected in the signal.",
        'range_adapted': "Integration ranges adapted for data length.",
        'baseline_drift': "Significant baseline drift detected.",
        'noise_high': "High noise levels detected in signal.",
        'correlation_low': "Low correlation between AI and manual results.",
        'processing_slow': "Processing is taking longer than expected.",
        'memory_usage_high': "High memory usage detected.",
        'excel_deviation': "Results deviate from expected Excel patterns.",
        'confidence_low': "Low confidence in analysis results."
    }
    
    INFO = {
        'processing_complete': "AI analysis completed successfully.",
        'validation_passed': "AI results validated against manual analysis.",
        'export_complete': "Results exported successfully.",
        'ranges_optimized': "Integration ranges automatically optimized.",
        'quality_good': "Signal quality is good for reliable analysis.",
        'excel_compatible': "Analysis performed in Excel-compatible mode.",
        'cache_used': "Cached results used to improve performance.",
        'auto_correction_applied': "Automatic corrections applied to improve results."
    }
    
    @classmethod
    def get_message(cls, category: str, key: str) -> str:
        """
        Get a specific message by category and key.
        
        Args:
            category: 'errors', 'warnings', or 'info'
            key: Message key
            
        Returns:
            str: Message text
        """
        category_map = {
            'errors': cls.ERRORS,
            'warnings': cls.WARNINGS,
            'info': cls.INFO
        }
        
        if category in category_map and key in category_map[category]:
            return category_map[category][key]
        
        return f"Unknown message: {category}.{key}"


class AIPerformanceMetrics:
    """Performance metrics and benchmarks for AI analysis."""
    
    BENCHMARKS = {
        'processing_time_target': 2.0,      # seconds
        'memory_usage_target': 100.0,       # MB
        'accuracy_target': 0.95,            # 95% accuracy vs manual
        'precision_target': 0.03,           # 3% precision (relative error)
        'recall_target': 0.95,              # 95% recall
        'throughput_target': 1000,          # data points per second
        'quality_score_target': 0.8         # quality score target
    }
    
    @classmethod
    def evaluate_performance(cls, processing_time: float, memory_usage: float, 
                           accuracy: float, quality_score: float = None) -> Dict:
        """
        Evaluate AI performance against benchmarks.
        
        Args:
            processing_time: Time taken for analysis (seconds)
            memory_usage: Memory used (MB)
            accuracy: Accuracy score (0-1)
            quality_score: Optional quality score (0-1)
            
        Returns:
            dict: Performance evaluation with detailed metrics
        """
        benchmarks = cls.BENCHMARKS
        
        # Individual performance checks
        time_performance = processing_time <= benchmarks['processing_time_target']
        memory_performance = memory_usage <= benchmarks['memory_usage_target']
        accuracy_performance = accuracy >= benchmarks['accuracy_target']
        
        # Quality performance (if provided)
        quality_performance = True
        if quality_score is not None:
            quality_performance = quality_score >= benchmarks['quality_score_target']
        
        # Overall performance
        performance_factors = [
            time_performance,
            memory_performance, 
            accuracy_performance,
            quality_performance
        ]
        overall_performance = all(performance_factors)
        
        # Calculate performance score (0-1)
        performance_score = (
            (1.0 if time_performance else 0.5) +
            (1.0 if memory_performance else 0.5) +
            (accuracy / benchmarks['accuracy_target'] if accuracy <= 1.0 else 1.0) +
            (quality_score / benchmarks['quality_score_target'] if quality_score is not None else 1.0)
        ) / 4.0
        
        return {
            'time_performance': time_performance,
            'memory_performance': memory_performance,
            'accuracy_performance': accuracy_performance,
            'quality_performance': quality_performance,
            'overall_performance': overall_performance,
            'performance_score': performance_score,
            'benchmarks_used': benchmarks,
            'metrics': {
                'processing_time': processing_time,
                'memory_usage': memory_usage,
                'accuracy': accuracy,
                'quality_score': quality_score
            },
            'recommendations': cls._generate_performance_recommendations(
                time_performance, memory_performance, accuracy_performance, quality_performance
            )
        }
    
    @classmethod
    def _generate_performance_recommendations(cls, time_perf: bool, memory_perf: bool, 
                                            accuracy_perf: bool, quality_perf: bool) -> List[str]:
        """Generate performance improvement recommendations."""
        recommendations = []
        
        if not time_perf:
            recommendations.append("Consider reducing data size or using 'fast' optimization mode")
        
        if not memory_perf:
            recommendations.append("Process data in smaller chunks to reduce memory usage")
        
        if not accuracy_perf:
            recommendations.append("Review input data quality and processing parameters")
        
        if not quality_perf:
            recommendations.append("Check signal quality and consider noise reduction")
        
        if all([time_perf, memory_perf, accuracy_perf, quality_perf]):
            recommendations.append("Performance is excellent - consider using 'accurate' mode for even better results")
        
        return recommendations


# Example usage and factory functions
def create_excel_compatible_config() -> Dict:
    """Create a configuration that exactly matches Excel analysis."""
    return AIConfig.get_excel_compatible_params()

def create_fast_analysis_config() -> Dict:
    """Create a configuration optimized for speed."""
    config = AIConfig.get_default_params()
    config.update(AIConfig.get_performance_config('fast'))
    return config

def create_accurate_analysis_config() -> Dict:
    """Create a configuration optimized for accuracy."""
    config = AIConfig.get_default_params()
    config.update(AIConfig.get_performance_config('accurate'))
    return config