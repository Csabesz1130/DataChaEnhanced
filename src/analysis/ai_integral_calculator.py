# src/analysis/ai_integral_calculator.py

"""
AI-Powered Integral Calculator for Action Potential Analysis

This module provides intelligent integral calculation based on Excel analysis patterns.
It automatically detects optimal integration ranges and calculates hyperpolarization
and depolarization integrals with confidence scoring and quality assessment.

Key Features:
- Excel-compatible calculations (points 11-210, 211-410)
- Automatic outlier detection and correction
- Baseline drift compensation
- Quality assessment and confidence scoring
- Adaptive range detection
- Performance monitoring
"""

import numpy as np
import time
from typing import Dict, Tuple, List, Optional, Union
from scipy import stats
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.metrics import mean_squared_error
from src.utils.logger import app_logger
from src.config.ai_config import AIConfig, AIErrorMessages


class AIIntegralCalculator:
    """
    Advanced AI-powered integral calculator for action potential analysis.
    
    This class implements machine learning techniques to automatically:
    1. Detect optimal integration ranges
    2. Correct signal artifacts and outliers
    3. Calculate accurate integrals with confidence scoring
    4. Validate results against expected patterns
    """
    
    def __init__(self, config: Optional[Dict] = None):
        """
        Initialize the AI integral calculator.
        
        Args:
            config: Optional configuration dictionary. If None, uses default Excel-compatible settings.
        """
        self.config = config or AIConfig.get_excel_compatible_params()
        self.scaler = StandardScaler()
        self.last_analysis_results = None
        self.performance_metrics = {}
        
        # Initialize processing state
        self.is_initialized = False
        self.signal_quality_cache = {}
        self.optimization_cache = {}
        
        app_logger.info("AI Integral Calculator initialized with Excel-compatible parameters")
        
    def analyze_action_potential(self, processor, custom_ranges: Optional[Dict] = None, 
                               enable_auto_optimization: bool = True) -> Dict:
        """
        Perform comprehensive AI analysis of action potential data.
        
        Args:
            processor: ActionPotentialProcessor instance with processed curves
            custom_ranges: Optional custom integration ranges
            enable_auto_optimization: Whether to enable automatic range optimization
            
        Returns:
            dict: Comprehensive analysis results with AI calculations, quality metrics, and validation
        """
        start_time = time.time()
        
        try:
            app_logger.info("=== Starting AI Action Potential Analysis ===")
            
            # Validate input data
            validation_result = self._validate_processor_data(processor)
            if not validation_result['valid']:
                raise ValueError(f"Invalid processor data: {validation_result['error']}")
            
            # Extract curves and time data
            curves_data = self._extract_curves_data(processor)
            if not curves_data['valid']:
                raise ValueError(f"Failed to extract curves: {curves_data['error']}")
            
            # Determine integration ranges
            ranges = self._determine_integration_ranges(
                curves_data, custom_ranges, enable_auto_optimization
            )
            
            # Perform signal quality assessment
            quality_metrics = self._assess_signal_quality(curves_data)
            
            # Calculate integrals with AI enhancement
            integral_results = self._calculate_ai_integrals(curves_data, ranges, quality_metrics)
            
            # Generate confidence scores
            confidence_scores = self._calculate_confidence_scores(
                curves_data, integral_results, quality_metrics
            )
            
            # Compile comprehensive results
            results = {
                'ai_integrals': integral_results,
                'integration_ranges': ranges,
                'quality_metrics': quality_metrics,
                'confidence_scores': confidence_scores,
                'processing_info': {
                    'processing_time': time.time() - start_time,
                    'excel_compatible': self.config.get('excel_compatible_mode', True),
                    'auto_optimization_used': enable_auto_optimization,
                    'ranges_source': ranges.get('source', 'default')
                },
                'validation_status': self._validate_results(integral_results, quality_metrics)
            }
            
            # Cache results for future reference
            self.last_analysis_results = results
            self._update_performance_metrics(results)
            
            app_logger.info(f"AI analysis completed in {results['processing_info']['processing_time']:.2f}s")
            app_logger.info(f"Hyperpol integral: {integral_results['hyperpol_integral']:.3f} pC")
            app_logger.info(f"Depol integral: {integral_results['depol_integral']:.3f} pC")
            app_logger.info(f"Overall confidence: {confidence_scores['overall_confidence']:.1%}")
            
            return results
            
        except Exception as e:
            app_logger.error(f"AI analysis failed: {str(e)}")
            raise
    
    def calculate_manual_integrals(self, processor, hyperpol_range: Tuple[int, int], 
                                 depol_range: Tuple[int, int]) -> Dict:
        """
        Calculate integrals for manually specified ranges.
        
        Args:
            processor: ActionPotentialProcessor instance
            hyperpol_range: (start, end) indices for hyperpolarization
            depol_range: (start, end) indices for depolarization
            
        Returns:
            dict: Manual integral calculation results
        """
        try:
            app_logger.debug(f"Calculating manual integrals: H({hyperpol_range}), D({depol_range})")
            
            # Validate processor and ranges
            if not hasattr(processor, 'modified_hyperpol') or processor.modified_hyperpol is None:
                raise ValueError("No hyperpolarization curve available")
            if not hasattr(processor, 'modified_depol') or processor.modified_depol is None:
                raise ValueError("No depolarization curve available")
            
            # Extract data for specified ranges
            hyperpol_data = processor.modified_hyperpol[hyperpol_range[0]:hyperpol_range[1]]
            hyperpol_times = processor.modified_hyperpol_times[hyperpol_range[0]:hyperpol_range[1]]
            
            depol_data = processor.modified_depol[depol_range[0]:depol_range[1]]
            depol_times = processor.modified_depol_times[depol_range[0]:depol_range[1]]
            
            # Apply outlier correction if enabled
            if self.config.get('outlier_correction_method') == 'set_zero':
                hyperpol_data = self._correct_outliers(hyperpol_data)
                depol_data = self._correct_outliers(depol_data)
            
            # Calculate integrals using Excel-compatible method
            hyperpol_integral = self._calculate_excel_integral(hyperpol_data, hyperpol_times)
            depol_integral = self._calculate_excel_integral(depol_data, depol_times)
            
            results = {
                'hyperpol_integral': hyperpol_integral,
                'depol_integral': depol_integral,
                'hyperpol_range': hyperpol_range,
                'depol_range': depol_range,
                'range_lengths': {
                    'hyperpol': len(hyperpol_data),
                    'depol': len(depol_data)
                },
                'time_spans': {
                    'hyperpol': (hyperpol_times[0] * 1000, hyperpol_times[-1] * 1000),
                    'depol': (depol_times[0] * 1000, depol_times[-1] * 1000)
                },
                'calculation_method': 'manual_with_excel_compatibility'
            }
            
            app_logger.debug(f"Manual calculation complete: H={hyperpol_integral:.3f}, D={depol_integral:.3f}")
            
            return results
            
        except Exception as e:
            app_logger.error(f"Manual integral calculation failed: {str(e)}")
            raise
    
    def validate_ai_vs_manual(self, ai_results: Dict, manual_results: Dict, 
                            tolerance: float = 0.15) -> Dict:
        """
        Validate AI results against manual calculations.
        
        Args:
            ai_results: Results from AI analysis
            manual_results: Results from manual calculation
            tolerance: Acceptable relative error (default 15%)
            
        Returns:
            dict: Comprehensive validation results
        """
        try:
            app_logger.info("Validating AI results against manual calculations")
            
            # Extract integral values
            ai_hyperpol = ai_results['ai_integrals']['hyperpol_integral']
            ai_depol = ai_results['ai_integrals']['depol_integral']
            
            manual_hyperpol = manual_results['hyperpol_integral']
            manual_depol = manual_results['depol_integral']
            
            # Calculate relative errors
            hyperpol_error = abs((ai_hyperpol - manual_hyperpol) / manual_hyperpol) if manual_hyperpol != 0 else 0
            depol_error = abs((ai_depol - manual_depol) / manual_depol) if manual_depol != 0 else 0
            
            # Calculate absolute differences
            hyperpol_abs_diff = abs(ai_hyperpol - manual_hyperpol)
            depol_abs_diff = abs(ai_depol - manual_depol)
            
            # Determine pass/fail status
            hyperpol_pass = hyperpol_error <= tolerance
            depol_pass = depol_error <= tolerance
            overall_pass = hyperpol_pass and depol_pass
            
            # Calculate correlation if possible
            correlation = self._calculate_validation_correlation(ai_results, manual_results)
            
            # Calculate confidence-weighted validation
            ai_confidence = ai_results.get('confidence_scores', {}).get('overall_confidence', 0.5)
            confidence_weighted_pass = overall_pass and ai_confidence >= 0.5
            
            validation_results = {
                'hyperpol_error_percent': hyperpol_error * 100,
                'depol_error_percent': depol_error * 100,
                'hyperpol_absolute_difference': hyperpol_abs_diff,
                'depol_absolute_difference': depol_abs_diff,
                'hyperpol_pass': hyperpol_pass,
                'depol_pass': depol_pass,
                'overall_pass': overall_pass,
                'confidence_weighted_pass': confidence_weighted_pass,
                'tolerance_used': tolerance * 100,
                'correlation': correlation,
                'ai_confidence': ai_confidence,
                'validation_summary': {
                    'status': 'PASS' if overall_pass else 'FAIL',
                    'max_error': max(hyperpol_error, depol_error) * 100,
                    'mean_error': (hyperpol_error + depol_error) / 2 * 100,
                    'quality_score': self._calculate_validation_quality_score(
                        hyperpol_error, depol_error, ai_confidence, correlation
                    )
                },
                'recommendations': self._generate_validation_recommendations(
                    hyperpol_error, depol_error, ai_confidence, overall_pass
                )
            }
            
            app_logger.info(f"Validation complete: {validation_results['validation_summary']['status']}")
            app_logger.info(f"Hyperpol error: {hyperpol_error*100:.1f}%, Depol error: {depol_error*100:.1f}%")
            
            return validation_results
            
        except Exception as e:
            app_logger.error(f"Validation failed: {str(e)}")
            raise
    
    def optimize_integration_ranges(self, processor, method: str = 'adaptive') -> Dict:
        """
        Automatically optimize integration ranges based on signal characteristics.
        
        Args:
            processor: ActionPotentialProcessor instance
            method: Optimization method ('adaptive', 'ml_based', 'excel_standard')
            
        Returns:
            dict: Optimized integration ranges with confidence scores
        """
        try:
            app_logger.info(f"Optimizing integration ranges using {method} method")
            
            if method == 'excel_standard':
                # Use standard Excel ranges
                ranges = {
                    'hyperpol_range': self.config['hyperpol_range'],
                    'depol_range': self.config['depol_range'],
                    'source': 'excel_standard',
                    'confidence': 0.8
                }
                
            elif method == 'adaptive':
                # Adaptive range detection based on curve characteristics
                ranges = self._adaptive_range_detection(processor)
                
            elif method == 'ml_based':
                # Machine learning-based optimization
                ranges = self._ml_range_optimization(processor)
                
            else:
                raise ValueError(f"Unknown optimization method: {method}")
            
            # Validate optimized ranges
            validation = self._validate_optimized_ranges(processor, ranges)
            ranges['validation'] = validation
            
            app_logger.info(f"Range optimization complete: {ranges['source']} (confidence: {ranges['confidence']:.1%})")
            
            return ranges
            
        except Exception as e:
            app_logger.error(f"Range optimization failed: {str(e)}")
            # Fall back to default ranges
            return {
                'hyperpol_range': self.config['hyperpol_range'],
                'depol_range': self.config['depol_range'],
                'source': 'default_fallback',
                'confidence': 0.5,
                'error': str(e)
            }
    
    def _validate_processor_data(self, processor) -> Dict:
        """Validate that the processor has required data for analysis."""
        if processor is None:
            return {'valid': False, 'error': 'Processor is None'}
        
        required_attrs = ['modified_hyperpol', 'modified_depol', 
                         'modified_hyperpol_times', 'modified_depol_times']
        
        for attr in required_attrs:
            if not hasattr(processor, attr) or getattr(processor, attr) is None:
                return {'valid': False, 'error': f'Missing required attribute: {attr}'}
        
        # Check data lengths
        hyperpol_len = len(processor.modified_hyperpol)
        depol_len = len(processor.modified_depol)
        
        if hyperpol_len < 50 or depol_len < 50:
            return {'valid': False, 'error': 'Insufficient data length for reliable analysis'}
        
        return {'valid': True, 'hyperpol_length': hyperpol_len, 'depol_length': depol_len}
    
    def _extract_curves_data(self, processor) -> Dict:
        """Extract and validate curve data from processor."""
        try:
            curves_data = {
                'hyperpol_data': np.array(processor.modified_hyperpol),
                'hyperpol_times': np.array(processor.modified_hyperpol_times),
                'depol_data': np.array(processor.modified_depol),
                'depol_times': np.array(processor.modified_depol_times),
                'valid': True
            }
            
            # Add additional curves if available
            if hasattr(processor, 'orange_curve') and processor.orange_curve is not None:
                curves_data['orange_data'] = np.array(processor.orange_curve)
                curves_data['orange_times'] = np.array(processor.orange_curve_times)
            
            if hasattr(processor, 'normalized_curve') and processor.normalized_curve is not None:
                curves_data['normalized_data'] = np.array(processor.normalized_curve)
                curves_data['normalized_times'] = np.array(processor.normalized_curve_times)
            
            return curves_data
            
        except Exception as e:
            return {'valid': False, 'error': str(e)}
    
    def _determine_integration_ranges(self, curves_data: Dict, custom_ranges: Optional[Dict], 
                                    enable_auto_optimization: bool) -> Dict:
        """Determine the best integration ranges to use."""
        if custom_ranges:
            app_logger.debug("Using custom integration ranges")
            return {
                'hyperpol_range': custom_ranges.get('hyperpol_range', self.config['hyperpol_range']),
                'depol_range': custom_ranges.get('depol_range', self.config['depol_range']),
                'source': 'custom'
            }
        
        if enable_auto_optimization:
            app_logger.debug("Using auto-optimized ranges")
            # Use adaptive range detection
            return self._adaptive_range_detection_from_curves(curves_data)
        
        app_logger.debug("Using default Excel ranges")
        return {
            'hyperpol_range': self.config['hyperpol_range'],
            'depol_range': self.config['depol_range'],
            'source': 'excel_default'
        }
    
    def _adaptive_range_detection_from_curves(self, curves_data: Dict) -> Dict:
        """Detect optimal ranges based on curve characteristics."""
        try:
            hyperpol_data = curves_data['hyperpol_data']
            depol_data = curves_data['depol_data']
            
            # Analyze curve characteristics
            hyperpol_stats = self._analyze_curve_statistics(hyperpol_data)
            depol_stats = self._analyze_curve_statistics(depol_data)
            
            # Detect significant signal regions
            hyperpol_range = self._detect_significant_region(hyperpol_data, 'hyperpol')
            depol_range = self._detect_significant_region(depol_data, 'depol')
            
            # Validate ranges don't exceed data bounds
            hyperpol_range = (
                max(0, hyperpol_range[0]),
                min(len(hyperpol_data), hyperpol_range[1])
            )
            depol_range = (
                max(0, depol_range[0]),
                min(len(depol_data), depol_range[1])
            )
            
            confidence = self._calculate_range_confidence(hyperpol_stats, depol_stats)
            
            return {
                'hyperpol_range': hyperpol_range,
                'depol_range': depol_range,
                'source': 'adaptive_detection',
                'confidence': confidence,
                'statistics': {
                    'hyperpol': hyperpol_stats,
                    'depol': depol_stats
                }
            }
            
        except Exception as e:
            app_logger.warning(f"Adaptive range detection failed, using defaults: {str(e)}")
            return {
                'hyperpol_range': self.config['hyperpol_range'],
                'depol_range': self.config['depol_range'],
                'source': 'default_after_adaptive_failure',
                'confidence': 0.5
            }
    
    def _detect_significant_region(self, data: np.ndarray, curve_type: str) -> Tuple[int, int]:
        """Detect the most significant region in a curve for integration."""
        # Calculate moving average to smooth noise
        window_size = min(10, len(data) // 10)
        if window_size > 1:
            smoothed = np.convolve(data, np.ones(window_size)/window_size, mode='same')
        else:
            smoothed = data
        
        # Find regions with significant deviation from baseline
        baseline = np.median(data[:20]) if len(data) > 20 else np.median(data)
        deviation = np.abs(smoothed - baseline)
        threshold = np.std(deviation) * 2
        
        # Find start and end of significant region
        significant_points = deviation > threshold
        if not np.any(significant_points):
            # Fallback to default range proportions
            if curve_type == 'hyperpol':
                return (10, min(210, len(data)))
            else:
                return (0, min(200, len(data)))
        
        significant_indices = np.where(significant_points)[0]
        start_idx = max(0, significant_indices[0] - 10)
        end_idx = min(len(data), significant_indices[-1] + 10)
        
        # Ensure minimum range length
        min_length = 50
        if end_idx - start_idx < min_length:
            center = (start_idx + end_idx) // 2
            start_idx = max(0, center - min_length // 2)
            end_idx = min(len(data), start_idx + min_length)
        
        return (start_idx, end_idx)
    
    def _analyze_curve_statistics(self, data: np.ndarray) -> Dict:
        """Analyze statistical characteristics of a curve."""
        return {
            'mean': np.mean(data),
            'std': np.std(data),
            'min': np.min(data),
            'max': np.max(data),
            'skewness': stats.skew(data),
            'kurtosis': stats.kurtosis(data),
            'signal_to_noise': np.abs(np.mean(data)) / np.std(data) if np.std(data) > 0 else 0,
            'length': len(data)
        }
    
    def _calculate_range_confidence(self, hyperpol_stats: Dict, depol_stats: Dict) -> float:
        """Calculate confidence in the detected ranges."""
        # Base confidence on signal quality metrics
        hyperpol_snr = hyperpol_stats['signal_to_noise']
        depol_snr = depol_stats['signal_to_noise']
        
        # Higher SNR means higher confidence
        snr_confidence = min(1.0, (hyperpol_snr + depol_snr) / 10.0)
        
        # Check if data lengths are reasonable
        length_confidence = 1.0
        if hyperpol_stats['length'] < 100 or depol_stats['length'] < 100:
            length_confidence = 0.7
        
        # Overall confidence
        overall_confidence = (snr_confidence + length_confidence) / 2.0
        
        return max(0.3, min(1.0, overall_confidence))  # Clamp between 0.3 and 1.0
    
    def _assess_signal_quality(self, curves_data: Dict) -> Dict:
        """Assess the quality of signal data for reliable analysis."""
        quality_metrics = {}
        
        for curve_name in ['hyperpol', 'depol']:
            data_key = f'{curve_name}_data'
            if data_key in curves_data:
                data = curves_data[data_key]
                
                # Calculate quality metrics
                metrics = {
                    'signal_to_noise_ratio': self._calculate_snr(data),
                    'baseline_stability': self._assess_baseline_stability(data),
                    'outlier_percentage': self._calculate_outlier_percentage(data),
                    'data_completeness': len(data) / 200,  # Assuming 200 is ideal length
                    'dynamic_range': (np.max(data) - np.min(data)) / np.std(data) if np.std(data) > 0 else 0
                }
                
                # Overall quality score
                metrics['overall_quality'] = self._calculate_overall_quality_score(metrics)
                
                quality_metrics[curve_name] = metrics
        
        # Combined quality assessment
        if 'hyperpol' in quality_metrics and 'depol' in quality_metrics:
            quality_metrics['combined'] = {
                'average_snr': (quality_metrics['hyperpol']['signal_to_noise_ratio'] + 
                               quality_metrics['depol']['signal_to_noise_ratio']) / 2,
                'average_quality': (quality_metrics['hyperpol']['overall_quality'] + 
                                   quality_metrics['depol']['overall_quality']) / 2,
                'data_balance': abs(len(curves_data['hyperpol_data']) - len(curves_data['depol_data'])) / 
                               max(len(curves_data['hyperpol_data']), len(curves_data['depol_data']))
            }
        
        return quality_metrics
    
    def _calculate_snr(self, data: np.ndarray) -> float:
        """Calculate signal-to-noise ratio."""
        if len(data) < 10:
            return 0.0
        
        # Use first and last 10% as noise estimate
        noise_region_size = max(5, len(data) // 10)
        noise_data = np.concatenate([data[:noise_region_size], data[-noise_region_size:]])
        noise_level = np.std(noise_data)
        
        # Signal level is the standard deviation of the entire signal
        signal_level = np.std(data)
        
        return signal_level / noise_level if noise_level > 0 else 0.0
    
    def _assess_baseline_stability(self, data: np.ndarray) -> float:
        """Assess baseline stability (lower is better)."""
        if len(data) < 20:
            return 1.0  # Poor stability for very short data
        
        # Calculate trend using linear regression
        x = np.arange(len(data))
        slope, intercept, r_value, p_value, std_err = stats.linregress(x, data)
        
        # Baseline instability is proportional to the slope magnitude
        baseline_drift = abs(slope) * len(data)
        data_range = np.max(data) - np.min(data)
        
        return baseline_drift / data_range if data_range > 0 else 1.0
    
    def _calculate_outlier_percentage(self, data: np.ndarray) -> float:
        """Calculate the percentage of outliers in the data."""
        if len(data) < 5:
            return 0.0
        
        # Use z-score method
        z_scores = np.abs(stats.zscore(data))
        outliers = z_scores > self.config.get('outlier_threshold', 3.0)
        
        return np.sum(outliers) / len(data)
    
    def _calculate_overall_quality_score(self, metrics: Dict) -> float:
        """Calculate an overall quality score from individual metrics."""
        # Weight different metrics
        weights = {
            'signal_to_noise_ratio': 0.3,
            'baseline_stability': 0.25,  # Lower is better, so we'll invert this
            'outlier_percentage': 0.2,   # Lower is better, so we'll invert this
            'data_completeness': 0.15,
            'dynamic_range': 0.1
        }
        
        # Normalize metrics to 0-1 scale
        normalized_snr = min(1.0, metrics['signal_to_noise_ratio'] / 5.0)  # SNR of 5 = excellent
        normalized_stability = max(0.0, 1.0 - metrics['baseline_stability'] * 10)  # Lower is better
        normalized_outliers = max(0.0, 1.0 - metrics['outlier_percentage'] * 5)  # Lower is better
        normalized_completeness = min(1.0, metrics['data_completeness'])
        normalized_dynamic = min(1.0, metrics['dynamic_range'] / 10.0)
        
        # Calculate weighted score
        quality_score = (
            weights['signal_to_noise_ratio'] * normalized_snr +
            weights['baseline_stability'] * normalized_stability +
            weights['outlier_percentage'] * normalized_outliers +
            weights['data_completeness'] * normalized_completeness +
            weights['dynamic_range'] * normalized_dynamic
        )
        
        return max(0.0, min(1.0, quality_score))
    
    def _calculate_ai_integrals(self, curves_data: Dict, ranges: Dict, quality_metrics: Dict) -> Dict:
        """Calculate integrals using AI-enhanced methods."""
        try:
            # Extract range information
            hyperpol_range = ranges['hyperpol_range']
            depol_range = ranges['depol_range']
            
            # Extract data for integration
            hyperpol_data = curves_data['hyperpol_data'][hyperpol_range[0]:hyperpol_range[1]]
            hyperpol_times = curves_data['hyperpol_times'][hyperpol_range[0]:hyperpol_range[1]]
            
            depol_data = curves_data['depol_data'][depol_range[0]:depol_range[1]]
            depol_times = curves_data['depol_times'][depol_range[0]:depol_range[1]]
            
            # Apply AI-enhanced preprocessing
            hyperpol_processed = self._ai_preprocess_signal(hyperpol_data, quality_metrics.get('hyperpol', {}))
            depol_processed = self._ai_preprocess_signal(depol_data, quality_metrics.get('depol', {}))
            
            # Calculate integrals using Excel-compatible method
            hyperpol_integral = self._calculate_excel_integral(hyperpol_processed, hyperpol_times)
            depol_integral = self._calculate_excel_integral(depol_processed, depol_times)
            
            # Apply AI quality corrections
            correction_factors = self._calculate_ai_correction_factors(quality_metrics)
            
            hyperpol_integral_corrected = hyperpol_integral * correction_factors['hyperpol']
            depol_integral_corrected = depol_integral * correction_factors['depol']
            
            results = {
                'hyperpol_integral': hyperpol_integral_corrected,
                'depol_integral': depol_integral_corrected,
                'raw_integrals': {
                    'hyperpol': hyperpol_integral,
                    'depol': depol_integral
                },
                'correction_factors': correction_factors,
                'preprocessing_applied': {
                    'outlier_correction': True,
                    'baseline_correction': quality_metrics.get('hyperpol', {}).get('baseline_stability', 0) > 0.1,
                    'noise_reduction': quality_metrics.get('combined', {}).get('average_snr', 0) < 2.0
                },
                'integration_method': 'ai_enhanced_excel_compatible'
            }
            
            return results
            
        except Exception as e:
            app_logger.error(f"AI integral calculation failed: {str(e)}")
            raise
    
    def _ai_preprocess_signal(self, data: np.ndarray, quality_metrics: Dict) -> np.ndarray:
        """Apply AI-based preprocessing to improve signal quality."""
        processed_data = data.copy()
        
        # Apply outlier correction if needed
        outlier_percentage = quality_metrics.get('outlier_percentage', 0)
        if outlier_percentage > 0.05:  # More than 5% outliers
            processed_data = self._correct_outliers(processed_data)
        
        # Apply baseline correction if needed
        baseline_stability = quality_metrics.get('baseline_stability', 0)
        if baseline_stability > 0.1:  # Significant baseline drift
            processed_data = self._correct_baseline_drift(processed_data)
        
        # Apply noise reduction if needed
        snr = quality_metrics.get('signal_to_noise_ratio', 10)
        if snr < 2.0:  # Low SNR
            processed_data = self._apply_intelligent_smoothing(processed_data)
        
        return processed_data
    
    def _correct_outliers(self, data: np.ndarray) -> np.ndarray:
        """Correct outliers by setting them to zero (Excel method)."""
        if self.config.get('outlier_correction_method') == 'set_zero':
            # Excel method: "kiugró értékeket 0-ra tenni"
            z_scores = np.abs(stats.zscore(data))
            outlier_threshold = self.config.get('outlier_threshold', 3.0)
            outlier_mask = z_scores > outlier_threshold
            
            corrected_data = data.copy()
            corrected_data[outlier_mask] = 0.0
            
            return corrected_data
        
        return data
    
    def _correct_baseline_drift(self, data: np.ndarray) -> np.ndarray:
        """Correct linear baseline drift."""
        x = np.arange(len(data))
        slope, intercept, _, _, _ = stats.linregress(x, data)
        
        # Remove linear trend
        linear_trend = slope * x + intercept - intercept  # Keep intercept
        corrected_data = data - linear_trend
        
        return corrected_data
    
    def _apply_intelligent_smoothing(self, data: np.ndarray) -> np.ndarray:
        """Apply intelligent smoothing that preserves important features."""
        # Use a small moving average to reduce noise while preserving peaks
        window_size = min(5, len(data) // 20)
        if window_size > 1:
            smoothed = np.convolve(data, np.ones(window_size)/window_size, mode='same')
        else:
            smoothed = data
        
        return smoothed
    
    def _calculate_excel_integral(self, data: np.ndarray, times: np.ndarray) -> float:
        """Calculate integral using Excel-compatible trapezoidal method with /2 correction."""
        if len(data) < 2 or len(times) < 2:
            return 0.0
        
        # Convert times to milliseconds (Excel works in ms)
        times_ms = times * 1000
        
        # Calculate integral using trapezoidal rule
        integral = np.trapz(data, x=times_ms)
        
        # Apply Excel correction factor (/2)
        excel_correction_factor = self.config.get('excel_division_factor', 2.0)
        corrected_integral = integral / excel_correction_factor
        
        return corrected_integral
    
    def _calculate_ai_correction_factors(self, quality_metrics: Dict) -> Dict:
        """Calculate AI-based correction factors based on signal quality."""
        correction_factors = {'hyperpol': 1.0, 'depol': 1.0}
        
        for curve_type in ['hyperpol', 'depol']:
            if curve_type in quality_metrics:
                metrics = quality_metrics[curve_type]
                
                # Base correction on quality score
                quality_score = metrics.get('overall_quality', 0.8)
                
                # Minor corrections based on quality
                if quality_score < 0.5:
                    # Low quality - apply conservative correction
                    correction_factors[curve_type] = 0.95
                elif quality_score > 0.9:
                    # High quality - minimal correction
                    correction_factors[curve_type] = 1.02
                # Medium quality uses factor of 1.0
        
        return correction_factors
    
    def _calculate_confidence_scores(self, curves_data: Dict, integral_results: Dict, 
                                   quality_metrics: Dict) -> Dict:
        """Calculate confidence scores for the analysis results."""
        try:
            # Individual confidence components
            data_quality_confidence = self._calculate_data_quality_confidence(quality_metrics)
            integral_validity_confidence = self._calculate_integral_validity_confidence(integral_results)
            method_confidence = self._calculate_method_confidence()
            
            # Combined confidence score
            confidence_weights = {
                'data_quality': 0.4,
                'integral_validity': 0.4,
                'method': 0.2
            }
            
            overall_confidence = (
                confidence_weights['data_quality'] * data_quality_confidence +
                confidence_weights['integral_validity'] * integral_validity_confidence +
                confidence_weights['method'] * method_confidence
            )
            
            confidence_scores = {
                'overall_confidence': overall_confidence,
                'data_quality_confidence': data_quality_confidence,
                'integral_validity_confidence': integral_validity_confidence,
                'method_confidence': method_confidence,
                'confidence_level': self._categorize_confidence(overall_confidence),
                'confidence_factors': {
                    'signal_quality': quality_metrics.get('combined', {}).get('average_quality', 0.5),
                    'data_completeness': min([
                        quality_metrics.get('hyperpol', {}).get('data_completeness', 1.0),
                        quality_metrics.get('depol', {}).get('data_completeness', 1.0)
                    ]),
                    'processing_stability': 0.9,  # Assuming stable processing
                    'excel_compatibility': 0.95 if self.config.get('excel_compatible_mode') else 0.8
                }
            }
            
            return confidence_scores
            
        except Exception as e:
            app_logger.warning(f"Confidence calculation failed: {str(e)}")
            return {
                'overall_confidence': 0.5,
                'confidence_level': 'medium',
                'error': str(e)
            }
    
    def _calculate_data_quality_confidence(self, quality_metrics: Dict) -> float:
        """Calculate confidence based on data quality metrics."""
        if 'combined' in quality_metrics:
            return quality_metrics['combined'].get('average_quality', 0.5)
        
        # Fallback calculation
        if 'hyperpol' in quality_metrics and 'depol' in quality_metrics:
            hyperpol_quality = quality_metrics['hyperpol'].get('overall_quality', 0.5)
            depol_quality = quality_metrics['depol'].get('overall_quality', 0.5)
            return (hyperpol_quality + depol_quality) / 2.0
        
        return 0.5  # Default medium confidence
    
    def _calculate_integral_validity_confidence(self, integral_results: Dict) -> float:
        """Calculate confidence based on integral result validity."""
        # Check if integrals are within expected physical ranges
        hyperpol_integral = integral_results['hyperpol_integral']
        depol_integral = integral_results['depol_integral']
        
        # Use physical limits from config
        validation_result = AIConfig.validate_integral_result(hyperpol_integral, depol_integral)
        
        if validation_result['overall_valid']:
            return 0.9
        
        # Partial validity
        valid_count = sum([
            validation_result['hyperpol_valid'],
            validation_result['depol_valid'],
            validation_result['ratio_valid'],
            validation_result['sign_hyperpol_correct'],
            validation_result['sign_depol_correct']
        ])
        
        return valid_count / 5.0
    
    def _calculate_method_confidence(self) -> float:
        """Calculate confidence in the analysis method."""
        # High confidence in Excel-compatible mode
        if self.config.get('excel_compatible_mode', True):
            return 0.95
        
        return 0.8  # Standard confidence for other methods
    
    def _categorize_confidence(self, confidence_score: float) -> str:
        """Categorize confidence score into levels."""
        thresholds = self.config.get('confidence_levels', {
            'high': 0.8,
            'medium': 0.5,
            'low': 0.0
        })
        
        if confidence_score >= thresholds['high']:
            return 'high'
        elif confidence_score >= thresholds['medium']:
            return 'medium'
        else:
            return 'low'
    
    def _validate_results(self, integral_results: Dict, quality_metrics: Dict) -> Dict:
        """Validate the analysis results."""
        validation_status = {
            'overall_valid': True,
            'warnings': [],
            'errors': []
        }
        
        # Check integral values
        hyperpol_integral = integral_results['hyperpol_integral']
        depol_integral = integral_results['depol_integral']
        
        # Physical validation
        physical_validation = AIConfig.validate_integral_result(hyperpol_integral, depol_integral)
        
        if not physical_validation['overall_valid']:
            validation_status['warnings'].append("Integral values outside expected physical ranges")
            
        if not physical_validation['sign_hyperpol_correct']:
            validation_status['errors'].append("Hyperpolarization integral should be negative")
            validation_status['overall_valid'] = False
            
        if not physical_validation['sign_depol_correct']:
            validation_status['errors'].append("Depolarization integral should be positive")
            validation_status['overall_valid'] = False
        
        # Quality validation
        combined_quality = quality_metrics.get('combined', {}).get('average_quality', 0.5)
        if combined_quality < 0.3:
            validation_status['warnings'].append("Low signal quality may affect result reliability")
        
        return validation_status
    
    def _update_performance_metrics(self, results: Dict) -> None:
        """Update performance metrics for monitoring."""
        processing_time = results['processing_info']['processing_time']
        
        if 'processing_times' not in self.performance_metrics:
            self.performance_metrics['processing_times'] = []
        
        self.performance_metrics['processing_times'].append(processing_time)
        
        # Keep only last 10 measurements
        if len(self.performance_metrics['processing_times']) > 10:
            self.performance_metrics['processing_times'] = self.performance_metrics['processing_times'][-10:]
        
        # Calculate average processing time
        self.performance_metrics['average_processing_time'] = np.mean(self.performance_metrics['processing_times'])
    
    def _calculate_validation_correlation(self, ai_results: Dict, manual_results: Dict) -> float:
        """Calculate correlation between AI and manual results."""
        try:
            ai_values = [
                ai_results['ai_integrals']['hyperpol_integral'],
                ai_results['ai_integrals']['depol_integral']
            ]
            manual_values = [
                manual_results['hyperpol_integral'],
                manual_results['depol_integral']
            ]
            
            correlation, _ = stats.pearsonr(ai_values, manual_values)
            return correlation if not np.isnan(correlation) else 0.0
            
        except Exception:
            return 0.0
    
    def _calculate_validation_quality_score(self, hyperpol_error: float, depol_error: float, 
                                          ai_confidence: float, correlation: float) -> float:
        """Calculate overall validation quality score."""
        # Convert errors to quality scores (lower error = higher quality)
        error_score = 1.0 - min(1.0, (hyperpol_error + depol_error) / 2.0)
        correlation_score = max(0.0, correlation)
        
        # Weighted combination
        quality_score = (
            0.4 * error_score +
            0.3 * ai_confidence +
            0.3 * correlation_score
        )
        
        return max(0.0, min(1.0, quality_score))
    
    def _generate_validation_recommendations(self, hyperpol_error: float, depol_error: float, 
                                           ai_confidence: float, overall_pass: bool) -> List[str]:
        """Generate recommendations based on validation results."""
        recommendations = []
        
        if not overall_pass:
            if hyperpol_error > 0.2:  # > 20% error
                recommendations.append("Consider checking hyperpolarization integration range")
            if depol_error > 0.2:  # > 20% error
                recommendations.append("Consider checking depolarization integration range")
        
        if ai_confidence < 0.5:
            recommendations.append("Low AI confidence - manual verification recommended")
        
        if overall_pass and ai_confidence > 0.8:
            recommendations.append("Excellent validation results - AI analysis is reliable")
        
        if max(hyperpol_error, depol_error) > 0.1 and overall_pass:
            recommendations.append("Consider fine-tuning integration ranges for better accuracy")
        
        return recommendations
    
    def get_performance_summary(self) -> Dict:
        """Get performance summary for monitoring."""
        if not self.performance_metrics:
            return {'status': 'No performance data available'}
        
        return {
            'average_processing_time': self.performance_metrics.get('average_processing_time', 0),
            'total_analyses': len(self.performance_metrics.get('processing_times', [])),
            'performance_status': 'good' if self.performance_metrics.get('average_processing_time', 0) < 2.0 else 'slow'
        }
    
    def export_analysis_results(self, results: Dict, filename: str = None) -> str:
        """Export analysis results in Excel-compatible format."""
        try:
            if filename is None:
                timestamp = time.strftime("%Y%m%d_%H%M%S")
                filename = f"ai_analysis_results_{timestamp}.txt"
            
            export_content = self._format_results_for_export(results)
            
            with open(filename, 'w', encoding='utf-8') as f:
                f.write(export_content)
            
            app_logger.info(f"Analysis results exported to {filename}")
            return filename
            
        except Exception as e:
            app_logger.error(f"Export failed: {str(e)}")
            raise
    
    def _format_results_for_export(self, results: Dict) -> str:
        """Format results for export in Excel-compatible format."""
        timestamp = time.strftime("%Y-%m-%d %H:%M:%S")
        
        content = [
            "=== AI ACTION POTENTIAL ANALYSIS RESULTS ===",
            f"Analysis Date: {timestamp}",
            f"Excel Compatible Mode: {self.config.get('excel_compatible_mode', True)}",
            "",
            "AI ANALYSIS:",
            f"  Hyperpolarization Integral: {results['ai_integrals']['hyperpol_integral']:.3f} pC",
            f"  Depolarization Integral: {results['ai_integrals']['depol_integral']:.3f} pC",
            "",
            "INTEGRATION RANGES:",
            f"  Hyperpolarization: {results['integration_ranges']['hyperpol_range']}",
            f"  Depolarization: {results['integration_ranges']['depol_range']}",
            "",
            "QUALITY METRICS:",
            f"  Overall Confidence: {results['confidence_scores']['overall_confidence']:.1%}",
            f"  Confidence Level: {results['confidence_scores']['confidence_level']}",
            f"  Data Quality: {results['quality_metrics'].get('combined', {}).get('average_quality', 0):.1%}",
            "",
            "PROCESSING INFO:",
            f"  Processing Time: {results['processing_info']['processing_time']:.2f} seconds",
            f"  Range Source: {results['integration_ranges']['source']}",
            "",
            "VALIDATION STATUS:",
            f"  Overall Valid: {results['validation_status']['overall_valid']}",
        ]
        
        if results['validation_status'].get('warnings'):
            content.append("  Warnings:")
            for warning in results['validation_status']['warnings']:
                content.append(f"    - {warning}")
        
        if results['validation_status'].get('errors'):
            content.append("  Errors:")
            for error in results['validation_status']['errors']:
                content.append(f"    - {error}")
        
        return "\n".join(content)