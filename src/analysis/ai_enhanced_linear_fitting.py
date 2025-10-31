"""
AI-Enhanced Linear Fitting Integration Module

This module integrates the enhanced linear fitting capabilities with AI-powered
analysis, providing intelligent automation and quality assessment.

File: src/analysis/ai_enhanced_linear_fitting.py
"""

import numpy as np
from typing import Dict, List, Optional, Tuple
import json
from pathlib import Path

try:
    from src.utils.logger import app_logger
except ImportError:
    import logging
    app_logger = logging.getLogger(__name__)

try:
    from .linear_fitting import EnhancedLinearDriftCorrector
    from .robust_regression import RobustRegressionAnalyzer
except ImportError:
    # Fallback imports for testing
    try:
        from linear_fitting import EnhancedLinearDriftCorrector
        from robust_regression import RobustRegressionAnalyzer
    except ImportError:
        app_logger.error("Could not import required modules. Please check file placement.")
        raise

class AIEnhancedLinearFitting:
    """
    Main integration class combining enhanced linear fitting with AI capabilities.
    """
    
    def __init__(self, use_ai: bool = True, auto_validate: bool = True):
        """
        Initialize AI-enhanced linear fitting system.
        
        Args:
            use_ai: Enable AI-powered analysis (currently uses enhanced heuristics)
            auto_validate: Automatically validate results
        """
        self.drift_corrector = EnhancedLinearDriftCorrector()
        self.robust_analyzer = RobustRegressionAnalyzer()
        self.use_ai = use_ai
        self.auto_validate = auto_validate
        
        # Initialize AI components if available
        self.ai_confidence_estimator = None
        if use_ai:
            try:
                # This would connect to your existing AI system
                # For now, we use enhanced heuristics
                self.ai_confidence_estimator = self._create_mock_ai_estimator()
            except ImportError:
                app_logger.warning("AI components not available, using enhanced heuristics")
                self.use_ai = False
    
    def _create_mock_ai_estimator(self):
        """Create a mock AI estimator for demonstration purposes."""
        class MockAIEstimator:
            def predict_confidence(self, curve_data):
                """Mock confidence prediction based on data quality."""
                # Simple heuristics-based confidence estimation
                data_quality = self._assess_data_quality(curve_data)
                return {
                    'confidence_score': data_quality.get('overall_quality_score', 0.5),
                    'confidence_level': data_quality.get('quality_level', 'unknown'),
                    'recommendations': []
                }
            
            def _assess_data_quality(self, curve_data):
                """Assess data quality using heuristics."""
                current = curve_data['current']
                
                # Signal-to-noise ratio
                snr = np.abs(np.mean(current)) / (np.std(current) + 1e-6)
                
                # Data completeness
                completeness = 1.0 if not (np.any(np.isnan(current)) or np.any(np.isinf(current))) else 0.0
                
                # Overall quality
                quality_score = min(1.0, (snr / 10 * 0.6 + completeness * 0.4))
                
                if quality_score >= 0.8:
                    quality_level = 'excellent'
                elif quality_score >= 0.6:
                    quality_level = 'good'
                else:
                    quality_level = 'poor'
                
                return {
                    'overall_quality_score': quality_score,
                    'quality_level': quality_level,
                    'snr': snr,
                    'completeness': completeness
                }
        
        return MockAIEstimator()
    
    def process_purple_curves_enhanced(self, processor, manual_plateau_regions=None, 
                                     fitting_method='auto', validation_level='full'):
        """
        Enhanced purple curve processing with AI integration and comprehensive validation.
        
        Args:
            processor: Data processor with purple curve data
            manual_plateau_regions: Optional manual plateau regions
            fitting_method: 'auto', 'ols', 'ransac', 'huber', 'theil_sen'
            validation_level: 'basic', 'standard', 'full'
            
        Returns:
            Enhanced results with AI analysis and validation
        """
        app_logger.info("Starting AI-enhanced purple curve drift correction")
        
        try:
            results = {
                'processing_info': {
                    'method': fitting_method,
                    'validation_level': validation_level,
                    'ai_enabled': self.use_ai,
                    'timestamp': np.datetime64('now').astype(str),
                    'version': 'enhanced_v1.0'
                }
            }
            
            # Process hyperpolarization curve
            if hasattr(processor, 'modified_hyperpol') and processor.modified_hyperpol is not None:
                app_logger.info("Processing enhanced hyperpolarization curve")
                
                hyperpol_results = self._process_single_curve_enhanced(
                    processor.modified_hyperpol,
                    processor.modified_hyperpol_times,
                    'hyperpol',
                    manual_plateau_regions.get('hyperpol') if manual_plateau_regions else None,
                    fitting_method,
                    validation_level
                )
                
                results['hyperpol'] = hyperpol_results
            
            # Process depolarization curve
            if hasattr(processor, 'modified_depol') and processor.modified_depol is not None:
                app_logger.info("Processing enhanced depolarization curve")
                
                depol_results = self._process_single_curve_enhanced(
                    processor.modified_depol,
                    processor.modified_depol_times,
                    'depol',
                    manual_plateau_regions.get('depol') if manual_plateau_regions else None,
                    fitting_method,
                    validation_level
                )
                
                results['depol'] = depol_results
            
            # Generate enhanced summary with AI insights
            results['enhanced_summary'] = self._generate_enhanced_summary(results)
            
            # AI-powered recommendations if available
            if self.use_ai:
                results['ai_recommendations'] = self._generate_ai_recommendations(results)
            
            # Performance metrics
            results['performance_metrics'] = self._calculate_performance_metrics(results)
            
            app_logger.info("AI-enhanced purple curve drift correction completed successfully")
            return results
            
        except Exception as e:
            app_logger.error(f"Error in enhanced processing: {str(e)}")
            raise
    
    def _process_single_curve_enhanced(self, curve_data: np.ndarray, curve_times: np.ndarray,
                                     curve_type: str, manual_plateau: Optional[Tuple],
                                     fitting_method: str, validation_level: str) -> Dict:
        """
        Process a single curve with enhanced methods and AI integration.
        """
        try:
            # Step 1: Enhanced segment identification
            segment_info = self.drift_corrector.identify_regression_segments(
                curve_data, curve_times, curve_type, method='auto'
            )
            
            # Step 2: AI-powered segment optimization if available
            if self.use_ai and self.ai_confidence_estimator:
                optimized_segment = self._ai_optimize_segment(
                    segment_info, curve_data, curve_times, curve_type
                )
                if optimized_segment:
                    segment_info.update(optimized_segment)
            
            # Step 3: Robust regression fitting
            regression_results = self.robust_analyzer.fit_robust_linear_regression(
                curve_data, curve_times, segment_info, fitting_method
            )
            
            # Step 4: Enhanced drift correction
            correction_info = self.drift_corrector.apply_drift_correction(
                curve_data, curve_times, regression_results,
                manual_plateau[0] if manual_plateau else None,
                manual_plateau[1] if manual_plateau else None
            )
            
            # Step 5: Validation based on requested level
            validation_results = {}
            if validation_level in ['standard', 'full']:
                validation_results = self.robust_analyzer.validate_regression_assumptions(
                    regression_results
                )
            
            if validation_level == 'full':
                # Additional comprehensive validation
                validation_results['curve_specific'] = self._validate_curve_specific_assumptions(
                    curve_data, curve_times, curve_type, regression_results
                )
                
                # AI-powered validation if available
                if self.use_ai:
                    validation_results['ai_assessment'] = self._ai_validate_results(
                        curve_data, curve_times, curve_type, regression_results
                    )
            
            # Step 6: Quality assessment
            quality_assessment = self._assess_fitting_quality(
                regression_results, validation_results, curve_type
            )
            
            # Step 7: AI confidence estimation
            ai_confidence = {}
            if self.use_ai and self.ai_confidence_estimator:
                curve_data_dict = {'time': curve_times, 'current': curve_data}
                ai_confidence = self.ai_confidence_estimator.predict_confidence(curve_data_dict)
            
            # Compile results
            enhanced_results = {
                'segment_info': segment_info,
                'regression_results': regression_results,
                'correction_info': correction_info,
                'validation_results': validation_results,
                'quality_assessment': quality_assessment,
                'ai_confidence': ai_confidence,
                'original_times': curve_times,
                'processing_metadata': {
                    'fitting_method_used': regression_results.get('method_used', fitting_method),
                    'segment_detection_method': segment_info.get('detection_method', 'auto'),
                    'validation_level': validation_level,
                    'ai_optimization_applied': bool(self.use_ai)
                }
            }
            
            return enhanced_results
            
        except Exception as e:
            app_logger.error(f"Error processing {curve_type} curve: {str(e)}")
            raise
    
    def _ai_optimize_segment(self, segment_info: Dict, curve_data: np.ndarray,
                           curve_times: np.ndarray, curve_type: str) -> Optional[Dict]:
        """
        Use AI-enhanced heuristics to optimize segment selection.
        """
        try:
            # AI-enhanced segment optimization using multiple criteria
            current_score = segment_info.get('linearity_score', 0)
            
            # Try slight adjustments to improve linearity
            best_adjustment = {'start_adjustment': 0, 'end_adjustment': 0}
            best_score = current_score
            
            for start_adj in [-3, -2, -1, 0, 1, 2, 3]:
                for end_adj in [-3, -2, -1, 0, 1, 2, 3]:
                    test_start = max(0, segment_info['start_idx'] + start_adj)
                    test_end = min(len(curve_data), segment_info['end_idx'] + end_adj)
                    
                    if test_end - test_start < 10:  # Minimum segment length
                        continue
                    
                    # Calculate linearity score for adjusted segment
                    test_score = self.drift_corrector._calculate_linearity_score(
                        curve_data[test_start:test_end], 
                        curve_times[test_start:test_end]
                    )
                    
                    if test_score > best_score:
                        best_score = test_score
                        best_adjustment = {'start_adjustment': start_adj, 'end_adjustment': end_adj}
            
            # Calculate confidence based on improvement
            improvement = best_score - current_score
            confidence = min(0.95, 0.5 + improvement * 2)  # Scale improvement to confidence
            
            optimized_segment = {
                'ai_confidence_score': confidence,
                'ai_optimization_applied': True,
                'ai_suggested_adjustments': best_adjustment,
                'ai_improvement': improvement,
                'ai_optimized_linearity_score': best_score
            }
            
            # Apply adjustments if they improve the score significantly
            if improvement > 0.05:  # 5% improvement threshold
                segment_info['start_idx'] = max(0, segment_info['start_idx'] + best_adjustment['start_adjustment'])
                segment_info['end_idx'] = min(len(curve_data), segment_info['end_idx'] + best_adjustment['end_adjustment'])
                optimized_segment['adjustments_applied'] = True
            else:
                optimized_segment['adjustments_applied'] = False
            
            return optimized_segment
            
        except Exception as e:
            app_logger.debug(f"AI optimization failed: {str(e)}")
            return None
    
    def _validate_curve_specific_assumptions(self, curve_data: np.ndarray, curve_times: np.ndarray,
                                          curve_type: str, regression_results: Dict) -> Dict:
        """
        Validate assumptions specific to purple curve analysis.
        """
        try:
            validation = {}
            
            # 1. Biological plausibility checks
            slope = regression_results['slope']
            
            if curve_type == 'hyperpol':
                # Hyperpolarization should generally have positive slope (recovery)
                validation['slope_direction'] = {
                    'expected': 'positive',
                    'observed': 'positive' if slope > 0 else 'negative',
                    'status': 'good' if slope > 0 else 'warning',
                    'slope_magnitude': abs(slope)
                }
            else:  # depol
                # Depolarization should generally have negative slope (decay)
                validation['slope_direction'] = {
                    'expected': 'negative',
                    'observed': 'negative' if slope < 0 else 'positive',
                    'status': 'good' if slope < 0 else 'warning',
                    'slope_magnitude': abs(slope)
                }
            
            # 2. Temporal consistency
            segment_duration = (regression_results['x_segment'][-1] - 
                              regression_results['x_segment'][0])
            
            validation['segment_duration'] = {
                'duration_ms': segment_duration,
                'status': 'good' if 5 <= segment_duration <= 200 else 'warning',
                'duration_range_check': 'within_expected' if 5 <= segment_duration <= 200 else 'outside_expected'
            }
            
            # 3. Amplitude consistency
            y_segment = regression_results['y_segment']
            amplitude_range = np.ptp(y_segment)  # Peak-to-peak
            
            validation['amplitude_range'] = {
                'range_pA': amplitude_range,
                'status': 'good' if amplitude_range > 0.5 else 'warning',  # At least 0.5 pA change
                'signal_strength': 'strong' if amplitude_range > 5 else 'moderate' if amplitude_range > 1 else 'weak'
            }
            
            # 4. Smoothness check
            if len(y_segment) > 2:
                second_derivatives = np.diff(np.diff(y_segment))
                smoothness_score = 1.0 / (1.0 + np.std(second_derivatives))
                
                validation['smoothness'] = {
                    'score': smoothness_score,
                    'status': 'good' if smoothness_score > 0.6 else 'warning',
                    'noise_level': 'low' if smoothness_score > 0.8 else 'moderate' if smoothness_score > 0.6 else 'high'
                }
            
            # 5. Curve shape consistency
            derivative = np.gradient(y_segment)
            
            if curve_type == 'hyperpol':
                # Should be mostly increasing
                increasing_ratio = np.sum(derivative > 0) / len(derivative)
                validation['shape_consistency'] = {
                    'increasing_ratio': increasing_ratio,
                    'status': 'good' if increasing_ratio > 0.7 else 'warning'
                }
            else:  # depol
                # Should be mostly decreasing
                decreasing_ratio = np.sum(derivative < 0) / len(derivative)
                validation['shape_consistency'] = {
                    'decreasing_ratio': decreasing_ratio,
                    'status': 'good' if decreasing_ratio > 0.7 else 'warning'
                }
            
            return validation
            
        except Exception as e:
            app_logger.warning(f"Curve-specific validation failed: {str(e)}")
            return {'error': str(e)}
    
    def _ai_validate_results(self, curve_data: np.ndarray, curve_times: np.ndarray,
                           curve_type: str, regression_results: Dict) -> Dict:
        """
        AI-powered validation of results.
        """
        try:
            # Mock AI validation using enhanced heuristics
            validation = {}
            
            # 1. Consistency check across multiple metrics
            r_squared = regression_results['r_squared']
            data_quality = regression_results.get('data_quality', {})
            
            # Combined quality score
            quality_factors = [
                r_squared,
                data_quality.get('overall_quality_score', 0.5),
                1.0 - data_quality.get('outlier_ratio', 0.5)  # Lower outlier ratio is better
            ]
            
            combined_quality = np.mean(quality_factors)
            
            validation['combined_quality'] = {
                'score': combined_quality,
                'level': 'high' if combined_quality > 0.8 else 'medium' if combined_quality > 0.6 else 'low',
                'factors': {
                    'r_squared_factor': r_squared,
                    'data_quality_factor': data_quality.get('overall_quality_score', 0.5),
                    'outlier_factor': 1.0 - data_quality.get('outlier_ratio', 0.5)
                }
            }
            
            # 2. Predictive reliability estimation
            segment_length = len(regression_results['x_segment'])
            method_reliability = {
                'ols': 0.7,
                'ransac': 0.9,
                'huber': 0.85,
                'theil_sen': 0.9
            }.get(regression_results.get('method_used', 'ols'), 0.7)
            
            reliability_score = min(1.0, method_reliability * (segment_length / 20) * combined_quality)
            
            validation['predictive_reliability'] = {
                'score': reliability_score,
                'confidence_interval': '95%' if reliability_score > 0.8 else '68%' if reliability_score > 0.6 else 'low',
                'method_contribution': method_reliability,
                'data_contribution': combined_quality
            }
            
            # 3. Anomaly detection
            residuals = regression_results.get('residuals', [])
            if len(residuals) > 0:
                # Check for systematic patterns in residuals
                residual_trend = np.corrcoef(np.arange(len(residuals)), residuals)[0, 1] if len(residuals) > 1 else 0
                
                validation['anomaly_detection'] = {
                    'residual_trend': residual_trend,
                    'systematic_error': abs(residual_trend) > 0.3,
                    'status': 'clean' if abs(residual_trend) < 0.2 else 'suspicious' if abs(residual_trend) < 0.5 else 'problematic'
                }
            
            return validation
            
        except Exception as e:
            app_logger.warning(f"AI validation failed: {str(e)}")
            return {'error': str(e)}
    
    def _assess_fitting_quality(self, regression_results: Dict, validation_results: Dict,
                              curve_type: str) -> Dict:
        """
        Assess overall fitting quality with enhanced scoring.
        """
        try:
            quality_factors = []
            detailed_assessment = {}
            
            # 1. Statistical quality (R², p-value, etc.)
            r_squared = regression_results.get('r_squared', 0)
            quality_factors.append(r_squared)
            detailed_assessment['statistical_fit'] = {
                'r_squared': r_squared,
                'score': r_squared,
                'status': 'excellent' if r_squared > 0.95 else 'good' if r_squared > 0.8 else 'acceptable' if r_squared > 0.6 else 'poor'
            }
            
            # 2. Data quality
            data_quality = regression_results.get('data_quality', {})
            data_quality_score = data_quality.get('overall_quality_score', 0.5)
            quality_factors.append(data_quality_score)
            detailed_assessment['data_quality'] = {
                'score': data_quality_score,
                'level': data_quality.get('quality_level', 'unknown'),
                'outlier_impact': data_quality.get('outlier_ratio', 0)
            }
            
            # 3. Validation results
            if validation_results and 'overall' in validation_results:
                validation_status = validation_results['overall']['status']
                validation_score = {
                    'excellent': 1.0,
                    'acceptable': 0.7,
                    'problematic': 0.3,
                    'error': 0.1
                }.get(validation_status, 0.5)
                quality_factors.append(validation_score)
                detailed_assessment['assumption_validation'] = {
                    'status': validation_status,
                    'score': validation_score,
                    'details': validation_results['overall']
                }
            
            # 4. Method-specific quality
            method_used = regression_results.get('method_used', 'unknown')
            method_reliability = {
                'ols': 0.8,
                'ransac': 0.9,
                'huber': 0.85,
                'theil_sen': 0.9
            }.get(method_used, 0.7)
            quality_factors.append(method_reliability)
            detailed_assessment['method_reliability'] = {
                'method': method_used,
                'reliability_score': method_reliability,
                'robustness': 'high' if method_used in ['ransac', 'theil_sen'] else 'medium'
            }
            
            # 5. Curve-specific quality (if available)
            if 'curve_specific' in validation_results:
                curve_validation = validation_results['curve_specific']
                curve_quality_factors = []
                
                for check_name, check_result in curve_validation.items():
                    if isinstance(check_result, dict) and 'status' in check_result:
                        score = 1.0 if check_result['status'] == 'good' else 0.5
                        curve_quality_factors.append(score)
                
                if curve_quality_factors:
                    curve_quality_score = np.mean(curve_quality_factors)
                    quality_factors.append(curve_quality_score)
                    detailed_assessment['curve_specific_quality'] = {
                        'score': curve_quality_score,
                        'checks_passed': len([f for f in curve_quality_factors if f == 1.0]),
                        'total_checks': len(curve_quality_factors)
                    }
            
            # Calculate overall quality score
            overall_quality = np.mean(quality_factors) if quality_factors else 0.5
            
            # Enhanced quality level determination
            if overall_quality >= 0.9:
                quality_level = 'excellent'
                recommendation = 'Results are publication-ready with high confidence'
            elif overall_quality >= 0.8:
                quality_level = 'very_good'
                recommendation = 'Results are highly reliable for analysis'
            elif overall_quality >= 0.7:
                quality_level = 'good'
                recommendation = 'Results are reliable for most analyses'
            elif overall_quality >= 0.6:
                quality_level = 'acceptable'
                recommendation = 'Results are usable but consider additional validation'
            elif overall_quality >= 0.4:
                quality_level = 'questionable'
                recommendation = 'Results should be carefully reviewed and possibly re-analyzed'
            else:
                quality_level = 'poor'
                recommendation = 'Results are unreliable and require re-analysis with different methods'
            
            # Risk assessment
            risk_factors = []
            if r_squared < 0.6:
                risk_factors.append('Low R-squared indicates poor fit')
            if data_quality.get('outlier_ratio', 0) > 0.2:
                risk_factors.append('High outlier ratio may affect results')
            if method_used == 'ols' and data_quality.get('outlier_ratio', 0) > 0.1:
                risk_factors.append('OLS method may be sensitive to detected outliers')
            
            return {
                'overall_score': overall_quality,
                'quality_level': quality_level,
                'recommendation': recommendation,
                'detailed_assessment': detailed_assessment,
                'quality_factors': quality_factors,
                'risk_factors': risk_factors,
                'confidence_intervals_available': 'prediction_intervals' in regression_results,
                'robustness_score': method_reliability
            }
            
        except Exception as e:
            app_logger.error(f"Error assessing fitting quality: {str(e)}")
            return {
                'overall_score': 0.0,
                'quality_level': 'error',
                'recommendation': f'Quality assessment failed: {str(e)}',
                'error': str(e)
            }
    
    def _generate_enhanced_summary(self, results: Dict) -> Dict:
        """
        Generate enhanced summary with comprehensive statistics.
        """
        summary = {
            'processing_timestamp': results['processing_info']['timestamp'],
            'curves_processed': [],
            'overall_quality': 'unknown',
            'key_findings': [],
            'recommendations': [],
            'method_effectiveness': {},
            'data_characteristics': {}
        }
        
        quality_scores = []
        methods_used = set()
        
        for curve_type in ['hyperpol', 'depol']:
            if curve_type in results:
                curve_data = results[curve_type]
                summary['curves_processed'].append(curve_type)
                
                # Extract key metrics
                regression = curve_data['regression_results']
                quality = curve_data['quality_assessment']
                segment_info = curve_data['segment_info']
                
                quality_scores.append(quality['overall_score'])
                methods_used.add(regression.get('method_used', 'unknown'))
                
                # Add key findings
                finding = {
                    'curve_type': curve_type,
                    'slope': regression['slope'],
                    'slope_units': 'pA/ms',
                    'intercept': regression['intercept'],
                    'r_squared': regression['r_squared'],
                    'quality_level': quality['quality_level'],
                    'method_used': regression.get('method_used', 'unknown'),
                    'detection_method': segment_info.get('detection_method', 'unknown'),
                    'segment_length': segment_info['end_idx'] - segment_info['start_idx'],
                    'confidence_available': 'prediction_intervals' in regression
                }
                
                # Add AI confidence if available
                if 'ai_confidence' in curve_data and curve_data['ai_confidence']:
                    finding['ai_confidence'] = curve_data['ai_confidence'].get('confidence_score', 'N/A')
                
                summary['key_findings'].append(finding)
                
                # Add recommendations if quality is poor
                if quality['overall_score'] < 0.7:
                    summary['recommendations'].append(
                        f"{curve_type}: {quality['recommendation']}"
                    )
                
                # Collect data characteristics
                data_quality = regression.get('data_quality', {})
                summary['data_characteristics'][curve_type] = {
                    'sample_size': data_quality.get('sample_size', 'unknown'),
                    'outlier_ratio': data_quality.get('outlier_ratio', 'unknown'),
                    'quality_level': data_quality.get('quality_level', 'unknown')
                }
        
        # Overall quality assessment
        if quality_scores:
            overall_quality_score = np.mean(quality_scores)
            if overall_quality_score >= 0.85:
                summary['overall_quality'] = 'excellent'
            elif overall_quality_score >= 0.7:
                summary['overall_quality'] = 'good'
            elif overall_quality_score >= 0.5:
                summary['overall_quality'] = 'acceptable'
            else:
                summary['overall_quality'] = 'needs_attention'
            
            summary['overall_quality_score'] = overall_quality_score
        
        # Method effectiveness summary
        summary['method_effectiveness'] = {
            'methods_used': list(methods_used),
            'automatic_detection': all(
                finding.get('detection_method') != 'manual' 
                for finding in summary['key_findings']
            ),
            'robust_methods_used': any(
                method in ['ransac', 'huber', 'theil_sen'] 
                for method in methods_used
            )
        }
        
        return summary
    
    def _generate_ai_recommendations(self, results: Dict) -> Dict:
        """
        Generate AI-powered recommendations for improvement.
        """
        try:
            recommendations = {
                'data_preprocessing': [],
                'fitting_methods': [],
                'validation_improvements': [],
                'general_suggestions': [],
                'next_steps': []
            }
            
            # Analyze results and generate recommendations
            overall_quality = 0
            quality_count = 0
            
            for curve_type in ['hyperpol', 'depol']:
                if curve_type in results:
                    curve_data = results[curve_type]
                    quality = curve_data['quality_assessment']
                    overall_quality += quality['overall_score']
                    quality_count += 1
                    
                    if quality['overall_score'] < 0.7:
                        # Suggest alternative methods
                        current_method = curve_data['regression_results'].get('method_used', 'ols')
                        if current_method == 'ols':
                            recommendations['fitting_methods'].append(
                                f"Try robust methods (RANSAC/Huber) for {curve_type} curve to handle potential outliers"
                            )
                        
                        # Data quality issues
                        data_quality = curve_data['regression_results'].get('data_quality', {})
                        if data_quality.get('outlier_ratio', 0) > 0.15:
                            recommendations['data_preprocessing'].append(
                                f"High outlier ratio ({data_quality.get('outlier_ratio', 0):.1%}) detected in {curve_type} - consider data cleaning or filtering"
                            )
                        
                        if data_quality.get('overall_quality_score', 1) < 0.6:
                            recommendations['data_preprocessing'].append(
                                f"Low data quality score for {curve_type} - check data acquisition parameters"
                            )
                        
                        # Validation issues
                        validation = curve_data.get('validation_results', {})
                        if validation.get('overall', {}).get('status') == 'problematic':
                            recommendations['validation_improvements'].append(
                                f"Multiple assumption violations in {curve_type} - review experimental conditions"
                            )
                        
                        # R-squared specific recommendations
                        r_squared = curve_data['regression_results'].get('r_squared', 0)
                        if r_squared < 0.6:
                            recommendations['fitting_methods'].append(
                                f"Low R² ({r_squared:.3f}) for {curve_type} - consider non-linear fitting or different segment selection"
                            )
            
            # Overall recommendations based on average quality
            if quality_count > 0:
                avg_quality = overall_quality / quality_count
                
                if avg_quality < 0.5:
                    recommendations['general_suggestions'].extend([
                        "Overall analysis quality is low - consider reviewing experimental protocol",
                        "Check electrode condition and recording environment",
                        "Verify that purple curve conditions are optimal"
                    ])
                elif avg_quality < 0.7:
                    recommendations['general_suggestions'].extend([
                        "Consider optimizing data acquisition parameters",
                        "Implement automated outlier detection in data processing pipeline"
                    ])
                else:
                    recommendations['general_suggestions'].append(
                        "Analysis quality is good - results are reliable for further analysis"
                    )
            
            # AI-powered next steps
            recommendations['next_steps'] = [
                "Consider ensemble methods for more robust results",
                "Implement cross-validation for method selection",
                "Use bootstrap resampling for confidence intervals",
                "Consider Bayesian regression for uncertainty quantification"
            ]
            
            # Method-specific recommendations
            methods_used = set()
            for curve_type in ['hyperpol', 'depol']:
                if curve_type in results:
                    method = results[curve_type]['regression_results'].get('method_used', 'unknown')
                    methods_used.add(method)
            
            if 'ols' in methods_used:
                recommendations['fitting_methods'].append(
                    "OLS was used - consider robust alternatives if outliers are present"
                )
            
            if len(methods_used) == 1 and 'ols' in methods_used:
                recommendations['fitting_methods'].append(
                    "Consider using multiple methods and comparing results for validation"
                )
            
            return recommendations
            
        except Exception as e:
            app_logger.error(f"Error generating AI recommendations: {str(e)}")
            return {'error': str(e)}
    
    def _calculate_performance_metrics(self, results: Dict) -> Dict:
        """
        Calculate performance metrics for the analysis.
        """
        try:
            metrics = {
                'processing_time': 'N/A',  # Would be calculated in real implementation
                'curves_analyzed': len([k for k in ['hyperpol', 'depol'] if k in results]),
                'methods_comparison': {},
                'accuracy_indicators': {},
                'efficiency_metrics': {}
            }
            
            # Accuracy indicators
            r_squared_values = []
            quality_scores = []
            
            for curve_type in ['hyperpol', 'depol']:
                if curve_type in results:
                    r_squared = results[curve_type]['regression_results'].get('r_squared', 0)
                    quality = results[curve_type]['quality_assessment']['overall_score']
                    
                    r_squared_values.append(r_squared)
                    quality_scores.append(quality)
            
            if r_squared_values:
                metrics['accuracy_indicators'] = {
                    'average_r_squared': np.mean(r_squared_values),
                    'min_r_squared': np.min(r_squared_values),
                    'max_r_squared': np.max(r_squared_values),
                    'average_quality_score': np.mean(quality_scores),
                    'consistency': np.std(quality_scores)  # Lower is better
                }
            
            # Method effectiveness
            method_performance = {}
            for curve_type in ['hyperpol', 'depol']:
                if curve_type in results:
                    method = results[curve_type]['regression_results'].get('method_used', 'unknown')
                    r_squared = results[curve_type]['regression_results'].get('r_squared', 0)
                    
                    if method not in method_performance:
                        method_performance[method] = []
                    method_performance[method].append(r_squared)
            
            metrics['methods_comparison'] = {
                method: {
                    'average_r_squared': np.mean(scores),
                    'count': len(scores)
                }
                for method, scores in method_performance.items()
            }
            
            return metrics
            
        except Exception as e:
            app_logger.error(f"Error calculating performance metrics: {str(e)}")
            return {'error': str(e)}

    def export_enhanced_results(self, results: Dict, filepath: str = None) -> str:
        """
        Export enhanced results with comprehensive reporting.
        """
        try:
            if filepath is None:
                try:
                    from tkinter import filedialog
                    filepath = filedialog.asksaveasfilename(
                        defaultextension=".json",
                        filetypes=[("JSON files", "*.json"), ("Excel files", "*.xlsx"), ("All files", "*.*")],
                        title="Export AI-Enhanced Linear Fitting Results"
                    )
                except ImportError:
                    # Fallback filename
                    from datetime import datetime
                    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                    filepath = f"ai_enhanced_linear_fitting_results_{timestamp}.json"
            
            if not filepath:
                return None
            
            # Prepare export data (convert numpy arrays to lists)
            export_data = self._prepare_export_data(results)
            
            # Add metadata
            export_data['export_metadata'] = {
                'export_timestamp': np.datetime64('now').astype(str),
                'analysis_version': 'AI-Enhanced v1.0',
                'feature_set': [
                    'Enhanced Segment Detection',
                    'Robust Regression',
                    'AI-Powered Optimization',
                    'Comprehensive Validation',
                    'Quality Assessment'
                ]
            }
            
            if filepath.endswith('.xlsx'):
                # Export to Excel using the enhanced corrector's method
                return self.drift_corrector.export_correction_report(
                    results, 
                    results.get('corrected_integrals'), 
                    filepath
                )
            else:
                # Export to JSON
                with open(filepath, 'w') as f:
                    json.dump(export_data, f, indent=2, default=str)
            
            app_logger.info(f"AI-enhanced results exported to: {filepath}")
            return filepath
            
        except Exception as e:
            app_logger.error(f"Error exporting enhanced results: {str(e)}")
            raise
    
    def _prepare_export_data(self, results: Dict) -> Dict:
        """
        Prepare results data for export by converting numpy arrays.
        """
        def convert_numpy(obj):
            if isinstance(obj, np.ndarray):
                return obj.tolist()
            elif isinstance(obj, dict):
                return {k: convert_numpy(v) for k, v in obj.items()}
            elif isinstance(obj, list):
                return [convert_numpy(item) for item in obj]
            elif isinstance(obj, (np.integer, np.floating)):
                return obj.item()
            elif isinstance(obj, np.bool_):
                return bool(obj)
            else:
                return obj
        
        return convert_numpy(results)

    def calculate_corrected_integrals(self, results: Dict, integration_ranges: Dict) -> Dict:
        """
        Calculate corrected integrals using the enhanced drift corrector.
        """
        return self.drift_corrector.calculate_corrected_integrals(results, integration_ranges)
    
    def create_visualization(self, results: Dict, save_path: str = None):
        """
        Create comprehensive visualization using the enhanced visualization tools.
        """
        try:
            # Try to import enhanced visualization tools
            from .enhanced_visualization import EnhancedVisualizationTools
            viz_tools = EnhancedVisualizationTools()
            return viz_tools.create_comprehensive_analysis_plot(results, save_path)
        except ImportError:
            # Fallback to basic visualization
            return self.drift_corrector.plot_correction_analysis(results, save_path)

# Backward compatibility
AILinearFitting = AIEnhancedLinearFitting