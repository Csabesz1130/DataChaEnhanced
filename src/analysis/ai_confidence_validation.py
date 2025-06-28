# src/analysis/ai_confidence_validation.py
"""
AI Confidence and Validation System

This module provides confidence estimation, validation, and quality control
for AI predictions, ensuring reliable automated curve fitting.
"""

import numpy as np
from sklearn.model_selection import cross_val_score, LeaveOneOut
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.ensemble import IsolationForest
import matplotlib.pyplot as plt
from scipy import stats
from src.utils.logger import app_logger

class PredictionConfidenceEstimator:
    """
    Estimates confidence intervals and uncertainty for AI predictions.
    """
    
    def __init__(self, ai_system):
        self.ai_system = ai_system
        self.prediction_history = []
        self.confidence_thresholds = {
            'high': 0.8,
            'medium': 0.5,
            'low': 0.0
        }
        
    def predict_with_confidence(self, curve_data):
        """
        Make predictions with confidence intervals.
        
        Returns:
            dict: Predictions with confidence scores and intervals
        """
        # Get base predictions
        predictions = self.ai_system.predict_optimal_points(curve_data)
        
        if predictions is None:
            return None
        
        # Add confidence estimation
        confidence_results = {}
        
        for curve_type, curve_predictions in predictions.items():
            confidence_results[curve_type] = {
                'predictions': curve_predictions,
                'confidence': self._estimate_confidence(curve_data[curve_type], curve_predictions),
                'intervals': self._calculate_prediction_intervals(curve_data[curve_type], curve_predictions),
                'anomaly_score': self._calculate_anomaly_score(curve_data[curve_type])
            }
        
        return confidence_results
    
    def _estimate_confidence(self, curve_data, predictions):
        """
        Estimate confidence based on multiple factors.
        """
        confidence_factors = []
        
        # 1. Prediction consistency (if using ensemble)
        if hasattr(self.ai_system, 'models'):
            prediction_variance = self._estimate_prediction_variance(curve_data, predictions)
            consistency_score = 1.0 / (1.0 + prediction_variance)
            confidence_factors.append(consistency_score)
        
        # 2. Feature reliability
        feature_quality = self._assess_feature_quality(curve_data)
        confidence_factors.append(feature_quality)
        
        # 3. Training data similarity
        similarity_score = self._calculate_training_similarity(curve_data)
        confidence_factors.append(similarity_score)
        
        # 4. Prediction feasibility
        feasibility_score = self._check_prediction_feasibility(predictions, len(curve_data['current']))
        confidence_factors.append(feasibility_score)
        
        # Combine factors
        overall_confidence = np.mean(confidence_factors)
        
        return {
            'overall': overall_confidence,
            'level': self._get_confidence_level(overall_confidence),
            'factors': {
                'consistency': confidence_factors[0] if len(confidence_factors) > 0 else 0,
                'feature_quality': confidence_factors[1] if len(confidence_factors) > 1 else 0,
                'similarity': confidence_factors[2] if len(confidence_factors) > 2 else 0,
                'feasibility': confidence_factors[3] if len(confidence_factors) > 3 else 0
            }
        }
    
    def _estimate_prediction_variance(self, curve_data, predictions):
        """Estimate variance in predictions from ensemble."""
        # This would use the RandomForest's built-in variance estimation
        # For now, return a placeholder
        return 0.1
    
    def _assess_feature_quality(self, curve_data):
        """Assess quality of extracted features."""
        try:
            current = curve_data['current']
            
            # Check for data quality issues
            quality_checks = []
            
            # 1. Signal-to-noise ratio
            snr = np.mean(current) / np.std(current) if np.std(current) > 0 else 0
            quality_checks.append(min(1.0, snr / 10))  # Normalize to 0-1
            
            # 2. Data completeness (no NaN/inf values)
            completeness = 1.0 if not (np.any(np.isnan(current)) or np.any(np.isinf(current))) else 0.0
            quality_checks.append(completeness)
            
            # 3. Sufficient data points
            sufficient_points = min(1.0, len(current) / 100)  # Assume 100 points is ideal minimum
            quality_checks.append(sufficient_points)
            
            return np.mean(quality_checks)
            
        except:
            return 0.5
    
    def _calculate_training_similarity(self, curve_data):
        """Calculate similarity to training data."""
        if len(self.ai_system.training_data) == 0:
            return 0.5
        
        try:
            # Extract features from current curve
            features = self.ai_system._extract_curve_features({'test': curve_data})['test']
            
            # Compare to training data features
            similarities = []
            
            for training_example in self.ai_system.training_data[-10:]:  # Last 10 examples
                if 'curve_features' in training_example:
                    for curve_type in ['hyperpol', 'depol']:
                        if curve_type in training_example['curve_features']:
                            train_features = training_example['curve_features'][curve_type]
                            
                            # Calculate cosine similarity
                            similarity = self._cosine_similarity(features, train_features)
                            similarities.append(similarity)
            
            return np.mean(similarities) if similarities else 0.5
            
        except:
            return 0.5
    
    def _cosine_similarity(self, features1, features2):
        """Calculate cosine similarity between feature sets."""
        common_keys = set(features1.keys()) & set(features2.keys())
        
        if not common_keys:
            return 0.0
        
        vec1 = np.array([features1[k] for k in common_keys])
        vec2 = np.array([features2[k] for k in common_keys])
        
        dot_product = np.dot(vec1, vec2)
        norm_product = np.linalg.norm(vec1) * np.linalg.norm(vec2)
        
        return dot_product / norm_product if norm_product > 0 else 0.0
    
    def _check_prediction_feasibility(self, predictions, data_length):
        """Check if predictions are feasible given data constraints."""
        feasibility_checks = []
        
        # Check if predictions are within data bounds
        for key, value in predictions.items():
            if 0 <= value < data_length:
                feasibility_checks.append(1.0)
            else:
                feasibility_checks.append(0.0)
        
        # Check logical ordering
        if 'linear_start' in predictions and 'linear_end' in predictions:
            if predictions['linear_start'] < predictions['linear_end']:
                feasibility_checks.append(1.0)
            else:
                feasibility_checks.append(0.0)
        
        if 'linear_end' in predictions and 'exp_start' in predictions:
            if predictions['linear_end'] <= predictions['exp_start']:
                feasibility_checks.append(1.0)
            else:
                feasibility_checks.append(0.5)  # Partial penalty
        
        return np.mean(feasibility_checks) if feasibility_checks else 0.0
    
    def _calculate_prediction_intervals(self, curve_data, predictions):
        """Calculate prediction intervals for each predicted point."""
        intervals = {}
        
        # Use bootstrap or ensemble variance for intervals
        # For now, use heuristic based on data characteristics
        data_std = np.std(curve_data['current'])
        
        for key, value in predictions.items():
            # Interval width proportional to data variability
            interval_width = max(5, int(0.1 * len(curve_data['current'])))
            
            intervals[key] = {
                'lower': max(0, value - interval_width),
                'upper': min(len(curve_data['current']) - 1, value + interval_width),
                'width': interval_width
            }
        
        return intervals
    
    def _calculate_anomaly_score(self, curve_data):
        """Calculate anomaly score for the curve."""
        try:
            # Use Isolation Forest for anomaly detection
            current = curve_data['current'].reshape(-1, 1)
            
            detector = IsolationForest(contamination=0.1, random_state=42)
            anomaly_scores = detector.fit_predict(current)
            
            # Convert to 0-1 score (1 = normal, 0 = anomaly)
            anomaly_ratio = np.sum(anomaly_scores == -1) / len(anomaly_scores)
            
            return 1.0 - anomaly_ratio
            
        except:
            return 0.5
    
    def _get_confidence_level(self, confidence_score):
        """Convert numerical confidence to categorical level."""
        if confidence_score >= self.confidence_thresholds['high']:
            return 'high'
        elif confidence_score >= self.confidence_thresholds['medium']:
            return 'medium'
        else:
            return 'low'

class ModelValidator:
    """
    Validates AI model performance and provides quality metrics.
    """
    
    def __init__(self, ai_system):
        self.ai_system = ai_system
        self.validation_results = {}
        
    def perform_cross_validation(self):
        """
        Perform leave-one-out cross validation on training data.
        """
        if len(self.ai_system.training_data) < 3:
            app_logger.warning("Insufficient data for cross-validation")
            return None
        
        app_logger.info("Performing cross-validation...")
        
        results = {
            'hyperpol': {},
            'depol': {}
        }
        
        for curve_type in ['hyperpol', 'depol']:
            # Prepare data
            X, y = self.ai_system._prepare_training_data(curve_type)
            
            if len(X) < 3:
                continue
            
            # Scale features
            X_scaled = self.ai_system.scalers[curve_type].fit_transform(X)
            
            # Validate each target
            for target_name, target_values in y.items():
                if len(target_values) < 3:
                    continue
                
                model_key = f"{curve_type}_{target_name}"
                model = self.ai_system.models[model_key]
                
                if model is None:
                    continue
                
                # Leave-one-out cross validation
                loo = LeaveOneOut()
                cv_scores = cross_val_score(
                    model, X_scaled, target_values,
                    cv=loo, scoring='neg_mean_squared_error'
                )
                
                results[curve_type][target_name] = {
                    'cv_mse': -np.mean(cv_scores),
                    'cv_std': np.std(cv_scores),
                    'cv_scores': cv_scores
                }
        
        self.validation_results = results
        return results
    
    def analyze_prediction_errors(self):
        """
        Analyze prediction errors to identify systematic issues.
        """
        if not self.validation_results:
            self.perform_cross_validation()
        
        error_analysis = {}
        
        for curve_type, targets in self.validation_results.items():
            error_analysis[curve_type] = {}
            
            for target_name, metrics in targets.items():
                if 'cv_scores' in metrics:
                    errors = -metrics['cv_scores']  # Convert back to positive MSE
                    
                    analysis = {
                        'mean_error': np.mean(errors),
                        'median_error': np.median(errors),
                        'error_skewness': stats.skew(errors),
                        'error_kurtosis': stats.kurtosis(errors),
                        'outlier_ratio': np.sum(errors > np.mean(errors) + 2*np.std(errors)) / len(errors)
                    }
                    
                    error_analysis[curve_type][target_name] = analysis
        
        return error_analysis
    
    def generate_validation_report(self, save_path=None):
        """
        Generate comprehensive validation report.
        """
        report_lines = [
            "=" * 60,
            "AI MODEL VALIDATION REPORT",
            "=" * 60,
            f"Generated: {np.datetime64('now')}",
            f"Training Examples: {len(self.ai_system.training_data)}",
            ""
        ]
        
        # Cross-validation results
        if self.validation_results:
            report_lines.append("CROSS-VALIDATION RESULTS:")
            report_lines.append("-" * 40)
            
            for curve_type, targets in self.validation_results.items():
                report_lines.append(f"\n{curve_type.upper()} Curve:")
                
                for target_name, metrics in targets.items():
                    report_lines.append(f"  {target_name}:")
                    report_lines.append(f"    Mean MSE: {metrics.get('cv_mse', 'N/A'):.2f}")
                    report_lines.append(f"    Std Dev: {metrics.get('cv_std', 'N/A'):.2f}")
        
        # Error analysis
        error_analysis = self.analyze_prediction_errors()
        if error_analysis:
            report_lines.extend([
                "",
                "ERROR ANALYSIS:",
                "-" * 40
            ])
            
            for curve_type, targets in error_analysis.items():
                report_lines.append(f"\n{curve_type.upper()} Curve:")
                
                for target_name, analysis in targets.items():
                    report_lines.append(f"  {target_name}:")
                    report_lines.append(f"    Error Skewness: {analysis.get('error_skewness', 'N/A'):.2f}")
                    report_lines.append(f"    Outlier Ratio: {analysis.get('outlier_ratio', 'N/A'):.2%}")
        
        # Recommendations
        report_lines.extend([
            "",
            "RECOMMENDATIONS:",
            "-" * 40
        ])
        
        if len(self.ai_system.training_data) < 10:
            report_lines.append("- Collect more training examples (minimum 10 recommended)")
        
        if any(metrics.get('cv_mse', 0) > 100 for targets in self.validation_results.values() 
               for metrics in targets.values()):
            report_lines.append("- High prediction errors detected - review feature engineering")
        
        report_lines.append("")
        
        # Save or return report
        report = "\n".join(report_lines)
        
        if save_path:
            with open(save_path, 'w') as f:
                f.write(report)
            app_logger.info(f"Validation report saved to: {save_path}")
        
        return report

class ActiveLearningSelector:
    """
    Identifies which curves would benefit most from expert review.
    """
    
    def __init__(self, confidence_estimator):
        self.confidence_estimator = confidence_estimator
        self.selection_history = []
        
    def select_for_review(self, curve_batch, max_selections=5):
        """
        Select curves that would most improve the model if labeled.
        """
        candidates = []
        
        for i, curve_data in enumerate(curve_batch):
            # Get confidence estimates
            confidence_results = self.confidence_estimator.predict_with_confidence(curve_data)
            
            if confidence_results is None:
                continue
            
            # Calculate informativeness score
            informativeness = self._calculate_informativeness(confidence_results)
            
            candidates.append({
                'index': i,
                'curve_data': curve_data,
                'confidence': confidence_results,
                'informativeness': informativeness
            })
        
        # Sort by informativeness (descending)
        candidates.sort(key=lambda x: x['informativeness'], reverse=True)
        
        # Select top candidates
        selected = candidates[:max_selections]
        
        # Record selection
        self.selection_history.append({
            'timestamp': np.datetime64('now'),
            'selected_indices': [c['index'] for c in selected],
            'scores': [c['informativeness'] for c in selected]
        })
        
        return selected
    
    def _calculate_informativeness(self, confidence_results):
        """
        Calculate how informative this example would be for training.
        
        High uncertainty + high anomaly = very informative
        """
        informativeness_scores = []
        
        for curve_type, results in confidence_results.items():
            confidence = results['confidence']['overall']
            anomaly = results['anomaly_score']
            
            # Uncertainty (inverse of confidence)
            uncertainty = 1.0 - confidence
            
            # Combine uncertainty and anomaly
            # High uncertainty is informative
            # Moderate anomaly is informative (not too extreme)
            anomaly_factor = 2 * anomaly * (1 - anomaly)  # Peaks at 0.5
            
            informativeness = uncertainty * 0.7 + anomaly_factor * 0.3
            informativeness_scores.append(informativeness)
        
        return np.mean(informativeness_scores)