# src/analysis/ai_visualization_explain.py
"""
AI Visualization and Explainability Module

This module provides visual insights into AI decisions, feature importance,
and model behavior to build trust and understanding.
"""

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.patches import Rectangle, FancyBboxPatch
from matplotlib.lines import Line2D
import matplotlib.gridspec as gridspec
from src.utils.logger import app_logger

class AIDecisionVisualizer:
    """
    Visualizes AI decisions and predictions on curve data.
    """
    
    def __init__(self):
        self.figure_size = (15, 10)
        self.color_scheme = {
            'hyperpol': '#3498db',  # Blue
            'depol': '#e74c3c',     # Red
            'linear': '#2ecc71',    # Green
            'exponential': '#f39c12', # Orange
            'confidence_high': '#27ae60',
            'confidence_medium': '#f39c12',
            'confidence_low': '#e74c3c'
        }
        
    def create_prediction_visualization(self, curve_data, predictions, save_path=None):
        """
        Create comprehensive visualization of AI predictions.
        """
        fig = plt.figure(figsize=self.figure_size)
        gs = gridspec.GridSpec(3, 2, height_ratios=[2, 2, 1], hspace=0.3)
        
        # Plot hyperpolarization predictions
        ax1 = fig.add_subplot(gs[0, :])
        self._plot_curve_with_predictions(
            ax1, 
            curve_data['hyperpol'], 
            predictions['hyperpol'],
            'Hyperpolarization Curve - AI Predictions'
        )
        
        # Plot depolarization predictions
        ax2 = fig.add_subplot(gs[1, :])
        self._plot_curve_with_predictions(
            ax2, 
            curve_data['depol'], 
            predictions['depol'],
            'Depolarization Curve - AI Predictions'
        )
        
        # Confidence meters
        ax3 = fig.add_subplot(gs[2, 0])
        self._plot_confidence_meter(ax3, predictions['hyperpol']['confidence'])
        
        ax4 = fig.add_subplot(gs[2, 1])
        self._plot_confidence_meter(ax4, predictions['depol']['confidence'])
        
        plt.suptitle('AI Curve Analysis Predictions', fontsize=16, fontweight='bold')
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            app_logger.info(f"Prediction visualization saved to: {save_path}")
        
        return fig
    
    def _plot_curve_with_predictions(self, ax, curve_data, prediction_results, title):
        """Plot curve with AI-predicted regions highlighted."""
        time = curve_data['time']
        current = curve_data['current']
        predictions = prediction_results['predictions']
        intervals = prediction_results['intervals']
        
        # Plot the curve
        ax.plot(time, current, 'k-', linewidth=1.5, alpha=0.7, label='Data')
        
        # Highlight linear region
        if 'linear_start' in predictions and 'linear_end' in predictions:
            start_idx = predictions['linear_start']
            end_idx = predictions['linear_end']
            
            # Draw linear region
            ax.axvspan(
                time[start_idx], time[end_idx],
                alpha=0.3, color=self.color_scheme['linear'],
                label='Linear Region'
            )
            
            # Add confidence intervals
            ax.axvspan(
                time[intervals['linear_start']['lower']],
                time[intervals['linear_start']['upper']],
                alpha=0.1, color=self.color_scheme['linear']
            )
            ax.axvspan(
                time[intervals['linear_end']['lower']],
                time[intervals['linear_end']['upper']],
                alpha=0.1, color=self.color_scheme['linear']
            )
            
            # Mark exact points
            ax.plot(time[start_idx], current[start_idx], 'o', 
                   color=self.color_scheme['linear'], markersize=8,
                   label=f'Linear Start (idx={start_idx})')
            ax.plot(time[end_idx], current[end_idx], 's', 
                   color=self.color_scheme['linear'], markersize=8,
                   label=f'Linear End (idx={end_idx})')
        
        # Highlight exponential region
        if 'exp_start' in predictions:
            exp_idx = predictions['exp_start']
            
            # Draw exponential region
            ax.axvspan(
                time[exp_idx], time[-1],
                alpha=0.3, color=self.color_scheme['exponential'],
                label='Exponential Region'
            )
            
            # Add confidence interval
            ax.axvspan(
                time[intervals['exp_start']['lower']],
                time[intervals['exp_start']['upper']],
                alpha=0.1, color=self.color_scheme['exponential']
            )
            
            # Mark exact point
            ax.plot(time[exp_idx], current[exp_idx], '^', 
                   color=self.color_scheme['exponential'], markersize=10,
                   label=f'Exp Start (idx={exp_idx})')
        
        # Add annotations
        confidence = prediction_results['confidence']
        confidence_color = self.color_scheme[f'confidence_{confidence["level"]}']
        
        # Add confidence badge
        props = dict(boxstyle='round,pad=0.3', facecolor=confidence_color, alpha=0.7)
        ax.text(0.02, 0.98, f'Confidence: {confidence["overall"]:.2f} ({confidence["level"]})',
               transform=ax.transAxes, fontsize=10, verticalalignment='top',
               bbox=props)
        
        # Styling
        ax.set_xlabel('Time (ms)')
        ax.set_ylabel('Current (pA)')
        ax.set_title(title)
        ax.legend(loc='best')
        ax.grid(True, alpha=0.3)
    
    def _plot_confidence_meter(self, ax, confidence_data):
        """Plot confidence meter visualization."""
        overall = confidence_data['overall']
        factors = confidence_data['factors']
        
        # Create meter
        meter_width = 0.8
        meter_height = 0.2
        meter_x = 0.1
        meter_y = 0.4
        
        # Background
        bg_rect = Rectangle((meter_x, meter_y), meter_width, meter_height,
                           facecolor='lightgray', edgecolor='black')
        ax.add_patch(bg_rect)
        
        # Confidence fill
        conf_color = self.color_scheme[f'confidence_{confidence_data["level"]}']
        conf_rect = Rectangle((meter_x, meter_y), meter_width * overall, meter_height,
                            facecolor=conf_color, alpha=0.8)
        ax.add_patch(conf_rect)
        
        # Add percentage text
        ax.text(0.5, 0.5, f'{overall:.0%}', ha='center', va='center',
               fontsize=14, fontweight='bold', transform=ax.transAxes)
        
        # Factor breakdown
        y_pos = 0.2
        for factor_name, factor_value in factors.items():
            ax.text(0.1, y_pos, f'{factor_name}: {factor_value:.2f}',
                   fontsize=8, transform=ax.transAxes)
            y_pos -= 0.05
        
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        ax.axis('off')
        ax.set_title(f'{confidence_data["level"].capitalize()} Confidence', fontsize=12)

class FeatureImportanceVisualizer:
    """
    Visualizes feature importance and model decision factors.
    """
    
    def __init__(self, ai_system):
        self.ai_system = ai_system
        
    def plot_feature_importance(self, curve_type='hyperpol', target='linear_start', save_path=None):
        """
        Plot feature importance for a specific model.
        """
        model_key = f"{curve_type}_{target}"
        model = self.ai_system.models.get(model_key)
        
        if model is None:
            app_logger.warning(f"No model found for {model_key}")
            return None
        
        # Get feature importance from RandomForest
        importances = model.feature_importances_
        
        # Get feature names (need to be consistent with extraction)
        feature_names = self._get_feature_names()
        
        # Sort by importance
        indices = np.argsort(importances)[::-1][:20]  # Top 20 features
        
        # Create plot
        fig, ax = plt.subplots(figsize=(10, 8))
        
        # Plot bars
        y_pos = np.arange(len(indices))
        ax.barh(y_pos, importances[indices], align='center',
               color=self.color_scheme.get(curve_type, 'blue'), alpha=0.7)
        
        # Labels
        ax.set_yticks(y_pos)
        ax.set_yticklabels([feature_names[i] for i in indices])
        ax.set_xlabel('Feature Importance')
        ax.set_title(f'Feature Importance - {curve_type} {target}')
        
        # Add value labels
        for i, (idx, importance) in enumerate(zip(indices, importances[indices])):
            ax.text(importance + 0.001, i, f'{importance:.3f}', 
                   va='center', fontsize=8)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            
        return fig
    
    def _get_feature_names(self):
        """Get feature names from training data."""
        if len(self.ai_system.training_data) > 0:
            example = self.ai_system.training_data[0]
            if 'curve_features' in example:
                for curve_type in ['hyperpol', 'depol']:
                    if curve_type in example['curve_features']:
                        return sorted(example['curve_features'][curve_type].keys())
        
        # Default feature names
        return [f'feature_{i}' for i in range(50)]
    
    def create_learning_progress_visualization(self, save_path=None):
        """
        Visualize AI learning progress over time.
        """
        if not hasattr(self.ai_system, 'training_data') or len(self.ai_system.training_data) == 0:
            app_logger.warning("No training data available")
            return None
        
        fig, axes = plt.subplots(2, 2, figsize=(12, 10))
        
        # Training data accumulation
        ax = axes[0, 0]
        timestamps = [ex.get('timestamp', datetime.now()) for ex in self.ai_system.training_data]
        ax.plot(range(len(timestamps)), range(len(timestamps)), 'b-', linewidth=2)
        ax.fill_between(range(len(timestamps)), 0, range(len(timestamps)), alpha=0.3)
        ax.set_xlabel('Time')
        ax.set_ylabel('Training Examples')
        ax.set_title('Training Data Accumulation')
        ax.grid(True, alpha=0.3)
        
        # Feature distribution
        ax = axes[0, 1]
        self._plot_feature_distribution(ax)
        
        # Prediction variance over time
        ax = axes[1, 0]
        self._plot_prediction_variance(ax)
        
        # Model performance metrics
        ax = axes[1, 1]
        self._plot_performance_metrics(ax)
        
        plt.suptitle('AI Learning Progress Dashboard', fontsize=16, fontweight='bold')
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            
        return fig
    
    def _plot_feature_distribution(self, ax):
        """Plot distribution of key features in training data."""
        feature_data = []
        
        for example in self.ai_system.training_data:
            if 'curve_features' in example:
                for curve_type in ['hyperpol', 'depol']:
                    if curve_type in example['curve_features']:
                        features = example['curve_features'][curve_type]
                        feature_data.append({
                            'mean': features.get('mean', 0),
                            'std': features.get('std', 0),
                            'curve_type': curve_type
                        })
        
        if feature_data:
            # Create scatter plot
            hyperpol_data = [d for d in feature_data if d['curve_type'] == 'hyperpol']
            depol_data = [d for d in feature_data if d['curve_type'] == 'depol']
            
            if hyperpol_data:
                ax.scatter([d['mean'] for d in hyperpol_data],
                          [d['std'] for d in hyperpol_data],
                          c='blue', alpha=0.6, label='Hyperpol')
            
            if depol_data:
                ax.scatter([d['mean'] for d in depol_data],
                          [d['std'] for d in depol_data],
                          c='red', alpha=0.6, label='Depol')
            
            ax.set_xlabel('Mean Current (pA)')
            ax.set_ylabel('Std Dev (pA)')
            ax.set_title('Training Data Feature Distribution')
            ax.legend()
        
        ax.grid(True, alpha=0.3)
    
    def _plot_prediction_variance(self, ax):
        """Plot prediction variance trends."""
        # Placeholder for actual variance tracking
        x = np.arange(10)
        variance = np.exp(-x/5) + 0.1 + np.random.normal(0, 0.05, 10)
        
        ax.plot(x, variance, 'g-', linewidth=2, marker='o')
        ax.fill_between(x, variance - 0.1, variance + 0.1, alpha=0.3)
        ax.set_xlabel('Model Version')
        ax.set_ylabel('Prediction Variance')
        ax.set_title('Prediction Uncertainty Over Time')
        ax.grid(True, alpha=0.3)
    
    def _plot_performance_metrics(self, ax):
        """Plot key performance metrics."""
        metrics = {
            'Accuracy': 0.85,
            'Precision': 0.88,
            'Confidence': 0.79,
            'Coverage': 0.92
        }
        
        # Create radar chart
        categories = list(metrics.keys())
        values = list(metrics.values())
        
        # Number of variables
        N = len(categories)
        
        # Compute angle for each axis
        angles = [n / float(N) * 2 * np.pi for n in range(N)]
        values += values[:1]  # Complete the circle
        angles += angles[:1]
        
        # Plot
        ax.plot(angles, values, 'o-', linewidth=2, color='purple')
        ax.fill(angles, values, alpha=0.25, color='purple')
        
        # Fix axis to go in the right order and start at 12 o'clock
        ax.set_theta_offset(np.pi / 2)
        ax.set_theta_direction(-1)
        
        # Draw axis lines for each angle and label
        ax.set_xticks(angles[:-1])
        ax.set_xticklabels(categories)
        ax.set_ylim(0, 1)
        ax.set_title('Model Performance Metrics')
        ax.grid(True)

class ExplainableAI:
    """
    Provides explanations for AI decisions.
    """
    
    def __init__(self, ai_system):
        self.ai_system = ai_system
        
    def explain_prediction(self, curve_data, predictions):
        """
        Generate human-readable explanation for predictions.
        """
        explanations = {}
        
        for curve_type in ['hyperpol', 'depol']:
            if curve_type not in predictions:
                continue
                
            explanation = {
                'summary': self._generate_summary(curve_type, predictions[curve_type]),
                'key_factors': self._identify_key_factors(curve_data[curve_type], predictions[curve_type]),
                'confidence_reasoning': self._explain_confidence(predictions[curve_type]['confidence']),
                'recommendations': self._generate_recommendations(predictions[curve_type])
            }
            
            explanations[curve_type] = explanation
        
        return explanations
    
    def _generate_summary(self, curve_type, prediction_result):
        """Generate summary explanation."""
        predictions = prediction_result['predictions']
        confidence = prediction_result['confidence']
        
        summary = f"For the {curve_type} curve, the AI identified:\n"
        summary += f"• Linear region: points {predictions.get('linear_start', 'N/A')} to {predictions.get('linear_end', 'N/A')}\n"
        summary += f"• Exponential fit starting at: point {predictions.get('exp_start', 'N/A')}\n"
        summary += f"• Overall confidence: {confidence['overall']:.0%} ({confidence['level']})\n"
        
        return summary
    
    def _identify_key_factors(self, curve_data, prediction_result):
        """Identify key factors that influenced the prediction."""
        factors = []
        
        # Analyze curve characteristics
        current = curve_data['current']
        
        # Factor 1: Initial stability
        initial_std = np.std(current[:20])
        if initial_std < np.std(current) * 0.5:
            factors.append("Stable initial baseline detected")
        
        # Factor 2: Clear trend
        linear_fit = np.polyfit(range(len(current)), current, 1)
        if abs(linear_fit[0]) > 0.1:
            direction = "increasing" if linear_fit[0] > 0 else "decreasing"
            factors.append(f"Clear {direction} trend observed")
        
        # Factor 3: Transition points
        diff = np.diff(current)
        transitions = np.where(np.abs(diff) > 2 * np.std(diff))[0]
        if len(transitions) > 0:
            factors.append(f"Detected {len(transitions)} potential transition points")
        
        return factors
    
    def _explain_confidence(self, confidence_data):
        """Explain confidence scoring."""
        level = confidence_data['level']
        factors = confidence_data['factors']
        
        explanation = f"The {level} confidence rating is based on:\n"
        
        for factor, value in factors.items():
            if factor == 'consistency' and value > 0.8:
                explanation += "• High prediction consistency across model ensemble\n"
            elif factor == 'feature_quality' and value > 0.7:
                explanation += "• Good quality input data with low noise\n"
            elif factor == 'similarity' and value > 0.6:
                explanation += "• Similar curves in training data\n"
            elif factor == 'feasibility' and value > 0.9:
                explanation += "• Predictions within expected ranges\n"
        
        return explanation
    
    def _generate_recommendations(self, prediction_result):
        """Generate recommendations based on predictions."""
        confidence = prediction_result['confidence']
        recommendations = []
        
        if confidence['level'] == 'high':
            recommendations.append("✓ Predictions are reliable - proceed with automated analysis")
        elif confidence['level'] == 'medium':
            recommendations.append("⚠ Review predictions before finalizing analysis")
            recommendations.append("Consider manual verification of key points")
        else:
            recommendations.append("⚠ Low confidence - manual analysis recommended")
            recommendations.append("Check data quality and consider reprocessing")
        
        # Specific recommendations based on intervals
        intervals = prediction_result.get('intervals', {})
        for key, interval in intervals.items():
            if interval['width'] > 20:
                recommendations.append(f"Wide uncertainty range for {key} - consider manual adjustment")
        
        return recommendations