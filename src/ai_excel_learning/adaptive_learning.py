#!/usr/bin/env python3
"""
Adaptive Learning System for AI Excel Learning

This module provides adaptive learning rate adjustment and performance
optimization based on learning outcomes and data characteristics.
"""

import pandas as pd
import numpy as np
import logging
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass
from datetime import datetime, timedelta
import json
from pathlib import Path

logger = logging.getLogger(__name__)

@dataclass
class LearningRateConfig:
    """Configuration for learning rate adaptation"""
    component: str
    base_rate: float
    min_rate: float
    max_rate: float
    adaptation_factor: float
    patience: int  # Number of attempts before rate adjustment
    decay_factor: float
    boost_factor: float

@dataclass
class PerformanceMetrics:
    """Performance metrics for learning components"""
    accuracy: float
    loss: float
    training_time: float
    convergence_rate: float
    stability_score: float
    timestamp: datetime

@dataclass
class AdaptiveLearningState:
    """Current state of adaptive learning"""
    component: str
    current_rate: float
    performance_history: List[PerformanceMetrics]
    adaptation_count: int
    last_improvement: datetime
    best_performance: float
    learning_phase: str  # 'exploration', 'exploitation', 'refinement'

class AdaptiveLearningRate:
    """
    Adaptive learning rate system for dynamic optimization
    """
    
    def __init__(self):
        self.components = {
            'formula_learning': LearningRateConfig(
                component='formula_learning',
                base_rate=0.01,
                min_rate=0.001,
                max_rate=0.05,
                adaptation_factor=0.1,
                patience=3,
                decay_factor=0.9,
                boost_factor=1.1
            ),
            'chart_learning': LearningRateConfig(
                component='chart_learning',
                base_rate=0.005,
                min_rate=0.0005,
                max_rate=0.02,
                adaptation_factor=0.08,
                patience=5,
                decay_factor=0.92,
                boost_factor=1.08
            ),
            'pattern_recognition': LearningRateConfig(
                component='pattern_recognition',
                base_rate=0.02,
                min_rate=0.002,
                max_rate=0.08,
                adaptation_factor=0.12,
                patience=2,
                decay_factor=0.85,
                boost_factor=1.15
            ),
            'data_generation': LearningRateConfig(
                component='data_generation',
                base_rate=0.015,
                min_rate=0.0015,
                max_rate=0.06,
                adaptation_factor=0.1,
                patience=4,
                decay_factor=0.9,
                boost_factor=1.1
            )
        }
        
        self.learning_states: Dict[str, AdaptiveLearningState] = {}
        self.performance_history: Dict[str, List[PerformanceMetrics]] = {}
        self.adaptation_log: List[Dict[str, Any]] = []
        
        # Initialize learning states
        for component_name, config in self.components.items():
            self.learning_states[component_name] = AdaptiveLearningState(
                component=component_name,
                current_rate=config.base_rate,
                performance_history=[],
                adaptation_count=0,
                last_improvement=datetime.now(),
                best_performance=0.0,
                learning_phase='exploration'
            )
            self.performance_history[component_name] = []
    
    def get_learning_rate(self, component: str) -> float:
        """Get current learning rate for a component"""
        if component not in self.learning_states:
            logger.warning(f"Unknown component: {component}, using default rate")
            return 0.01
        
        return self.learning_states[component].current_rate
    
    def update_performance(self, 
                          component: str, 
                          metrics: PerformanceMetrics) -> Dict[str, Any]:
        """
        Update performance metrics and adjust learning rate if needed
        
        Args:
            component: Learning component name
            metrics: Performance metrics
            
        Returns:
            Dict with adaptation information
        """
        if component not in self.learning_states:
            logger.warning(f"Unknown component: {component}")
            return {}
        
        state = self.learning_states[component]
        config = self.components[component]
        
        # Add to performance history
        state.performance_history.append(metrics)
        self.performance_history[component].append(metrics)
        
        # Check if performance improved
        performance_improved = metrics.accuracy > state.best_performance
        
        if performance_improved:
            state.best_performance = metrics.accuracy
            state.last_improvement = datetime.now()
            state.adaptation_count = 0
            logger.info(f"{component}: Performance improved to {metrics.accuracy:.3f}")
        else:
            state.adaptation_count += 1
            logger.info(f"{component}: Performance did not improve, attempt {state.adaptation_count}")
        
        # Determine if learning rate should be adjusted
        adaptation_info = self._should_adjust_learning_rate(component, metrics)
        
        if adaptation_info['should_adjust']:
            self._adjust_learning_rate(component, adaptation_info['adjustment_type'])
        
        # Update learning phase
        self._update_learning_phase(component)
        
        return adaptation_info
    
    def _should_adjust_learning_rate(self, 
                                   component: str, 
                                   metrics: PerformanceMetrics) -> Dict[str, Any]:
        """Determine if learning rate should be adjusted"""
        state = self.learning_states[component]
        config = self.components[component]
        
        # Check if we've been patient enough
        if state.adaptation_count < config.patience:
            return {
                'should_adjust': False,
                'reason': f'Patience threshold not met ({state.adaptation_count}/{config.patience})',
                'adjustment_type': None
            }
        
        # Check for different adjustment scenarios
        recent_performance = self._get_recent_performance(component, window=5)
        
        if not recent_performance:
            return {
                'should_adjust': False,
                'reason': 'Insufficient performance history',
                'adjustment_type': None
            }
        
        # Calculate performance trends
        accuracy_trend = self._calculate_trend([p.accuracy for p in recent_performance])
        loss_trend = self._calculate_trend([p.loss for p in recent_performance])
        stability = self._calculate_stability(recent_performance)
        
        # Determine adjustment type
        if accuracy_trend < -0.05 and loss_trend > 0.05:
            # Performance declining - reduce learning rate
            return {
                'should_adjust': True,
                'reason': 'Performance declining, reducing learning rate',
                'adjustment_type': 'decrease'
            }
        elif accuracy_trend > 0.05 and loss_trend < -0.05 and stability > 0.8:
            # Performance improving and stable - increase learning rate
            return {
                'should_adjust': True,
                'reason': 'Performance improving and stable, increasing learning rate',
                'adjustment_type': 'increase'
            }
        elif stability < 0.5:
            # Unstable performance - reduce learning rate
            return {
                'should_adjust': True,
                'reason': 'Unstable performance, reducing learning rate',
                'adjustment_type': 'decrease'
            }
        
        return {
            'should_adjust': False,
            'reason': 'Performance stable, no adjustment needed',
            'adjustment_type': None
        }
    
    def _adjust_learning_rate(self, component: str, adjustment_type: str):
        """Adjust learning rate for a component"""
        state = self.learning_states[component]
        config = self.components[component]
        
        old_rate = state.current_rate
        
        if adjustment_type == 'decrease':
            # Decrease learning rate
            new_rate = state.current_rate * config.decay_factor
            new_rate = max(new_rate, config.min_rate)
            state.current_rate = new_rate
            logger.info(f"{component}: Learning rate decreased from {old_rate:.6f} to {new_rate:.6f}")
            
        elif adjustment_type == 'increase':
            # Increase learning rate
            new_rate = state.current_rate * config.boost_factor
            new_rate = min(new_rate, config.max_rate)
            state.current_rate = new_rate
            logger.info(f"{component}: Learning rate increased from {old_rate:.6f} to {new_rate:.6f}")
        
        # Log adaptation
        self.adaptation_log.append({
            'timestamp': datetime.now().isoformat(),
            'component': component,
            'adjustment_type': adjustment_type,
            'old_rate': old_rate,
            'new_rate': state.current_rate,
            'reason': 'Performance-based adjustment'
        })
        
        # Reset adaptation counter
        state.adaptation_count = 0
    
    def _update_learning_phase(self, component: str):
        """Update learning phase based on performance history"""
        state = self.learning_states[component]
        
        if len(state.performance_history) < 10:
            return
        
        recent_performance = state.performance_history[-10:]
        accuracy_values = [p.accuracy for p in recent_performance]
        
        # Calculate performance statistics
        avg_accuracy = np.mean(accuracy_values)
        accuracy_std = np.std(accuracy_values)
        accuracy_trend = self._calculate_trend(accuracy_values)
        
        # Determine learning phase
        if avg_accuracy < 0.6:
            new_phase = 'exploration'
        elif avg_accuracy > 0.8 and accuracy_std < 0.1:
            new_phase = 'exploitation'
        elif avg_accuracy > 0.7 and accuracy_trend > 0.02:
            new_phase = 'refinement'
        else:
            new_phase = state.learning_phase
        
        if new_phase != state.learning_phase:
            logger.info(f"{component}: Learning phase changed from {state.learning_phase} to {new_phase}")
            state.learning_phase = new_phase
    
    def _get_recent_performance(self, component: str, window: int = 5) -> List[PerformanceMetrics]:
        """Get recent performance metrics"""
        history = self.performance_history[component]
        return history[-window:] if len(history) >= window else history
    
    def _calculate_trend(self, values: List[float]) -> float:
        """Calculate trend (slope) of values"""
        if len(values) < 2:
            return 0.0
        
        x = np.arange(len(values))
        y = np.array(values)
        
        # Linear regression slope
        slope = np.polyfit(x, y, 1)[0]
        return slope
    
    def _calculate_stability(self, performance: List[PerformanceMetrics]) -> float:
        """Calculate stability score based on performance variance"""
        if len(performance) < 2:
            return 1.0
        
        accuracy_values = [p.accuracy for p in performance]
        variance = np.var(accuracy_values)
        
        # Convert variance to stability score (0-1, higher is more stable)
        # Normalize variance to reasonable range
        normalized_variance = min(variance / 0.1, 1.0)  # 0.1 variance = 0 stability
        stability = 1.0 - normalized_variance
        
        return max(0.0, min(1.0, stability))
    
    def get_learning_recommendations(self, component: str) -> List[str]:
        """Get recommendations for improving learning performance"""
        if component not in self.learning_states:
            return ["Component not found"]
        
        state = self.learning_states[component]
        config = self.components[component]
        
        recommendations = []
        
        # Check learning rate
        if state.current_rate < config.min_rate * 1.1:
            recommendations.append(
                f"Learning rate is very low ({state.current_rate:.6f}). "
                f"Consider resetting to base rate ({config.base_rate:.6f})."
            )
        elif state.current_rate > config.max_rate * 0.9:
            recommendations.append(
                f"Learning rate is very high ({state.current_rate:.6f}). "
                f"Consider reducing to avoid instability."
            )
        
        # Check performance trends
        if len(state.performance_history) >= 5:
            recent_performance = state.performance_history[-5:]
            accuracy_trend = self._calculate_trend([p.accuracy for p in recent_performance])
            
            if accuracy_trend < -0.05:
                recommendations.append(
                    "Performance is declining. Consider reducing learning rate "
                    "or reviewing training data quality."
                )
            elif accuracy_trend > 0.05:
                recommendations.append(
                    "Performance is improving. Current learning rate seems appropriate."
                )
        
        # Check learning phase
        if state.learning_phase == 'exploration':
            recommendations.append(
                "Currently in exploration phase. Focus on data diversity "
                "and avoid overfitting."
            )
        elif state.learning_phase == 'exploitation':
            recommendations.append(
                "Currently in exploitation phase. Fine-tune parameters "
                "and optimize for specific patterns."
            )
        elif state.learning_phase == 'refinement':
            recommendations.append(
                "Currently in refinement phase. Focus on edge cases "
                "and improving robustness."
            )
        
        if not recommendations:
            recommendations.append("Learning performance is optimal. No changes needed.")
        
        return recommendations
    
    def reset_learning_rate(self, component: str):
        """Reset learning rate to base value"""
        if component not in self.components:
            logger.warning(f"Unknown component: {component}")
            return
        
        config = self.components[component]
        state = self.learning_states[component]
        
        old_rate = state.current_rate
        state.current_rate = config.base_rate
        state.adaptation_count = 0
        
        logger.info(f"{component}: Learning rate reset from {old_rate:.6f} to {config.base_rate:.6f}")
        
        # Log reset
        self.adaptation_log.append({
            'timestamp': datetime.now().isoformat(),
            'component': component,
            'adjustment_type': 'reset',
            'old_rate': old_rate,
            'new_rate': config.base_rate,
            'reason': 'Manual reset to base rate'
        })
    
    def get_adaptation_summary(self) -> Dict[str, Any]:
        """Get summary of all learning rate adaptations"""
        summary = {
            'total_adaptations': len(self.adaptation_log),
            'components': {},
            'recent_adaptations': self.adaptation_log[-10:] if self.adaptation_log else []
        }
        
        for component_name, state in self.learning_states.items():
            summary['components'][component_name] = {
                'current_rate': state.current_rate,
                'base_rate': self.components[component_name].base_rate,
                'learning_phase': state.learning_phase,
                'best_performance': state.best_performance,
                'adaptation_count': state.adaptation_count,
                'last_improvement': state.last_improvement.isoformat() if state.last_improvement else None
            }
        
        return summary
    
    def export_adaptation_log(self, output_path: str):
        """Export adaptation log to file"""
        log_data = {
            'adaptation_log': self.adaptation_log,
            'learning_states': {
                name: {
                    'current_rate': state.current_rate,
                    'learning_phase': state.learning_phase,
                    'best_performance': state.best_performance,
                    'adaptation_count': state.adaptation_count
                }
                for name, state in self.learning_states.items()
            },
            'export_timestamp': datetime.now().isoformat()
        }
        
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(log_data, f, indent=2, ensure_ascii=False, default=str)
        
        logger.info(f"Adaptation log exported to {output_path}")
    
    def optimize_for_component(self, component: str, target_accuracy: float = 0.9) -> Dict[str, Any]:
        """
        Optimize learning parameters for a specific component
        
        Args:
            component: Component to optimize
            target_accuracy: Target accuracy to achieve
            
        Returns:
            Optimization results and recommendations
        """
        if component not in self.learning_states:
            return {'error': f'Component {component} not found'}
        
        state = self.learning_states[component]
        config = self.components[component]
        
        # Analyze current performance
        if len(state.performance_history) < 5:
            return {'error': 'Insufficient performance history for optimization'}
        
        recent_performance = state.performance_history[-10:]
        current_accuracy = np.mean([p.accuracy for p in recent_performance])
        
        optimization_result = {
            'component': component,
            'current_accuracy': current_accuracy,
            'target_accuracy': target_accuracy,
            'current_learning_rate': state.current_rate,
            'recommendations': [],
            'parameter_suggestions': {}
        }
        
        # Generate optimization recommendations
        if current_accuracy < target_accuracy - 0.1:
            # Significant improvement needed
            if state.current_rate > config.base_rate * 1.5:
                optimization_result['recommendations'].append(
                    "Learning rate is high. Consider reducing to improve stability."
                )
                optimization_result['parameter_suggestions']['learning_rate'] = config.base_rate
            else:
                optimization_result['recommendations'].append(
                    "Performance below target. Consider increasing learning rate gradually."
                )
                optimization_result['parameter_suggestions']['learning_rate'] = min(
                    state.current_rate * 1.2, config.max_rate
                )
        
        elif current_accuracy > target_accuracy + 0.05:
            # Performance exceeds target
            optimization_result['recommendations'].append(
                "Performance exceeds target. Consider reducing learning rate for fine-tuning."
            )
            optimization_result['parameter_suggestions']['learning_rate'] = max(
                state.current_rate * 0.8, config.min_rate
            )
        
        else:
            # Performance close to target
            optimization_result['recommendations'].append(
                "Performance close to target. Current parameters are appropriate."
            )
        
        # Add phase-specific recommendations
        if state.learning_phase == 'exploration':
            optimization_result['recommendations'].append(
                "In exploration phase. Focus on data diversity and avoid premature optimization."
            )
        elif state.learning_phase == 'exploitation':
            optimization_result['recommendations'].append(
                "In exploitation phase. Fine-tune parameters for maximum performance."
            )
        
        return optimization_result
