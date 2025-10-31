#!/usr/bin/env python3
"""
Performance Monitor for AI Excel Learning

This module provides real-time performance monitoring, trend analysis,
and automatic optimization triggers for the learning system.
"""

import pandas as pd
import numpy as np
import logging
from typing import Dict, List, Any, Optional, Tuple, Callable
from dataclasses import dataclass
from datetime import datetime, timedelta
import json
from pathlib import Path
import threading
import time
from collections import deque
import matplotlib.pyplot as plt
import seaborn as sns

logger = logging.getLogger(__name__)

@dataclass
class PerformanceMetric:
    """Individual performance metric"""
    timestamp: datetime
    component: str
    metric_name: str
    value: float
    metadata: Dict[str, Any]

@dataclass
class PerformanceAlert:
    """Performance alert/notification"""
    timestamp: datetime
    alert_type: str  # 'warning', 'critical', 'info'
    component: str
    message: str
    severity: float  # 0.0 - 1.0
    recommendations: List[str]

@dataclass
class OptimizationTrigger:
    """Trigger for automatic optimization"""
    timestamp: datetime
    trigger_type: str  # 'accuracy_drop', 'speed_decrease', 'memory_increase'
    component: str
    current_value: float
    threshold_value: float
    action_required: str

class PerformanceMonitor:
    """
    Real-time performance monitoring and optimization system
    """
    
    def __init__(self, 
                 alert_thresholds: Optional[Dict[str, float]] = None,
                 optimization_triggers: Optional[Dict[str, float]] = None,
                 history_window: int = 1000):
        
        # Default thresholds
        self.alert_thresholds = alert_thresholds or {
            'accuracy_drop': 0.05,      # 5% accuracy decrease
            'speed_decrease': 0.2,      # 20% speed decrease
            'memory_increase': 0.3,     # 30% memory increase
            'error_rate_increase': 0.1, # 10% error rate increase
            'convergence_slowdown': 0.25 # 25% convergence slowdown
        }
        
        # Default optimization triggers
        self.optimization_triggers = optimization_triggers or {
            'accuracy_drop': 0.05,
            'speed_decrease': 0.2,
            'memory_increase': 0.3,
            'stability_decrease': 0.15,
            'resource_inefficiency': 0.4
        }
        
        # Performance history
        self.history_window = history_window
        self.performance_history: Dict[str, deque] = {}
        self.alert_history: List[PerformanceAlert] = []
        self.optimization_history: List[OptimizationTrigger] = []
        
        # Monitoring state
        self.is_monitoring = False
        self.monitoring_thread = None
        self.monitoring_interval = 5.0  # seconds
        
        # Callbacks for external systems
        self.alert_callbacks: List[Callable[[PerformanceAlert], None]] = []
        self.optimization_callbacks: List[Callable[[OptimizationTrigger], None]] = []
        
        # Performance baselines
        self.baselines: Dict[str, Dict[str, float]] = {}
        self.baseline_window = 50  # Number of samples for baseline calculation
        
        # Initialize monitoring for all components
        self._initialize_component_monitoring()
    
    def _initialize_component_monitoring(self):
        """Initialize monitoring for all learning components"""
        components = [
            'formula_learning', 'chart_learning', 'pattern_recognition',
            'data_generation', 'excel_generation', 'overall_system'
        ]
        
        for component in components:
            self.performance_history[component] = deque(maxlen=self.history_window)
            self.baselines[component] = {}
    
    def start_monitoring(self):
        """Start continuous performance monitoring"""
        if self.is_monitoring:
            logger.warning("Performance monitoring is already running")
            return
        
        self.is_monitoring = True
        self.monitoring_thread = threading.Thread(target=self._monitoring_loop, daemon=True)
        self.monitoring_thread.start()
        logger.info("Performance monitoring started")
    
    def stop_monitoring(self):
        """Stop continuous performance monitoring"""
        if not self.is_monitoring:
            return
        
        self.is_monitoring = False
        if self.monitoring_thread:
            self.monitoring_thread.join(timeout=5.0)
        logger.info("Performance monitoring stopped")
    
    def _monitoring_loop(self):
        """Main monitoring loop"""
        while self.is_monitoring:
            try:
                self._check_performance_trends()
                self._detect_anomalies()
                self._update_baselines()
                time.sleep(self.monitoring_interval)
            except Exception as e:
                logger.error(f"Error in monitoring loop: {e}")
                time.sleep(self.monitoring_interval)
    
    def record_metric(self, 
                     component: str, 
                     metric_name: str, 
                     value: float, 
                     metadata: Optional[Dict[str, Any]] = None):
        """
        Record a performance metric
        
        Args:
            component: Learning component name
            metric_name: Name of the metric
            value: Metric value
            metadata: Additional metadata
        """
        if component not in self.performance_history:
            self.performance_history[component] = deque(maxlen=self.history_window)
        
        metric = PerformanceMetric(
            timestamp=datetime.now(),
            component=component,
            metric_name=metric_name,
            value=value,
            metadata=metadata or {}
        )
        
        self.performance_history[component].append(metric)
        
        # Check for immediate alerts
        self._check_immediate_alerts(metric)
        
        logger.debug(f"Recorded metric: {component}.{metric_name} = {value}")
    
    def _check_immediate_alerts(self, metric: PerformanceMetric):
        """Check for immediate alert conditions"""
        if metric.metric_name == 'accuracy' and metric.value < 0.5:
            self._create_alert(
                alert_type='critical',
                component=metric.component,
                message=f"Critical accuracy drop: {metric.value:.3f}",
                severity=0.9,
                recommendations=[
                    "Check training data quality",
                    "Review model parameters",
                    "Consider reducing learning rate"
                ]
            )
        
        elif metric.metric_name == 'error_rate' and metric.value > 0.2:
            self._create_alert(
                alert_type='warning',
                component=metric.component,
                message=f"High error rate: {metric.value:.3f}",
                severity=0.7,
                recommendations=[
                    "Review error patterns",
                    "Check data preprocessing",
                    "Validate model assumptions"
                ]
            )
    
    def _check_performance_trends(self):
        """Check for performance trend changes"""
        for component in self.performance_history:
            if len(self.performance_history[component]) < 10:
                continue
            
            # Check accuracy trends
            accuracy_metrics = [
                m for m in self.performance_history[component] 
                if m.metric_name == 'accuracy'
            ]
            
            if len(accuracy_metrics) >= 5:
                recent_accuracy = [m.value for m in accuracy_metrics[-5:]]
                earlier_accuracy = [m.value for m in accuracy_metrics[-10:-5]]
                
                if earlier_accuracy and recent_accuracy:
                    recent_avg = np.mean(recent_accuracy)
                    earlier_avg = np.mean(earlier_accuracy)
                    accuracy_change = recent_avg - earlier_avg
                    
                    if abs(accuracy_change) > self.optimization_triggers['accuracy_drop']:
                        self._create_optimization_trigger(
                            trigger_type='accuracy_drop',
                            component=component,
                            current_value=recent_avg,
                            threshold_value=earlier_avg,
                            action_required='Review learning parameters and data quality'
                        )
            
            # Check speed trends
            speed_metrics = [
                m for m in self.performance_history[component] 
                if m.metric_name == 'training_time'
            ]
            
            if len(speed_metrics) >= 5:
                recent_speed = [m.value for m in speed_metrics[-5:]]
                earlier_speed = [m.value for m in speed_metrics[-10:-5]]
                
                if earlier_speed and recent_speed:
                    recent_avg = np.mean(recent_speed)
                    earlier_avg = np.mean(earlier_speed)
                    speed_change = (recent_avg - earlier_avg) / earlier_avg
                    
                    if speed_change > self.optimization_triggers['speed_decrease']:
                        self._create_optimization_trigger(
                            trigger_type='speed_decrease',
                            component=component,
                            current_value=recent_avg,
                            threshold_value=earlier_avg,
                            action_required='Optimize algorithm efficiency and resource usage'
                        )
    
    def _detect_anomalies(self):
        """Detect performance anomalies using statistical methods"""
        for component in self.performance_history:
            if len(self.performance_history[component]) < 20:
                continue
            
            # Group metrics by type
            metric_groups = {}
            for metric in self.performance_history[component]:
                if metric.metric_name not in metric_groups:
                    metric_groups[metric.metric_name] = []
                metric_groups[metric.metric_name].append(metric.value)
            
            # Check for anomalies in each metric type
            for metric_name, values in metric_groups.items():
                if len(values) < 10:
                    continue
                
                values_array = np.array(values)
                mean_val = np.mean(values_array)
                std_val = np.std(values_array)
                
                if std_val > 0:
                    # Z-score based anomaly detection
                    z_scores = np.abs((values_array - mean_val) / std_val)
                    anomalies = z_scores > 3.0  # 3-sigma rule
                    
                    if np.any(anomalies):
                        anomaly_indices = np.where(anomalies)[0]
                        for idx in anomaly_indices:
                            self._create_alert(
                                alert_type='warning',
                                component=component,
                                message=f"Anomaly detected in {metric_name}: {values_array[idx]:.3f} "
                                       f"(mean: {mean_val:.3f}, std: {std_val:.3f})",
                                severity=0.6,
                                recommendations=[
                                    "Investigate data quality",
                                    "Check for system changes",
                                    "Review recent modifications"
                                ]
                            )
    
    def _update_baselines(self):
        """Update performance baselines"""
        for component in self.performance_history:
            if len(self.performance_history[component]) < self.baseline_window:
                continue
            
            # Calculate baselines for recent performance
            recent_metrics = list(self.performance_history[component])[-self.baseline_window:]
            
            # Group by metric type
            metric_groups = {}
            for metric in recent_metrics:
                if metric.metric_name not in metric_groups:
                    metric_groups[metric.metric_name] = []
                metric_groups[metric.metric_name].append(metric.value)
            
            # Update baselines
            for metric_name, values in metric_groups.items():
                if len(values) >= 10:
                    self.baselines[component][metric_name] = {
                        'mean': np.mean(values),
                        'std': np.std(values),
                        'min': np.min(values),
                        'max': np.max(values),
                        'median': np.median(values),
                        'last_updated': datetime.now()
                    }
    
    def _create_alert(self, 
                     alert_type: str, 
                     component: str, 
                     message: str, 
                     severity: float, 
                     recommendations: List[str]):
        """Create and dispatch a performance alert"""
        alert = PerformanceAlert(
            timestamp=datetime.now(),
            alert_type=alert_type,
            component=component,
            message=message,
            severity=severity,
            recommendations=recommendations
        )
        
        self.alert_history.append(alert)
        
        # Dispatch to callbacks
        for callback in self.alert_callbacks:
            try:
                callback(alert)
            except Exception as e:
                logger.error(f"Error in alert callback: {e}")
        
        logger.warning(f"Performance alert: {component} - {message}")
    
    def _create_optimization_trigger(self, 
                                   trigger_type: str, 
                                   component: str, 
                                   current_value: float, 
                                   threshold_value: float, 
                                   action_required: str):
        """Create and dispatch an optimization trigger"""
        trigger = OptimizationTrigger(
            timestamp=datetime.now(),
            trigger_type=trigger_type,
            component=component,
            current_value=current_value,
            threshold_value=threshold_value,
            action_required=action_required
        )
        
        self.optimization_history.append(trigger)
        
        # Dispatch to callbacks
        for callback in self.optimization_callbacks:
            try:
                callback(trigger)
            except Exception as e:
                logger.error(f"Error in optimization callback: {e}")
        
        logger.info(f"Optimization trigger: {component} - {trigger_type}")
    
    def get_performance_summary(self, 
                              component: Optional[str] = None, 
                              time_window: Optional[timedelta] = None) -> Dict[str, Any]:
        """Get performance summary for specified component and time window"""
        if component and component not in self.performance_history:
            return {'error': f'Component {component} not found'}
        
        components_to_check = [component] if component else list(self.performance_history.keys())
        summary = {}
        
        for comp in components_to_check:
            if comp not in self.performance_history:
                continue
            
            metrics = list(self.performance_history[comp])
            
            # Filter by time window if specified
            if time_window:
                cutoff_time = datetime.now() - time_window
                metrics = [m for m in metrics if m.timestamp >= cutoff_time]
            
            if not metrics:
                summary[comp] = {'error': 'No metrics in time window'}
                continue
            
            # Group metrics by type
            metric_groups = {}
            for metric in metrics:
                if metric.metric_name not in metric_groups:
                    metric_groups[metric.metric_name] = []
                metric_groups[metric.metric_name].append(metric.value)
            
            # Calculate statistics for each metric type
            comp_summary = {}
            for metric_name, values in metric_groups.items():
                if len(values) >= 2:
                    comp_summary[metric_name] = {
                        'count': len(values),
                        'mean': np.mean(values),
                        'std': np.std(values),
                        'min': np.min(values),
                        'max': np.max(values),
                        'trend': self._calculate_trend(values),
                        'last_value': values[-1],
                        'last_timestamp': metrics[-1].timestamp.isoformat()
                    }
                else:
                    comp_summary[metric_name] = {
                        'count': len(values),
                        'value': values[0] if values else None,
                        'timestamp': metrics[-1].timestamp.isoformat() if metrics else None
                    }
            
            summary[comp] = comp_summary
        
        return summary
    
    def _calculate_trend(self, values: List[float]) -> str:
        """Calculate trend direction for a list of values"""
        if len(values) < 3:
            return 'insufficient_data'
        
        # Simple linear trend calculation
        x = np.arange(len(values))
        y = np.array(values)
        
        try:
            slope = np.polyfit(x, y, 1)[0]
            if slope > 0.01:
                return 'increasing'
            elif slope < -0.01:
                return 'decreasing'
            else:
                return 'stable'
        except:
            return 'unknown'
    
    def get_alerts_summary(self, 
                          alert_type: Optional[str] = None, 
                          severity_threshold: float = 0.0) -> List[Dict[str, Any]]:
        """Get summary of performance alerts"""
        alerts = self.alert_history
        
        # Filter by type if specified
        if alert_type:
            alerts = [a for a in alerts if a.alert_type == alert_type]
        
        # Filter by severity if specified
        if severity_threshold > 0:
            alerts = [a for a in alerts if a.severity >= severity_threshold]
        
        # Convert to dict format for JSON serialization
        alert_summaries = []
        for alert in alerts[-100:]:  # Last 100 alerts
            alert_summaries.append({
                'timestamp': alert.timestamp.isoformat(),
                'alert_type': alert.alert_type,
                'component': alert.component,
                'message': alert.message,
                'severity': alert.severity,
                'recommendations': alert.recommendations
            })
        
        return alert_summaries
    
    def get_optimization_summary(self) -> Dict[str, Any]:
        """Get summary of optimization triggers"""
        summary = {
            'total_triggers': len(self.optimization_history),
            'triggers_by_type': {},
            'recent_triggers': []
        }
        
        # Count triggers by type
        for trigger in self.optimization_history:
            trigger_type = trigger.trigger_type
            if trigger_type not in summary['triggers_by_type']:
                summary['triggers_by_type'][trigger_type] = 0
            summary['triggers_by_type'][trigger_type] += 1
        
        # Recent triggers
        for trigger in self.optimization_history[-20:]:  # Last 20 triggers
            summary['recent_triggers'].append({
                'timestamp': trigger.timestamp.isoformat(),
                'trigger_type': trigger.trigger_type,
                'component': trigger.component,
                'current_value': trigger.current_value,
                'threshold_value': trigger.threshold_value,
                'action_required': trigger.action_required
            })
        
        return summary
    
    def add_alert_callback(self, callback: Callable[[PerformanceAlert], None]):
        """Add callback for performance alerts"""
        self.alert_callbacks.append(callback)
    
    def add_optimization_callback(self, callback: Callable[[OptimizationTrigger], None]):
        """Add callback for optimization triggers"""
        self.optimization_callbacks.append(callback)
    
    def export_performance_report(self, output_path: str, 
                                include_plots: bool = True):
        """Export comprehensive performance report"""
        report_data = {
            'export_timestamp': datetime.now().isoformat(),
            'performance_summary': self.get_performance_summary(),
            'alerts_summary': self.get_alerts_summary(),
            'optimization_summary': self.get_optimization_summary(),
            'baselines': self.baselines
        }
        
        # Export to JSON
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(report_data, f, indent=2, ensure_ascii=False, default=str)
        
        logger.info(f"Performance report exported to {output_path}")
        
        # Generate plots if requested
        if include_plots:
            self._generate_performance_plots(output_path.replace('.json', '_plots.png'))
    
    def _generate_performance_plots(self, output_path: str):
        """Generate performance visualization plots"""
        try:
            fig, axes = plt.subplots(2, 2, figsize=(15, 10))
            fig.suptitle('AI Excel Learning Performance Overview', fontsize=16)
            
            # Plot 1: Accuracy trends
            ax1 = axes[0, 0]
            self._plot_accuracy_trends(ax1)
            
            # Plot 2: Training time trends
            ax2 = axes[0, 1]
            self._plot_training_time_trends(ax2)
            
            # Plot 3: Alert distribution
            ax3 = axes[1, 0]
            self._plot_alert_distribution(ax3)
            
            # Plot 4: Component performance comparison
            ax4 = axes[1, 1]
            self._plot_component_comparison(ax4)
            
            plt.tight_layout()
            plt.savefig(output_path, dpi=300, bbox_inches='tight')
            plt.close()
            
            logger.info(f"Performance plots generated: {output_path}")
            
        except Exception as e:
            logger.error(f"Error generating performance plots: {e}")
    
    def _plot_accuracy_trends(self, ax):
        """Plot accuracy trends for all components"""
        for component in self.performance_history:
            accuracy_metrics = [
                m for m in self.performance_history[component] 
                if m.metric_name == 'accuracy'
            ]
            
            if accuracy_metrics:
                timestamps = [m.timestamp for m in accuracy_metrics]
                values = [m.value for m in accuracy_metrics]
                ax.plot(timestamps, values, label=component, marker='o', markersize=3)
        
        ax.set_title('Accuracy Trends')
        ax.set_xlabel('Time')
        ax.set_ylabel('Accuracy')
        ax.legend()
        ax.grid(True, alpha=0.3)
    
    def _plot_training_time_trends(self, ax):
        """Plot training time trends for all components"""
        for component in self.performance_history:
            time_metrics = [
                m for m in self.performance_history[component] 
                if m.metric_name == 'training_time'
            ]
            
            if time_metrics:
                timestamps = [m.timestamp for m in time_metrics]
                values = [m.value for m in time_metrics]
                ax.plot(timestamps, values, label=component, marker='s', markersize=3)
        
        ax.set_title('Training Time Trends')
        ax.set_xlabel('Time')
        ax.set_ylabel('Training Time (seconds)')
        ax.legend()
        ax.grid(True, alpha=0.3)
    
    def _plot_alert_distribution(self, ax):
        """Plot distribution of performance alerts"""
        if not self.alert_history:
            ax.text(0.5, 0.5, 'No alerts recorded', ha='center', va='center')
            return
        
        alert_types = [alert.alert_type for alert in self.alert_history]
        alert_counts = pd.Series(alert_types).value_counts()
        
        ax.pie(alert_counts.values, labels=alert_counts.index, autopct='%1.1f%%')
        ax.set_title('Alert Distribution by Type')
    
    def _plot_component_comparison(self, ax):
        """Plot performance comparison across components"""
        components = list(self.performance_history.keys())
        avg_accuracies = []
        
        for component in components:
            accuracy_metrics = [
                m for m in self.performance_history[component] 
                if m.metric_name == 'accuracy'
            ]
            
            if accuracy_metrics:
                avg_accuracies.append(np.mean([m.value for m in accuracy_metrics]))
            else:
                avg_accuracies.append(0.0)
        
        bars = ax.bar(components, avg_accuracies)
        ax.set_title('Average Accuracy by Component')
        ax.set_xlabel('Component')
        ax.set_ylabel('Average Accuracy')
        ax.set_ylim(0, 1.0)
        
        # Add value labels on bars
        for bar, value in zip(bars, avg_accuracies):
            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                   f'{value:.3f}', ha='center', va='bottom')
