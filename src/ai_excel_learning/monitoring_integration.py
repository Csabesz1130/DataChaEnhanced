"""
AI Monitoring Integration

This module provides seamless integration with existing AI components
without requiring changes to their core functionality. It uses decorators
and monkey patching to add monitoring capabilities.
"""

import time
import functools
import threading
from typing import Dict, Any, Callable, Optional
from datetime import datetime
import psutil
import os

from .ai_monitor import (
    get_ai_monitor, MetricType, record_metric, 
    monitor_ai_operation, ComponentType
)


class MonitoringIntegration:
    """Handles integration of monitoring with existing AI components"""
    
    def __init__(self):
        self.monitor = get_ai_monitor()
        self.integrated_components = set()
        self.original_methods = {}
    
    def integrate_component(self, component_name: str, component_instance: Any):
        """Integrate monitoring with a component instance"""
        if component_name in self.integrated_components:
            return  # Already integrated
        
        # Get all methods that should be monitored
        methods_to_monitor = self._get_monitorable_methods(component_instance)
        
        for method_name, method in methods_to_monitor.items():
            self._wrap_method(component_instance, method_name, method, component_name)
        
        self.integrated_components.add(component_name)
        print(f"✅ Monitoring integrated with {component_name}")
    
    def _get_monitorable_methods(self, component_instance: Any) -> Dict[str, Callable]:
        """Get methods that should be monitored"""
        monitorable_methods = {}
        
        # Common AI method patterns
        method_patterns = [
            'analyze', 'learn', 'train', 'predict', 'generate', 
            'process', 'extract', 'detect', 'classify', 'optimize',
            'fit', 'transform', 'evaluate', 'validate'
        ]
        
        for attr_name in dir(component_instance):
            if not attr_name.startswith('_'):  # Skip private methods
                attr = getattr(component_instance, attr_name)
                if callable(attr) and not isinstance(attr, type):
                    # Check if method name matches patterns
                    if any(pattern in attr_name.lower() for pattern in method_patterns):
                        monitorable_methods[attr_name] = attr
        
        return monitorable_methods
    
    def _wrap_method(self, component_instance: Any, method_name: str, 
                    original_method: Callable, component_name: str):
        """Wrap a method with monitoring"""
        @functools.wraps(original_method)
        def monitored_method(*args, **kwargs):
            start_time = time.time()
            session_id = f"{component_name}_{method_name}_{int(start_time)}"
            
            # Record resource usage before
            process = psutil.Process(os.getpid())
            memory_before = process.memory_info().rss / 1024 / 1024  # MB
            cpu_before = process.cpu_percent()
            
            try:
                # Execute the original method
                result = original_method(*args, **kwargs)
                
                # Calculate metrics
                execution_time = time.time() - start_time
                memory_after = process.memory_info().rss / 1024 / 1024
                cpu_after = process.cpu_percent()
                
                # Record success metrics
                self.monitor.record_metric(
                    component=component_name,
                    metric_type=MetricType.RESPONSE_TIME,
                    value=execution_time,
                    metadata={
                        "method": method_name,
                        "status": "success",
                        "result_type": type(result).__name__,
                        "args_count": len(args),
                        "kwargs_count": len(kwargs)
                    },
                    session_id=session_id
                )
                
                # Record resource usage
                self.monitor.record_metric(
                    component=component_name,
                    metric_type=MetricType.MEMORY_USAGE,
                    value=memory_after - memory_before,
                    metadata={"method": method_name, "memory_peak": memory_after},
                    session_id=session_id
                )
                
                self.monitor.record_metric(
                    component=component_name,
                    metric_type=MetricType.CPU_USAGE,
                    value=(cpu_before + cpu_after) / 2,
                    metadata={"method": method_name},
                    session_id=session_id
                )
                
                # Record success rate
                self.monitor.record_metric(
                    component=component_name,
                    metric_type=MetricType.SUCCESS_RATE,
                    value=1.0,  # Success
                    metadata={"method": method_name},
                    session_id=session_id
                )
                
                return result
                
            except Exception as e:
                # Calculate metrics for failed execution
                execution_time = time.time() - start_time
                memory_after = process.memory_info().rss / 1024 / 1024
                
                # Record error metrics
                self.monitor.record_metric(
                    component=component_name,
                    metric_type=MetricType.ERROR_RATE,
                    value=1.0,  # Error occurred
                    metadata={
                        "method": method_name,
                        "error_type": type(e).__name__,
                        "error_message": str(e)
                    },
                    session_id=session_id
                )
                
                # Record response time for failed operations
                self.monitor.record_metric(
                    component=component_name,
                    metric_type=MetricType.RESPONSE_TIME,
                    value=execution_time,
                    metadata={"method": method_name, "status": "error"},
                    session_id=session_id
                )
                
                # Record success rate (failure)
                self.monitor.record_metric(
                    component=component_name,
                    metric_type=MetricType.SUCCESS_RATE,
                    value=0.0,  # Failure
                    metadata={"method": method_name},
                    session_id=session_id
                )
                
                raise
        
        # Replace the original method
        setattr(component_instance, method_name, monitored_method)
        
        # Store original method for potential restoration
        self.original_methods[f"{component_name}.{method_name}"] = original_method
    
    def restore_original_methods(self, component_name: str):
        """Restore original methods (for testing/debugging)"""
        for key, original_method in self.original_methods.items():
            if key.startswith(f"{component_name}."):
                method_name = key.split('.')[1]
                # This would need the component instance to restore
                # For now, just remove from tracking
                del self.original_methods[key]
        
        self.integrated_components.discard(component_name)


# Global integration instance
_integration = None

def get_monitoring_integration() -> MonitoringIntegration:
    """Get the global monitoring integration instance"""
    global _integration
    if _integration is None:
        _integration = MonitoringIntegration()
    return _integration


def auto_integrate_monitoring():
    """Automatically integrate monitoring with all AI components"""
    integration = get_monitoring_integration()
    
    try:
        # Import existing components
        from .excel_analyzer import ExcelAnalyzer
        from .chart_learner import ChartLearner
        from .formula_learner import FormulaLearner
        from .ml_models import ExcelMLModels
        from .learning_pipeline import LearningPipeline
        from .background_processor import BackgroundProcessor
        from .research_extensions import ResearchExtensions
        
        # Integrate with each component
        components_to_integrate = [
            ("excel_analyzer", ExcelAnalyzer),
            ("chart_learner", ChartLearner),
            ("formula_learner", FormulaLearner),
            ("ml_models", ExcelMLModels),
            ("learning_pipeline", LearningPipeline),
            ("background_processor", BackgroundProcessor),
            ("research_extensions", ResearchExtensions)
        ]
        
        for component_name, component_class in components_to_integrate:
            try:
                # Create a temporary instance to integrate with
                temp_instance = component_class()
                integration.integrate_component(component_name, temp_instance)
                
                # Apply the same monitoring to the class
                integration._apply_monitoring_to_class(component_class, component_name)
                
            except Exception as e:
                print(f"⚠️ Could not integrate monitoring with {component_name}: {e}")
        
        print("✅ Auto-integration completed")
        
    except ImportError as e:
        print(f"⚠️ Some components not available for auto-integration: {e}")


def monitor_function(component: str, metric_type: MetricType = MetricType.RESPONSE_TIME):
    """Decorator to monitor individual functions"""
    def decorator(func):
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            return monitor_ai_operation(component, metric_type, func)(*args, **kwargs)
        return wrapper
    return decorator


def monitor_class_methods(component: str):
    """Decorator to monitor all methods in a class"""
    def decorator(cls):
        for attr_name in dir(cls):
            if not attr_name.startswith('_'):
                attr = getattr(cls, attr_name)
                if callable(attr) and not isinstance(attr, type):
                    # Wrap the method
                    setattr(cls, attr_name, monitor_ai_operation(component, MetricType.RESPONSE_TIME)(attr))
        return cls
    return decorator


# Convenience functions for manual monitoring
def record_ai_operation(component: str, operation: str, success: bool, 
                       duration: float, metadata: Dict[str, Any] = None):
    """Manually record an AI operation"""
    session_id = f"{component}_{operation}_{int(time.time())}"
    
    # Record response time
    record_metric(
        component=component,
        metric_type=MetricType.RESPONSE_TIME,
        value=duration,
        metadata=metadata or {},
        session_id=session_id
    )
    
    # Record success/failure
    record_metric(
        component=component,
        metric_type=MetricType.SUCCESS_RATE if success else MetricType.ERROR_RATE,
        value=1.0 if success else 1.0,  # 1.0 for success, 1.0 for error occurrence
        metadata=metadata or {},
        session_id=session_id
    )


def record_ai_accuracy(component: str, accuracy: float, metadata: Dict[str, Any] = None):
    """Record AI accuracy metrics"""
    record_metric(
        component=component,
        metric_type=MetricType.ACCURACY,
        value=accuracy,
        metadata=metadata or {},
        session_id=f"{component}_accuracy_{int(time.time())}"
    )


def record_resource_usage(component: str, memory_mb: float, cpu_percent: float):
    """Record resource usage metrics"""
    session_id = f"{component}_resources_{int(time.time())}"
    
    record_metric(
        component=component,
        metric_type=MetricType.MEMORY_USAGE,
        value=memory_mb,
        metadata={"type": "current_usage"},
        session_id=session_id
    )
    
    record_metric(
        component=component,
        metric_type=MetricType.CPU_USAGE,
        value=cpu_percent,
        metadata={"type": "current_usage"},
        session_id=session_id
    )


# Context manager for monitoring operations
class MonitoredOperation:
    """Context manager for monitoring AI operations"""
    
    def __init__(self, component: str, operation: str, metadata: Dict[str, Any] = None):
        self.component = component
        self.operation = operation
        self.metadata = metadata or {}
        self.start_time = None
        self.session_id = None
    
    def __enter__(self):
        self.start_time = time.time()
        self.session_id = f"{self.component}_{self.operation}_{int(self.start_time)}"
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        duration = time.time() - self.start_time
        success = exc_type is None
        
        # Record the operation
        record_ai_operation(
            component=self.component,
            operation=self.operation,
            success=success,
            duration=duration,
            metadata={
                **self.metadata,
                "error_type": exc_type.__name__ if exc_type else None,
                "error_message": str(exc_val) if exc_val else None
            }
        )
        
        # Don't suppress exceptions
        return False


# Example usage functions
def example_monitored_function():
    """Example of how to use the monitoring system"""
    
    # Method 1: Using decorator
    @monitor_function("example_component", MetricType.RESPONSE_TIME)
    def some_ai_operation():
        time.sleep(0.1)  # Simulate work
        return "result"
    
    # Method 2: Using context manager
    with MonitoredOperation("example_component", "batch_processing", {"batch_size": 100}):
        time.sleep(0.2)  # Simulate work
        # Any exceptions will be automatically recorded
    
    # Method 3: Manual recording
    start_time = time.time()
    try:
        # Do some work
        time.sleep(0.1)
        result = "success"
        success = True
    except Exception as e:
        success = False
        result = str(e)
    
    duration = time.time() - start_time
    record_ai_operation(
        component="example_component",
        operation="manual_operation",
        success=success,
        duration=duration,
        metadata={"result": result}
    )
    
    return some_ai_operation()


# Integration with existing background processor
def integrate_with_background_processor():
    """Integrate monitoring with the background processor"""
    try:
        from .background_processor import BackgroundProcessor
        
        # Monitor task processing
        original_process_task = BackgroundProcessor._process_task
        
        @functools.wraps(original_process_task)
        def monitored_process_task(self, task):
            start_time = time.time()
            session_id = f"background_processor_{task.id}"
            
            try:
                result = original_process_task(self, task)
                
                # Record success metrics
                duration = time.time() - start_time
                record_metric(
                    component="background_processor",
                    metric_type=MetricType.RESPONSE_TIME,
                    value=duration,
                    metadata={
                        "task_id": task.id,
                        "task_type": task.task_type,
                        "status": "completed"
                    },
                    session_id=session_id
                )
                
                return result
                
            except Exception as e:
                # Record error metrics
                duration = time.time() - start_time
                record_metric(
                    component="background_processor",
                    metric_type=MetricType.ERROR_RATE,
                    value=1.0,
                    metadata={
                        "task_id": task.id,
                        "task_type": task.task_type,
                        "error": str(e)
                    },
                    session_id=session_id
                )
                raise
        
        # Replace the method
        BackgroundProcessor._process_task = monitored_process_task
        print("✅ Background processor monitoring integrated")
        
    except ImportError:
        print("⚠️ Background processor not available for integration")


# Initialize monitoring integration
def initialize_monitoring():
    """Initialize the monitoring system"""
    print("🚀 Initializing AI Monitoring System...")
    
    # Auto-integrate with existing components
    auto_integrate_monitoring()
    
    # Integrate with background processor
    integrate_with_background_processor()
    
    print("✅ AI Monitoring System initialized")


if __name__ == "__main__":
    # Test the monitoring system
    initialize_monitoring()
    example_monitored_function()
