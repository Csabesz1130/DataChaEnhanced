"""
AI Excel Learning Module

This module provides advanced AI/ML capabilities for learning from Excel files
and generating similar files automatically. It includes:

- Excel file analysis and pattern recognition
- Machine learning models for data generation
- Chart and visualization learning
- Automated Excel file generation
- Continuous learning and model refinement
"""

from .excel_analyzer import ExcelAnalyzer
from .ml_models import ExcelMLModels
from .chart_learner import ChartLearner
from .data_generator import DataGenerator
from .excel_generator import ExcelGenerator
from .learning_pipeline import LearningPipeline
from .model_manager import ModelManager
from .formula_learner import FormulaLearner, FormulaLogic, FilterCondition, CalculationPattern, FormulaType
from .background_processor import BackgroundProcessor, LearningTask, TaskStatus, Notification, NotificationType
from .research_extensions import ResearchExtensions, ResearchProject, DataQualityReport, StatisticalAnalysis
from .ai_monitor import (
    AIMonitor, MetricsCollector, AnomalyDetector, AlertManager, 
    PerformanceAnalyzer, OptimizationEngine, AIMetrics, Alert, 
    OptimizationRecommendation, MetricType, AlertSeverity, ComponentType,
    get_ai_monitor, record_metric, get_performance_summary, 
    get_active_alerts, get_recommendations, monitor_ai_operation
)
from .monitoring_integration import (
    MonitoringIntegration, get_monitoring_integration, auto_integrate_monitoring,
    monitor_function, monitor_class_methods, record_ai_operation, 
    record_ai_accuracy, record_resource_usage, MonitoredOperation,
    initialize_monitoring
)

__version__ = "1.0.0"
__author__ = "DataChaEnhanced AI Team"

__all__ = [
    "ExcelAnalyzer",
    "ExcelMLModels",
    "ChartLearner",
    "DataGenerator",
    "ExcelGenerator",
    "LearningPipeline",
    "ModelManager",
    "FormulaLearner",
    "FormulaLogic",
    "FilterCondition",
    "CalculationPattern",
    "FormulaType",
    "BackgroundProcessor",
    "LearningTask",
    "TaskStatus",
    "Notification",
    "NotificationType",
    "ResearchExtensions",
    "ResearchProject",
    "DataQualityReport",
    "StatisticalAnalysis",
    # AI Monitoring System
    "AIMonitor",
    "MetricsCollector",
    "AnomalyDetector",
    "AlertManager",
    "PerformanceAnalyzer",
    "OptimizationEngine",
    "AIMetrics",
    "Alert",
    "OptimizationRecommendation",
    "MetricType",
    "AlertSeverity",
    "ComponentType",
    "get_ai_monitor",
    "record_metric",
    "get_performance_summary",
    "get_active_alerts",
    "get_recommendations",
    "monitor_ai_operation",
    "MonitoringIntegration",
    "get_monitoring_integration",
    "auto_integrate_monitoring",
    "monitor_function",
    "monitor_class_methods",
    "record_ai_operation",
    "record_ai_accuracy",
    "record_resource_usage",
    "MonitoredOperation",
    "initialize_monitoring"
]
