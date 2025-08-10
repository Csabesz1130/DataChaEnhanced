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

# Core modules - try to import but make them conditional
try:
    from .excel_analyzer import ExcelAnalyzer
    EXCEL_ANALYZER_AVAILABLE = True
except ImportError:
    EXCEL_ANALYZER_AVAILABLE = False
    ExcelAnalyzer = None

try:
    from .formula_learner import FormulaLearner, FormulaLogic, FilterCondition, CalculationPattern, FormulaType
    FORMULA_LEARNER_AVAILABLE = True
except ImportError:
    FORMULA_LEARNER_AVAILABLE = False
    FormulaLearner = FormulaLogic = FilterCondition = CalculationPattern = FormulaType = None

# Try to import ML-dependent modules
try:
    from .ml_models import ExcelMLModels
    ML_MODELS_AVAILABLE = True
except ImportError:
    ML_MODELS_AVAILABLE = False
    ExcelMLModels = None

try:
    from .chart_learner import ChartLearner
    CHART_LEARNER_AVAILABLE = True
except ImportError:
    CHART_LEARNER_AVAILABLE = False
    ChartLearner = None

try:
    from .data_generator import DataGenerator
    DATA_GENERATOR_AVAILABLE = True
except ImportError:
    DATA_GENERATOR_AVAILABLE = False
    DataGenerator = None

try:
    from .excel_generator import ExcelGenerator
    EXCEL_GENERATOR_AVAILABLE = True
except ImportError:
    EXCEL_GENERATOR_AVAILABLE = False
    ExcelGenerator = None

try:
    from .learning_pipeline import LearningPipeline
    LEARNING_PIPELINE_AVAILABLE = True
except ImportError:
    LEARNING_PIPELINE_AVAILABLE = False
    LearningPipeline = None

try:
    from .model_manager import ModelManager
    MODEL_MANAGER_AVAILABLE = True
except ImportError:
    MODEL_MANAGER_AVAILABLE = False
    ModelManager = None

# Background processing - try simplified first, then full
try:
    from .background_processor_simple import SimpleBackgroundProcessor
    SIMPLE_BACKGROUND_PROCESSOR_AVAILABLE = True
except ImportError:
    SIMPLE_BACKGROUND_PROCESSOR_AVAILABLE = False
    SimpleBackgroundProcessor = None

try:
    from .background_processor import BackgroundProcessor, LearningTask, TaskStatus, Notification, NotificationType
    BACKGROUND_PROCESSOR_AVAILABLE = True
except ImportError:
    BACKGROUND_PROCESSOR_AVAILABLE = False
    BackgroundProcessor = LearningTask = TaskStatus = Notification = NotificationType = None

# Research extensions
try:
    from .research_extensions import ResearchExtensions, ResearchProject, DataQualityReport, StatisticalAnalysis
    RESEARCH_EXTENSIONS_AVAILABLE = True
except ImportError:
    RESEARCH_EXTENSIONS_AVAILABLE = False
    ResearchExtensions = ResearchProject = DataQualityReport = StatisticalAnalysis = None

# AI Monitoring - optional
try:
    from .ai_monitor import (
        AIMonitor, MetricsCollector, AnomalyDetector, AlertManager, 
        PerformanceAnalyzer, OptimizationEngine, AIMetrics, Alert, 
        OptimizationRecommendation, MetricType, AlertSeverity, ComponentType,
        get_ai_monitor, record_metric, get_performance_summary, 
        get_active_alerts, get_recommendations, monitor_ai_operation
    )
    AI_MONITOR_AVAILABLE = True
except ImportError:
    AI_MONITOR_AVAILABLE = False
    AIMonitor = MetricsCollector = AnomalyDetector = AlertManager = None
    PerformanceAnalyzer = OptimizationEngine = AIMetrics = Alert = None
    OptimizationRecommendation = MetricType = AlertSeverity = ComponentType = None
    get_ai_monitor = record_metric = get_performance_summary = None
    get_active_alerts = get_recommendations = monitor_ai_operation = None

try:
    from .monitoring_integration import (
        MonitoringIntegration, get_monitoring_integration, auto_integrate_monitoring,
        monitor_function, monitor_class_methods, record_ai_operation, 
        record_ai_accuracy, record_resource_usage, MonitoredOperation,
        initialize_monitoring
    )
    MONITORING_INTEGRATION_AVAILABLE = True
except ImportError:
    MONITORING_INTEGRATION_AVAILABLE = False
    MonitoringIntegration = get_monitoring_integration = auto_integrate_monitoring = None
    monitor_function = monitor_class_methods = record_ai_operation = None
    record_ai_accuracy = record_resource_usage = MonitoredOperation = None
    initialize_monitoring = None

__version__ = "1.0.0"
__author__ = "DataChaEnhanced AI Team"

# Build __all__ dynamically based on what's available
__all__ = []

# Add core modules if available
if EXCEL_ANALYZER_AVAILABLE:
    __all__.append("ExcelAnalyzer")

if FORMULA_LEARNER_AVAILABLE:
    __all__.extend(["FormulaLearner", "FormulaLogic", "FilterCondition", "CalculationPattern", "FormulaType"])

# Add available modules to __all__
if ML_MODELS_AVAILABLE:
    __all__.append("ExcelMLModels")

if CHART_LEARNER_AVAILABLE:
    __all__.append("ChartLearner")

if DATA_GENERATOR_AVAILABLE:
    __all__.append("DataGenerator")

if EXCEL_GENERATOR_AVAILABLE:
    __all__.append("ExcelGenerator")

if LEARNING_PIPELINE_AVAILABLE:
    __all__.append("LearningPipeline")

if MODEL_MANAGER_AVAILABLE:
    __all__.append("ModelManager")

if SIMPLE_BACKGROUND_PROCESSOR_AVAILABLE:
    __all__.append("SimpleBackgroundProcessor")

if BACKGROUND_PROCESSOR_AVAILABLE:
    __all__.extend(["BackgroundProcessor", "LearningTask", "TaskStatus", "Notification", "NotificationType"])

if RESEARCH_EXTENSIONS_AVAILABLE:
    __all__.extend(["ResearchExtensions", "ResearchProject", "DataQualityReport", "StatisticalAnalysis"])

if AI_MONITOR_AVAILABLE:
    __all__.extend([
        "AIMonitor", "MetricsCollector", "AnomalyDetector", "AlertManager",
        "PerformanceAnalyzer", "OptimizationEngine", "AIMetrics", "Alert", 
        "OptimizationRecommendation", "MetricType", "AlertSeverity", "ComponentType",
        "get_ai_monitor", "record_metric", "get_performance_summary",
        "get_active_alerts", "get_recommendations", "monitor_ai_operation"
    ])

if MONITORING_INTEGRATION_AVAILABLE:
    __all__.extend([
        "MonitoringIntegration", "get_monitoring_integration", "auto_integrate_monitoring",
        "monitor_function", "monitor_class_methods", "record_ai_operation",
        "record_ai_accuracy", "record_resource_usage", "MonitoredOperation",
        "initialize_monitoring"
    ])
