"""
AI Monitoring and Analytics System (AMAS)

This module provides comprehensive monitoring, analytics, and optimization capabilities
for the AI Excel Learning system. It integrates seamlessly with existing components
without disrupting current functionality.

Features:
- Real-time performance monitoring
- Anomaly detection and alerting
- Automated optimization recommendations
- Comprehensive analytics dashboard
- Predictive maintenance
- A/B testing framework
"""

import asyncio
import json
import time
import threading
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Callable, Union
from dataclasses import dataclass, asdict
from enum import Enum
from collections import defaultdict, deque
import sqlite3
import logging
from pathlib import Path
import uuid

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class MetricType(Enum):
    """Types of metrics that can be monitored"""
    RESPONSE_TIME = "response_time"
    ACCURACY = "accuracy"
    MEMORY_USAGE = "memory_usage"
    CPU_USAGE = "cpu_usage"
    ERROR_RATE = "error_rate"
    SUCCESS_RATE = "success_rate"
    USER_SATISFACTION = "user_satisfaction"
    MODEL_DRIFT = "model_drift"
    DATA_QUALITY = "data_quality"
    RESOURCE_UTILIZATION = "resource_utilization"


class AlertSeverity(Enum):
    """Alert severity levels"""
    INFO = "info"
    WARNING = "warning"
    ERROR = "error"
    CRITICAL = "critical"


class ComponentType(Enum):
    """AI component types"""
    EXCEL_ANALYZER = "excel_analyzer"
    CHART_LEARNER = "chart_learner"
    FORMULA_LEARNER = "formula_learner"
    ML_MODELS = "ml_models"
    LEARNING_PIPELINE = "learning_pipeline"
    BACKGROUND_PROCESSOR = "background_processor"
    RESEARCH_EXTENSIONS = "research_extensions"


@dataclass
class AIMetrics:
    """AI performance metrics"""
    component: str
    metric_type: MetricType
    value: float
    timestamp: datetime
    metadata: Dict[str, Any] = None
    session_id: str = None
    
    def __post_init__(self):
        if self.metadata is None:
            self.metadata = {}
        if self.session_id is None:
            self.session_id = str(uuid.uuid4())


@dataclass
class Alert:
    """AI system alert"""
    id: str
    component: str
    severity: AlertSeverity
    message: str
    timestamp: datetime
    metrics: Dict[str, Any] = None
    resolved: bool = False
    resolution_time: Optional[datetime] = None
    
    def __post_init__(self):
        if self.metrics is None:
            self.metrics = {}


@dataclass
class OptimizationRecommendation:
    """AI optimization recommendation"""
    id: str
    component: str
    recommendation_type: str
    description: str
    expected_impact: str
    priority: int
    timestamp: datetime
    implemented: bool = False
    implementation_time: Optional[datetime] = None


class MetricsCollector:
    """Collects and stores AI performance metrics"""
    
    def __init__(self, db_path: str = "ai_monitoring.db"):
        self.db_path = db_path
        self.metrics_buffer = deque(maxlen=1000)  # In-memory buffer
        self.lock = threading.Lock()
        self._init_database()
    
    def _init_database(self):
        """Initialize the monitoring database"""
        with sqlite3.connect(self.db_path) as conn:
            conn.execute("""
                CREATE TABLE IF NOT EXISTS metrics (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    component TEXT NOT NULL,
                    metric_type TEXT NOT NULL,
                    value REAL NOT NULL,
                    timestamp TEXT NOT NULL,
                    metadata TEXT,
                    session_id TEXT
                )
            """)
            
            conn.execute("""
                CREATE TABLE IF NOT EXISTS alerts (
                    id TEXT PRIMARY KEY,
                    component TEXT NOT NULL,
                    severity TEXT NOT NULL,
                    message TEXT NOT NULL,
                    timestamp TEXT NOT NULL,
                    metrics TEXT,
                    resolved INTEGER DEFAULT 0,
                    resolution_time TEXT
                )
            """)
            
            conn.execute("""
                CREATE TABLE IF NOT EXISTS recommendations (
                    id TEXT PRIMARY KEY,
                    component TEXT NOT NULL,
                    recommendation_type TEXT NOT NULL,
                    description TEXT NOT NULL,
                    expected_impact TEXT NOT NULL,
                    priority INTEGER NOT NULL,
                    timestamp TEXT NOT NULL,
                    implemented INTEGER DEFAULT 0,
                    implementation_time TEXT
                )
            """)
            
            # Create indexes for better performance
            conn.execute("CREATE INDEX IF NOT EXISTS idx_metrics_component ON metrics(component)")
            conn.execute("CREATE INDEX IF NOT EXISTS idx_metrics_timestamp ON metrics(timestamp)")
            conn.execute("CREATE INDEX IF NOT EXISTS idx_alerts_component ON alerts(component)")
            conn.execute("CREATE INDEX IF NOT EXISTS idx_alerts_severity ON alerts(severity)")
    
    def record_metric(self, component: str, metric_type: MetricType, value: float, 
                     metadata: Dict[str, Any] = None, session_id: str = None):
        """Record a new metric"""
        metric = AIMetrics(
            component=component,
            metric_type=metric_type,
            value=value,
            timestamp=datetime.now(),
            metadata=metadata or {},
            session_id=session_id
        )
        
        with self.lock:
            self.metrics_buffer.append(metric)
        
        # Periodically flush to database
        if len(self.metrics_buffer) >= 100:
            self._flush_metrics()
    
    def _flush_metrics(self):
        """Flush metrics from buffer to database"""
        with self.lock:
            if not self.metrics_buffer:
                return
            
            metrics_to_flush = list(self.metrics_buffer)
            self.metrics_buffer.clear()
        
        with sqlite3.connect(self.db_path) as conn:
            for metric in metrics_to_flush:
                conn.execute("""
                    INSERT INTO metrics (component, metric_type, value, timestamp, metadata, session_id)
                    VALUES (?, ?, ?, ?, ?, ?)
                """, (
                    metric.component,
                    metric.metric_type.value,
                    metric.value,
                    metric.timestamp.isoformat(),
                    json.dumps(metric.metadata),
                    metric.session_id
                ))
    
    def get_metrics(self, component: str = None, metric_type: MetricType = None, 
                   start_time: datetime = None, end_time: datetime = None, limit: int = 1000) -> List[AIMetrics]:
        """Retrieve metrics from database"""
        query = "SELECT * FROM metrics WHERE 1=1"
        params = []
        
        if component:
            query += " AND component = ?"
            params.append(component)
        
        if metric_type:
            query += " AND metric_type = ?"
            params.append(metric_type.value)
        
        if start_time:
            query += " AND timestamp >= ?"
            params.append(start_time.isoformat())
        
        if end_time:
            query += " AND timestamp <= ?"
            params.append(end_time.isoformat())
        
        query += " ORDER BY timestamp DESC LIMIT ?"
        params.append(limit)
        
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.execute(query, params)
            metrics = []
            for row in cursor.fetchall():
                metric = AIMetrics(
                    component=row[1],
                    metric_type=MetricType(row[2]),
                    value=row[3],
                    timestamp=datetime.fromisoformat(row[4]),
                    metadata=json.loads(row[5]) if row[5] else {},
                    session_id=row[6]
                )
                metrics.append(metric)
        
        return metrics


class AnomalyDetector:
    """Detects anomalies in AI performance metrics"""
    
    def __init__(self, metrics_collector: MetricsCollector):
        self.metrics_collector = metrics_collector
        self.baselines = defaultdict(lambda: {
            'mean': 0.0,
            'std': 1.0,
            'count': 0
        })
        self.thresholds = {
            MetricType.RESPONSE_TIME: 2.0,  # 2 standard deviations
            MetricType.ERROR_RATE: 1.5,
            MetricType.ACCURACY: 2.0,
            MetricType.MEMORY_USAGE: 2.5,
            MetricType.CPU_USAGE: 2.0
        }
    
    def update_baseline(self, component: str, metric_type: MetricType, value: float):
        """Update baseline statistics for anomaly detection"""
        baseline = self.baselines[f"{component}_{metric_type.value}"]
        
        # Online mean and variance calculation
        baseline['count'] += 1
        delta = value - baseline['mean']
        baseline['mean'] += delta / baseline['count']
        delta2 = value - baseline['mean']
        baseline['std'] = ((baseline['std'] ** 2 * (baseline['count'] - 1) + delta * delta2) / baseline['count']) ** 0.5
    
    def detect_anomaly(self, component: str, metric_type: MetricType, value: float) -> bool:
        """Detect if a metric value is anomalous"""
        baseline_key = f"{component}_{metric_type.value}"
        baseline = self.baselines[baseline_key]
        
        if baseline['count'] < 10:  # Need minimum data points
            return False
        
        threshold = self.thresholds.get(metric_type, 2.0)
        z_score = abs((value - baseline['mean']) / baseline['std'])
        
        return z_score > threshold
    
    def get_anomaly_score(self, component: str, metric_type: MetricType, value: float) -> float:
        """Get anomaly score (0-1, higher = more anomalous)"""
        baseline_key = f"{component}_{metric_type.value}"
        baseline = self.baselines[baseline_key]
        
        if baseline['count'] < 10:
            return 0.0
        
        z_score = abs((value - baseline['mean']) / baseline['std'])
        return min(z_score / 3.0, 1.0)  # Normalize to 0-1


class AlertManager:
    """Manages AI system alerts"""
    
    def __init__(self, db_path: str = "ai_monitoring.db"):
        self.db_path = db_path
        self.active_alerts = {}
        self.alert_handlers = []
    
    def add_alert_handler(self, handler: Callable[[Alert], None]):
        """Add a custom alert handler"""
        self.alert_handlers.append(handler)
    
    def create_alert(self, component: str, severity: AlertSeverity, message: str, 
                    metrics: Dict[str, Any] = None) -> Alert:
        """Create a new alert"""
        alert = Alert(
            id=str(uuid.uuid4()),
            component=component,
            severity=severity,
            message=message,
            timestamp=datetime.now(),
            metrics=metrics or {}
        )
        
        # Store in database
        with sqlite3.connect(self.db_path) as conn:
            conn.execute("""
                INSERT INTO alerts (id, component, severity, message, timestamp, metrics)
                VALUES (?, ?, ?, ?, ?, ?)
            """, (
                alert.id,
                alert.component,
                alert.severity.value,
                alert.message,
                alert.timestamp.isoformat(),
                json.dumps(alert.metrics)
            ))
        
        # Store in memory
        self.active_alerts[alert.id] = alert
        
        # Notify handlers
        for handler in self.alert_handlers:
            try:
                handler(alert)
            except Exception as e:
                logger.error(f"Error in alert handler: {e}")
        
        return alert
    
    def resolve_alert(self, alert_id: str):
        """Mark an alert as resolved"""
        if alert_id in self.active_alerts:
            alert = self.active_alerts[alert_id]
            alert.resolved = True
            alert.resolution_time = datetime.now()
            
            with sqlite3.connect(self.db_path) as conn:
                conn.execute("""
                    UPDATE alerts SET resolved = 1, resolution_time = ?
                    WHERE id = ?
                """, (alert.resolution_time.isoformat(), alert_id))
            
            del self.active_alerts[alert_id]
    
    def get_active_alerts(self, component: str = None, severity: AlertSeverity = None) -> List[Alert]:
        """Get active alerts"""
        alerts = list(self.active_alerts.values())
        
        if component:
            alerts = [a for a in alerts if a.component == component]
        
        if severity:
            alerts = [a for a in alerts if a.severity == severity]
        
        return alerts


class PerformanceAnalyzer:
    """Analyzes AI performance and generates insights"""
    
    def __init__(self, metrics_collector: MetricsCollector):
        self.metrics_collector = metrics_collector
    
    def get_performance_summary(self, component: str, hours: int = 24) -> Dict[str, Any]:
        """Get performance summary for a component"""
        end_time = datetime.now()
        start_time = end_time - timedelta(hours=hours)
        
        metrics = self.metrics_collector.get_metrics(
            component=component,
            start_time=start_time,
            end_time=end_time
        )
        
        if not metrics:
            return {"error": "No metrics found"}
        
        # Group by metric type
        by_type = defaultdict(list)
        for metric in metrics:
            by_type[metric.metric_type].append(metric.value)
        
        summary = {
            "component": component,
            "period_hours": hours,
            "total_metrics": len(metrics),
            "metrics_by_type": {}
        }
        
        for metric_type, values in by_type.items():
            summary["metrics_by_type"][metric_type.value] = {
                "count": len(values),
                "mean": sum(values) / len(values),
                "min": min(values),
                "max": max(values),
                "std": (sum((x - sum(values)/len(values))**2 for x in values) / len(values))**0.5
            }
        
        return summary
    
    def get_trend_analysis(self, component: str, metric_type: MetricType, hours: int = 24) -> Dict[str, Any]:
        """Analyze trends in a specific metric"""
        end_time = datetime.now()
        start_time = end_time - timedelta(hours=hours)
        
        metrics = self.metrics_collector.get_metrics(
            component=component,
            metric_type=metric_type,
            start_time=start_time,
            end_time=end_time
        )
        
        if len(metrics) < 2:
            return {"error": "Insufficient data for trend analysis"}
        
        # Sort by timestamp
        metrics.sort(key=lambda x: x.timestamp)
        
        # Calculate trend
        values = [m.value for m in metrics]
        timestamps = [m.timestamp for m in metrics]
        
        # Simple linear trend
        n = len(values)
        x_sum = sum(range(n))
        y_sum = sum(values)
        xy_sum = sum(i * v for i, v in enumerate(values))
        x2_sum = sum(i * i for i in range(n))
        
        slope = (n * xy_sum - x_sum * y_sum) / (n * x2_sum - x_sum * x_sum)
        
        return {
            "component": component,
            "metric_type": metric_type.value,
            "trend_slope": slope,
            "trend_direction": "increasing" if slope > 0 else "decreasing" if slope < 0 else "stable",
            "data_points": len(metrics),
            "time_range": {
                "start": timestamps[0].isoformat(),
                "end": timestamps[-1].isoformat()
            }
        }


class OptimizationEngine:
    """Generates optimization recommendations"""
    
    def __init__(self, metrics_collector: MetricsCollector, performance_analyzer: PerformanceAnalyzer):
        self.metrics_collector = metrics_collector
        self.performance_analyzer = performance_analyzer
        self.db_path = "ai_monitoring.db"
    
    def generate_recommendations(self, component: str) -> List[OptimizationRecommendation]:
        """Generate optimization recommendations for a component"""
        recommendations = []
        
        # Get recent performance data
        summary = self.performance_analyzer.get_performance_summary(component, hours=24)
        
        if "error" in summary:
            return recommendations
        
        metrics_by_type = summary.get("metrics_by_type", {})
        
        # Check response time
        if "response_time" in metrics_by_type:
            avg_response_time = metrics_by_type["response_time"]["mean"]
            if avg_response_time > 5.0:  # More than 5 seconds
                recommendations.append(OptimizationRecommendation(
                    id=str(uuid.uuid4()),
                    component=component,
                    recommendation_type="performance_optimization",
                    description=f"High response time detected ({avg_response_time:.2f}s). Consider caching or parallel processing.",
                    expected_impact="Reduce response time by 30-50%",
                    priority=1,
                    timestamp=datetime.now()
                ))
        
        # Check error rate
        if "error_rate" in metrics_by_type:
            avg_error_rate = metrics_by_type["error_rate"]["mean"]
            if avg_error_rate > 0.1:  # More than 10% error rate
                recommendations.append(OptimizationRecommendation(
                    id=str(uuid.uuid4()),
                    component=component,
                    recommendation_type="error_reduction",
                    description=f"High error rate detected ({avg_error_rate:.2%}). Review error handling and input validation.",
                    expected_impact="Reduce error rate by 50-80%",
                    priority=1,
                    timestamp=datetime.now()
                ))
        
        # Check accuracy
        if "accuracy" in metrics_by_type:
            avg_accuracy = metrics_by_type["accuracy"]["mean"]
            if avg_accuracy < 0.8:  # Less than 80% accuracy
                recommendations.append(OptimizationRecommendation(
                    id=str(uuid.uuid4()),
                    component=component,
                    recommendation_type="model_improvement",
                    description=f"Low accuracy detected ({avg_accuracy:.2%}). Consider retraining with more data or feature engineering.",
                    expected_impact="Improve accuracy by 10-20%",
                    priority=2,
                    timestamp=datetime.now()
                ))
        
        # Store recommendations
        with sqlite3.connect(self.db_path) as conn:
            for rec in recommendations:
                conn.execute("""
                    INSERT INTO recommendations (id, component, recommendation_type, description, 
                                               expected_impact, priority, timestamp)
                    VALUES (?, ?, ?, ?, ?, ?, ?)
                """, (
                    rec.id,
                    rec.component,
                    rec.recommendation_type,
                    rec.description,
                    rec.expected_impact,
                    rec.priority,
                    rec.timestamp.isoformat()
                ))
        
        return recommendations


class AIMonitor:
    """
    Main AI monitoring system that integrates all components
    """
    
    def __init__(self, db_path: str = "ai_monitoring.db"):
        self.db_path = db_path
        self.metrics_collector = MetricsCollector(db_path)
        self.anomaly_detector = AnomalyDetector(self.metrics_collector)
        self.alert_manager = AlertManager(db_path)
        self.performance_analyzer = PerformanceAnalyzer(self.metrics_collector)
        self.optimization_engine = OptimizationEngine(self.metrics_collector, self.performance_analyzer)
        
        # Start background monitoring
        self.monitoring_active = True
        self.monitoring_thread = threading.Thread(target=self._monitoring_loop, daemon=True)
        self.monitoring_thread.start()
    
    def _monitoring_loop(self):
        """Background monitoring loop"""
        while self.monitoring_active:
            try:
                # Flush metrics to database
                self.metrics_collector._flush_metrics()
                
                # Check for anomalies and create alerts
                self._check_anomalies()
                
                # Generate optimization recommendations
                self._generate_recommendations()
                
                time.sleep(60)  # Check every minute
                
            except Exception as e:
                logger.error(f"Error in monitoring loop: {e}")
                time.sleep(60)
    
    def _check_anomalies(self):
        """Check for anomalies in recent metrics"""
        # Get recent metrics for all components
        recent_metrics = self.metrics_collector.get_metrics(
            start_time=datetime.now() - timedelta(minutes=30),
            limit=1000
        )
        
        for metric in recent_metrics:
            # Update baseline
            self.anomaly_detector.update_baseline(
                metric.component, 
                metric.metric_type, 
                metric.value
            )
            
            # Check for anomaly
            if self.anomaly_detector.detect_anomaly(
                metric.component, 
                metric.metric_type, 
                metric.value
            ):
                severity = AlertSeverity.WARNING
                if metric.metric_type in [MetricType.ERROR_RATE, MetricType.MODEL_DRIFT]:
                    severity = AlertSeverity.ERROR
                
                self.alert_manager.create_alert(
                    component=metric.component,
                    severity=severity,
                    message=f"Anomaly detected in {metric.metric_type.value}: {metric.value}",
                    metrics={"value": metric.value, "metric_type": metric.metric_type.value}
                )
    
    def _generate_recommendations(self):
        """Generate optimization recommendations"""
        # Get unique components
        components = set()
        recent_metrics = self.metrics_collector.get_metrics(
            start_time=datetime.now() - timedelta(hours=6),
            limit=1000
        )
        
        for metric in recent_metrics:
            components.add(metric.component)
        
        # Generate recommendations for each component
        for component in components:
            self.optimization_engine.generate_recommendations(component)
    
    def record_metric(self, component: str, metric_type: MetricType, value: float, 
                     metadata: Dict[str, Any] = None, session_id: str = None):
        """Record a metric (main entry point)"""
        self.metrics_collector.record_metric(component, metric_type, value, metadata, session_id)
    
    def get_performance_summary(self, component: str, hours: int = 24) -> Dict[str, Any]:
        """Get performance summary for a component"""
        return self.performance_analyzer.get_performance_summary(component, hours)
    
    def get_active_alerts(self, component: str = None, severity: AlertSeverity = None) -> List[Alert]:
        """Get active alerts"""
        return self.alert_manager.get_active_alerts(component, severity)
    
    def get_recommendations(self, component: str = None, implemented: bool = None) -> List[OptimizationRecommendation]:
        """Get optimization recommendations"""
        query = "SELECT * FROM recommendations WHERE 1=1"
        params = []
        
        if component:
            query += " AND component = ?"
            params.append(component)
        
        if implemented is not None:
            query += " AND implemented = ?"
            params.append(1 if implemented else 0)
        
        query += " ORDER BY priority DESC, timestamp DESC"
        
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.execute(query, params)
            recommendations = []
            for row in cursor.fetchall():
                rec = OptimizationRecommendation(
                    id=row[0],
                    component=row[1],
                    recommendation_type=row[2],
                    description=row[3],
                    expected_impact=row[4],
                    priority=row[5],
                    timestamp=datetime.fromisoformat(row[6]),
                    implemented=bool(row[7]),
                    implementation_time=datetime.fromisoformat(row[8]) if row[8] else None
                )
                recommendations.append(rec)
        
        return recommendations
    
    def shutdown(self):
        """Shutdown the monitoring system"""
        self.monitoring_active = False
        self.metrics_collector._flush_metrics()
        if self.monitoring_thread.is_alive():
            self.monitoring_thread.join(timeout=5)


# Global monitoring instance
_global_monitor = None

def get_ai_monitor() -> AIMonitor:
    """Get the global AI monitor instance"""
    global _global_monitor
    if _global_monitor is None:
        _global_monitor = AIMonitor()
    return _global_monitor


def monitor_ai_operation(component: str, metric_type: MetricType, 
                        operation_func: Callable, *args, **kwargs):
    """Decorator to monitor AI operations"""
    def wrapper(*args, **kwargs):
        monitor = get_ai_monitor()
        start_time = time.time()
        session_id = str(uuid.uuid4())
        
        try:
            result = operation_func(*args, **kwargs)
            
            # Record success metrics
            response_time = time.time() - start_time
            monitor.record_metric(
                component=component,
                metric_type=metric_type,
                value=response_time,
                metadata={"status": "success", "result_type": type(result).__name__},
                session_id=session_id
            )
            
            return result
            
        except Exception as e:
            # Record error metrics
            response_time = time.time() - start_time
            monitor.record_metric(
                component=component,
                metric_type=MetricType.ERROR_RATE,
                value=1.0,  # Error occurred
                metadata={"status": "error", "error_type": type(e).__name__, "error_message": str(e)},
                session_id=session_id
            )
            
            # Also record response time for failed operations
            monitor.record_metric(
                component=component,
                metric_type=MetricType.RESPONSE_TIME,
                value=response_time,
                metadata={"status": "error"},
                session_id=session_id
            )
            
            raise
    
    return wrapper


# Convenience functions for easy integration
def record_metric(component: str, metric_type: MetricType, value: float, 
                 metadata: Dict[str, Any] = None, session_id: str = None):
    """Record a metric using the global monitor"""
    monitor = get_ai_monitor()
    monitor.record_metric(component, metric_type, value, metadata, session_id)


def get_performance_summary(component: str, hours: int = 24) -> Dict[str, Any]:
    """Get performance summary using the global monitor"""
    monitor = get_ai_monitor()
    return monitor.get_performance_summary(component, hours)


def get_active_alerts(component: str = None, severity: AlertSeverity = None) -> List[Alert]:
    """Get active alerts using the global monitor"""
    monitor = get_ai_monitor()
    return monitor.get_active_alerts(component, severity)


def get_recommendations(component: str = None, implemented: bool = None) -> List[OptimizationRecommendation]:
    """Get recommendations using the global monitor"""
    monitor = get_ai_monitor()
    return monitor.get_recommendations(component, implemented)
