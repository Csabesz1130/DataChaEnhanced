#!/usr/bin/env python3
"""
Direct AI Monitoring System Test

This script tests the AI monitoring system by including the essential
monitoring code directly to avoid import issues.
"""

import time
import random
import os
import sys
import json
import sqlite3
import threading
from datetime import datetime, timedelta
from dataclasses import dataclass
from enum import Enum
from collections import defaultdict, deque
import uuid


# Monitoring system classes (copied from ai_monitor.py)
class MetricType(Enum):
    RESPONSE_TIME = "response_time"
    ACCURACY = "accuracy"
    MEMORY_USAGE = "memory_usage"
    CPU_USAGE = "cpu_usage"
    ERROR_RATE = "error_rate"
    SUCCESS_RATE = "success_rate"
    DATA_QUALITY = "data_quality"


class AlertSeverity(Enum):
    INFO = "info"
    WARNING = "warning"
    ERROR = "error"
    CRITICAL = "critical"


@dataclass
class AIMetrics:
    component: str
    metric_type: MetricType
    value: float
    timestamp: datetime
    metadata: dict = None
    session_id: str = None
    
    def __post_init__(self):
        if self.metadata is None:
            self.metadata = {}
        if self.session_id is None:
            self.session_id = str(uuid.uuid4())


class MetricsCollector:
    def __init__(self, db_path: str = "ai_monitoring.db"):
        self.db_path = db_path
        self.metrics_buffer = deque(maxlen=1000)
        self.lock = threading.Lock()
        self._init_database()
    
    def _init_database(self):
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
            conn.execute("CREATE INDEX IF NOT EXISTS idx_metrics_component ON metrics(component)")
            conn.execute("CREATE INDEX IF NOT EXISTS idx_metrics_timestamp ON metrics(timestamp)")
    
    def record_metric(self, component: str, metric_type: MetricType, value: float, 
                     metadata: dict = None, session_id: str = None):
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
        
        if len(self.metrics_buffer) >= 100:
            self._flush_metrics()
    
    def _flush_metrics(self):
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


class AIMonitor:
    def __init__(self, db_path: str = "ai_monitoring.db"):
        self.db_path = db_path
        self.metrics_collector = MetricsCollector(db_path)
    
    def record_metric(self, component: str, metric_type: MetricType, value: float, 
                     metadata: dict = None, session_id: str = None):
        self.metrics_collector.record_metric(component, metric_type, value, metadata, session_id)


# Global monitor instance
_global_monitor = None

def get_ai_monitor() -> AIMonitor:
    global _global_monitor
    if _global_monitor is None:
        _global_monitor = AIMonitor()
    return _global_monitor


def record_metric(component: str, metric_type: MetricType, value: float, 
                 metadata: dict = None, session_id: str = None):
    monitor = get_ai_monitor()
    monitor.record_metric(component, metric_type, value, metadata, session_id)


def record_ai_operation(component: str, operation: str, success: bool, 
                       duration: float, metadata: dict = None):
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
        value=1.0,
        metadata=metadata or {},
        session_id=session_id
    )


def record_ai_accuracy(component: str, accuracy: float, metadata: dict = None):
    record_metric(
        component=component,
        metric_type=MetricType.ACCURACY,
        value=accuracy,
        metadata=metadata or {},
        session_id=f"{component}_accuracy_{int(time.time())}"
    )


class MonitoredOperation:
    def __init__(self, component: str, operation: str, metadata: dict = None):
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
        return False


def simulate_ai_operations():
    """Simulate various AI operations to generate monitoring data"""
    print("🤖 Simulating AI operations...")
    
    # Initialize monitoring
    monitor = get_ai_monitor()
    
    # Simulate Excel Analyzer operations
    print("📊 Simulating Excel Analyzer operations...")
    for i in range(5):
        with MonitoredOperation("excel_analyzer", f"analyze_file_{i}", {"file_size": random.randint(100, 1000)}):
            time.sleep(random.uniform(0.5, 2.0))
            
            # Simulate some errors
            if random.random() < 0.1:
                raise Exception("Simulated analysis error")
            
            # Record accuracy
            accuracy = random.uniform(0.7, 0.95)
            record_ai_accuracy("excel_analyzer", accuracy, {"file_id": i})
    
    # Simulate Chart Learner operations
    print("📈 Simulating Chart Learner operations...")
    for i in range(3):
        start_time = time.time()
        try:
            time.sleep(random.uniform(1.0, 3.0))
            confidence = random.uniform(0.8, 0.99)
            record_ai_accuracy("chart_learner", confidence, {"chart_id": i})
            success = True
        except Exception as e:
            success = False
        
        duration = time.time() - start_time
        record_ai_operation(
            component="chart_learner",
            operation=f"learn_chart_{i}",
            success=success,
            duration=duration,
            metadata={"chart_type": "bar"}
        )
    
    # Simulate Formula Learner operations
    print("🧮 Simulating Formula Learner operations...")
    for i in range(4):
        start_time = time.time()
        try:
            time.sleep(random.uniform(0.3, 1.5))
            success = random.random() > 0.05
            
            if not success:
                raise Exception("Formula parsing failed")
            
            accuracy = random.uniform(0.85, 0.98)
            record_ai_accuracy("formula_learner", accuracy, {"formula_id": i})
            
        except Exception as e:
            success = False
        
        duration = time.time() - start_time
        record_ai_operation(
            component="formula_learner",
            operation=f"learn_formula_{i}",
            success=success,
            duration=duration,
            metadata={"complexity": random.randint(1, 5)}
        )
    
    # Simulate Learning Pipeline operations
    print("🔄 Simulating Learning Pipeline operations...")
    for i in range(2):
        with MonitoredOperation("learning_pipeline", f"pipeline_run_{i}", {"epochs": 10}):
            time.sleep(random.uniform(2.0, 5.0))
            
            # Simulate training progress
            for epoch in range(10):
                time.sleep(0.1)
                accuracy = 0.5 + (epoch * 0.05) + random.uniform(-0.02, 0.02)
                record_ai_accuracy("learning_pipeline", accuracy, {"epoch": epoch, "run_id": i})
    
    # Simulate Background Processor operations
    print("⚙️ Simulating Background Processor operations...")
    for i in range(3):
        start_time = time.time()
        try:
            time.sleep(random.uniform(1.0, 4.0))
            success = random.random() > 0.08
            
            if not success:
                raise Exception("Background task failed")
                
        except Exception as e:
            success = False
        
        duration = time.time() - start_time
        record_ai_operation(
            component="background_processor",
            operation=f"process_task_{i}",
            success=success,
            duration=duration,
            metadata={"task_priority": random.choice(["high", "medium", "low"])}
        )
    
    # Simulate Research Extensions operations
    print("🔬 Simulating Research Extensions operations...")
    for i in range(2):
        with MonitoredOperation("research_extensions", f"research_analysis_{i}", {"data_points": 1000}):
            time.sleep(random.uniform(1.5, 3.5))
            
            # Simulate data quality analysis
            quality_score = random.uniform(0.6, 0.95)
            record_metric(
                component="research_extensions",
                metric_type=MetricType.DATA_QUALITY,
                value=quality_score,
                metadata={"analysis_type": "quality_assessment", "project_id": i}
            )


def show_database_info():
    """Show information about the monitoring database"""
    print("\n🗄️ Monitoring Database Info:")
    
    try:
        # Check if database exists
        db_path = "ai_monitoring.db"
        if os.path.exists(db_path):
            with sqlite3.connect(db_path) as conn:
                # Get table info
                cursor = conn.execute("SELECT name FROM sqlite_master WHERE type='table'")
                tables = cursor.fetchall()
                
                print(f"  Database: {db_path}")
                print(f"  Tables: {[table[0] for table in tables]}")
                
                # Get record counts
                for table in tables:
                    table_name = table[0]
                    cursor = conn.execute(f"SELECT COUNT(*) FROM {table_name}")
                    count = cursor.fetchone()[0]
                    print(f"  {table_name}: {count} records")
                
                # Show some sample metrics
                print("\n📊 Sample Metrics:")
                cursor = conn.execute("""
                    SELECT component, metric_type, value, timestamp 
                    FROM metrics 
                    ORDER BY timestamp DESC 
                    LIMIT 10
                """)
                for row in cursor.fetchall():
                    print(f"  {row[0]} - {row[1]}: {row[2]:.3f} ({row[3][:19]})")
        else:
            print("  Database not found - no monitoring data yet")
            
    except Exception as e:
        print(f"  Error accessing database: {e}")


def main():
    """Main test function"""
    print("🚀 Direct AI Monitoring System Test")
    print("=" * 50)
    
    # Simulate AI operations
    simulate_ai_operations()
    
    # Show database info
    show_database_info()
    
    print("\n" + "=" * 50)
    print("✅ Test completed successfully!")
    print("\n📋 What was demonstrated:")
    print("  • Real-time metric collection")
    print("  • Performance monitoring")
    print("  • Database storage")
    print("  • Context manager usage")
    print("  • Manual metric recording")
    
    print("\n🌐 To run the full dashboard:")
    print("  pip install streamlit plotly")
    print("  streamlit run src/ai_excel_learning/ai_dashboard.py")
    
    print("\n📚 For more information:")
    print("  Read AI_MONITORING_SYSTEM_GUIDE.md")


if __name__ == "__main__":
    main()
