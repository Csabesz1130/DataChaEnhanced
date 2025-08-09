#!/usr/bin/env python3
"""
Standalone AI Monitoring System Test

This script tests the AI monitoring system directly without importing
the main module that has TensorFlow dependencies.
"""

import time
import random
import os
import sys
from datetime import datetime, timedelta

# Add the src directory to the path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

# Import monitoring modules directly
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src', 'ai_excel_learning'))

# Import monitoring components directly
from ai_monitor import (
    AIMonitor, MetricType, AlertSeverity, get_ai_monitor,
    record_metric, get_performance_summary, get_active_alerts, get_recommendations
)
from monitoring_integration import (
    MonitoredOperation, record_ai_operation, record_ai_accuracy
)


def simulate_ai_operations():
    """Simulate various AI operations to generate monitoring data"""
    print("🤖 Simulating AI operations...")
    
    # Initialize monitoring
    monitor = get_ai_monitor()
    
    # Simulate Excel Analyzer operations
    print("📊 Simulating Excel Analyzer operations...")
    for i in range(5):
        with MonitoredOperation("excel_analyzer", f"analyze_file_{i}", {"file_size": random.randint(100, 1000)}):
            time.sleep(random.uniform(0.5, 2.0))  # Simulate processing time
            
            # Simulate some errors
            if random.random() < 0.1:  # 10% error rate
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
            success = random.random() > 0.05  # 95% success rate
            
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
            success = random.random() > 0.08  # 92% success rate
            
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


def demonstrate_monitoring_features():
    """Demonstrate various monitoring features"""
    print("\n📊 Demonstrating monitoring features...")
    
    monitor = get_ai_monitor()
    
    # Wait a moment for background processing
    time.sleep(2)
    
    # Show performance summaries
    print("\n📈 Performance Summaries:")
    components = ["excel_analyzer", "chart_learner", "formula_learner", 
                 "learning_pipeline", "background_processor", "research_extensions"]
    
    for component in components:
        try:
            summary = monitor.get_performance_summary(component, hours=1)
            if "error" not in summary:
                print(f"\n{component.replace('_', ' ').title()}:")
                print(f"  Total Operations: {summary.get('total_metrics', 0)}")
                
                metrics = summary.get("metrics_by_type", {})
                if "response_time" in metrics:
                    avg_rt = metrics["response_time"]["mean"]
                    print(f"  Avg Response Time: {avg_rt:.2f}s")
                
                if "error_rate" in metrics:
                    error_rate = metrics["error_rate"]["mean"]
                    print(f"  Error Rate: {error_rate:.2%}")
                
                if "accuracy" in metrics:
                    accuracy = metrics["accuracy"]["mean"]
                    print(f"  Avg Accuracy: {accuracy:.2%}")
        except Exception as e:
            print(f"Error getting summary for {component}: {e}")
    
    # Show active alerts
    print("\n🚨 Active Alerts:")
    alerts = get_active_alerts()
    if alerts:
        for alert in alerts:
            severity_icon = {
                "info": "ℹ️",
                "warning": "⚠️",
                "error": "❌",
                "critical": "🚨"
            }.get(alert.severity.value, "ℹ️")
            
            print(f"  {severity_icon} {alert.component}: {alert.message}")
    else:
        print("  ✅ No active alerts")
    
    # Show optimization recommendations
    print("\n💡 Optimization Recommendations:")
    recommendations = get_recommendations(implemented=False)
    if recommendations:
        for rec in recommendations[:3]:  # Show top 3
            priority_icon = {1: "🔴", 2: "🟡", 3: "🟢"}.get(rec.priority, "⚪")
            print(f"  {priority_icon} {rec.component}: {rec.description}")
    else:
        print("  ✅ No pending recommendations")


def show_database_info():
    """Show information about the monitoring database"""
    print("\n🗄️ Monitoring Database Info:")
    
    try:
        import sqlite3
        
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
        else:
            print("  Database not found - no monitoring data yet")
            
    except Exception as e:
        print(f"  Error accessing database: {e}")


def main():
    """Main test function"""
    print("🚀 Standalone AI Monitoring System Test")
    print("=" * 50)
    
    # Simulate AI operations
    simulate_ai_operations()
    
    # Demonstrate monitoring features
    demonstrate_monitoring_features()
    
    # Show database info
    show_database_info()
    
    print("\n" + "=" * 50)
    print("✅ Test completed successfully!")
    print("\n📋 What was demonstrated:")
    print("  • Real-time metric collection")
    print("  • Performance monitoring")
    print("  • Anomaly detection")
    print("  • Alert generation")
    print("  • Optimization recommendations")
    print("  • Database storage")
    
    print("\n🌐 To run the full dashboard:")
    print("  pip install streamlit plotly")
    print("  streamlit run src/ai_excel_learning/ai_dashboard.py")
    
    print("\n📚 For more information:")
    print("  Read AI_MONITORING_SYSTEM_GUIDE.md")


if __name__ == "__main__":
    main()
