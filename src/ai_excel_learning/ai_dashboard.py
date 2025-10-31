"""
AI Monitoring Dashboard

A Streamlit-based dashboard for visualizing AI performance metrics,
alerts, and optimization recommendations.
"""

import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime, timedelta
import json
from typing import Dict, List, Any
import sqlite3
import time

from .ai_monitor import (
    get_ai_monitor, MetricType, AlertSeverity, 
    get_performance_summary, get_active_alerts, get_recommendations
)


def create_dashboard():
    """Create the main AI monitoring dashboard"""
    st.set_page_config(
        page_title="AI Monitoring Dashboard",
        page_icon="🤖",
        layout="wide",
        initial_sidebar_state="expanded"
    )
    
    st.title("🤖 AI Monitoring Dashboard")
    st.markdown("Real-time monitoring and analytics for AI Excel Learning System")
    
    # Sidebar for controls
    with st.sidebar:
        st.header("📊 Dashboard Controls")
        
        # Time range selector
        time_range = st.selectbox(
            "Time Range",
            ["Last Hour", "Last 6 Hours", "Last 24 Hours", "Last 7 Days"],
            index=2
        )
        
        # Component filter
        components = ["All Components", "excel_analyzer", "chart_learner", 
                     "formula_learner", "ml_models", "learning_pipeline", 
                     "background_processor", "research_extensions"]
        selected_component = st.selectbox("Component", components, index=0)
        
        # Refresh button
        if st.button("🔄 Refresh Data"):
            st.rerun()
    
    # Convert time range to hours
    time_range_hours = {
        "Last Hour": 1,
        "Last 6 Hours": 6,
        "Last 24 Hours": 24,
        "Last 7 Days": 168
    }[time_range]
    
    # Get monitoring data
    monitor = get_ai_monitor()
    
    # Create dashboard layout
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.subheader("📈 Performance Overview")
        create_performance_overview(monitor, selected_component, time_range_hours)
    
    with col2:
        st.subheader("🚨 Active Alerts")
        create_alerts_panel()
    
    # Second row
    col3, col4 = st.columns(2)
    
    with col3:
        st.subheader("📊 Detailed Metrics")
        create_detailed_metrics(monitor, selected_component, time_range_hours)
    
    with col4:
        st.subheader("💡 Optimization Recommendations")
        create_recommendations_panel()
    
    # Third row - full width
    st.subheader("🔄 Real-time Metrics")
    create_realtime_metrics(monitor, selected_component)


def create_performance_overview(monitor, component: str, hours: int):
    """Create performance overview section"""
    if component == "All Components":
        # Show summary for all components
        components = ["excel_analyzer", "chart_learner", "formula_learner", 
                     "ml_models", "learning_pipeline", "background_processor"]
        
        overview_data = []
        for comp in components:
            try:
                summary = monitor.get_performance_summary(comp, hours)
                if "error" not in summary:
                    metrics = summary.get("metrics_by_type", {})
                    
                    # Calculate overall health score
                    health_score = 0
                    if "response_time" in metrics:
                        avg_rt = metrics["response_time"]["mean"]
                        health_score += max(0, 100 - avg_rt * 10)  # Lower response time = better
                    
                    if "error_rate" in metrics:
                        avg_er = metrics["error_rate"]["mean"]
                        health_score += max(0, 100 - avg_er * 100)  # Lower error rate = better
                    
                    if "accuracy" in metrics:
                        avg_acc = metrics["accuracy"]["mean"]
                        health_score += avg_acc * 100  # Higher accuracy = better
                    
                    health_score = min(100, health_score / 3)  # Average and cap at 100
                    
                    overview_data.append({
                        "Component": comp.replace("_", " ").title(),
                        "Health Score": health_score,
                        "Total Operations": summary.get("total_metrics", 0),
                        "Avg Response Time": metrics.get("response_time", {}).get("mean", 0),
                        "Error Rate": metrics.get("error_rate", {}).get("mean", 0)
                    })
            except Exception as e:
                st.error(f"Error getting data for {comp}: {e}")
        
        if overview_data:
            df = pd.DataFrame(overview_data)
            
            # Health score gauge
            fig = px.bar(df, x="Component", y="Health Score", 
                        color="Health Score", 
                        color_continuous_scale="RdYlGn",
                        title="Component Health Scores")
            fig.update_layout(height=300)
            st.plotly_chart(fig, use_container_width=True)
            
            # Metrics table
            st.dataframe(df, use_container_width=True)
        else:
            st.info("No performance data available")
    else:
        # Show detailed view for specific component
        try:
            summary = monitor.get_performance_summary(component, hours)
            if "error" not in summary:
                metrics = summary.get("metrics_by_type", {})
                
                # Create metric cards
                col1, col2, col3, col4 = st.columns(4)
                
                with col1:
                    st.metric(
                        "Total Operations",
                        summary.get("total_metrics", 0)
                    )
                
                with col2:
                    avg_rt = metrics.get("response_time", {}).get("mean", 0)
                    st.metric(
                        "Avg Response Time",
                        f"{avg_rt:.2f}s"
                    )
                
                with col3:
                    error_rate = metrics.get("error_rate", {}).get("mean", 0)
                    st.metric(
                        "Error Rate",
                        f"{error_rate:.2%}"
                    )
                
                with col4:
                    accuracy = metrics.get("accuracy", {}).get("mean", 0)
                    st.metric(
                        "Accuracy",
                        f"{accuracy:.2%}"
                    )
                
                # Create trend chart
                create_trend_chart(monitor, component, hours)
            else:
                st.warning("No data available for this component")
        except Exception as e:
            st.error(f"Error loading component data: {e}")


def create_trend_chart(monitor, component: str, hours: int):
    """Create trend chart for a component"""
    try:
        # Get metrics for different types
        metric_types = [MetricType.RESPONSE_TIME, MetricType.ERROR_RATE, MetricType.ACCURACY]
        
        trend_data = []
        for metric_type in metric_types:
            metrics = monitor.metrics_collector.get_metrics(
                component=component,
                metric_type=metric_type,
                start_time=datetime.now() - timedelta(hours=hours),
                limit=1000
            )
            
            for metric in metrics:
                trend_data.append({
                    "Timestamp": metric.timestamp,
                    "Value": metric.value,
                    "Metric Type": metric_type.value.replace("_", " ").title()
                })
        
        if trend_data:
            df = pd.DataFrame(trend_data)
            df = df.sort_values("Timestamp")
            
            fig = px.line(df, x="Timestamp", y="Value", color="Metric Type",
                         title=f"{component.replace('_', ' ').title()} - Performance Trends")
            fig.update_layout(height=300)
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("No trend data available")
    except Exception as e:
        st.error(f"Error creating trend chart: {e}")


def create_alerts_panel():
    """Create alerts panel"""
    try:
        alerts = get_active_alerts()
        
        if not alerts:
            st.success("✅ No active alerts")
            return
        
        # Group alerts by severity
        alerts_by_severity = {}
        for alert in alerts:
            severity = alert.severity.value
            if severity not in alerts_by_severity:
                alerts_by_severity[severity] = []
            alerts_by_severity[severity].append(alert)
        
        # Display alerts by severity
        for severity, alert_list in alerts_by_severity.items():
            severity_icon = {
                "info": "ℹ️",
                "warning": "⚠️",
                "error": "❌",
                "critical": "🚨"
            }.get(severity, "ℹ️")
            
            st.subheader(f"{severity_icon} {severity.title()} ({len(alert_list)})")
            
            for alert in alert_list:
                with st.expander(f"{alert.component} - {alert.message[:50]}..."):
                    st.write(f"**Message:** {alert.message}")
                    st.write(f"**Component:** {alert.component}")
                    st.write(f"**Time:** {alert.timestamp.strftime('%Y-%m-%d %H:%M:%S')}")
                    
                    if alert.metrics:
                        st.write("**Metrics:**")
                        st.json(alert.metrics)
                    
                    if st.button(f"Resolve Alert", key=f"resolve_{alert.id}"):
                        monitor = get_ai_monitor()
                        monitor.alert_manager.resolve_alert(alert.id)
                        st.success("Alert resolved!")
                        st.rerun()
    
    except Exception as e:
        st.error(f"Error loading alerts: {e}")


def create_detailed_metrics(monitor, component: str, hours: int):
    """Create detailed metrics section"""
    if component == "All Components":
        st.info("Select a specific component to view detailed metrics")
        return
    
    try:
        summary = monitor.get_performance_summary(component, hours)
        if "error" in summary:
            st.warning("No data available")
            return
        
        metrics = summary.get("metrics_by_type", {})
        
        # Create detailed metrics visualization
        for metric_type, data in metrics.items():
            st.subheader(metric_type.replace("_", " ").title())
            
            col1, col2, col3 = st.columns(3)
            
            with col1:
                st.metric("Mean", f"{data['mean']:.3f}")
            
            with col2:
                st.metric("Min", f"{data['min']:.3f}")
            
            with col3:
                st.metric("Max", f"{data['max']:.3f}")
            
            # Create distribution chart
            if data['count'] > 1:
                # Get actual values for histogram
                actual_metrics = monitor.metrics_collector.get_metrics(
                    component=component,
                    metric_type=MetricType(metric_type),
                    start_time=datetime.now() - timedelta(hours=hours),
                    limit=1000
                )
                
                if actual_metrics:
                    values = [m.value for m in actual_metrics]
                    fig = px.histogram(x=values, title=f"{metric_type} Distribution")
                    fig.update_layout(height=200)
                    st.plotly_chart(fig, use_container_width=True)
    
    except Exception as e:
        st.error(f"Error loading detailed metrics: {e}")


def create_recommendations_panel():
    """Create recommendations panel"""
    try:
        recommendations = get_recommendations(implemented=False)
        
        if not recommendations:
            st.success("✅ No pending recommendations")
            return
        
        # Sort by priority
        recommendations.sort(key=lambda x: x.priority, reverse=True)
        
        for rec in recommendations:
            priority_color = {
                1: "🔴",
                2: "🟡",
                3: "🟢"
            }.get(rec.priority, "⚪")
            
            with st.expander(f"{priority_color} {rec.recommendation_type.replace('_', ' ').title()}"):
                st.write(f"**Component:** {rec.component}")
                st.write(f"**Description:** {rec.description}")
                st.write(f"**Expected Impact:** {rec.expected_impact}")
                st.write(f"**Priority:** {rec.priority}")
                st.write(f"**Generated:** {rec.timestamp.strftime('%Y-%m-%d %H:%M:%S')}")
                
                if st.button(f"Mark as Implemented", key=f"implement_{rec.id}"):
                    # Mark as implemented
                    monitor = get_ai_monitor()
                    with monitor.metrics_collector.lock:
                        with sqlite3.connect(monitor.db_path) as conn:
                            conn.execute("""
                                UPDATE recommendations 
                                SET implemented = 1, implementation_time = ?
                                WHERE id = ?
                            """, (datetime.now().isoformat(), rec.id))
                    
                    st.success("Recommendation marked as implemented!")
                    st.rerun()
    
    except Exception as e:
        st.error(f"Error loading recommendations: {e}")


def create_realtime_metrics(monitor, component: str):
    """Create real-time metrics section"""
    try:
        # Get recent metrics (last 10 minutes)
        recent_metrics = monitor.metrics_collector.get_metrics(
            component=component if component != "All Components" else None,
            start_time=datetime.now() - timedelta(minutes=10),
            limit=100
        )
        
        if not recent_metrics:
            st.info("No recent metrics available")
            return
        
        # Create real-time chart
        chart_data = []
        for metric in recent_metrics:
            chart_data.append({
                "Timestamp": metric.timestamp,
                "Value": metric.value,
                "Metric Type": metric.metric_type.value,
                "Component": metric.component
            })
        
        df = pd.DataFrame(chart_data)
        df = df.sort_values("Timestamp")
        
        fig = px.scatter(df, x="Timestamp", y="Value", color="Metric Type",
                        title="Real-time Metrics (Last 10 Minutes)",
                        hover_data=["Component"])
        fig.update_layout(height=400)
        st.plotly_chart(fig, use_container_width=True)
        
        # Auto-refresh
        st.empty()
        time.sleep(5)
        st.rerun()
    
    except Exception as e:
        st.error(f"Error loading real-time metrics: {e}")


def main():
    """Main function to run the dashboard"""
    create_dashboard()


if __name__ == "__main__":
    main()
