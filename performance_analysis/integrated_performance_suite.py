#!/usr/bin/env python3
"""
Integrated Performance Suite for Signal Analyzer Application
==========================================================

Comprehensive performance monitoring system that combines all monitoring modules
into a unified, easy-to-use interface with automated analysis and reporting.

Features:
    - Unified monitoring of all performance aspects
    - Automated startup and coordination of all monitors
    - Comprehensive dashboard with performance scoring
    - Integrated analysis with cross-component insights
    - Performance regression detection
    - Automated optimization recommendations
    - System health monitoring
    - Beautiful HTML dashboard with real-time metrics
    - Performance baseline establishment
    - Alert system for critical performance issues

Usage:
    # Option 1: Complete automated monitoring
    from performance_analysis.integrated_performance_suite import IntegratedPerformanceSuite
    
    suite = IntegratedPerformanceSuite()
    suite.start_comprehensive_monitoring(duration=300)
    # ... run your application ...
    suite.stop_monitoring()
    
    # Option 2: Context manager
    with IntegratedPerformanceSuite() as suite:
        # Your application code
        run_signal_analyzer()
        
    # Option 3: Custom configuration
    suite = IntegratedPerformanceSuite(
        enable_gui_monitoring=True,
        enable_memory_monitoring=True,
        enable_io_monitoring=True,
        monitor_duration=600
    )
    suite.start_monitoring('comprehensive')
"""

import time
import threading
import os
import json
import logging
import psutil
import gc
from collections import defaultdict, deque
from datetime import datetime, timedelta
from pathlib import Path
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import seaborn as sns
import numpy as np
from typing import Dict, List, Any, Optional, Callable, Union
import contextlib
import webbrowser
import warnings

# Configure matplotlib and seaborn
plt.switch_backend('Agg')
sns.set_style("whitegrid")
warnings.filterwarnings('ignore', category=UserWarning, module='matplotlib')

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class PerformanceAlert:
    """Represents a performance alert with severity and recommendations."""
    
    def __init__(self, alert_type: str, severity: str, message: str, 
                 component: str, metric_value: float = None, threshold: float = None):
        self.alert_type = alert_type
        self.severity = severity  # 'critical', 'warning', 'info'
        self.message = message
        self.component = component
        self.metric_value = metric_value
        self.threshold = threshold
        self.timestamp = datetime.now()
        self.acknowledged = False
        
    def to_dict(self):
        """Convert alert to dictionary."""
        return {
            'alert_type': self.alert_type,
            'severity': self.severity,
            'message': self.message,
            'component': self.component,
            'metric_value': self.metric_value,
            'threshold': self.threshold,
            'timestamp': self.timestamp.isoformat(),
            'acknowledged': self.acknowledged
        }


class PerformanceScorer:
    """Calculates performance scores for different components."""
    
    @staticmethod
    def calculate_overall_score(component_scores: Dict[str, float]) -> float:
        """Calculate weighted overall performance score."""
        weights = {
            'memory': 0.25,
            'cpu': 0.20,
            'io': 0.20,
            'gui': 0.15,
            'function_calls': 0.10,
            'threads': 0.10
        }
        
        weighted_score = 0
        total_weight = 0
        
        for component, score in component_scores.items():
            weight = weights.get(component, 0.1)
            weighted_score += score * weight
            total_weight += weight
            
        return weighted_score / total_weight if total_weight > 0 else 0
        
    @staticmethod
    def score_memory_performance(memory_data: Dict) -> float:
        """Score memory performance (0-100)."""
        score = 100
        
        # Memory usage percentage
        if 'memory_usage_percent' in memory_data:
            usage = memory_data['memory_usage_percent']
            if usage > 90:
                score -= 40
            elif usage > 80:
                score -= 25
            elif usage > 70:
                score -= 10
                
        # Memory leaks detected
        if memory_data.get('memory_leaks_detected', 0) > 0:
            score -= 30
            
        # Memory growth rate
        growth_rate = memory_data.get('memory_growth_rate_mb_per_hour', 0)
        if growth_rate > 100:  # > 100MB/hour
            score -= 25
        elif growth_rate > 50:
            score -= 15
            
        return max(0, score)
        
    @staticmethod
    def score_cpu_performance(cpu_data: Dict) -> float:
        """Score CPU performance (0-100)."""
        score = 100
        
        # CPU usage percentage
        if 'cpu_usage_percent' in cpu_data:
            usage = cpu_data['cpu_usage_percent']
            if usage > 95:
                score -= 30
            elif usage > 85:
                score -= 20
            elif usage > 75:
                score -= 10
                
        # CPU spikes
        spikes = cpu_data.get('cpu_spikes_detected', 0)
        if spikes > 10:
            score -= 20
        elif spikes > 5:
            score -= 10
            
        return max(0, score)
        
    @staticmethod
    def score_io_performance(io_data: Dict) -> float:
        """Score I/O performance (0-100)."""
        score = 100
        
        # Slow operations
        slow_ops = io_data.get('slow_operations_count', 0)
        total_ops = io_data.get('total_operations', 1)
        slow_percentage = (slow_ops / total_ops) * 100
        
        if slow_percentage > 20:
            score -= 40
        elif slow_percentage > 10:
            score -= 25
        elif slow_percentage > 5:
            score -= 15
            
        # Transfer rate
        avg_rate = io_data.get('avg_transfer_rate_mbps', 0)
        if avg_rate < 1:  # < 1 MB/s
            score -= 25
        elif avg_rate < 5:
            score -= 15
            
        # Cache hit rate
        cache_rate = io_data.get('cache_hit_rate_percent', 0)
        if cache_rate < 50:
            score -= 20
        elif cache_rate < 70:
            score -= 10
            
        return max(0, score)
        
    @staticmethod
    def score_gui_performance(gui_data: Dict) -> float:
        """Score GUI performance (0-100)."""
        score = 100
        
        # Frame rate
        frame_rate = gui_data.get('average_frame_rate', 60)
        if frame_rate < 15:
            score -= 50
        elif frame_rate < 30:
            score -= 30
        elif frame_rate < 45:
            score -= 15
            
        # UI freezes
        freezes = gui_data.get('ui_freezes_detected', 0)
        if freezes > 5:
            score -= 40
        elif freezes > 2:
            score -= 25
        elif freezes > 0:
            score -= 15
            
        # Slow events percentage
        slow_percentage = gui_data.get('slow_events_percentage', 0)
        if slow_percentage > 15:
            score -= 30
        elif slow_percentage > 10:
            score -= 20
        elif slow_percentage > 5:
            score -= 10
            
        return max(0, score)


class IntegratedPerformanceSuite:
    """Comprehensive performance monitoring suite."""
    
    def __init__(self, 
                 monitor_duration=300,
                 enable_gui_monitoring=True,
                 enable_memory_monitoring=True,
                 enable_io_monitoring=True,
                 enable_function_tracing=True,
                 enable_thread_monitoring=True,
                 auto_report=True,
                 alert_thresholds=None):
        """
        Initialize integrated performance suite.
        
        Args:
            monitor_duration: Default monitoring duration in seconds
            enable_gui_monitoring: Enable GUI responsiveness monitoring
            enable_memory_monitoring: Enable memory leak detection
            enable_io_monitoring: Enable I/O performance monitoring
            enable_function_tracing: Enable function call tracing
            enable_thread_monitoring: Enable thread performance monitoring
            auto_report: Automatically generate reports on completion
            alert_thresholds: Custom alert thresholds
        """
        self.monitor_duration = monitor_duration
        self.enable_gui_monitoring = enable_gui_monitoring
        self.enable_memory_monitoring = enable_memory_monitoring
        self.enable_io_monitoring = enable_io_monitoring
        self.enable_function_tracing = enable_function_tracing
        self.enable_thread_monitoring = enable_thread_monitoring
        self.auto_report = auto_report
        
        # Alert thresholds
        self.alert_thresholds = alert_thresholds or {
            'memory_usage_percent': 85,
            'cpu_usage_percent': 90,
            'memory_growth_rate_mb_per_hour': 100,
            'slow_io_operations_percent': 15,
            'ui_freeze_count': 3,
            'function_avg_time_ms': 100
        }
        
        # Monitor instances
        self.monitors = {}
        self.active_monitors = []
        
        # Performance data
        self.performance_data = {}
        self.performance_scores = {}
        self.performance_alerts = deque(maxlen=1000)
        self.performance_baseline = {}
        
        # Monitoring state
        self.monitoring_active = False
        self.start_time = None
        self.coordinator_thread = None
        
        # Reports
        self.last_report_path = None
        
        # Thread safety
        self.lock = threading.Lock()
        
        logger.info("Integrated Performance Suite initialized")
        
    def start_comprehensive_monitoring(self, duration=None):
        """
        Start comprehensive monitoring of all enabled components.
        
        Args:
            duration: Monitoring duration in seconds (uses default if None)
        """
        if self.monitoring_active:
            logger.warning("Monitoring already active")
            return
            
        duration = duration or self.monitor_duration
        
        logger.info(f"🚀 Starting comprehensive performance monitoring (duration: {duration}s)")
        
        try:
            self.monitoring_active = True
            self.start_time = time.time()
            
            # Initialize all monitors
            self._initialize_monitors()
            
            # Start all enabled monitors
            self._start_all_monitors()
            
            # Start coordination thread
            self.coordinator_thread = threading.Thread(
                target=self._coordination_loop,
                args=(duration,),
                name="PerformanceCoordinator",
                daemon=True
            )
            self.coordinator_thread.start()
            
            logger.info("✅ Comprehensive monitoring started successfully")
            
        except Exception as e:
            logger.error(f"❌ Failed to start comprehensive monitoring: {e}")
            self.stop_monitoring()
            raise
            
    def _initialize_monitors(self):
        """Initialize all monitoring components."""
        try:
            # Section monitor (always enabled)
            from .monitors import _global_monitor
            self.monitors['section'] = _global_monitor
            
            # Memory leak detector
            if self.enable_memory_monitoring:
                from .memory_leak_detector import MemoryLeakDetector
                self.monitors['memory'] = MemoryLeakDetector()
                
            # GUI responsiveness monitor
            if self.enable_gui_monitoring:
                from .gui_responsiveness_monitor import GUIResponsivenessMonitor
                self.monitors['gui'] = GUIResponsivenessMonitor()
                
            # I/O performance monitor
            if self.enable_io_monitoring:
                from .io_performance_monitor import IOPerformanceMonitor
                self.monitors['io'] = IOPerformanceMonitor()
                
            # Function call tracer
            if self.enable_function_tracing:
                from .function_call_tracer import FunctionCallTracer
                self.monitors['function'] = FunctionCallTracer()
                
            # Thread performance monitor
            if self.enable_thread_monitoring:
                from .thread_performance_monitor import ThreadPerformanceMonitor
                self.monitors['thread'] = ThreadPerformanceMonitor()
                
            # Performance analyzer (system-wide)
            from .performance_analyzer import AdvancedPerformanceAnalyzer
            self.monitors['system'] = AdvancedPerformanceAnalyzer(
                monitor_duration=self.monitor_duration
            )
            
            logger.info(f"Initialized {len(self.monitors)} monitoring components")
            
        except ImportError as e:
            logger.warning(f"Some monitoring modules not available: {e}")
            
    def _start_all_monitors(self):
        """Start all initialized monitors."""
        for name, monitor in self.monitors.items():
            try:
                if hasattr(monitor, 'start_monitoring'):
                    if name == 'memory':
                        monitor.start_monitoring(self.monitor_duration)
                    else:
                        monitor.start_monitoring()
                elif hasattr(monitor, 'start_tracing'):
                    monitor.start_tracing()
                    
                self.active_monitors.append(name)
                logger.debug(f"Started {name} monitor")
                
            except Exception as e:
                logger.error(f"Failed to start {name} monitor: {e}")
                
    def _coordination_loop(self, duration):
        """Main coordination loop for monitoring."""
        logger.debug("Performance coordination loop started")
        
        end_time = time.time() + duration
        sample_interval = 5  # Sample every 5 seconds
        
        while self.monitoring_active and time.time() < end_time:
            try:
                # Collect performance data from all monitors
                self._collect_performance_data()
                
                # Calculate performance scores
                self._calculate_performance_scores()
                
                # Check for alerts
                self._check_performance_alerts()
                
                # Sleep until next sample
                time.sleep(sample_interval)
                
            except Exception as e:
                logger.error(f"Error in coordination loop: {e}")
                time.sleep(sample_interval)
                
        # Auto-stop monitoring when duration reached
        if self.monitoring_active:
            logger.info("⏰ Monitoring duration reached, stopping automatically")
            self.stop_monitoring()
            
        logger.debug("Performance coordination loop ended")
        
    def _collect_performance_data(self):
        """Collect performance data from all active monitors."""
        current_data = {
            'timestamp': time.time(),
            'monitoring_duration': time.time() - self.start_time
        }
        
        # Collect system metrics
        try:
            process = psutil.Process()
            current_data['system'] = {
                'cpu_percent': process.cpu_percent(),
                'memory_percent': process.memory_percent(),
                'memory_mb': process.memory_info().rss / (1024 * 1024),
                'num_threads': process.num_threads(),
                'num_fds': process.num_fds() if hasattr(process, 'num_fds') else 0
            }
        except:
            pass
            
        # Collect data from each monitor
        for name, monitor in self.monitors.items():
            try:
                if hasattr(monitor, 'get_performance_summary'):
                    monitor_data = monitor.get_performance_summary()
                    if monitor_data:
                        current_data[name] = monitor_data
                        
            except Exception as e:
                logger.debug(f"Error collecting data from {name} monitor: {e}")
                
        with self.lock:
            self.performance_data[current_data['timestamp']] = current_data
            
    def _calculate_performance_scores(self):
        """Calculate performance scores for all components."""
        if not self.performance_data:
            return
            
        # Get latest performance data
        latest_timestamp = max(self.performance_data.keys())
        latest_data = self.performance_data[latest_timestamp]
        
        scores = {}
        
        # Calculate component scores
        if 'system' in latest_data:
            system_data = latest_data['system']
            scores['cpu'] = PerformanceScorer.score_cpu_performance({
                'cpu_usage_percent': system_data.get('cpu_percent', 0)
            })
            scores['memory'] = PerformanceScorer.score_memory_performance({
                'memory_usage_percent': system_data.get('memory_percent', 0)
            })
            
        if 'memory' in latest_data:
            scores['memory'] = PerformanceScorer.score_memory_performance(latest_data['memory'])
            
        if 'io' in latest_data:
            scores['io'] = PerformanceScorer.score_io_performance(latest_data['io'])
            
        if 'gui' in latest_data:
            scores['gui'] = PerformanceScorer.score_gui_performance(latest_data['gui'])
            
        # Calculate overall score
        scores['overall'] = PerformanceScorer.calculate_overall_score(scores)
        
        with self.lock:
            self.performance_scores[latest_timestamp] = scores
            
    def _check_performance_alerts(self):
        """Check for performance issues and generate alerts."""
        if not self.performance_data:
            return
            
        latest_timestamp = max(self.performance_data.keys())
        latest_data = self.performance_data[latest_timestamp]
        latest_scores = self.performance_scores.get(latest_timestamp, {})
        
        alerts = []
        
        # Check system metrics
        if 'system' in latest_data:
            system = latest_data['system']
            
            # Memory usage alert
            memory_percent = system.get('memory_percent', 0)
            if memory_percent > self.alert_thresholds['memory_usage_percent']:
                alerts.append(PerformanceAlert(
                    'high_memory_usage',
                    'critical' if memory_percent > 95 else 'warning',
                    f'High memory usage: {memory_percent:.1f}%',
                    'system',
                    memory_percent,
                    self.alert_thresholds['memory_usage_percent']
                ))
                
            # CPU usage alert
            cpu_percent = system.get('cpu_percent', 0)
            if cpu_percent > self.alert_thresholds['cpu_usage_percent']:
                alerts.append(PerformanceAlert(
                    'high_cpu_usage',
                    'critical' if cpu_percent > 98 else 'warning',
                    f'High CPU usage: {cpu_percent:.1f}%',
                    'system',
                    cpu_percent,
                    self.alert_thresholds['cpu_usage_percent']
                ))
                
        # Check overall performance score
        overall_score = latest_scores.get('overall', 100)
        if overall_score < 50:
            alerts.append(PerformanceAlert(
                'low_performance_score',
                'critical' if overall_score < 30 else 'warning',
                f'Low overall performance score: {overall_score:.1f}/100',
                'overall',
                overall_score,
                50
            ))
            
        # Add alerts to queue
        with self.lock:
            for alert in alerts:
                self.performance_alerts.append(alert)
                logger.warning(f"🚨 Performance Alert: {alert.message}")
                
    def stop_monitoring(self):
        """Stop all performance monitoring."""
        if not self.monitoring_active:
            logger.info("No active monitoring to stop")
            return
            
        logger.info("🛑 Stopping comprehensive performance monitoring...")
        
        self.monitoring_active = False
        
        # Stop coordination thread
        if self.coordinator_thread and self.coordinator_thread.is_alive():
            self.coordinator_thread.join(timeout=5.0)
            
        # Stop all monitors
        for name, monitor in self.monitors.items():
            try:
                if hasattr(monitor, 'stop_monitoring'):
                    monitor.stop_monitoring()
                elif hasattr(monitor, 'stop_tracing'):
                    monitor.stop_tracing()
                    
            except Exception as e:
                logger.error(f"Error stopping {name} monitor: {e}")
                
        # Generate final report if enabled
        if self.auto_report:
            try:
                self.generate_comprehensive_report()
            except Exception as e:
                logger.error(f"Error generating final report: {e}")
                
        logger.info("✅ Comprehensive monitoring stopped")
        
    def generate_comprehensive_report(self):
        """Generate comprehensive performance report combining all monitors."""
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        
        # Create reports directory
        reports_dir = Path('performance_reports')
        reports_dir.mkdir(exist_ok=True)
        
        logger.info("📊 Generating comprehensive performance report...")
        
        # Generate JSON report
        json_report = self._generate_comprehensive_json_report()
        json_file = reports_dir / f'comprehensive_performance_report_{timestamp}.json'
        with open(json_file, 'w') as f:
            json.dump(json_report, f, indent=2, default=str)
            
        # Generate HTML dashboard
        html_report = self._generate_performance_dashboard()
        html_file = reports_dir / f'performance_dashboard_{timestamp}.html'
        with open(html_file, 'w') as f:
            f.write(html_report)
            
        # Generate comprehensive visualizations
        self._generate_comprehensive_visualizations(timestamp)
        
        # Generate individual monitor reports
        individual_reports = {}
        for name, monitor in self.monitors.items():
            try:
                if hasattr(monitor, 'generate_report'):
                    report_info = monitor.generate_report()
                    individual_reports[name] = report_info
                elif hasattr(monitor, 'generate_comprehensive_report'):
                    report_info = monitor.generate_comprehensive_report()
                    individual_reports[name] = report_info
            except Exception as e:
                logger.warning(f"Failed to generate {name} report: {e}")
                
        self.last_report_path = str(html_file)
        
        logger.info(f"📁 Comprehensive reports generated:")
        logger.info(f"  Main Dashboard: {html_file}")
        logger.info(f"  JSON Report: {json_file}")
        logger.info(f"  Individual Reports: {len(individual_reports)} generated")
        
        return {
            'dashboard': str(html_file),
            'json_report': str(json_file),
            'individual_reports': individual_reports,
            'timestamp': timestamp
        }
        
    def _generate_comprehensive_json_report(self):
        """Generate comprehensive JSON report."""
        with self.lock:
            # Calculate final scores
            final_scores = {}
            if self.performance_scores:
                latest_scores = list(self.performance_scores.values())[-1]
                final_scores = latest_scores
                
            # Calculate monitoring statistics
            monitoring_duration = time.time() - self.start_time if self.start_time else 0
            
            # Collect all monitor summaries
            monitor_summaries = {}
            for name, monitor in self.monitors.items():
                try:
                    if hasattr(monitor, 'get_performance_summary'):
                        summary = monitor.get_performance_summary()
                        if summary:
                            monitor_summaries[name] = summary
                except Exception as e:
                    logger.debug(f"Error getting summary from {name}: {e}")
                    
            # Performance trend analysis
            performance_trends = self._analyze_performance_trends()
            
            report = {
                'metadata': {
                    'timestamp': datetime.now().isoformat(),
                    'monitoring_duration': monitoring_duration,
                    'monitors_enabled': self.active_monitors,
                    'alert_thresholds': self.alert_thresholds
                },
                'performance_scores': {
                    'final_scores': final_scores,
                    'score_history': [
                        {'timestamp': ts, 'scores': scores}
                        for ts, scores in self.performance_scores.items()
                    ]
                },
                'monitor_summaries': monitor_summaries,
                'performance_alerts': [
                    alert.to_dict() for alert in self.performance_alerts
                ],
                'performance_trends': performance_trends,
                'system_information': self._get_system_information(),
                'optimization_recommendations': self._generate_comprehensive_recommendations(),
                'performance_baseline': self.performance_baseline
            }
            
        return report
        
    def _generate_performance_dashboard(self):
        """Generate comprehensive HTML performance dashboard."""
        timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        
        # Get latest performance data
        with self.lock:
            latest_scores = {}
            if self.performance_scores:
                latest_scores = list(self.performance_scores.values())[-1]
                
            monitoring_duration = time.time() - self.start_time if self.start_time else 0
            
            # Count alerts by severity
            alert_counts = {'critical': 0, 'warning': 0, 'info': 0}
            for alert in self.performance_alerts:
                alert_counts[alert.severity] = alert_counts.get(alert.severity, 0) + 1
                
        overall_score = latest_scores.get('overall', 0)
        score_color = self._get_score_color(overall_score)
        
        html = f"""
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Performance Dashboard - Signal Analyzer</title>
    <style>
        * {{ margin: 0; padding: 0; box-sizing: border-box; }}
        
        body {{ 
            font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', 'Roboto', sans-serif;
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            min-height: 100vh;
            padding: 20px;
        }}
        
        .dashboard {{ 
            max-width: 1600px; 
            margin: 0 auto; 
            background: white; 
            border-radius: 20px;
            box-shadow: 0 25px 80px rgba(0,0,0,0.15);
            overflow: hidden;
        }}
        
        .header {{ 
            background: linear-gradient(135deg, #1e3c72 0%, #2a5298 100%);
            color: white; 
            padding: 40px 50px; 
            position: relative;
            overflow: hidden;
        }}
        
        .header::before {{
            content: '';
            position: absolute;
            top: -50%;
            right: -20%;
            width: 100%;
            height: 200%;
            background: radial-gradient(circle, rgba(255,255,255,0.1) 0%, transparent 60%);
            animation: float 8s ease-in-out infinite;
        }}
        
        @keyframes float {{
            0%, 100% {{ transform: translateY(0px) rotate(0deg); }}
            50% {{ transform: translateY(-30px) rotate(180deg); }}
        }}
        
        .header h1 {{ 
            font-size: 3.5em; 
            margin-bottom: 15px; 
            font-weight: 700;
            position: relative;
            z-index: 1;
        }}
        
        .header .subtitle {{ 
            font-size: 1.3em; 
            opacity: 0.9; 
            position: relative;
            z-index: 1;
        }}
        
        .content {{ padding: 50px; }}
        
        .score-section {{
            text-align: center;
            margin: 40px 0 60px 0;
            padding: 40px;
            background: linear-gradient(135deg, #f8f9ff 0%, #e8f4fd 100%);
            border-radius: 20px;
            border: 1px solid #e3f2fd;
        }}
        
        .overall-score {{
            font-size: 6em;
            font-weight: bold;
            color: {score_color};
            text-shadow: 2px 2px 4px rgba(0,0,0,0.1);
            margin: 20px 0;
        }}
        
        .score-label {{
            font-size: 1.8em;
            color: #666;
            margin-bottom: 20px;
        }}
        
        .score-status {{
            font-size: 1.3em;
            font-weight: 600;
            color: {score_color};
            background: rgba({self._hex_to_rgb(score_color)}, 0.1);
            padding: 15px 30px;
            border-radius: 50px;
            display: inline-block;
            border: 2px solid {score_color};
        }}
        
        .metrics-grid {{ 
            display: grid; 
            grid-template-columns: repeat(auto-fit, minmax(300px, 1fr)); 
            gap: 30px; 
            margin: 50px 0;
        }}
        
        .metric-card {{ 
            background: white;
            padding: 35px; 
            border-radius: 20px; 
            text-align: center;
            border: 1px solid #e0e0e0;
            transition: all 0.3s ease;
            position: relative;
            overflow: hidden;
        }}
        
        .metric-card::before {{
            content: '';
            position: absolute;
            top: 0;
            left: -100%;
            width: 100%;
            height: 100%;
            background: linear-gradient(90deg, transparent, rgba(255,255,255,0.6), transparent);
            transition: left 0.6s;
        }}
        
        .metric-card:hover {{
            transform: translateY(-8px);
            box-shadow: 0 15px 40px rgba(0,0,0,0.1);
        }}
        
        .metric-card:hover::before {{
            left: 100%;
        }}
        
        .metric-icon {{ 
            font-size: 4em; 
            margin-bottom: 20px; 
            display: block;
        }}
        
        .metric-value {{ 
            font-size: 3em; 
            font-weight: bold; 
            margin-bottom: 15px;
        }}
        
        .metric-label {{ 
            font-size: 1.2em; 
            color: #666; 
            font-weight: 500;
        }}
        
        .component-scores {{
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
            gap: 25px;
            margin: 40px 0;
        }}
        
        .component-score {{
            background: white;
            padding: 25px;
            border-radius: 15px;
            text-align: center;
            border-left: 5px solid;
            box-shadow: 0 3px 15px rgba(0,0,0,0.05);
        }}
        
        .component-score.excellent {{ border-left-color: #4caf50; }}
        .component-score.good {{ border-left-color: #8bc34a; }}
        .component-score.fair {{ border-left-color: #ff9800; }}
        .component-score.poor {{ border-left-color: #f44336; }}
        
        .component-score h3 {{
            margin-bottom: 15px;
            color: #333;
            font-size: 1.1em;
        }}
        
        .component-score .score {{
            font-size: 2.5em;
            font-weight: bold;
            margin-bottom: 10px;
        }}
        
        .component-score.excellent .score {{ color: #4caf50; }}
        .component-score.good .score {{ color: #8bc34a; }}
        .component-score.fair .score {{ color: #ff9800; }}
        .component-score.poor .score {{ color: #f44336; }}
        
        .alerts-section {{
            background: linear-gradient(135deg, #fff3e0 0%, #ffe0b2 100%);
            padding: 30px;
            border-radius: 15px;
            margin: 40px 0;
            border-left: 5px solid #ff9800;
        }}
        
        .alert-summary {{
            display: flex;
            justify-content: space-around;
            margin: 20px 0;
        }}
        
        .alert-count {{
            text-align: center;
            padding: 15px 25px;
            border-radius: 10px;
            background: white;
            box-shadow: 0 2px 10px rgba(0,0,0,0.1);
        }}
        
        .alert-count.critical {{ border-left: 4px solid #f44336; }}
        .alert-count.warning {{ border-left: 4px solid #ff9800; }}
        .alert-count.info {{ border-left: 4px solid #2196f3; }}
        
        .chart-grid {{
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(500px, 1fr));
            gap: 40px;
            margin: 50px 0;
        }}
        
        .chart-container {{ 
            background: white;
            padding: 30px; 
            border-radius: 15px;
            box-shadow: 0 5px 20px rgba(0,0,0,0.05);
            text-align: center;
        }}
        
        .chart-container h3 {{ 
            color: #1976d2; 
            margin-bottom: 25px; 
            font-size: 1.5em;
        }}
        
        .chart-container img {{ 
            max-width: 100%; 
            height: auto; 
            border-radius: 10px;
        }}
        
        .recommendations {{
            background: linear-gradient(135deg, #e8f5e8 0%, #c8e6c9 100%);
            padding: 40px;
            border-radius: 15px;
            border-left: 5px solid #4caf50;
            margin: 40px 0;
        }}
        
        .recommendations h2 {{
            color: #2e7d32;
            margin-bottom: 25px;
            font-size: 1.8em;
        }}
        
        .recommendations ul {{
            list-style: none;
            padding: 0;
        }}
        
        .recommendations li {{
            margin: 15px 0;
            padding: 20px;
            background: rgba(255,255,255,0.8);
            border-radius: 10px;
            border-left: 3px solid #4caf50;
            line-height: 1.6;
            transition: all 0.2s ease;
        }}
        
        .recommendations li:hover {{
            background: rgba(255,255,255,0.95);
            transform: translateX(5px);
        }}
        
        .system-info {{
            background: linear-gradient(135deg, #f3e5f5 0%, #e1bee7 100%);
            padding: 30px;
            border-radius: 15px;
            margin: 40px 0;
            border-left: 5px solid #9c27b0;
        }}
        
        .system-info-grid {{
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(250px, 1fr));
            gap: 20px;
            margin-top: 20px;
        }}
        
        .system-metric {{
            background: rgba(255,255,255,0.7);
            padding: 15px 20px;
            border-radius: 8px;
            display: flex;
            justify-content: space-between;
            align-items: center;
        }}
        
        .badge {{
            display: inline-block;
            padding: 5px 12px;
            border-radius: 15px;
            font-size: 0.85em;
            font-weight: 600;
            text-transform: uppercase;
            letter-spacing: 0.5px;
        }}
        
        .badge.excellent {{ background: #4caf50; color: white; }}
        .badge.good {{ background: #8bc34a; color: white; }}
        .badge.fair {{ background: #ff9800; color: white; }}
        .badge.poor {{ background: #f44336; color: white; }}
        
        @media (max-width: 768px) {{
            .header {{ padding: 30px 25px; }}
            .header h1 {{ font-size: 2.5em; }}
            .content {{ padding: 25px; }}
            .metrics-grid {{ grid-template-columns: 1fr; }}
            .overall-score {{ font-size: 4em; }}
        }}
    </style>
</head>
<body>
    <div class="dashboard">
        <div class="header">
            <h1>🎯 Performance Dashboard</h1>
            <p class="subtitle">Comprehensive Performance Analysis for Signal Analyzer</p>
            <p>Generated: {timestamp} | Duration: {monitoring_duration/60:.1f} minutes</p>
        </div>
        
        <div class="content">
            <div class="score-section">
                <div class="score-label">Overall Performance Score</div>
                <div class="overall-score">{overall_score:.0f}</div>
                <div class="score-status">{self._get_score_status(overall_score)}</div>
            </div>
            
            <div class="component-scores">
                {self._generate_component_scores_html(latest_scores)}
            </div>
            
            <div class="metrics-grid">
                <div class="metric-card">
                    <span class="metric-icon">⏱️</span>
                    <div class="metric-value">{monitoring_duration/60:.1f}</div>
                    <div class="metric-label">Monitoring Duration (minutes)</div>
                </div>
                <div class="metric-card">
                    <span class="metric-icon">📊</span>
                    <div class="metric-value">{len(self.active_monitors)}</div>
                    <div class="metric-label">Active Monitors</div>
                </div>
                <div class="metric-card">
                    <span class="metric-icon">🚨</span>
                    <div class="metric-value">{len(self.performance_alerts)}</div>
                    <div class="metric-label">Performance Alerts</div>
                </div>
                <div class="metric-card">
                    <span class="metric-icon">📈</span>
                    <div class="metric-value">{len(self.performance_data)}</div>
                    <div class="metric-label">Data Points Collected</div>
                </div>
            </div>
            
            {self._generate_alerts_section_html(alert_counts) if self.performance_alerts else ''}
            
            <div class="chart-grid">
                <div class="chart-container">
                    <h3>📈 Performance Score Timeline</h3>
                    <img src="performance_score_timeline_{datetime.now().strftime('%Y%m%d_%H%M%S')}.png" 
                         alt="Performance Score Timeline">
                </div>
                <div class="chart-container">
                    <h3>🔍 Component Performance Breakdown</h3>
                    <img src="component_performance_radar_{datetime.now().strftime('%Y%m%d_%H%M%S')}.png" 
                         alt="Component Performance Radar">
                </div>
                <div class="chart-container">
                    <h3>💾 System Resource Usage</h3>
                    <img src="system_resource_usage_{datetime.now().strftime('%Y%m%d_%H%M%S')}.png" 
                         alt="System Resource Usage">
                </div>
                <div class="chart-container">
                    <h3>🚨 Alert Timeline</h3>
                    <img src="alert_timeline_{datetime.now().strftime('%Y%m%d_%H%M%S')}.png" 
                         alt="Alert Timeline">
                </div>
            </div>
            
            <div class="system-info">
                <h2>🖥️ System Information</h2>
                {self._generate_system_info_html()}
            </div>
            
            <div class="recommendations">
                <h2>🎯 Performance Optimization Recommendations</h2>
                <ul>
                    {''.join(f'<li>{rec}</li>' for rec in self._generate_comprehensive_recommendations())}
                </ul>
            </div>
        </div>
    </div>
</body>
</html>
        """
        
        return html
        
    def _get_score_color(self, score):
        """Get color for performance score."""
        if score >= 80:
            return '#4caf50'  # Green
        elif score >= 60:
            return '#8bc34a'  # Light green
        elif score >= 40:
            return '#ff9800'  # Orange
        else:
            return '#f44336'  # Red
            
    def _get_score_status(self, score):
        """Get status text for performance score."""
        if score >= 80:
            return 'Excellent Performance'
        elif score >= 60:
            return 'Good Performance'
        elif score >= 40:
            return 'Fair Performance'
        else:
            return 'Poor Performance'
            
    def _get_score_class(self, score):
        """Get CSS class for performance score."""
        if score >= 80:
            return 'excellent'
        elif score >= 60:
            return 'good'
        elif score >= 40:
            return 'fair'
        else:
            return 'poor'
            
    def _hex_to_rgb(self, hex_color):
        """Convert hex color to RGB values."""
        hex_color = hex_color.lstrip('#')
        return ','.join(str(int(hex_color[i:i+2], 16)) for i in (0, 2, 4))
        
    def _generate_component_scores_html(self, scores):
        """Generate HTML for component scores."""
        components = {
            'memory': '💾 Memory',
            'cpu': '🔥 CPU',
            'io': '💾 I/O',
            'gui': '🖥️ GUI',
            'function': '⚡ Functions',
            'thread': '🧵 Threads'
        }
        
        html = ""
        for component, label in components.items():
            score = scores.get(component, 0)
            score_class = self._get_score_class(score)
            
            html += f'''
            <div class="component-score {score_class}">
                <h3>{label}</h3>
                <div class="score">{score:.0f}</div>
                <div class="badge {score_class}">{self._get_score_status(score).split()[0]}</div>
            </div>
            '''
            
        return html
        
    def _generate_alerts_section_html(self, alert_counts):
        """Generate HTML for alerts section."""
        if not any(alert_counts.values()):
            return ""
            
        html = f'''
        <div class="alerts-section">
            <h2>🚨 Performance Alerts Summary</h2>
            <div class="alert-summary">
                <div class="alert-count critical">
                    <h3>{alert_counts['critical']}</h3>
                    <p>Critical</p>
                </div>
                <div class="alert-count warning">
                    <h3>{alert_counts['warning']}</h3>
                    <p>Warning</p>
                </div>
                <div class="alert-count info">
                    <h3>{alert_counts['info']}</h3>
                    <p>Info</p>
                </div>
            </div>
        </div>
        '''
        
        return html
        
    def _generate_system_info_html(self):
        """Generate HTML for system information."""
        system_info = self._get_system_information()
        
        html = '<div class="system-info-grid">'
        
        for key, value in system_info.items():
            display_key = key.replace('_', ' ').title()
            html += f'''
            <div class="system-metric">
                <span><strong>{display_key}:</strong></span>
                <span>{value}</span>
            </div>
            '''
            
        html += '</div>'
        return html
        
    def _get_system_information(self):
        """Get comprehensive system information."""
        try:
            import platform
            
            # Get system info
            system_info = {
                'platform': platform.platform(),
                'processor': platform.processor() or 'Unknown',
                'python_version': platform.python_version(),
                'architecture': platform.architecture()[0],
            }
            
            # Get process info
            try:
                process = psutil.Process()
                system_info.update({
                    'pid': process.pid,
                    'memory_mb': f"{process.memory_info().rss / (1024*1024):.1f} MB",
                    'cpu_percent': f"{process.cpu_percent():.1f}%",
                    'num_threads': process.num_threads(),
                })
            except:
                pass
                
            # Get system resources
            try:
                system_info.update({
                    'cpu_count': psutil.cpu_count(),
                    'memory_total_gb': f"{psutil.virtual_memory().total / (1024**3):.1f} GB",
                    'disk_usage_percent': f"{psutil.disk_usage('/').percent:.1f}%"
                })
            except:
                pass
                
            return system_info
            
        except Exception as e:
            logger.debug(f"Error getting system info: {e}")
            return {'error': 'Unable to retrieve system information'}
            
    def _analyze_performance_trends(self):
        """Analyze performance trends over time."""
        if len(self.performance_scores) < 2:
            return {}
            
        # Calculate trends for each component
        timestamps = sorted(self.performance_scores.keys())
        trends = {}
        
        for component in ['overall', 'memory', 'cpu', 'io', 'gui']:
            scores = [self.performance_scores[ts].get(component, 0) for ts in timestamps]
            
            if len(scores) >= 2:
                # Simple linear trend calculation
                x = np.arange(len(scores))
                y = np.array(scores)
                
                if len(x) > 1:
                    slope = np.polyfit(x, y, 1)[0]
                    trends[component] = {
                        'trend': 'improving' if slope > 1 else 'declining' if slope < -1 else 'stable',
                        'slope': float(slope),
                        'start_score': float(scores[0]),
                        'end_score': float(scores[-1]),
                        'change': float(scores[-1] - scores[0])
                    }
                    
        return trends
        
    def _generate_comprehensive_recommendations(self):
        """Generate comprehensive optimization recommendations."""
        recommendations = []
        
        # Get latest performance data
        with self.lock:
            if not self.performance_scores:
                return ["No performance data available for recommendations"]
                
            latest_scores = list(self.performance_scores.values())[-1]
            
        # Analyze component scores and generate recommendations
        if latest_scores.get('overall', 100) < 70:
            recommendations.append(
                "🎯 <strong>Overall Performance:</strong> Performance score is below optimal. "
                "Focus on the lowest-scoring components first."
            )
            
        if latest_scores.get('memory', 100) < 60:
            recommendations.append(
                "💾 <strong>Memory Optimization:</strong> Implement memory pooling, "
                "reduce object creation in loops, and consider using generators for large datasets."
            )
            
        if latest_scores.get('cpu', 100) < 60:
            recommendations.append(
                "🔥 <strong>CPU Optimization:</strong> Profile CPU-intensive functions, "
                "consider multi-threading for parallel operations, and optimize algorithms."
            )
            
        if latest_scores.get('io', 100) < 60:
            recommendations.append(
                "💾 <strong>I/O Optimization:</strong> Implement async I/O operations, "
                "use file caching for frequently accessed data, and optimize file access patterns."
            )
            
        if latest_scores.get('gui', 100) < 60:
            recommendations.append(
                "🖥️ <strong>GUI Optimization:</strong> Reduce plot complexity, "
                "implement progressive rendering, and move heavy operations to background threads."
            )
            
        # Check alert patterns
        with self.lock:
            critical_alerts = [a for a in self.performance_alerts if a.severity == 'critical']
            
        if critical_alerts:
            alert_types = set(a.alert_type for a in critical_alerts)
            recommendations.append(
                f"🚨 <strong>Critical Issues:</strong> Address {len(critical_alerts)} critical alerts: "
                f"{', '.join(alert_types)}"
            )
            
        # General recommendations
        if len(recommendations) < 3:
            recommendations.extend([
                "📊 <strong>Monitoring:</strong> Continue regular performance monitoring to catch regressions early",
                "🔧 <strong>Profiling:</strong> Use detailed profiling tools to identify specific bottlenecks",
                "⚡ <strong>Caching:</strong> Implement intelligent caching strategies for computed results",
                "🧵 <strong>Concurrency:</strong> Consider using multiprocessing for CPU-bound operations"
            ])
            
        return recommendations[:8]  # Limit to 8 recommendations
        
    def _generate_comprehensive_visualizations(self, timestamp):
        """Generate comprehensive performance visualizations."""
        try:
            # Create main visualization figure
            fig = plt.figure(figsize=(20, 16))
            gs = fig.add_gridspec(3, 3, hspace=0.3, wspace=0.3)
            
            with self.lock:
                # 1. Performance Score Timeline
                ax1 = fig.add_subplot(gs[0, :2])
                if self.performance_scores:
                    timestamps = list(self.performance_scores.keys())
                    start_time = min(timestamps)
                    times = [(ts - start_time) / 60 for ts in timestamps]  # Minutes
                    
                    scores = {
                        'Overall': [scores.get('overall', 0) for scores in self.performance_scores.values()],
                        'Memory': [scores.get('memory', 0) for scores in self.performance_scores.values()],
                        'CPU': [scores.get('cpu', 0) for scores in self.performance_scores.values()],
                        'I/O': [scores.get('io', 0) for scores in self.performance_scores.values()],
                        'GUI': [scores.get('gui', 0) for scores in self.performance_scores.values()]
                    }
                    
                    colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd']
                    for i, (component, score_list) in enumerate(scores.items()):
                        if any(s > 0 for s in score_list):
                            ax1.plot(times, score_list, 'o-', color=colors[i], 
                                   label=component, linewidth=2, markersize=4)
                    
                    ax1.set_xlabel('Time (minutes)')
                    ax1.set_ylabel('Performance Score')
                    ax1.set_title('Performance Score Timeline', fontsize=14, fontweight='bold')
                    ax1.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
                    ax1.grid(True, alpha=0.3)
                    ax1.set_ylim(0, 100)
                
                # 2. Component Performance Radar Chart
                ax2 = fig.add_subplot(gs[0, 2], projection='polar')
                if self.performance_scores:
                    latest_scores = list(self.performance_scores.values())[-1]
                    components = ['Memory', 'CPU', 'I/O', 'GUI', 'Functions']
                    scores = [latest_scores.get(comp.lower(), 0) for comp in components]
                    
                    # Add the first score at the end to close the radar chart
                    scores.append(scores[0])
                    
                    angles = np.linspace(0, 2 * np.pi, len(components), endpoint=False).tolist()
                    angles.append(angles[0])
                    
                    ax2.plot(angles, scores, 'o-', linewidth=2, color='blue')
                    ax2.fill(angles, scores, alpha=0.25, color='blue')
                    ax2.set_xticks(angles[:-1])
                    ax2.set_xticklabels(components)
                    ax2.set_ylim(0, 100)
                    ax2.set_title('Component Performance Radar', fontsize=12, fontweight='bold', y=1.08)
                    ax2.grid(True)
                
                # 3. System Resource Usage
                ax3 = fig.add_subplot(gs[1, 0])
                if self.performance_data:
                    timestamps = list(self.performance_data.keys())
                    start_time = min(timestamps)
                    times = [(ts - start_time) / 60 for ts in timestamps]
                    
                    memory_usage = []
                    cpu_usage = []
                    
                    for ts in timestamps:
                        data = self.performance_data[ts]
                        if 'system' in data:
                            memory_usage.append(data['system'].get('memory_percent', 0))
                            cpu_usage.append(data['system'].get('cpu_percent', 0))
                        else:
                            memory_usage.append(0)
                            cpu_usage.append(0)
                    
                    ax3.plot(times, memory_usage, 'o-', color='red', label='Memory %', linewidth=2)
                    ax3.plot(times, cpu_usage, 'o-', color='blue', label='CPU %', linewidth=2)
                    ax3.set_xlabel('Time (minutes)')
                    ax3.set_ylabel('Usage (%)')
                    ax3.set_title('System Resource Usage', fontsize=12, fontweight='bold')
                    ax3.legend()
                    ax3.grid(True, alpha=0.3)
                    ax3.set_ylim(0, 100)
                
                # 4. Alert Timeline
                ax4 = fig.add_subplot(gs[1, 1])
                if self.performance_alerts:
                    alert_times = []
                    alert_severities = []
                    severity_colors = {'critical': 'red', 'warning': 'orange', 'info': 'blue'}
                    
                    start_time = self.start_time if self.start_time else time.time()
                    
                    for alert in self.performance_alerts:
                        alert_time = (alert.timestamp.timestamp() - start_time) / 60
                        alert_times.append(alert_time)
                        alert_severities.append(alert.severity)
                    
                    for severity in ['critical', 'warning', 'info']:
                        severity_times = [t for t, s in zip(alert_times, alert_severities) if s == severity]
                        severity_values = [{'critical': 3, 'warning': 2, 'info': 1}[severity]] * len(severity_times)
                        
                        if severity_times:
                            ax4.scatter(severity_times, severity_values, 
                                      color=severity_colors[severity], 
                                      label=severity.title(), s=50, alpha=0.7)
                    
                    ax4.set_xlabel('Time (minutes)')
                    ax4.set_ylabel('Alert Severity')
                    ax4.set_title('Alert Timeline', fontsize=12, fontweight='bold')
                    ax4.set_yticks([1, 2, 3])
                    ax4.set_yticklabels(['Info', 'Warning', 'Critical'])
                    ax4.legend()
                    ax4.grid(True, alpha=0.3)
                else:
                    ax4.text(0.5, 0.5, 'No Alerts Generated\n🎉', 
                           ha='center', va='center', transform=ax4.transAxes, 
                           fontsize=16, bbox=dict(boxstyle="round,pad=0.3", facecolor="lightgreen"))
                    ax4.set_title('Alert Timeline', fontsize=12, fontweight='bold')
                
                # 5. Monitor Activity Distribution
                ax5 = fig.add_subplot(gs[1, 2])
                if self.monitors:
                    monitor_names = list(self.monitors.keys())
                    monitor_counts = [1] * len(monitor_names)  # Simplified - could be activity counts
                    
                    colors = plt.cm.Set3(np.linspace(0, 1, len(monitor_names)))
                    wedges, texts, autotexts = ax5.pie(monitor_counts, labels=monitor_names, 
                                                     autopct='%1.0f%%', colors=colors, startangle=90)
                    ax5.set_title('Active Monitors', fontsize=12, fontweight='bold')
                    
                    # Improve text readability
                    for autotext in autotexts:
                        autotext.set_color('white')
                        autotext.set_fontweight('bold')
                
                # 6. Performance Distribution Histogram
                ax6 = fig.add_subplot(gs[2, 0])
                if self.performance_scores:
                    all_overall_scores = [scores.get('overall', 0) 
                                        for scores in self.performance_scores.values()]
                    
                    ax6.hist(all_overall_scores, bins=20, alpha=0.7, color='skyblue', 
                           edgecolor='black', density=True)
                    ax6.axvline(np.mean(all_overall_scores), color='red', linestyle='--', 
                              label=f'Mean: {np.mean(all_overall_scores):.1f}')
                    ax6.set_xlabel('Performance Score')
                    ax6.set_ylabel('Density')
                    ax6.set_title('Performance Score Distribution', fontsize=12, fontweight='bold')
                    ax6.legend()
                    ax6.grid(True, alpha=0.3)
                
                # 7. Component Score Comparison
                ax7 = fig.add_subplot(gs[2, 1])
                if self.performance_scores:
                    latest_scores = list(self.performance_scores.values())[-1]
                    components = ['memory', 'cpu', 'io', 'gui']
                    scores = [latest_scores.get(comp, 0) for comp in components]
                    
                    colors = ['#ff9999', '#66b3ff', '#99ff99', '#ffcc99']
                    bars = ax7.bar(components, scores, color=colors, alpha=0.8)
                    
                    # Add value labels on bars
                    for bar, score in zip(bars, scores):
                        height = bar.get_height()
                        ax7.text(bar.get_x() + bar.get_width()/2., height + 1,
                               f'{score:.0f}', ha='center', va='bottom', fontweight='bold')
                    
                    ax7.set_ylabel('Performance Score')
                    ax7.set_title('Current Component Scores', fontsize=12, fontweight='bold')
                    ax7.set_ylim(0, 105)
                    ax7.grid(True, alpha=0.3, axis='y')
                
                # 8. Trend Analysis
                ax8 = fig.add_subplot(gs[2, 2])
                trends = self._analyze_performance_trends()
                if trends:
                    components = list(trends.keys())
                    changes = [trends[comp]['change'] for comp in components]
                    
                    colors = ['green' if c > 0 else 'red' if c < 0 else 'gray' for c in changes]
                    bars = ax8.barh(components, changes, color=colors, alpha=0.7)
                    
                    ax8.set_xlabel('Score Change')
                    ax8.set_title('Performance Trends', fontsize=12, fontweight='bold')
                    ax8.axvline(0, color='black', linestyle='-', alpha=0.5)
                    ax8.grid(True, alpha=0.3, axis='x')
                    
                    # Add value labels
                    for bar, change in zip(bars, changes):
                        width = bar.get_width()
                        ax8.text(width + (1 if width >= 0 else -1), bar.get_y() + bar.get_height()/2,
                               f'{change:+.1f}', ha='left' if width >= 0 else 'right', 
                               va='center', fontweight='bold')
            
            plt.suptitle('Signal Analyzer - Comprehensive Performance Analysis', 
                        fontsize=18, fontweight='bold', y=0.98)
            
            # Save the comprehensive visualization
            plot_file = f'performance_reports/comprehensive_performance_analysis_{timestamp}.png'
            plt.savefig(plot_file, dpi=300, bbox_inches='tight', facecolor='white')
            plt.close()
            
            # Generate individual chart files for the dashboard
            self._generate_individual_charts(timestamp)
            
            logger.info(f"Comprehensive visualizations saved to: {plot_file}")
            
        except Exception as e:
            logger.error(f"Error generating comprehensive visualizations: {e}")
            
    def _generate_individual_charts(self, timestamp):
        """Generate individual chart files for the dashboard."""
        try:
            with self.lock:
                # Performance Score Timeline
                plt.figure(figsize=(12, 6))
                if self.performance_scores:
                    timestamps = list(self.performance_scores.keys())
                    start_time = min(timestamps)
                    times = [(ts - start_time) / 60 for ts in timestamps]
                    
                    overall_scores = [scores.get('overall', 0) for scores in self.performance_scores.values()]
                    plt.plot(times, overall_scores, 'o-', color='blue', linewidth=3, markersize=6)
                    plt.fill_between(times, overall_scores, alpha=0.3, color='blue')
                    
                plt.xlabel('Time (minutes)', fontsize=12)
                plt.ylabel('Performance Score', fontsize=12)
                plt.title('Performance Score Over Time', fontsize=14, fontweight='bold')
                plt.grid(True, alpha=0.3)
                plt.ylim(0, 100)
                plt.tight_layout()
                plt.savefig(f'performance_reports/performance_score_timeline_{timestamp}.png', 
                          dpi=300, bbox_inches='tight')
                plt.close()
                
                # Component Performance Radar
                fig, ax = plt.subplots(figsize=(8, 8), subplot_kw=dict(projection='polar'))
                if self.performance_scores:
                    latest_scores = list(self.performance_scores.values())[-1]
                    components = ['Memory', 'CPU', 'I/O', 'GUI', 'Functions']
                    scores = [latest_scores.get(comp.lower(), 0) for comp in components]
                    scores.append(scores[0])  # Close the radar
                    
                    angles = np.linspace(0, 2 * np.pi, len(components), endpoint=False).tolist()
                    angles.append(angles[0])
                    
                    ax.plot(angles, scores, 'o-', linewidth=3, color='green', markersize=8)
                    ax.fill(angles, scores, alpha=0.25, color='green')
                    ax.set_xticks(angles[:-1])
                    ax.set_xticklabels(components, fontsize=12)
                    ax.set_ylim(0, 100)
                    ax.grid(True)
                    
                plt.title('Component Performance Breakdown', fontsize=14, fontweight='bold', y=1.08)
                plt.tight_layout()
                plt.savefig(f'performance_reports/component_performance_radar_{timestamp}.png', 
                          dpi=300, bbox_inches='tight')
                plt.close()
                
                # System Resource Usage
                plt.figure(figsize=(12, 6))
                if self.performance_data:
                    timestamps = list(self.performance_data.keys())
                    start_time = min(timestamps)
                    times = [(ts - start_time) / 60 for ts in timestamps]
                    
                    memory_usage = []
                    cpu_usage = []
                    
                    for ts in timestamps:
                        data = self.performance_data[ts]
                        if 'system' in data:
                            memory_usage.append(data['system'].get('memory_percent', 0))
                            cpu_usage.append(data['system'].get('cpu_percent', 0))
                        else:
                            memory_usage.append(0)
                            cpu_usage.append(0)
                    
                    plt.plot(times, memory_usage, 'o-', color='red', label='Memory Usage %', linewidth=2)
                    plt.plot(times, cpu_usage, 'o-', color='blue', label='CPU Usage %', linewidth=2)
                    
                plt.xlabel('Time (minutes)', fontsize=12)
                plt.ylabel('Usage (%)', fontsize=12)
                plt.title('System Resource Usage', fontsize=14, fontweight='bold')
                plt.legend(fontsize=12)
                plt.grid(True, alpha=0.3)
                plt.ylim(0, 100)
                plt.tight_layout()
                plt.savefig(f'performance_reports/system_resource_usage_{timestamp}.png', 
                          dpi=300, bbox_inches='tight')
                plt.close()
                
                # Alert Timeline
                plt.figure(figsize=(12, 6))
                if self.performance_alerts:
                    alert_times = []
                    alert_severities = []
                    severity_colors = {'critical': 'red', 'warning': 'orange', 'info': 'blue'}
                    
                    start_time = self.start_time if self.start_time else time.time()
                    
                    for alert in self.performance_alerts:
                        alert_time = (alert.timestamp.timestamp() - start_time) / 60
                        alert_times.append(alert_time)
                        alert_severities.append(alert.severity)
                    
                    for severity in ['critical', 'warning', 'info']:
                        severity_times = [t for t, s in zip(alert_times, alert_severities) if s == severity]
                        severity_values = [{'critical': 3, 'warning': 2, 'info': 1}[severity]] * len(severity_times)
                        
                        if severity_times:
                            plt.scatter(severity_times, severity_values, 
                                      color=severity_colors[severity], 
                                      label=severity.title(), s=100, alpha=0.7)
                    
                    plt.xlabel('Time (minutes)', fontsize=12)
                    plt.ylabel('Alert Severity', fontsize=12)
                    plt.yticks([1, 2, 3], ['Info', 'Warning', 'Critical'])
                    plt.legend(fontsize=12)
                    plt.grid(True, alpha=0.3)
                else:
                    plt.text(0.5, 0.5, 'No Performance Alerts\nGenerated! 🎉', 
                           ha='center', va='center', transform=plt.gca().transAxes, 
                           fontsize=20, bbox=dict(boxstyle="round,pad=0.5", facecolor="lightgreen"))
                    
                plt.title('Performance Alert Timeline', fontsize=14, fontweight='bold')
                plt.tight_layout()
                plt.savefig(f'performance_reports/alert_timeline_{timestamp}.png', 
                          dpi=300, bbox_inches='tight')
                plt.close()
                
        except Exception as e:
            logger.error(f"Error generating individual charts: {e}")
            
    def __enter__(self):
        """Context manager entry."""
        self.start_comprehensive_monitoring()
        return self
        
    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit."""
        self.stop_monitoring()
        
    def open_dashboard(self):
        """Open the last generated dashboard in web browser."""
        if self.last_report_path and Path(self.last_report_path).exists():
            try:
                webbrowser.open(f'file://{Path(self.last_report_path).absolute()}')
                logger.info(f"Opened dashboard in browser: {self.last_report_path}")
            except Exception as e:
                logger.error(f"Failed to open dashboard: {e}")
        else:
            logger.warning("No dashboard available to open")
            
    def get_performance_summary(self):
        """Get current performance summary."""
        with self.lock:
            if not self.performance_scores:
                return {}
                
            latest_scores = list(self.performance_scores.values())[-1]
            monitoring_duration = time.time() - self.start_time if self.start_time else 0
            
            return {
                'overall_score': latest_scores.get('overall', 0),
                'component_scores': latest_scores,
                'monitoring_duration': monitoring_duration,
                'active_monitors': len(self.active_monitors),
                'alerts_generated': len(self.performance_alerts),
                'critical_alerts': len([a for a in self.performance_alerts if a.severity == 'critical']),
                'data_points_collected': len(self.performance_data)
            }


# Convenience functions for easy access
def start_comprehensive_monitoring(duration=300, **kwargs):
    """Start comprehensive performance monitoring with default settings."""
    suite = IntegratedPerformanceSuite(monitor_duration=duration, **kwargs)
    suite.start_comprehensive_monitoring()
    return suite

def quick_performance_check(duration=60):
    """Quick performance check with minimal overhead."""
    suite = IntegratedPerformanceSuite(
        monitor_duration=duration,
        enable_function_tracing=False,  # Disable for quick check
        enable_thread_monitoring=False
    )
    
    with suite:
        time.sleep(duration)
        
    return suite.get_performance_summary()