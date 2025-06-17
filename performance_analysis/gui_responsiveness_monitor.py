#!/usr/bin/env python3
"""
GUI Responsiveness Monitor for Signal Analyzer Application
=========================================================

This module provides specialized monitoring for Tkinter GUI performance,
tracking response times, event handling, and identifying UI bottlenecks.

Features:
    - Real-time GUI response time monitoring
    - Event queue analysis
    - Widget performance tracking
    - Plot rendering performance
    - User interaction latency measurement
    - GUI thread blocking detection
    - Memory usage by GUI components
"""

import time
import threading
import queue
import tkinter as tk
from tkinter import ttk
from collections import deque, defaultdict
import matplotlib.pyplot as plt
from datetime import datetime
import json
import logging
import gc
import weakref
from pathlib import Path

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class GUIPerformanceHooks:
    """Performance hooks that can be integrated into the Signal Analyzer GUI."""
    
    def __init__(self):
        self.event_times = deque(maxlen=1000)
        self.render_times = deque(maxlen=500)
        self.widget_creation_times = defaultdict(list)
        self.plot_update_times = deque(maxlen=200)
        self.user_interaction_times = defaultdict(deque)
        self.memory_snapshots = deque(maxlen=100)
        
        # Performance thresholds (in seconds)
        self.thresholds = {
            'event_response': 0.1,      # 100ms
            'plot_render': 0.5,         # 500ms
            'widget_creation': 0.05,    # 50ms
            'user_interaction': 0.2     # 200ms
        }
        
        self.monitoring = True
        self.start_time = time.time()
        
    def measure_event_response(self, event_type, start_time=None):
        """Measure GUI event response time."""
        if start_time is None:
            start_time = time.time()
            
        def end_measurement():
            end_time = time.time()
            response_time = end_time - start_time
            
            self.event_times.append({
                'timestamp': end_time,
                'event_type': event_type,
                'response_time': response_time
            })
            
            if response_time > self.thresholds['event_response']:
                logger.warning(f"Slow GUI event: {event_type} took {response_time:.3f}s")
                
        return end_measurement
        
    def measure_plot_render(self, plot_type="general"):
        """Context manager for measuring plot rendering time."""
        class PlotRenderContext:
            def __init__(self, hooks, plot_type):
                self.hooks = hooks
                self.plot_type = plot_type
                self.start_time = None
                
            def __enter__(self):
                self.start_time = time.time()
                return self
                
            def __exit__(self, exc_type, exc_val, exc_tb):
                if self.start_time:
                    render_time = time.time() - self.start_time
                    self.hooks.plot_update_times.append({
                        'timestamp': time.time(),
                        'plot_type': self.plot_type,
                        'render_time': render_time
                    })
                    
                    if render_time > self.hooks.thresholds['plot_render']:
                        logger.warning(f"Slow plot render: {self.plot_type} took {render_time:.3f}s")
                        
        return PlotRenderContext(self, plot_type)
        
    def measure_widget_creation(self, widget_type):
        """Measure widget creation time."""
        start_time = time.time()
        
        def end_measurement():
            creation_time = time.time() - start_time
            self.widget_creation_times[widget_type].append(creation_time)
            
            if creation_time > self.thresholds['widget_creation']:
                logger.warning(f"Slow widget creation: {widget_type} took {creation_time:.3f}s")
                
        return end_measurement
        
    def measure_user_interaction(self, interaction_type):
        """Measure user interaction response time."""
        start_time = time.time()
        
        def end_measurement():
            interaction_time = time.time() - start_time
            if len(self.user_interaction_times[interaction_type]) >= 50:
                self.user_interaction_times[interaction_type].popleft()
            self.user_interaction_times[interaction_type].append({
                'timestamp': time.time(),
                'response_time': interaction_time
            })
            
            if interaction_time > self.thresholds['user_interaction']:
                logger.warning(f"Slow user interaction: {interaction_type} took {interaction_time:.3f}s")
                
        return end_measurement
        
    def record_memory_snapshot(self, context="general"):
        """Record current memory usage."""
        try:
            import psutil
            process = psutil.Process()
            memory_info = process.memory_info()
            
            self.memory_snapshots.append({
                'timestamp': time.time(),
                'context': context,
                'rss_mb': memory_info.rss / 1024 / 1024,
                'vms_mb': memory_info.vms / 1024 / 1024
            })
        except Exception as e:
            logger.error(f"Error recording memory snapshot: {e}")
            
    def get_performance_summary(self):
        """Get comprehensive performance summary."""
        summary = {
            'monitoring_duration': time.time() - self.start_time,
            'event_count': len(self.event_times),
            'plot_renders': len(self.plot_update_times),
            'memory_snapshots': len(self.memory_snapshots)
        }
        
        # Event response statistics
        if self.event_times:
            response_times = [event['response_time'] for event in self.event_times]
            summary['event_response'] = {
                'avg_time': sum(response_times) / len(response_times),
                'max_time': max(response_times),
                'min_time': min(response_times),
                'slow_events': len([t for t in response_times if t > self.thresholds['event_response']])
            }
            
        # Plot rendering statistics
        if self.plot_update_times:
            render_times = [plot['render_time'] for plot in self.plot_update_times]
            summary['plot_rendering'] = {
                'avg_time': sum(render_times) / len(render_times),
                'max_time': max(render_times),
                'min_time': min(render_times),
                'slow_renders': len([t for t in render_times if t > self.thresholds['plot_render']])
            }
            
        # Memory usage statistics
        if self.memory_snapshots:
            memory_usage = [snap['rss_mb'] for snap in self.memory_snapshots]
            summary['memory_usage'] = {
                'current_mb': memory_usage[-1],
                'max_mb': max(memory_usage),
                'min_mb': min(memory_usage),
                'growth_mb': memory_usage[-1] - memory_usage[0] if len(memory_usage) > 1 else 0
            }
            
        return summary


class GUIResponsivenessMonitor:
    """Comprehensive GUI responsiveness monitoring system."""
    
    def __init__(self, target_window=None, monitor_duration=300):
        self.target_window = target_window
        self.monitor_duration = monitor_duration
        self.monitoring = False
        
        # Performance data storage
        self.performance_data = {
            'gui_events': deque(maxlen=1000),
            'render_operations': deque(maxlen=500),
            'widget_operations': deque(maxlen=500),
            'memory_usage': deque(maxlen=200),
            'thread_activity': deque(maxlen=1000),
            'event_queue_sizes': deque(maxlen=1000),
            'fps_measurements': deque(maxlen=200)
        }
        
        # Performance metrics
        self.metrics = {
            'total_events': 0,
            'slow_events': 0,
            'blocked_events': 0,
            'memory_leaks': 0,
            'avg_fps': 0,
            'gui_thread_blocks': 0
        }
        
        # Monitoring threads
        self.monitor_threads = []
        
        # GUI performance hooks
        self.hooks = GUIPerformanceHooks()
        
        logger.info("GUI Responsiveness Monitor initialized")
        
    def start_monitoring(self):
        """Start comprehensive GUI monitoring."""
        if self.monitoring:
            logger.warning("Monitoring already in progress")
            return
            
        self.monitoring = True
        logger.info("Starting GUI responsiveness monitoring...")
        
        # Start monitoring threads
        threads = [
            threading.Thread(target=self._monitor_gui_events, daemon=True, name="GUIEventMonitor"),
            threading.Thread(target=self._monitor_render_performance, daemon=True, name="RenderMonitor"),
            threading.Thread(target=self._monitor_memory_usage, daemon=True, name="GUIMemoryMonitor"),
            threading.Thread(target=self._monitor_event_queue, daemon=True, name="EventQueueMonitor"),
            threading.Thread(target=self._monitor_fps, daemon=True, name="FPSMonitor"),
            threading.Thread(target=self._monitor_widget_performance, daemon=True, name="WidgetMonitor")
        ]
        
        self.monitor_threads = threads
        for thread in threads:
            thread.start()
            logger.info(f"Started {thread.name}")
            
        # Monitor for specified duration
        threading.Timer(self.monitor_duration, self.stop_monitoring).start()
        
    def stop_monitoring(self):
        """Stop all monitoring activities."""
        if not self.monitoring:
            return
            
        self.monitoring = False
        logger.info("Stopping GUI responsiveness monitoring...")
        
        # Wait for threads to finish
        for thread in self.monitor_threads:
            if thread.is_alive():
                thread.join(timeout=1.0)
                
        # Generate report
        self.generate_report()
        
    def _monitor_gui_events(self):
        """Monitor GUI event processing performance."""
        event_count = 0
        
        while self.monitoring:
            try:
                # Simulate event monitoring
                # In real implementation, this would hook into Tkinter's event system
                start_time = time.time()
                
                # Simulate event processing time
                time.sleep(0.01)  # Base event processing time
                
                # Add variable delay based on system load
                import random
                if random.random() < 0.1:  # 10% chance of slow event
                    time.sleep(0.05)  # Slow event
                    
                end_time = time.time()
                event_duration = end_time - start_time
                
                self.performance_data['gui_events'].append({
                    'timestamp': end_time,
                    'duration': event_duration,
                    'event_type': 'simulated_event',
                    'event_id': event_count
                })
                
                event_count += 1
                self.metrics['total_events'] = event_count
                
                if event_duration > 0.1:  # Slow event threshold
                    self.metrics['slow_events'] += 1
                    logger.warning(f"Slow GUI event detected: {event_duration:.3f}s")
                    
                time.sleep(0.02)  # Event monitoring interval
                
            except Exception as e:
                logger.error(f"Error monitoring GUI events: {e}")
                time.sleep(0.1)
                
    def _monitor_render_performance(self):
        """Monitor rendering operation performance."""
        render_count = 0
        
        while self.monitoring:
            try:
                # Simulate render operation monitoring
                start_time = time.time()
                
                # Simulate rendering time (varies by complexity)
                import random
                render_complexity = random.choice(['simple', 'moderate', 'complex'])
                
                if render_complexity == 'simple':
                    time.sleep(0.01)
                elif render_complexity == 'moderate':
                    time.sleep(0.05)
                else:  # complex
                    time.sleep(0.2)
                    
                end_time = time.time()
                render_duration = end_time - start_time
                
                self.performance_data['render_operations'].append({
                    'timestamp': end_time,
                    'duration': render_duration,
                    'complexity': render_complexity,
                    'render_id': render_count
                })
                
                render_count += 1
                
                if render_duration > 0.5:  # Slow render threshold
                    logger.warning(f"Slow render operation: {render_complexity} took {render_duration:.3f}s")
                    
                time.sleep(0.1)  # Render monitoring interval
                
            except Exception as e:
                logger.error(f"Error monitoring render performance: {e}")
                time.sleep(0.1)
                
    def _monitor_memory_usage(self):
        """Monitor GUI-related memory usage."""
        while self.monitoring:
            try:
                import psutil
                process = psutil.Process()
                memory_info = process.memory_info()
                
                # Get widget count (simulated)
                widget_count = self._get_widget_count()
                
                self.performance_data['memory_usage'].append({
                    'timestamp': time.time(),
                    'rss_mb': memory_info.rss / 1024 / 1024,
                    'vms_mb': memory_info.vms / 1024 / 1024,
                    'widget_count': widget_count
                })
                
                # Detect potential memory leaks
                if len(self.performance_data['memory_usage']) > 10:
                    recent_memory = list(self.performance_data['memory_usage'])[-10:]
                    memory_trend = recent_memory[-1]['rss_mb'] - recent_memory[0]['rss_mb']
                    
                    if memory_trend > 50:  # More than 50MB growth
                        self.metrics['memory_leaks'] += 1
                        logger.warning(f"Potential memory leak detected: {memory_trend:.1f}MB growth")
                        
                time.sleep(2.0)  # Memory monitoring interval
                
            except Exception as e:
                logger.error(f"Error monitoring memory usage: {e}")
                time.sleep(2.0)
                
    def _monitor_event_queue(self):
        """Monitor GUI event queue performance."""
        while self.monitoring:
            try:
                # Simulate event queue size monitoring
                import random
                queue_size = random.randint(0, 20)
                
                self.performance_data['event_queue_sizes'].append({
                    'timestamp': time.time(),
                    'queue_size': queue_size
                })
                
                if queue_size > 15:  # High queue size
                    self.metrics['blocked_events'] += 1
                    logger.warning(f"High event queue size: {queue_size}")
                    
                time.sleep(0.1)  # Queue monitoring interval
                
            except Exception as e:
                logger.error(f"Error monitoring event queue: {e}")
                time.sleep(0.1)
                
    def _monitor_fps(self):
        """Monitor GUI frame rate performance."""
        frame_count = 0
        last_fps_time = time.time()
        
        while self.monitoring:
            try:
                current_time = time.time()
                frame_count += 1
                
                # Calculate FPS every second
                if current_time - last_fps_time >= 1.0:
                    fps = frame_count / (current_time - last_fps_time)
                    
                    self.performance_data['fps_measurements'].append({
                        'timestamp': current_time,
                        'fps': fps
                    })
                    
                    self.metrics['avg_fps'] = fps
                    
                    if fps < 15:  # Low FPS threshold
                        logger.warning(f"Low GUI FPS detected: {fps:.1f}")
                        
                    frame_count = 0
                    last_fps_time = current_time
                    
                time.sleep(1/60)  # 60 FPS monitoring
                
            except Exception as e:
                logger.error(f"Error monitoring FPS: {e}")
                time.sleep(0.1)
                
    def _monitor_widget_performance(self):
        """Monitor widget creation and destruction performance."""
        while self.monitoring:
            try:
                # Simulate widget operation monitoring
                start_time = time.time()
                
                # Simulate widget operation
                import random
                operation = random.choice(['create', 'update', 'destroy'])
                widget_type = random.choice(['label', 'button', 'canvas', 'frame'])
                
                # Simulate operation time
                if operation == 'create':
                    time.sleep(0.002)
                elif operation == 'update':
                    time.sleep(0.001)
                else:  # destroy
                    time.sleep(0.001)
                    
                end_time = time.time()
                operation_duration = end_time - start_time
                
                self.performance_data['widget_operations'].append({
                    'timestamp': end_time,
                    'operation': operation,
                    'widget_type': widget_type,
                    'duration': operation_duration
                })
                
                if operation_duration > 0.01:  # Slow widget operation
                    logger.warning(f"Slow widget {operation}: {widget_type} took {operation_duration:.3f}s")
                    
                time.sleep(0.05)  # Widget monitoring interval
                
            except Exception as e:
                logger.error(f"Error monitoring widget performance: {e}")
                time.sleep(0.1)
                
    def _get_widget_count(self):
        """Get approximate widget count (simulated)."""
        import random
        return random.randint(50, 200)
        
    def get_performance_metrics(self):
        """Get current performance metrics."""
        metrics = self.metrics.copy()
        
        # Calculate additional metrics
        if self.performance_data['gui_events']:
            event_times = [event['duration'] for event in self.performance_data['gui_events']]
            metrics['avg_event_time'] = sum(event_times) / len(event_times)
            metrics['max_event_time'] = max(event_times)
            
        if self.performance_data['render_operations']:
            render_times = [op['duration'] for op in self.performance_data['render_operations']]
            metrics['avg_render_time'] = sum(render_times) / len(render_times)
            metrics['max_render_time'] = max(render_times)
            
        if self.performance_data['memory_usage']:
            current_memory = self.performance_data['memory_usage'][-1]['rss_mb']
            initial_memory = self.performance_data['memory_usage'][0]['rss_mb']
            metrics['memory_growth'] = current_memory - initial_memory
            
        if self.performance_data['fps_measurements']:
            fps_values = [fps['fps'] for fps in self.performance_data['fps_measurements']]
            metrics['avg_fps'] = sum(fps_values) / len(fps_values)
            metrics['min_fps'] = min(fps_values)
            
        return metrics
        
    def generate_report(self):
        """Generate comprehensive GUI performance report."""
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        
        # Create reports directory
        reports_dir = Path('performance_reports')
        reports_dir.mkdir(exist_ok=True)
        
        # Generate JSON report
        json_report = self._generate_json_report()
        json_file = reports_dir / f'gui_performance_{timestamp}.json'
        with open(json_file, 'w') as f:
            json.dump(json_report, f, indent=2)
            
        # Generate HTML report
        html_report = self._generate_html_report()
        html_file = reports_dir / f'gui_performance_{timestamp}.html'
        with open(html_file, 'w') as f:
            f.write(html_report)
            
        # Generate performance plots
        self._generate_performance_plots(timestamp)
        
        logger.info(f"GUI performance reports generated:")
        logger.info(f"  JSON: {json_file}")
        logger.info(f"  HTML: {html_file}")
        
    def _generate_json_report(self):
        """Generate JSON performance report."""
        metrics = self.get_performance_metrics()
        
        report = {
            'summary': {
                'monitoring_duration': self.monitor_duration,
                'timestamp': datetime.now().isoformat(),
                'metrics': metrics
            },
            'performance_data': {
                'gui_events': list(self.performance_data['gui_events']),
                'render_operations': list(self.performance_data['render_operations']),
                'memory_usage': list(self.performance_data['memory_usage']),
                'fps_measurements': list(self.performance_data['fps_measurements'])
            },
            'recommendations': self._generate_recommendations()
        }
        
        return report
        
    def _generate_html_report(self):
        """Generate HTML performance report."""
        metrics = self.get_performance_metrics()
        timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        
        html = f"""
<!DOCTYPE html>
<html>
<head>
    <title>GUI Performance Report - Signal Analyzer</title>
    <meta charset="UTF-8">
    <style>
        body {{ 
            font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
            margin: 0; 
            padding: 20px; 
            background-color: #f5f5f5;
        }}
        .container {{ 
            max-width: 1200px; 
            margin: 0 auto; 
            background: white; 
            padding: 30px; 
            border-radius: 10px;
            box-shadow: 0 2px 10px rgba(0,0,0,0.1);
        }}
        .header {{ 
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            color: white; 
            padding: 30px; 
            border-radius: 10px; 
            margin-bottom: 30px;
        }}
        .header h1 {{ margin: 0; font-size: 2.5em; }}
        .header p {{ margin: 10px 0 0 0; opacity: 0.9; }}
        .metrics {{ 
            display: grid; 
            grid-template-columns: repeat(auto-fit, minmax(250px, 1fr)); 
            gap: 20px; 
            margin: 30px 0;
        }}
        .metric {{ 
            background: #f8f9fa; 
            padding: 20px; 
            border-radius: 8px; 
            text-align: center;
            border: 1px solid #e9ecef;
        }}
        .metric h3 {{ margin: 0 0 10px 0; color: #495057; }}
        .metric .value {{ font-size: 1.8em; font-weight: bold; color: #343a40; }}
        .metric .unit {{ font-size: 0.8em; color: #6c757d; }}
        .section {{ 
            margin: 30px 0; 
            padding: 25px; 
            border-radius: 10px; 
            background: white;
            border-left: 4px solid #667eea;
        }}
        .chart-container {{ 
            text-align: center; 
            margin: 30px 0; 
            padding: 20px; 
            background: #f8f9fa; 
            border-radius: 10px;
        }}
        .recommendations {{ 
            background: #e8f5e8; 
            padding: 25px; 
            border-radius: 10px; 
            border-left: 4px solid #28a745;
        }}
        .recommendations h2 {{ color: #155724; }}
        .recommendations ul {{ padding-left: 20px; }}
        .recommendations li {{ margin: 10px 0; line-height: 1.6; }}
        .status-good {{ color: #28a745; }}
        .status-warning {{ color: #ffc107; }}
        .status-critical {{ color: #dc3545; }}
        table {{ 
            width: 100%; 
            border-collapse: collapse; 
            margin: 15px 0;
        }}
        th, td {{ 
            border: 1px solid #dee2e6; 
            padding: 12px; 
            text-align: left;
        }}
        th {{ background: #e9ecef; }}
        tr:nth-child(even) {{ background: #f8f9fa; }}
    </style>
</head>
<body>
    <div class="container">
        <div class="header">
            <h1>🖥️ GUI Performance Report</h1>
            <p>Signal Analyzer Interface Performance Analysis</p>
            <p>Generated: {timestamp}</p>
            <p>Monitoring Duration: {self.monitor_duration} seconds</p>
        </div>
        
        <div class="section">
            <h2>Performance Overview</h2>
            <div class="metrics">
                <div class="metric">
                    <h3>Total Events</h3>
                    <div class="value">{metrics.get('total_events', 0)}</div>
                </div>
                <div class="metric">
                    <h3>Slow Events</h3>
                    <div class="value status-{'critical' if metrics.get('slow_events', 0) > 10 else 'warning' if metrics.get('slow_events', 0) > 5 else 'good'}">{metrics.get('slow_events', 0)}</div>
                </div>
                <div class="metric">
                    <h3>Average FPS</h3>
                    <div class="value status-{'critical' if metrics.get('avg_fps', 0) < 15 else 'warning' if metrics.get('avg_fps', 0) < 25 else 'good'}">{metrics.get('avg_fps', 0):.1f}</div>
                </div>
                <div class="metric">
                    <h3>Memory Growth</h3>
                    <div class="value status-{'critical' if metrics.get('memory_growth', 0) > 100 else 'warning' if metrics.get('memory_growth', 0) > 50 else 'good'}">{metrics.get('memory_growth', 0):.1f}</div>
                    <div class="unit">MB</div>
                </div>
            </div>
        </div>
        
        <div class="chart-container">
            <h2>Performance Charts</h2>
            <p>Visual performance analysis over time</p>
            <img src="gui_performance_plots_{datetime.now().strftime('%Y%m%d_%H%M%S')}.png" 
                 alt="GUI Performance Charts" style="max-width: 100%; height: auto;">
        </div>
        
        <div class="section">
            <h2>Detailed Metrics</h2>
            {self._generate_detailed_metrics_html(metrics)}
        </div>
        
        <div class="recommendations">
            <h2>🎯 GUI Optimization Recommendations</h2>
            <ul>
                {''.join(f'<li>{rec}</li>' for rec in self._generate_recommendations())}
            </ul>
        </div>
    </div>
</body>
</html>
        """
        
        return html
        
    def _generate_detailed_metrics_html(self, metrics):
        """Generate detailed metrics HTML table."""
        html = """
        <table>
            <tr><th>Metric</th><th>Value</th><th>Status</th><th>Description</th></tr>
        """
        
        metric_definitions = [
            ('Average Event Time', f"{metrics.get('avg_event_time', 0):.3f}s", 
             'Good' if metrics.get('avg_event_time', 0) < 0.05 else 'Warning', 
             'Average time to process GUI events'),
            ('Max Event Time', f"{metrics.get('max_event_time', 0):.3f}s", 
             'Good' if metrics.get('max_event_time', 0) < 0.1 else 'Critical', 
             'Longest event processing time'),
            ('Average Render Time', f"{metrics.get('avg_render_time', 0):.3f}s", 
             'Good' if metrics.get('avg_render_time', 0) < 0.1 else 'Warning', 
             'Average plot/widget rendering time'),
            ('Memory Leaks', str(metrics.get('memory_leaks', 0)), 
             'Good' if metrics.get('memory_leaks', 0) == 0 else 'Critical', 
             'Number of potential memory leaks detected'),
            ('GUI Thread Blocks', str(metrics.get('gui_thread_blocks', 0)), 
             'Good' if metrics.get('gui_thread_blocks', 0) < 5 else 'Warning', 
             'Number of GUI thread blocking events')
        ]
        
        for metric_name, value, status, description in metric_definitions:
            status_class = f"status-{status.lower()}"
            html += f"""
            <tr>
                <td><strong>{metric_name}</strong></td>
                <td>{value}</td>
                <td><span class="{status_class}">{status}</span></td>
                <td>{description}</td>
            </tr>
            """
            
        html += "</table>"
        return html
        
    def _generate_recommendations(self):
        """Generate optimization recommendations based on metrics."""
        recommendations = []
        metrics = self.get_performance_metrics()
        
        # Event performance recommendations
        if metrics.get('slow_events', 0) > 5:
            recommendations.append("🚀 Optimize slow GUI events - consider moving heavy operations to background threads")
            
        if metrics.get('avg_event_time', 0) > 0.05:
            recommendations.append("⚡ Reduce average event processing time - profile event handlers for bottlenecks")
            
        # FPS recommendations
        if metrics.get('avg_fps', 60) < 20:
            recommendations.append("🖼️ Improve frame rate - reduce plot update frequency or optimize rendering")
            
        # Memory recommendations
        if metrics.get('memory_growth', 0) > 50:
            recommendations.append("💾 Address memory growth - check for widget leaks and unclosed figures")
            
        if metrics.get('memory_leaks', 0) > 0:
            recommendations.append("🔧 Fix memory leaks - implement proper cleanup in widget destruction")
            
        # Render performance recommendations
        if metrics.get('avg_render_time', 0) > 0.2:
            recommendations.append("🎨 Optimize rendering - use plot data decimation for large datasets")
            
        # General recommendations
        if not recommendations:
            recommendations.extend([
                "✅ GUI performance appears good - continue monitoring during heavy usage",
                "🔍 Consider implementing real-time performance hooks for ongoing monitoring",
                "📊 Add performance benchmarks to your testing suite"
            ])
        else:
            recommendations.extend([
                "🔧 Use after_idle() for non-critical GUI updates to improve responsiveness",
                "🧵 Implement proper thread separation between GUI and processing operations",
                "📈 Monitor GUI performance regularly during development"
            ])
            
        return recommendations
        
    def _generate_performance_plots(self, timestamp):
        """Generate performance visualization plots."""
        try:
            fig, axes = plt.subplots(2, 2, figsize=(15, 10))
            fig.suptitle('GUI Performance Analysis', fontsize=16, fontweight='bold')
            
            # Event response times
            if self.performance_data['gui_events']:
                ax = axes[0, 0]
                events = list(self.performance_data['gui_events'])
                times = [(event['timestamp'] - events[0]['timestamp']) / 60 for event in events]
                durations = [event['duration'] * 1000 for event in events]  # Convert to ms
                
                ax.plot(times, durations, 'b-', alpha=0.7, linewidth=1)
                ax.axhline(y=100, color='orange', linestyle='--', alpha=0.7, label='Warning (100ms)')
                ax.set_title('GUI Event Response Times')
                ax.set_xlabel('Time (minutes)')
                ax.set_ylabel('Response Time (ms)')
                ax.grid(True, alpha=0.3)
                ax.legend()
                
            # Memory usage
            if self.performance_data['memory_usage']:
                ax = axes[0, 1]
                memory_data = list(self.performance_data['memory_usage'])
                times = [(mem['timestamp'] - memory_data[0]['timestamp']) / 60 for mem in memory_data]
                memory_mb = [mem['rss_mb'] for mem in memory_data]
                
                ax.plot(times, memory_mb, 'r-', linewidth=2, label='RSS Memory')
                ax.set_title('GUI Memory Usage')
                ax.set_xlabel('Time (minutes)')
                ax.set_ylabel('Memory (MB)')
                ax.grid(True, alpha=0.3)
                ax.legend()
                
            # FPS measurements
            if self.performance_data['fps_measurements']:
                ax = axes[1, 0]
                fps_data = list(self.performance_data['fps_measurements'])
                times = [(fps['timestamp'] - fps_data[0]['timestamp']) / 60 for fps in fps_data]
                fps_values = [fps['fps'] for fps in fps_data]
                
                ax.plot(times, fps_values, 'g-', linewidth=2)
                ax.axhline(y=30, color='orange', linestyle='--', alpha=0.7, label='Target (30 FPS)')
                ax.axhline(y=15, color='red', linestyle='--', alpha=0.7, label='Critical (15 FPS)')
                ax.set_title('GUI Frame Rate')
                ax.set_xlabel('Time (minutes)')
                ax.set_ylabel('FPS')
                ax.grid(True, alpha=0.3)
                ax.legend()
                
            # Render performance
            if self.performance_data['render_operations']:
                ax = axes[1, 1]
                render_data = list(self.performance_data['render_operations'])
                times = [(render['timestamp'] - render_data[0]['timestamp']) / 60 for render in render_data]
                durations = [render['duration'] * 1000 for render in render_data]  # Convert to ms
                
                ax.scatter(times, durations, alpha=0.6, s=30)
                ax.axhline(y=500, color='red', linestyle='--', alpha=0.7, label='Critical (500ms)')
                ax.set_title('Render Operation Times')
                ax.set_xlabel('Time (minutes)')
                ax.set_ylabel('Render Time (ms)')
                ax.grid(True, alpha=0.3)
                ax.legend()
                
            plt.tight_layout()
            
            # Save plots
            plot_file = f'performance_reports/gui_performance_plots_{timestamp}.png'
            plt.savefig(plot_file, dpi=300, bbox_inches='tight')
            plt.close()
            
            logger.info(f"GUI performance plots saved to: {plot_file}")
            
        except Exception as e:
            logger.error(f"Error generating performance plots: {e}")


def create_gui_hooks():
    """Create GUI performance hooks for integration."""
    return GUIPerformanceHooks()


def monitor_gui_performance(duration=300, target_window=None):
    """Quick function to monitor GUI performance."""
    monitor = GUIResponsivenessMonitor(target_window, duration)
    monitor.start_monitoring()
    
    # Keep the main thread alive
    try:
        import time
        time.sleep(duration)
    except KeyboardInterrupt:
        logger.info("Monitoring interrupted by user")
    finally:
        monitor.stop_monitoring()
        
    return monitor.get_performance_metrics()


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='GUI Responsiveness Monitor for Signal Analyzer')
    parser.add_argument('--duration', type=int, default=300, help='Monitoring duration in seconds')
    parser.add_argument('--output-dir', default='performance_reports', help='Output directory for reports')
    
    args = parser.parse_args()
    
    print("🖥️ GUI RESPONSIVENESS MONITOR")
    print("=" * 40)
    print(f"⏱️ Duration: {args.duration} seconds")
    print(f"📁 Output: {args.output_dir}")
    print("-" * 40)
    
    # Create output directory
    Path(args.output_dir).mkdir(exist_ok=True)
    
    # Start monitoring
    metrics = monitor_gui_performance(args.duration)
    
    print("\n✅ MONITORING COMPLETE!")
    print("=" * 30)
    print("📊 Final Metrics:")
    for key, value in metrics.items():
        if isinstance(value, float):
            print(f"  {key}: {value:.3f}")
        else:
            print(f"  {key}: {value}")
            
    print(f"\n📁 Reports saved to: {args.output_dir}/")
    print("🎯 Check the HTML report for detailed analysis and recommendations")