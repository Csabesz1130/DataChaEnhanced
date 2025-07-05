#!/usr/bin/env python3
"""
Advanced Code Section Monitoring for Signal Analyzer Application
==============================================================

Comprehensive performance monitoring system for tracking specific code sections
with advanced analysis, visualization, and optimization recommendations.

Features:
    - Real-time section performance monitoring
    - Memory usage tracking per section
    - Nested section analysis and visualization
    - Hot path detection and bottleneck identification
    - Performance regression detection
    - Interactive HTML reports with charts
    - Section dependency mapping
    - Automated optimization recommendations
    - Thread-safe operation for GUI applications
"""

import time
import contextlib
import threading
import tracemalloc
import psutil
from collections import defaultdict, deque
from datetime import datetime, timedelta
import os
import json
import logging
from pathlib import Path
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import seaborn as sns
import numpy as np
from typing import Dict, List, Any, Optional, Callable
import warnings

# Configure matplotlib for non-interactive use
plt.switch_backend('Agg')
warnings.filterwarnings('ignore', category=UserWarning, module='matplotlib')

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class SectionCall:
    """Represents a single section execution with comprehensive metrics."""
    
    def __init__(self, section_name: str, category: str = "general", parent=None):
        self.section_name = section_name
        self.category = category
        self.full_name = f"{category}.{section_name}" if category != "general" else section_name
        self.start_time = time.time()
        self.end_time = None
        self.duration = None
        self.thread_id = threading.get_ident()
        self.parent = parent
        self.children = []
        self.call_depth = parent.call_depth + 1 if parent else 0
        self.exception_raised = None
        
        # Memory tracking
        self.memory_start = None
        self.memory_end = None
        self.memory_delta = None
        self.memory_peak = None
        
        # Performance metrics
        self.cpu_percent_start = None
        self.cpu_percent_end = None
        
        # Initialize memory tracking
        self._start_memory_tracking()
        
    def _start_memory_tracking(self):
        """Initialize memory tracking for this section."""
        try:
            process = psutil.Process()
            self.memory_start = process.memory_info().rss
            self.cpu_percent_start = process.cpu_percent()
            
            # Start tracemalloc if not already running
            if not tracemalloc.is_tracing():
                tracemalloc.start()
                
        except Exception as e:
            logger.debug(f"Memory tracking initialization failed: {e}")
            
    def end_section(self, exception=None):
        """Mark the end of the section execution."""
        self.end_time = time.time()
        self.duration = self.end_time - self.start_time
        self.exception_raised = exception
        
        # Finalize memory tracking
        self._end_memory_tracking()
        
    def _end_memory_tracking(self):
        """Finalize memory tracking and calculate metrics."""
        try:
            process = psutil.Process()
            self.memory_end = process.memory_info().rss
            self.memory_delta = self.memory_end - self.memory_start if self.memory_start else 0
            self.cpu_percent_end = process.cpu_percent()
            
            # Get peak memory usage if available
            try:
                self.memory_peak = process.memory_info().peak_wss if hasattr(process.memory_info(), 'peak_wss') else self.memory_end
            except:
                self.memory_peak = self.memory_end
                
        except Exception as e:
            logger.debug(f"Memory tracking finalization failed: {e}")
            
    def add_child(self, child):
        """Add a child section."""
        child.parent = self
        self.children.append(child)
        
    def get_total_child_time(self):
        """Get total execution time of all child sections."""
        return sum(child.duration for child in self.children if child.duration)
        
    def get_self_time(self):
        """Get execution time excluding child sections."""
        child_time = self.get_total_child_time()
        return self.duration - child_time if self.duration else 0
        
    def to_dict(self):
        """Convert section call to dictionary for serialization."""
        return {
            'section_name': self.section_name,
            'category': self.category,
            'full_name': self.full_name,
            'start_time': self.start_time,
            'end_time': self.end_time,
            'duration': self.duration,
            'thread_id': self.thread_id,
            'call_depth': self.call_depth,
            'memory_delta': self.memory_delta,
            'memory_peak': self.memory_peak,
            'cpu_usage': self.cpu_percent_end - self.cpu_percent_start if self.cpu_percent_end and self.cpu_percent_start else None,
            'exception': str(self.exception_raised) if self.exception_raised else None,
            'children_count': len(self.children),
            'self_time': self.get_self_time()
        }


class AdvancedSectionMonitor:
    """Advanced monitoring system for code sections with comprehensive analysis."""
    
    def __init__(self, max_sections=50000, enable_memory_tracking=True):
        self.max_sections = max_sections
        self.enable_memory_tracking = enable_memory_tracking
        
        # Section tracking
        self.active_sections = defaultdict(list)  # Thread ID -> section stack
        self.completed_sections = deque(maxlen=max_sections)
        self.section_stats = defaultdict(lambda: {
            'count': 0,
            'total_time': 0,
            'avg_time': 0,
            'max_time': 0,
            'min_time': float('inf'),
            'total_memory': 0,
            'avg_memory': 0,
            'max_memory': 0,
            'error_count': 0,
            'last_executed': None,
            'call_depths': [],
            'durations': deque(maxlen=1000)  # Last 1000 durations for trend analysis
        })
        
        # Performance analysis data
        self.hot_sections = []
        self.slow_sections = deque(maxlen=500)
        self.memory_intensive_sections = deque(maxlen=200)
        self.section_dependencies = defaultdict(set)
        self.performance_trends = defaultdict(list)
        
        # Configuration
        self.slow_threshold = float(os.getenv('SLOW_SECTION_THRESHOLD', '0.5'))  # 500ms
        self.memory_threshold = int(os.getenv('MEMORY_THRESHOLD', str(10 * 1024 * 1024)))  # 10MB
        self.verbose_logging = os.getenv('VERBOSE_SECTION_LOGGING', '0') == '1'
        self.enable_trends = os.getenv('ENABLE_PERFORMANCE_TRENDS', '1') == '1'
        
        # Thread safety
        self.lock = threading.Lock()
        
        # Monitoring state
        self.monitoring_start_time = time.time()
        self.total_sections_monitored = 0
        
        logger.info("Advanced Section Monitor initialized")
        
    @contextlib.contextmanager
    def monitor_section(self, section_name: str, category: str = "general"):
        """
        Advanced context manager for monitoring code sections.
        
        Args:
            section_name: Name of the section being monitored
            category: Category for grouping (e.g., 'filtering', 'gui', 'io')
            
        Usage:
            with monitor.monitor_section('apply_filters', 'signal_processing'):
                result = apply_complex_filter(data)
        """
        thread_id = threading.get_ident()
        
        with self.lock:
            # Get current call stack for this thread
            call_stack = self.active_sections[thread_id]
            
            # Create section call object
            parent_section = call_stack[-1] if call_stack else None
            section_call = SectionCall(section_name, category, parent_section)
            
            # Add to call stack and update parent-child relationships
            call_stack.append(section_call)
            if parent_section:
                parent_section.add_child(section_call)
                self.section_dependencies[parent_section.full_name].add(section_call.full_name)
                
        if self.verbose_logging:
            indent = "  " * section_call.call_depth
            print(f"{indent}🔍 Starting section: {section_call.full_name}")
            
        try:
            yield section_call
            
        except Exception as e:
            section_call.exception_raised = e
            if self.verbose_logging:
                print(f"❌ Exception in section {section_call.full_name}: {e}")
            raise
            
        finally:
            # End the section call
            section_call.end_section()
            
            with self.lock:
                # Remove from active call stack
                if call_stack and call_stack[-1] == section_call:
                    call_stack.pop()
                
                # Add to completed sections
                self.completed_sections.append(section_call)
                self.total_sections_monitored += 1
                
                # Update statistics
                self._update_section_stats(section_call)
                
                # Check for performance issues
                self._analyze_section_performance(section_call)
                
            if self.verbose_logging:
                indent = "  " * section_call.call_depth
                print(f"{indent}✅ Completed section: {section_call.full_name} ({section_call.duration:.3f}s)")
                
    @contextlib.contextmanager
    def monitor_memory_section(self, section_name: str, category: str = "general"):
        """
        Context manager with enhanced memory tracking.
        
        Usage:
            with monitor.monitor_memory_section('load_large_file', 'io'):
                data = load_massive_dataset()
        """
        # Use the main monitor_section but ensure memory tracking is enabled
        original_memory_setting = self.enable_memory_tracking
        self.enable_memory_tracking = True
        
        try:
            with self.monitor_section(section_name, category) as section:
                yield section
        finally:
            self.enable_memory_tracking = original_memory_setting
            
    @contextlib.contextmanager
    def monitor_io_section(self, section_name: str, filepath: str = None, operation: str = "read"):
        """
        Specialized context manager for I/O operations.
        
        Args:
            section_name: Name of the I/O operation
            filepath: Optional file path for additional context
            operation: Type of operation (read, write, process)
            
        Usage:
            with monitor.monitor_io_section('load_atf_file', '/path/to/file.atf', 'read'):
                data = load_file(filepath)
        """
        full_section_name = section_name
        if filepath:
            try:
                file_size = Path(filepath).stat().st_size / 1024 / 1024  # MB
                full_section_name = f"{section_name}_({file_size:.1f}MB)"
            except:
                pass
                
        with self.monitor_section(full_section_name, f"io_{operation}") as section:
            # Add file-specific metadata
            if hasattr(section, 'metadata'):
                section.metadata = {'filepath': filepath, 'operation': operation}
            yield section
            
    def _update_section_stats(self, section_call: SectionCall):
        """Update comprehensive statistics for a section call."""
        full_name = section_call.full_name
        stats = self.section_stats[full_name]
        
        # Basic timing statistics
        stats['count'] += 1
        stats['total_time'] += section_call.duration
        stats['avg_time'] = stats['total_time'] / stats['count']
        stats['max_time'] = max(stats['max_time'], section_call.duration)
        stats['min_time'] = min(stats['min_time'], section_call.duration)
        stats['last_executed'] = datetime.now()
        stats['call_depths'].append(section_call.call_depth)
        stats['durations'].append(section_call.duration)
        
        # Memory statistics
        if section_call.memory_delta:
            stats['total_memory'] += abs(section_call.memory_delta)
            stats['avg_memory'] = stats['total_memory'] / stats['count']
            stats['max_memory'] = max(stats['max_memory'], abs(section_call.memory_delta))
            
        # Error tracking
        if section_call.exception_raised:
            stats['error_count'] += 1
            
        # Performance trends
        if self.enable_trends:
            current_time = time.time()
            self.performance_trends[full_name].append({
                'timestamp': current_time,
                'duration': section_call.duration,
                'memory_delta': section_call.memory_delta or 0
            })
            
            # Keep only last hour of trend data
            cutoff_time = current_time - 3600  # 1 hour
            self.performance_trends[full_name] = [
                trend for trend in self.performance_trends[full_name]
                if trend['timestamp'] > cutoff_time
            ]
            
    def _analyze_section_performance(self, section_call: SectionCall):
        """Analyze section performance and identify issues."""
        # Check for slow sections
        if section_call.duration > self.slow_threshold:
            self.slow_sections.append({
                'section': section_call.full_name,
                'duration': section_call.duration,
                'timestamp': datetime.now(),
                'call_depth': section_call.call_depth,
                'memory_delta': section_call.memory_delta or 0
            })
            
            if self.verbose_logging or section_call.duration > self.slow_threshold * 2:
                print(f"⏱️ Slow section detected: {section_call.full_name} took {section_call.duration:.3f}s")
                
        # Check for memory-intensive sections
        if section_call.memory_delta and abs(section_call.memory_delta) > self.memory_threshold:
            self.memory_intensive_sections.append({
                'section': section_call.full_name,
                'memory_delta': section_call.memory_delta,
                'duration': section_call.duration,
                'timestamp': datetime.now()
            })
            
            action = "allocated" if section_call.memory_delta > 0 else "freed"
            print(f"💾 High memory usage: {section_call.full_name} {action} {abs(section_call.memory_delta) / 1024 / 1024:.1f}MB")
            
    def generate_comprehensive_report(self):
        """Generate comprehensive performance monitoring report."""
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        
        # Create reports directory
        reports_dir = Path('performance_reports')
        reports_dir.mkdir(exist_ok=True)
        
        # Generate JSON report
        json_report = self._generate_json_report()
        json_file = reports_dir / f'section_monitor_report_{timestamp}.json'
        with open(json_file, 'w') as f:
            json.dump(json_report, f, indent=2, default=str)
            
        # Generate HTML report
        html_report = self._generate_html_report()
        html_file = reports_dir / f'section_monitor_report_{timestamp}.html'
        with open(html_file, 'w') as f:
            f.write(html_report)
            
        # Generate performance visualizations
        self._generate_performance_visualizations(timestamp)
        
        # Generate section dependency graph
        self._generate_dependency_visualization(timestamp)
        
        logger.info(f"Comprehensive section monitoring reports generated:")
        logger.info(f"  JSON: {json_file}")
        logger.info(f"  HTML: {html_file}")
        
        return {
            'json_report': str(json_file),
            'html_report': str(html_file),
            'timestamp': timestamp
        }
        
    def _generate_json_report(self):
        """Generate comprehensive JSON report."""
        with self.lock:
            # Calculate summary statistics
            total_execution_time = sum(section.duration for section in self.completed_sections if section.duration)
            unique_sections = len(self.section_stats)
            avg_section_time = total_execution_time / len(self.completed_sections) if self.completed_sections else 0
            
            # Get top sections by various metrics
            top_by_time = sorted(
                self.section_stats.items(),
                key=lambda x: x[1]['total_time'],
                reverse=True
            )[:20]
            
            top_by_calls = sorted(
                self.section_stats.items(),
                key=lambda x: x[1]['count'],
                reverse=True
            )[:20]
            
            top_by_memory = sorted(
                self.section_stats.items(),
                key=lambda x: x[1]['max_memory'],
                reverse=True
            )[:20]
            
            report = {
                'metadata': {
                    'timestamp': datetime.now().isoformat(),
                    'monitoring_duration': time.time() - self.monitoring_start_time,
                    'total_sections_monitored': self.total_sections_monitored,
                    'configuration': {
                        'slow_threshold': self.slow_threshold,
                        'memory_threshold': self.memory_threshold,
                        'max_sections': self.max_sections,
                        'memory_tracking_enabled': self.enable_memory_tracking
                    }
                },
                'summary': {
                    'unique_sections': unique_sections,
                    'total_section_calls': len(self.completed_sections),
                    'total_execution_time': total_execution_time,
                    'average_section_time': avg_section_time,
                    'slow_sections_detected': len(self.slow_sections),
                    'memory_intensive_sections': len(self.memory_intensive_sections),
                    'sections_with_errors': sum(1 for stats in self.section_stats.values() if stats['error_count'] > 0)
                },
                'top_sections': {
                    'by_total_time': [
                        {
                            'section': name,
                            'total_time': stats['total_time'],
                            'call_count': stats['count'],
                            'avg_time': stats['avg_time']
                        }
                        for name, stats in top_by_time
                    ],
                    'by_call_count': [
                        {
                            'section': name,
                            'call_count': stats['count'],
                            'total_time': stats['total_time'],
                            'avg_time': stats['avg_time']
                        }
                        for name, stats in top_by_calls
                    ],
                    'by_memory_usage': [
                        {
                            'section': name,
                            'max_memory_mb': stats['max_memory'] / 1024 / 1024 if stats['max_memory'] else 0,
                            'avg_memory_mb': stats['avg_memory'] / 1024 / 1024 if stats['avg_memory'] else 0,
                            'call_count': stats['count']
                        }
                        for name, stats in top_by_memory if stats['max_memory'] > 0
                    ]
                },
                'performance_issues': {
                    'slow_sections': [
                        {
                            'section': slow['section'],
                            'duration': slow['duration'],
                            'timestamp': slow['timestamp'].isoformat(),
                            'memory_delta_mb': slow['memory_delta'] / 1024 / 1024 if slow['memory_delta'] else 0
                        }
                        for slow in list(self.slow_sections)
                    ],
                    'memory_intensive_sections': [
                        {
                            'section': memory['section'],
                            'memory_delta_mb': memory['memory_delta'] / 1024 / 1024,
                            'duration': memory['duration'],
                            'timestamp': memory['timestamp'].isoformat()
                        }
                        for memory in list(self.memory_intensive_sections)
                    ]
                },
                'recommendations': self._generate_optimization_recommendations(),
                'section_dependencies': {
                    parent: list(children) 
                    for parent, children in self.section_dependencies.items()
                    if len(children) > 1
                }
            }
            
        return report
        
    def _generate_html_report(self):
        """Generate comprehensive HTML report with interactive visualizations."""
        timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        
        # Calculate summary metrics
        with self.lock:
            total_time = sum(section.duration for section in self.completed_sections if section.duration)
            unique_sections = len(self.section_stats)
            total_calls = len(self.completed_sections)
            avg_time = total_time / total_calls if total_calls > 0 else 0
            
        html = f"""
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Advanced Section Monitoring Report - Signal Analyzer</title>
    <style>
        * {{ margin: 0; padding: 0; box-sizing: border-box; }}
        
        body {{ 
            font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', 'Roboto', sans-serif;
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            min-height: 100vh;
            padding: 20px;
        }}
        
        .container {{ 
            max-width: 1400px; 
            margin: 0 auto; 
            background: white; 
            border-radius: 20px;
            box-shadow: 0 20px 60px rgba(0,0,0,0.1);
            overflow: hidden;
        }}
        
        .header {{ 
            background: linear-gradient(135deg, #2196F3 0%, #21CBF3 100%);
            color: white; 
            padding: 40px 50px; 
            position: relative;
            overflow: hidden;
        }}
        
        .header::before {{
            content: '';
            position: absolute;
            top: -50%;
            right: -50%;
            width: 200%;
            height: 200%;
            background: radial-gradient(circle, rgba(255,255,255,0.1) 0%, transparent 70%);
            animation: float 6s ease-in-out infinite;
        }}
        
        @keyframes float {{
            0%, 100% {{ transform: translateY(0px) rotate(0deg); }}
            50% {{ transform: translateY(-20px) rotate(180deg); }}
        }}
        
        .header h1 {{ 
            font-size: 3em; 
            margin-bottom: 15px; 
            font-weight: 700;
            position: relative;
            z-index: 1;
        }}
        
        .header p {{ 
            font-size: 1.2em; 
            opacity: 0.95; 
            position: relative;
            z-index: 1;
        }}
        
        .content {{ padding: 50px; }}
        
        .metrics-grid {{ 
            display: grid; 
            grid-template-columns: repeat(auto-fit, minmax(280px, 1fr)); 
            gap: 30px; 
            margin: 40px 0;
        }}
        
        .metric-card {{ 
            background: linear-gradient(135deg, #f8f9ff 0%, #e8f4fd 100%);
            padding: 30px; 
            border-radius: 15px; 
            text-align: center;
            border: 1px solid #e3f2fd;
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
            background: linear-gradient(90deg, transparent, rgba(255,255,255,0.4), transparent);
            transition: left 0.5s;
        }}
        
        .metric-card:hover {{
            transform: translateY(-5px);
            box-shadow: 0 10px 30px rgba(0,0,0,0.1);
        }}
        
        .metric-card:hover::before {{
            left: 100%;
        }}
        
        .metric-icon {{ 
            font-size: 3em; 
            margin-bottom: 15px; 
            display: block;
        }}
        
        .metric-value {{ 
            font-size: 2.5em; 
            font-weight: bold; 
            color: #1976d2; 
            margin-bottom: 10px;
        }}
        
        .metric-label {{ 
            font-size: 1.1em; 
            color: #666; 
            font-weight: 500;
        }}
        
        .metric-unit {{ 
            font-size: 0.9em; 
            color: #999; 
            margin-top: 5px;
        }}
        
        .section {{ 
            background: white;
            margin: 40px 0; 
            padding: 40px; 
            border-radius: 15px; 
            border-left: 5px solid #2196F3;
            box-shadow: 0 5px 20px rgba(0,0,0,0.05);
        }}
        
        .section h2 {{ 
            color: #1976d2; 
            margin-bottom: 25px; 
            font-size: 1.8em;
            display: flex;
            align-items: center;
            gap: 15px;
        }}
        
        .performance-table {{ 
            width: 100%; 
            border-collapse: collapse; 
            margin: 25px 0;
            background: white;
            border-radius: 10px;
            overflow: hidden;
            box-shadow: 0 3px 15px rgba(0,0,0,0.1);
        }}
        
        .performance-table th {{ 
            background: linear-gradient(135deg, #1976d2 0%, #1565c0 100%);
            color: white; 
            padding: 18px; 
            text-align: left;
            font-weight: 600;
            font-size: 0.95em;
            text-transform: uppercase;
            letter-spacing: 0.5px;
        }}
        
        .performance-table td {{ 
            border-bottom: 1px solid #f0f0f0; 
            padding: 15px 18px; 
            transition: background-color 0.2s;
        }}
        
        .performance-table tr:hover td {{ 
            background-color: #f8f9ff; 
        }}
        
        .performance-table tr:last-child td {{ 
            border-bottom: none; 
        }}
        
        .chart-container {{ 
            text-align: center; 
            margin: 40px 0; 
            padding: 30px; 
            background: linear-gradient(135deg, #f8f9ff 0%, #e8f4fd 100%);
            border-radius: 15px;
            border: 1px solid #e3f2fd;
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
            box-shadow: 0 5px 20px rgba(0,0,0,0.1);
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
        }}
        
        .recommendations ul {{ 
            list-style: none; 
            padding: 0;
        }}
        
        .recommendations li {{ 
            margin: 15px 0; 
            padding: 15px 20px;
            background: rgba(255,255,255,0.7);
            border-radius: 10px;
            border-left: 3px solid #4caf50;
            line-height: 1.6;
            transition: all 0.2s ease;
        }}
        
        .recommendations li:hover {{
            background: rgba(255,255,255,0.9);
            transform: translateX(5px);
        }}
        
        .code {{ 
            background: #263238; 
            color: #80cbc4; 
            padding: 4px 8px; 
            border-radius: 5px; 
            font-family: 'Monaco', 'Menlo', 'Ubuntu Mono', monospace;
            font-size: 0.9em;
            font-weight: 500;
        }}
        
        .status-indicator {{
            display: inline-block;
            width: 12px;
            height: 12px;
            border-radius: 50%;
            margin-right: 8px;
        }}
        
        .status-good {{ background-color: #4caf50; }}
        .status-warning {{ background-color: #ff9800; }}
        .status-critical {{ background-color: #f44336; }}
        
        .progress-bar {{
            width: 100%;
            height: 8px;
            background-color: #e0e0e0;
            border-radius: 4px;
            overflow: hidden;
            margin: 10px 0;
        }}
        
        .progress-fill {{
            height: 100%;
            background: linear-gradient(90deg, #4caf50 0%, #8bc34a 100%);
            border-radius: 4px;
            transition: width 0.3s ease;
        }}
        
        .tab-container {{
            margin: 30px 0;
        }}
        
        .tab-buttons {{
            display: flex;
            border-bottom: 2px solid #e0e0e0;
            margin-bottom: 30px;
        }}
        
        .tab-button {{
            padding: 15px 30px;
            background: none;
            border: none;
            font-size: 1.1em;
            cursor: pointer;
            color: #666;
            border-bottom: 3px solid transparent;
            transition: all 0.3s ease;
        }}
        
        .tab-button.active {{
            color: #1976d2;
            border-bottom-color: #1976d2;
            background: rgba(33, 150, 243, 0.05);
        }}
        
        .tab-content {{
            display: none;
        }}
        
        .tab-content.active {{
            display: block;
        }}
        
        @media (max-width: 768px) {{
            .header {{ padding: 30px 25px; }}
            .header h1 {{ font-size: 2em; }}
            .content {{ padding: 25px; }}
            .metrics-grid {{ grid-template-columns: 1fr; gap: 20px; }}
            .section {{ padding: 25px; margin: 25px 0; }}
        }}
    </style>
    <script>
        function showTab(tabName) {{
            // Hide all tab contents
            const contents = document.querySelectorAll('.tab-content');
            contents.forEach(content => content.classList.remove('active'));
            
            // Remove active class from all buttons
            const buttons = document.querySelectorAll('.tab-button');
            buttons.forEach(button => button.classList.remove('active'));
            
            // Show selected tab content
            document.getElementById(tabName).classList.add('active');
            event.target.classList.add('active');
        }}
        
        document.addEventListener('DOMContentLoaded', function() {{
            // Show first tab by default
            document.querySelector('.tab-button').click();
        }});
    </script>
</head>
<body>
    <div class="container">
        <div class="header">
            <h1>🔍 Advanced Section Monitoring Report</h1>
            <p>Comprehensive Performance Analysis for Signal Analyzer</p>
            <p>Generated: {timestamp}</p>
        </div>
        
        <div class="content">
            <div class="metrics-grid">
                <div class="metric-card">
                    <span class="metric-icon">📊</span>
                    <div class="metric-value">{unique_sections}</div>
                    <div class="metric-label">Unique Sections</div>
                </div>
                <div class="metric-card">
                    <span class="metric-icon">🔄</span>
                    <div class="metric-value">{total_calls:,}</div>
                    <div class="metric-label">Total Section Calls</div>
                </div>
                <div class="metric-card">
                    <span class="metric-icon">⏱️</span>
                    <div class="metric-value">{total_time:.2f}</div>
                    <div class="metric-label">Total Execution Time</div>
                    <div class="metric-unit">seconds</div>
                </div>
                <div class="metric-card">
                    <span class="metric-icon">📈</span>
                    <div class="metric-value">{avg_time*1000:.1f}</div>
                    <div class="metric-label">Average Section Time</div>
                    <div class="metric-unit">milliseconds</div>
                </div>
                <div class="metric-card">
                    <span class="metric-icon">⚠️</span>
                    <div class="metric-value">{len(self.slow_sections)}</div>
                    <div class="metric-label">Slow Sections Detected</div>
                </div>
                <div class="metric-card">
                    <span class="metric-icon">💾</span>
                    <div class="metric-value">{len(self.memory_intensive_sections)}</div>
                    <div class="metric-label">Memory Intensive Sections</div>
                </div>
            </div>
            
            <div class="chart-container">
                <h3>📊 Performance Visualizations</h3>
                <img src="section_performance_charts_{datetime.now().strftime('%Y%m%d_%H%M%S')}.png" 
                     alt="Section Performance Charts" style="max-width: 100%; height: auto;">
            </div>
            
            <div class="chart-container">
                <h3>🔗 Section Dependencies</h3>
                <img src="section_dependencies_{datetime.now().strftime('%Y%m%d_%H%M%S')}.png" 
                     alt="Section Dependencies Graph" style="max-width: 100%; height: auto;">
            </div>
            
            <div class="tab-container">
                <div class="tab-buttons">
                    <button class="tab-button" onclick="showTab('top-sections')">🏆 Top Sections</button>
                    <button class="tab-button" onclick="showTab('slow-sections')">⏱️ Slow Sections</button>
                    <button class="tab-button" onclick="showTab('memory-sections')">💾 Memory Usage</button>
                    <button class="tab-button" onclick="showTab('dependencies')">🔗 Dependencies</button>
                </div>
                
                <div id="top-sections" class="tab-content">
                    <div class="section">
                        <h2>🏆 Top Sections by Total Time</h2>
                        {self._generate_top_sections_table()}
                    </div>
                </div>
                
                <div id="slow-sections" class="tab-content">
                    <div class="section">
                        <h2>⏱️ Recent Slow Sections</h2>
                        {self._generate_slow_sections_table()}
                    </div>
                </div>
                
                <div id="memory-sections" class="tab-content">
                    <div class="section">
                        <h2>💾 Memory Intensive Sections</h2>
                        {self._generate_memory_sections_table()}
                    </div>
                </div>
                
                <div id="dependencies" class="tab-content">
                    <div class="section">
                        <h2>🔗 Section Dependencies</h2>
                        {self._generate_dependencies_table()}
                    </div>
                </div>
            </div>
            
            <div class="recommendations">
                <h2>🎯 Performance Optimization Recommendations</h2>
                <ul>
                    {''.join(f'<li>{rec}</li>' for rec in self._generate_optimization_recommendations())}
                </ul>
            </div>
        </div>
    </div>
</body>
</html>
        """
        
        return html
        
    def _generate_top_sections_table(self):
        """Generate HTML table for top performing sections."""
        with self.lock:
            if not self.section_stats:
                return '<p>No section data available.</p>'
                
            sorted_sections = sorted(
                self.section_stats.items(),
                key=lambda x: x[1]['total_time'],
                reverse=True
            )[:20]
            
        html = '''
        <table class="performance-table">
            <thead>
                <tr>
                    <th>Section</th>
                    <th>Call Count</th>
                    <th>Total Time (s)</th>
                    <th>Avg Time (ms)</th>
                    <th>Max Time (ms)</th>
                    <th>Memory (MB)</th>
                    <th>Status</th>
                </tr>
            </thead>
            <tbody>
        '''
        
        for section_name, stats in sorted_sections:
            avg_time_ms = stats['avg_time'] * 1000
            max_time_ms = stats['max_time'] * 1000
            avg_memory_mb = stats['avg_memory'] / 1024 / 1024 if stats['avg_memory'] else 0
            
            # Determine status
            if stats['error_count'] > 0:
                status = '<span class="status-indicator status-critical"></span>Critical'
            elif stats['avg_time'] > self.slow_threshold:
                status = '<span class="status-indicator status-warning"></span>Warning'
            else:
                status = '<span class="status-indicator status-good"></span>Good'
                
            short_name = section_name[:50] + '...' if len(section_name) > 50 else section_name
            
            html += f'''
            <tr>
                <td><span class="code">{short_name}</span></td>
                <td>{stats['count']:,}</td>
                <td>{stats['total_time']:.3f}</td>
                <td>{avg_time_ms:.2f}</td>
                <td>{max_time_ms:.2f}</td>
                <td>{avg_memory_mb:.2f}</td>
                <td>{status}</td>
            </tr>
            '''
            
        html += '</tbody></table>'
        return html
        
    def _generate_slow_sections_table(self):
        """Generate HTML table for slow sections."""
        if not self.slow_sections:
            return '<p>No slow sections detected. 🎉</p>'
            
        html = '''
        <table class="performance-table">
            <thead>
                <tr>
                    <th>Section</th>
                    <th>Duration (s)</th>
                    <th>Memory Impact (MB)</th>
                    <th>Call Depth</th>
                    <th>Timestamp</th>
                </tr>
            </thead>
            <tbody>
        '''
        
        for slow in sorted(self.slow_sections, key=lambda x: x['duration'], reverse=True)[:20]:
            memory_mb = slow['memory_delta'] / 1024 / 1024 if slow['memory_delta'] else 0
            timestamp_str = slow['timestamp'].strftime('%H:%M:%S')
            
            html += f'''
            <tr>
                <td><span class="code">{slow['section']}</span></td>
                <td>{slow['duration']:.3f}</td>
                <td>{memory_mb:+.2f}</td>
                <td>{slow['call_depth']}</td>
                <td>{timestamp_str}</td>
            </tr>
            '''
            
        html += '</tbody></table>'
        return html
        
    def _generate_memory_sections_table(self):
        """Generate HTML table for memory intensive sections."""
        if not self.memory_intensive_sections:
            return '<p>No memory intensive sections detected. 💾</p>'
            
        html = '''
        <table class="performance-table">
            <thead>
                <tr>
                    <th>Section</th>
                    <th>Memory Delta (MB)</th>
                    <th>Duration (s)</th>
                    <th>Memory/Time Ratio</th>
                    <th>Timestamp</th>
                </tr>
            </thead>
            <tbody>
        '''
        
        for memory in sorted(self.memory_intensive_sections, 
                           key=lambda x: abs(x['memory_delta']), reverse=True)[:20]:
            memory_mb = memory['memory_delta'] / 1024 / 1024
            ratio = abs(memory_mb) / memory['duration'] if memory['duration'] > 0 else 0
            timestamp_str = memory['timestamp'].strftime('%H:%M:%S')
            action_color = '#f44336' if memory_mb > 0 else '#4caf50'
            
            html += f'''
            <tr>
                <td><span class="code">{memory['section']}</span></td>
                <td style="color: {action_color};">{memory_mb:+.2f}</td>
                <td>{memory['duration']:.3f}</td>
                <td>{ratio:.2f} MB/s</td>
                <td>{timestamp_str}</td>
            </tr>
            '''
            
        html += '</tbody></table>'
        return html
        
    def _generate_dependencies_table(self):
        """Generate HTML table for section dependencies."""
        if not self.section_dependencies:
            return '<p>No section dependencies detected.</p>'
            
        html = '''
        <table class="performance-table">
            <thead>
                <tr>
                    <th>Parent Section</th>
                    <th>Child Sections</th>
                    <th>Dependency Count</th>
                    <th>Complexity</th>
                </tr>
            </thead>
            <tbody>
        '''
        
        sorted_deps = sorted(
            self.section_dependencies.items(),
            key=lambda x: len(x[1]),
            reverse=True
        )[:15]
        
        for parent, children in sorted_deps:
            child_count = len(children)
            complexity = "High" if child_count > 10 else "Medium" if child_count > 5 else "Low"
            complexity_color = "#f44336" if complexity == "High" else "#ff9800" if complexity == "Medium" else "#4caf50"
            
            child_list = ', '.join(list(children)[:5])
            if len(children) > 5:
                child_list += f', ... and {len(children) - 5} more'
                
            html += f'''
            <tr>
                <td><span class="code">{parent}</span></td>
                <td>{child_list}</td>
                <td>{child_count}</td>
                <td style="color: {complexity_color};">{complexity}</td>
            </tr>
            '''
            
        html += '</tbody></table>'
        return html
        
    def _generate_optimization_recommendations(self):
        """Generate actionable optimization recommendations."""
        recommendations = []
        
        with self.lock:
            if not self.section_stats:
                return ["No section monitoring data available for analysis"]
                
            # Analyze top time consumers
            top_time_sections = sorted(
                self.section_stats.items(),
                key=lambda x: x[1]['total_time'],
                reverse=True
            )[:5]
            
            if top_time_sections:
                top_section = top_time_sections[0]
                recommendations.append(
                    f"🔥 <strong>Primary Optimization Target:</strong> "
                    f"<span class='code'>{top_section[0]}</span> consumes "
                    f"{top_section[1]['total_time']:.3f}s ({top_section[1]['count']} calls)"
                )
                
            # Identify high-frequency sections
            high_freq_sections = [
                (name, stats) for name, stats in self.section_stats.items()
                if stats['count'] > 100 and stats['avg_time'] > 0.01
            ]
            
            if high_freq_sections:
                recommendations.append(
                    f"📈 <strong>Caching Opportunity:</strong> "
                    f"{len(high_freq_sections)} high-frequency sections could benefit from caching"
                )
                
            # Analyze slow sections patterns
            if self.slow_sections:
                slow_patterns = defaultdict(int)
                for slow in self.slow_sections:
                    category = slow['section'].split('.')[0]
                    slow_patterns[category] += 1
                    
                if slow_patterns:
                    worst_category = max(slow_patterns.items(), key=lambda x: x[1])
                    recommendations.append(
                        f"⏱️ <strong>Performance Hotspot:</strong> "
                        f"'{worst_category[0]}' category has {worst_category[1]} slow operations"
                    )
                    
            # Memory optimization recommendations
            if self.memory_intensive_sections:
                memory_patterns = defaultdict(list)
                for memory in self.memory_intensive_sections:
                    category = memory['section'].split('.')[0]
                    memory_patterns[category].append(abs(memory['memory_delta']))
                    
                if memory_patterns:
                    for category, deltas in memory_patterns.items():
                        avg_delta = sum(deltas) / len(deltas) / 1024 / 1024
                        if avg_delta > 50:  # > 50MB average
                            recommendations.append(
                                f"💾 <strong>Memory Optimization:</strong> "
                                f"'{category}' operations average {avg_delta:.1f}MB memory usage"
                            )
                            
            # Dependency complexity recommendations
            complex_dependencies = [
                (parent, len(children)) for parent, children in self.section_dependencies.items()
                if len(children) > 8
            ]
            
            if complex_dependencies:
                recommendations.append(
                    f"🔗 <strong>Architecture Review:</strong> "
                    f"{len(complex_dependencies)} sections have high dependency complexity"
                )
                
            # Performance trends (if available)
            if self.enable_trends and self.performance_trends:
                trending_up = []
                for section, trends in self.performance_trends.items():
                    if len(trends) > 10:
                        recent_avg = sum(t['duration'] for t in trends[-5:]) / 5
                        older_avg = sum(t['duration'] for t in trends[:5]) / 5
                        if recent_avg > older_avg * 1.2:  # 20% increase
                            trending_up.append(section)
                            
                if trending_up:
                    recommendations.append(
                        f"📊 <strong>Performance Regression:</strong> "
                        f"{len(trending_up)} sections showing performance degradation"
                    )
                    
            # General recommendations
            if len(recommendations) < 3:
                recommendations.extend([
                    "⚡ <strong>Optimization Strategy:</strong> Focus on sections with high call count × average time",
                    "🧵 <strong>Concurrency:</strong> Consider parallelizing independent section operations",
                    "📏 <strong>Monitoring:</strong> Continue monitoring during peak usage periods"
                ])
                
        return recommendations
        
    def _generate_performance_visualizations(self, timestamp):
        """Generate comprehensive performance visualization charts."""
        try:
            # Set up the plotting style
            plt.style.use('seaborn-v0_8')
            fig = plt.figure(figsize=(20, 16))
            
            # Create subplots
            gs = fig.add_gridspec(3, 3, hspace=0.3, wspace=0.3)
            
            with self.lock:
                if not self.section_stats:
                    plt.text(0.5, 0.5, 'No data available for visualization', 
                           ha='center', va='center', transform=plt.gca().transAxes, fontsize=16)
                    plt.savefig(f'performance_reports/section_performance_charts_{timestamp}.png', 
                              dpi=300, bbox_inches='tight')
                    plt.close()
                    return
                    
                # 1. Top sections by total time (horizontal bar chart)
                ax1 = fig.add_subplot(gs[0, :2])
                top_sections = sorted(self.section_stats.items(), key=lambda x: x[1]['total_time'], reverse=True)[:15]
                if top_sections:
                    sections = [name.split('.')[-1][:20] for name, _ in top_sections]
                    times = [stats['total_time'] for _, stats in top_sections]
                    
                    bars = ax1.barh(sections, times, color=plt.cm.Blues(np.linspace(0.4, 0.8, len(sections))))
                    ax1.set_xlabel('Total Time (seconds)')
                    ax1.set_title('Top 15 Sections by Total Execution Time', fontsize=14, fontweight='bold')
                    ax1.grid(axis='x', alpha=0.3)
                    
                    # Add value labels on bars
                    for i, (bar, time) in enumerate(zip(bars, times)):
                        ax1.text(bar.get_width() + max(times) * 0.01, bar.get_y() + bar.get_height()/2, 
                               f'{time:.3f}s', va='center', fontsize=9)
                
                # 2. Section call frequency distribution
                ax2 = fig.add_subplot(gs[0, 2])
                call_counts = [stats['count'] for stats in self.section_stats.values()]
                if call_counts:
                    ax2.hist(call_counts, bins=min(30, len(call_counts)), alpha=0.7, color='skyblue', edgecolor='black')
                    ax2.set_xlabel('Number of Calls')
                    ax2.set_ylabel('Number of Sections')
                    ax2.set_title('Call Frequency Distribution', fontsize=12, fontweight='bold')
                    ax2.set_yscale('log')
                    ax2.grid(alpha=0.3)
                
                # 3. Execution time distribution
                ax3 = fig.add_subplot(gs[1, 0])
                avg_times = [stats['avg_time'] * 1000 for stats in self.section_stats.values()]  # Convert to ms
                if avg_times:
                    ax3.hist(avg_times, bins=min(50, len(avg_times)), alpha=0.7, color='lightcoral', edgecolor='black')
                    ax3.set_xlabel('Average Time (milliseconds)')
                    ax3.set_ylabel('Number of Sections')
                    ax3.set_title('Execution Time Distribution', fontsize=12, fontweight='bold')
                    ax3.set_xscale('log')
                    ax3.grid(alpha=0.3)
                
                # 4. Memory usage vs execution time scatter plot
                ax4 = fig.add_subplot(gs[1, 1])
                if self.completed_sections:
                    durations = []
                    memory_deltas = []
                    colors = []
                    
                    for section in self.completed_sections:
                        if section.duration and section.memory_delta:
                            durations.append(section.duration * 1000)  # Convert to ms
                            memory_deltas.append(abs(section.memory_delta) / 1024 / 1024)  # Convert to MB
                            colors.append('red' if section.memory_delta > 0 else 'blue')
                    
                    if durations and memory_deltas:
                        scatter = ax4.scatter(durations, memory_deltas, c=colors, alpha=0.6, s=20)
                        ax4.set_xlabel('Duration (milliseconds)')
                        ax4.set_ylabel('Memory Usage (MB)')
                        ax4.set_title('Memory vs Time Correlation', fontsize=12, fontweight='bold')
                        ax4.set_xscale('log')
                        ax4.set_yscale('log')
                        ax4.grid(alpha=0.3)
                        
                        # Add legend
                        red_patch = mpatches.Patch(color='red', label='Memory Allocation')
                        blue_patch = mpatches.Patch(color='blue', label='Memory Deallocation')
                        ax4.legend(handles=[red_patch, blue_patch], loc='upper right')
                
                # 5. Performance trends over time (if available)
                ax5 = fig.add_subplot(gs[1, 2])
                if self.enable_trends and self.performance_trends:
                    # Show trends for top 5 sections
                    top_trending = sorted(self.performance_trends.items(), 
                                        key=lambda x: len(x[1]), reverse=True)[:5]
                    
                    colors = plt.cm.Set1(np.linspace(0, 1, len(top_trending)))
                    for i, (section_name, trends) in enumerate(top_trending):
                        if len(trends) > 5:
                            timestamps = [t['timestamp'] for t in trends]
                            durations = [t['duration'] * 1000 for t in trends]  # Convert to ms
                            
                            # Convert timestamps to relative time (minutes from start)
                            start_time = min(timestamps)
                            relative_times = [(t - start_time) / 60 for t in timestamps]
                            
                            ax5.plot(relative_times, durations, 'o-', color=colors[i], 
                                   label=section_name.split('.')[-1][:15], alpha=0.8, linewidth=2, markersize=4)
                    
                    ax5.set_xlabel('Time (minutes)')
                    ax5.set_ylabel('Duration (ms)')
                    ax5.set_title('Performance Trends', fontsize=12, fontweight='bold')
                    ax5.legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=8)
                    ax5.grid(alpha=0.3)
                else:
                    ax5.text(0.5, 0.5, 'No trend data available\\n(Enable with ENABLE_PERFORMANCE_TRENDS=1)', 
                           ha='center', va='center', transform=ax5.transAxes, fontsize=10)
                
                # 6. Category performance breakdown (pie chart)
                ax6 = fig.add_subplot(gs[2, 0])
                category_times = defaultdict(float)
                for section_name, stats in self.section_stats.items():
                    category = section_name.split('.')[0] if '.' in section_name else 'general'
                    category_times[category] += stats['total_time']
                
                if category_times:
                    categories = list(category_times.keys())
                    times = list(category_times.values())
                    colors = plt.cm.Set3(np.linspace(0, 1, len(categories)))
                    
                    wedges, texts, autotexts = ax6.pie(times, labels=categories, autopct='%1.1f%%', 
                                                     colors=colors, startangle=90)
                    ax6.set_title('Time Distribution by Category', fontsize=12, fontweight='bold')
                    
                    # Improve text readability
                    for autotext in autotexts:
                        autotext.set_color('white')
                        autotext.set_fontweight('bold')
                
                # 7. Error rate analysis
                ax7 = fig.add_subplot(gs[2, 1])
                error_sections = [(name, stats['error_count'], stats['count']) 
                                for name, stats in self.section_stats.items() 
                                if stats['error_count'] > 0]
                
                if error_sections:
                    error_sections.sort(key=lambda x: x[1] / x[2], reverse=True)  # Sort by error rate
                    error_sections = error_sections[:10]  # Top 10
                    
                    section_names = [name.split('.')[-1][:15] for name, _, _ in error_sections]
                    error_rates = [errors / total * 100 for _, errors, total in error_sections]
                    
                    bars = ax7.bar(range(len(section_names)), error_rates, color='red', alpha=0.7)
                    ax7.set_xlabel('Sections')
                    ax7.set_ylabel('Error Rate (%)')
                    ax7.set_title('Sections with Highest Error Rates', fontsize=12, fontweight='bold')
                    ax7.set_xticks(range(len(section_names)))
                    ax7.set_xticklabels(section_names, rotation=45, ha='right')
                    ax7.grid(axis='y', alpha=0.3)
                    
                    # Add value labels on bars
                    for bar, rate in zip(bars, error_rates):
                        ax7.text(bar.get_x() + bar.get_width()/2, bar.get_height() + max(error_rates) * 0.01,
                               f'{rate:.1f}%', ha='center', va='bottom', fontsize=9)
                else:
                    ax7.text(0.5, 0.5, 'No errors detected 🎉', 
                           ha='center', va='center', transform=ax7.transAxes, fontsize=14, 
                           bbox=dict(boxstyle="round,pad=0.3", facecolor="lightgreen", alpha=0.5))
                
                # 8. Call depth analysis
                ax8 = fig.add_subplot(gs[2, 2])
                all_depths = []
                for stats in self.section_stats.values():
                    all_depths.extend(stats['call_depths'])
                
                if all_depths:
                    depth_counts = {}
                    for depth in all_depths:
                        depth_counts[depth] = depth_counts.get(depth, 0) + 1
                    
                    depths = sorted(depth_counts.keys())
                    counts = [depth_counts[d] for d in depths]
                    
                    bars = ax8.bar(depths, counts, color='orange', alpha=0.7)
                    ax8.set_xlabel('Call Depth')
                    ax8.set_ylabel('Number of Calls')
                    ax8.set_title('Call Depth Distribution', fontsize=12, fontweight='bold')
                    ax8.grid(axis='y', alpha=0.3)
                    
                    # Add value labels on bars
                    for bar, count in zip(bars, counts):
                        ax8.text(bar.get_x() + bar.get_width()/2, bar.get_height() + max(counts) * 0.01,
                               str(count), ha='center', va='bottom', fontsize=9)
            
            plt.suptitle('Signal Analyzer - Section Performance Analysis', fontsize=18, fontweight='bold', y=0.98)
            
            # Save the plot
            plot_file = f'performance_reports/section_performance_charts_{timestamp}.png'
            plt.savefig(plot_file, dpi=300, bbox_inches='tight', facecolor='white')
            plt.close()
            
            logger.info(f"Performance visualizations saved to: {plot_file}")
            
        except Exception as e:
            logger.error(f"Error generating performance visualizations: {e}")
            # Create a simple error plot
            plt.figure(figsize=(10, 6))
            plt.text(0.5, 0.5, f'Error generating visualizations: {str(e)}', 
                   ha='center', va='center', fontsize=12)
            plt.savefig(f'performance_reports/section_performance_charts_{timestamp}.png', 
                      dpi=300, bbox_inches='tight')
            plt.close()
            
    def _generate_dependency_visualization(self, timestamp):
        """Generate section dependency graph visualization."""
        try:
            import networkx as nx
            
            if not self.section_dependencies:
                # Create empty graph placeholder
                plt.figure(figsize=(10, 6))
                plt.text(0.5, 0.5, 'No section dependencies detected', 
                       ha='center', va='center', fontsize=16)
                plt.title('Section Dependencies Graph', fontsize=14, fontweight='bold')
                plt.axis('off')
                plt.savefig(f'performance_reports/section_dependencies_{timestamp}.png', 
                          dpi=300, bbox_inches='tight')
                plt.close()
                return
            
            # Create directed graph
            G = nx.DiGraph()
            
            with self.lock:
                # Add nodes and edges
                for parent, children in self.section_dependencies.items():
                    if len(children) > 1:  # Only show sections with multiple dependencies
                        for child in children:
                            G.add_edge(parent, child)
            
            if G.number_of_nodes() == 0:
                plt.figure(figsize=(10, 6))
                plt.text(0.5, 0.5, 'No significant dependencies to visualize', 
                       ha='center', va='center', fontsize=16)
                plt.title('Section Dependencies Graph', fontsize=14, fontweight='bold')
                plt.axis('off')
                plt.savefig(f'performance_reports/section_dependencies_{timestamp}.png', 
                          dpi=300, bbox_inches='tight')
                plt.close()
                return
            
            # Limit to most connected nodes for readability
            if G.number_of_nodes() > 30:
                # Keep nodes with highest degree (most connections)
                degrees = dict(G.degree())
                top_nodes = sorted(degrees.items(), key=lambda x: x[1], reverse=True)[:30]
                top_node_names = [node for node, _ in top_nodes]
                G = G.subgraph(top_node_names)
            
            plt.figure(figsize=(16, 12))
            
            # Calculate layout
            try:
                pos = nx.spring_layout(G, k=3, iterations=50, seed=42)
            except:
                pos = nx.random_layout(G, seed=42)
            
            # Calculate node sizes based on degree
            degrees = dict(G.degree())
            max_degree = max(degrees.values()) if degrees else 1
            node_sizes = [min(degrees.get(node, 1) * 300, 2000) for node in G.nodes()]
            
            # Calculate node colors based on category
            categories = {}
            category_colors = {}
            color_map = plt.cm.Set3
            color_idx = 0
            
            for node in G.nodes():
                category = node.split('.')[0] if '.' in node else 'general'
                if category not in category_colors:
                    category_colors[category] = color_map(color_idx / 10)
                    color_idx += 1
                categories[node] = category
            
            node_colors = [category_colors[categories[node]] for node in G.nodes()]
            
            # Draw the graph
            nx.draw_networkx_nodes(G, pos, node_size=node_sizes, node_color=node_colors, 
                                 alpha=0.8, linewidths=2, edgecolors='black')
            
            # Draw edges with varying thickness based on frequency
            edge_weights = []
            for parent, children in self.section_dependencies.items():
                for child in children:
                    if G.has_edge(parent, child):
                        edge_weights.append(min(len(children), 5))  # Cap at 5 for readability
            
            if edge_weights:
                nx.draw_networkx_edges(G, pos, width=edge_weights, alpha=0.6, 
                                     edge_color='gray', arrows=True, arrowsize=20)
            
            # Draw labels (shortened names)
            labels = {}
            for node in G.nodes():
                # Show only the last part of the section name
                short_name = node.split('.')[-1]
                if len(short_name) > 12:
                    short_name = short_name[:10] + '..'
                labels[node] = short_name
            
            nx.draw_networkx_labels(G, pos, labels, font_size=8, font_weight='bold')
            
            plt.title('Section Dependencies Graph\\n(Node size = degree, Color = category)', 
                     fontsize=16, fontweight='bold', pad=20)
            
            # Create legend for categories
            legend_elements = [plt.Line2D([0], [0], marker='o', color='w', 
                                        markerfacecolor=color, markersize=10, label=category)
                             for category, color in category_colors.items()]
            plt.legend(handles=legend_elements, loc='upper left', bbox_to_anchor=(1, 1))
            
            plt.axis('off')
            plt.tight_layout()
            
            # Save the plot
            plot_file = f'performance_reports/section_dependencies_{timestamp}.png'
            plt.savefig(plot_file, dpi=300, bbox_inches='tight', facecolor='white')
            plt.close()
            
            logger.info(f"Dependencies visualization saved to: {plot_file}")
            
        except Exception as e:
            logger.error(f"Error generating dependencies visualization: {e}")
            # Create error placeholder
            plt.figure(figsize=(10, 6))
            plt.text(0.5, 0.5, f'Error generating dependencies graph: {str(e)}', 
                   ha='center', va='center', fontsize=12)
            plt.savefig(f'performance_reports/section_dependencies_{timestamp}.png', 
                      dpi=300, bbox_inches='tight')
            plt.close()
            
    def get_performance_summary(self):
        """Get comprehensive performance summary."""
        with self.lock:
            if not self.section_stats:
                return {}
                
            total_sections = len(self.section_stats)
            total_calls = sum(stats['count'] for stats in self.section_stats.values())
            total_time = sum(stats['total_time'] for stats in self.section_stats.values())
            avg_time = total_time / total_calls if total_calls > 0 else 0
            
            return {
                'monitoring_duration': time.time() - self.monitoring_start_time,
                'unique_sections': total_sections,
                'total_section_calls': total_calls,
                'total_execution_time': total_time,
                'average_section_time': avg_time,
                'slow_sections_detected': len(self.slow_sections),
                'memory_intensive_sections': len(self.memory_intensive_sections),
                'sections_with_dependencies': len(self.section_dependencies),
                'sections_with_errors': sum(1 for stats in self.section_stats.values() if stats['error_count'] > 0)
            }
            
    def reset_monitoring_data(self):
        """Reset all monitoring data."""
        with self.lock:
            self.active_sections.clear()
            self.completed_sections.clear()
            self.section_stats.clear()
            self.hot_sections.clear()
            self.slow_sections.clear()
            self.memory_intensive_sections.clear()
            self.section_dependencies.clear()
            self.performance_trends.clear()
            self.monitoring_start_time = time.time()
            self.total_sections_monitored = 0
            
        logger.info("Section monitoring data reset")


# Global monitor instance
_global_monitor = AdvancedSectionMonitor()

# Easy-to-use context managers that use the global monitor
@contextlib.contextmanager
def monitor_section(section_name: str, category: str = "general"):
    """
    Easy-to-use context manager for section monitoring.
    
    Usage:
        with monitor_section('data_processing', 'signal'):
            result = process_complex_data()
    """
    with _global_monitor.monitor_section(section_name, category) as section:
        yield section

@contextlib.contextmanager
def monitor_memory_section(section_name: str, category: str = "general"):
    """
    Easy-to-use context manager for memory monitoring.
    
    Usage:
        with monitor_memory_section('load_large_file', 'io'):
            data = load_huge_dataset()
    """
    with _global_monitor.monitor_memory_section(section_name, category) as section:
        yield section

@contextlib.contextmanager
def monitor_io_section(section_name: str, filepath: str = None, operation: str = "read"):
    """
    Easy-to-use context manager for I/O monitoring.
    
    Usage:
        with monitor_io_section('load_atf', '/path/to/file.atf', 'read'):
            data = load_file(filepath)
    """
    with _global_monitor.monitor_io_section(section_name, filepath, operation) as section:
        yield section

# Convenience functions
def get_section_summary():
    """Get section performance summary."""
    summary = _global_monitor.get_performance_summary()
    if summary:
        print("\\n📊 ADVANCED SECTION MONITORING SUMMARY:")
        print("=" * 50)
        for key, value in summary.items():
            if isinstance(value, float):
                if 'time' in key.lower():
                    print(f"  {key.replace('_', ' ').title()}: {value:.3f}s")
                else:
                    print(f"  {key.replace('_', ' ').title()}: {value:.2f}")
            else:
                print(f"  {key.replace('_', ' ').title()}: {value:,}")
    return summary

def generate_comprehensive_report():
    """Generate comprehensive section monitoring report."""
    return _global_monitor.generate_comprehensive_report()

def reset_section_monitoring():
    """Reset all section monitoring data."""
    _global_monitor.reset_monitoring_data()

# Auto-cleanup and reporting
def _section_monitoring_cleanup():
    """Cleanup function called on application exit."""
    try:
        summary = get_section_summary()
        if summary and summary.get('total_section_calls', 0) > 0:
            print("\\n🎯 Generating comprehensive section monitoring report...")
            report_info = generate_comprehensive_report()
            print(f"📁 Reports saved: {report_info.get('html_report', 'N/A')}")
    except Exception as e:
        logger.error(f"Error in section monitoring cleanup: {e}")

import atexit
atexit.register(_section_monitoring_cleanup)