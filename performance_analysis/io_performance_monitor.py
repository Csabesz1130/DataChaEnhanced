#!/usr/bin/env python3
"""
I/O Performance Monitor for Signal Analyzer Application
=====================================================

Advanced monitoring system for detecting and analyzing file I/O performance issues,
disk bottlenecks, and data transfer optimization opportunities.

Features:
    - Real-time file I/O operation monitoring
    - Disk read/write performance analysis
    - Network I/O tracking (if applicable)
    - File access pattern analysis
    - Transfer rate and throughput monitoring
    - Cache effectiveness analysis
    - Storage bottleneck identification
    - Interactive visualizations with I/O timelines
    - Automated I/O optimization recommendations

Usage:
    # Option 1: Automatic monitoring
    from performance_analysis.io_performance_monitor import IOPerformanceMonitor
    
    monitor = IOPerformanceMonitor()
    monitor.start_monitoring()
    # ... run your application with file operations ...
    monitor.stop_monitoring()
    
    # Option 2: Context manager for specific operations
    with IOPerformanceMonitor() as monitor:
        data = load_large_file('experiment.atf')
        save_results('output.json', results)
    
    # Option 3: Monitor specific I/O operations
    @monitor_io_operation
    def load_atf_file(filepath):
        return AtfHandler(filepath).load_atf()
"""

import time
import os
import threading
import psutil
from collections import defaultdict, deque
from datetime import datetime, timedelta
import json
import logging
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np
from typing import Dict, List, Any, Optional, Callable, Union
import hashlib
import contextlib
import functools

# Configure matplotlib for non-interactive use
plt.switch_backend('Agg')

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class IOOperation:
    """Represents a single I/O operation with comprehensive metrics."""
    
    def __init__(self, operation_type: str, filepath: str, operation_mode: str = 'unknown'):
        self.operation_type = operation_type  # 'read', 'write', 'delete', 'create'
        self.filepath = Path(filepath) if filepath else None
        self.operation_mode = operation_mode  # 'sequential', 'random', 'append'
        self.start_time = time.time()
        self.end_time = None
        self.duration = None
        
        # File metrics
        self.file_size = None
        self.bytes_transferred = 0
        self.transfer_rate = None  # bytes/second
        
        # System metrics
        self.cpu_percent_start = None
        self.cpu_percent_end = None
        self.memory_usage_start = None
        self.memory_usage_end = None
        self.disk_io_start = None
        self.disk_io_end = None
        
        # Performance metrics
        self.cache_hit = None
        self.disk_utilization = None
        self.queue_depth = None
        self.latency = None
        
        # Error tracking
        self.error_occurred = False
        self.error_message = None
        
        # Thread info
        self.thread_id = threading.get_ident()
        
        self._capture_start_metrics()
        
    def _capture_start_metrics(self):
        """Capture system metrics at operation start."""
        try:
            process = psutil.Process()
            self.cpu_percent_start = process.cpu_percent()
            self.memory_usage_start = process.memory_info().rss
            
            # Get disk I/O counters
            self.disk_io_start = psutil.disk_io_counters()
            
            # Get file size if file exists
            if self.filepath and self.filepath.exists():
                self.file_size = self.filepath.stat().st_size
                
        except Exception as e:
            logger.debug(f"Error capturing start metrics: {e}")
            
    def end_operation(self, bytes_transferred: int = None, error: Exception = None):
        """Mark the end of the I/O operation."""
        self.end_time = time.time()
        self.duration = self.end_time - self.start_time
        
        if bytes_transferred is not None:
            self.bytes_transferred = bytes_transferred
            
        if error:
            self.error_occurred = True
            self.error_message = str(error)
            
        self._capture_end_metrics()
        self._calculate_performance_metrics()
        
    def _capture_end_metrics(self):
        """Capture system metrics at operation end."""
        try:
            process = psutil.Process()
            self.cpu_percent_end = process.cpu_percent()
            self.memory_usage_end = process.memory_info().rss
            
            # Get disk I/O counters
            self.disk_io_end = psutil.disk_io_counters()
            
            # Update file size if changed
            if self.filepath and self.filepath.exists():
                new_size = self.filepath.stat().st_size
                if self.operation_type in ['write', 'create']:
                    self.bytes_transferred = max(self.bytes_transferred, new_size - (self.file_size or 0))
                    
        except Exception as e:
            logger.debug(f"Error capturing end metrics: {e}")
            
    def _calculate_performance_metrics(self):
        """Calculate derived performance metrics."""
        if self.duration and self.duration > 0:
            # Calculate transfer rate
            if self.bytes_transferred > 0:
                self.transfer_rate = self.bytes_transferred / self.duration
                
            # Estimate cache effectiveness
            if self.disk_io_start and self.disk_io_end and self.operation_type == 'read':
                disk_reads = self.disk_io_end.read_count - self.disk_io_start.read_count
                if disk_reads == 0 and self.bytes_transferred > 0:
                    self.cache_hit = True
                else:
                    self.cache_hit = False
                    
            # Calculate latency (time to first byte)
            if self.bytes_transferred > 0:
                self.latency = min(self.duration, 0.1)  # Cap at 100ms for estimation
                
    def get_performance_score(self):
        """Calculate a performance score (0-100) for this operation."""
        if not self.duration or self.error_occurred:
            return 0
            
        score = 100
        
        # Penalize slow operations
        if self.duration > 1.0:  # > 1 second
            score -= min(50, self.duration * 10)
            
        # Reward high transfer rates
        if self.transfer_rate:
            # Good: >10MB/s, Excellent: >50MB/s
            mb_per_sec = self.transfer_rate / (1024 * 1024)
            if mb_per_sec > 50:
                score += 10
            elif mb_per_sec > 10:
                score += 5
            elif mb_per_sec < 1:
                score -= 20
                
        # Reward cache hits
        if self.cache_hit:
            score += 15
            
        return max(0, min(100, score))
        
    def to_dict(self):
        """Convert to dictionary for serialization."""
        return {
            'operation_type': self.operation_type,
            'filepath': str(self.filepath) if self.filepath else None,
            'operation_mode': self.operation_mode,
            'start_time': self.start_time,
            'end_time': self.end_time,
            'duration': self.duration,
            'file_size': self.file_size,
            'bytes_transferred': self.bytes_transferred,
            'transfer_rate': self.transfer_rate,
            'cache_hit': self.cache_hit,
            'latency': self.latency,
            'error_occurred': self.error_occurred,
            'error_message': self.error_message,
            'performance_score': self.get_performance_score(),
            'thread_id': self.thread_id
        }


class IOPerformanceMonitor:
    """Advanced I/O performance monitoring system."""
    
    def __init__(self, 
                 slow_io_threshold=1.0,      # 1 second
                 large_file_threshold=10*1024*1024,  # 10MB
                 sample_interval=0.1,        # 100ms
                 max_operations=5000):
        """
        Initialize I/O performance monitor.
        
        Args:
            slow_io_threshold: Threshold for slow I/O operations (seconds)
            large_file_threshold: Threshold for large file operations (bytes)
            sample_interval: System monitoring sample interval (seconds)
            max_operations: Maximum number of operations to store
        """
        self.slow_io_threshold = slow_io_threshold
        self.large_file_threshold = large_file_threshold
        self.sample_interval = sample_interval
        self.max_operations = max_operations
        
        # Operation tracking
        self.io_operations = deque(maxlen=max_operations)
        self.active_operations = {}  # Thread ID -> IOOperation
        self.operation_stats = defaultdict(lambda: {
            'count': 0,
            'total_time': 0,
            'avg_time': 0,
            'max_time': 0,
            'min_time': float('inf'),
            'total_bytes': 0,
            'avg_transfer_rate': 0,
            'error_count': 0,
            'cache_hits': 0,
            'cache_misses': 0
        })
        
        # Performance metrics
        self.slow_operations = deque(maxlen=500)
        self.large_file_operations = deque(maxlen=200)
        self.file_access_patterns = defaultdict(list)  # filepath -> [access_times]
        self.directory_hotspots = defaultdict(int)      # directory -> access_count
        
        # System monitoring
        self.disk_usage_samples = deque(maxlen=1000)
        self.network_io_samples = deque(maxlen=1000)
        self.monitoring = False
        self.monitor_thread = None
        self.start_time = None
        
        # Thread safety
        self.lock = threading.Lock()
        
        # Performance counters
        self.total_operations = 0
        self.total_bytes_transferred = 0
        self.slow_operations_detected = 0
        self.cache_hit_rate = 0
        
        logger.info("I/O Performance Monitor initialized")
        
    def start_monitoring(self):
        """Start I/O performance monitoring."""
        if self.monitoring:
            logger.warning("I/O monitoring already active")
            return
            
        self.monitoring = True
        self.start_time = time.time()
        
        # Start system monitoring thread
        self.monitor_thread = threading.Thread(
            target=self._monitor_system_io,
            name="IOPerformanceMonitor",
            daemon=True
        )
        self.monitor_thread.start()
        
        logger.info("💾 I/O performance monitoring started")
        
    def stop_monitoring(self):
        """Stop I/O performance monitoring."""
        if not self.monitoring:
            return
            
        self.monitoring = False
        
        if self.monitor_thread and self.monitor_thread.is_alive():
            self.monitor_thread.join(timeout=2.0)
            
        self._analyze_io_performance()
        logger.info("🛑 I/O performance monitoring stopped")
        
    def _monitor_system_io(self):
        """Monitor system-wide I/O performance."""
        logger.debug("I/O system monitoring thread started")
        
        while self.monitoring:
            try:
                # Sample disk usage
                disk_usage = psutil.disk_usage('/')
                disk_io = psutil.disk_io_counters()
                
                sample = {
                    'timestamp': time.time(),
                    'disk_usage_percent': (disk_usage.used / disk_usage.total) * 100,
                    'disk_read_rate': 0,  # Will be calculated from deltas
                    'disk_write_rate': 0,
                    'disk_io_util': 0
                }
                
                # Calculate rates from previous sample
                if self.disk_usage_samples:
                    prev_sample = self.disk_usage_samples[-1]
                    time_delta = sample['timestamp'] - prev_sample['timestamp']
                    
                    if time_delta > 0:
                        # Note: This is simplified - real implementation would track per-disk
                        sample['disk_read_rate'] = disk_io.read_bytes / time_delta if disk_io else 0
                        sample['disk_write_rate'] = disk_io.write_bytes / time_delta if disk_io else 0
                
                self.disk_usage_samples.append(sample)
                
                # Monitor network I/O if applicable
                try:
                    net_io = psutil.net_io_counters()
                    net_sample = {
                        'timestamp': time.time(),
                        'bytes_sent': net_io.bytes_sent,
                        'bytes_recv': net_io.bytes_recv,
                        'packets_sent': net_io.packets_sent,
                        'packets_recv': net_io.packets_recv
                    }
                    self.network_io_samples.append(net_sample)
                except:
                    pass  # Network monitoring optional
                
                time.sleep(self.sample_interval)
                
            except Exception as e:
                logger.error(f"Error in I/O monitoring: {e}")
                time.sleep(self.sample_interval)
                
        logger.debug("I/O system monitoring thread ended")
        
    @contextlib.contextmanager
    def monitor_io_operation(self, operation_type: str, filepath: str, operation_mode: str = 'unknown'):
        """
        Context manager to monitor a specific I/O operation.
        
        Args:
            operation_type: Type of operation ('read', 'write', 'delete', 'create')
            filepath: Path to the file being accessed
            operation_mode: Mode of operation ('sequential', 'random', 'append')
            
        Usage:
            with monitor.monitor_io_operation('read', 'data.atf'):
                data = open('data.atf').read()
        """
        operation = IOOperation(operation_type, filepath, operation_mode)
        thread_id = threading.get_ident()
        
        with self.lock:
            self.active_operations[thread_id] = operation
            
        try:
            yield operation
            
        except Exception as e:
            operation.end_operation(error=e)
            raise
            
        finally:
            if not operation.end_time:  # If not already ended due to exception
                operation.end_operation()
                
            with self.lock:
                # Remove from active operations
                self.active_operations.pop(thread_id, None)
                
                # Add to completed operations
                self.io_operations.append(operation)
                self.total_operations += 1
                self.total_bytes_transferred += operation.bytes_transferred
                
                # Update statistics
                self._update_operation_stats(operation)
                
                # Track access patterns
                if operation.filepath:
                    self.file_access_patterns[str(operation.filepath)].append(operation.start_time)
                    self.directory_hotspots[str(operation.filepath.parent)] += 1
                
                # Check for performance issues
                self._analyze_operation_performance(operation)
                
    def _update_operation_stats(self, operation: IOOperation):
        """Update statistics for an I/O operation."""
        op_type = operation.operation_type
        stats = self.operation_stats[op_type]
        
        stats['count'] += 1
        stats['total_time'] += operation.duration
        stats['avg_time'] = stats['total_time'] / stats['count']
        stats['max_time'] = max(stats['max_time'], operation.duration)
        stats['min_time'] = min(stats['min_time'], operation.duration)
        stats['total_bytes'] += operation.bytes_transferred
        
        if operation.transfer_rate:
            # Calculate weighted average transfer rate
            total_ops = stats['count']
            prev_avg = stats['avg_transfer_rate']
            stats['avg_transfer_rate'] = ((prev_avg * (total_ops - 1)) + operation.transfer_rate) / total_ops
            
        if operation.error_occurred:
            stats['error_count'] += 1
            
        if operation.cache_hit is not None:
            if operation.cache_hit:
                stats['cache_hits'] += 1
            else:
                stats['cache_misses'] += 1
                
    def _analyze_operation_performance(self, operation: IOOperation):
        """Analyze individual operation performance."""
        # Check for slow operations
        if operation.duration > self.slow_io_threshold:
            self.slow_operations.append({
                'operation': operation.to_dict(),
                'timestamp': datetime.now(),
                'severity': 'critical' if operation.duration > 5.0 else 'major' if operation.duration > 2.0 else 'minor'
            })
            self.slow_operations_detected += 1
            
            logger.warning(f"🐌 Slow I/O operation: {operation.operation_type} {operation.filepath} took {operation.duration:.3f}s")
            
        # Check for large file operations
        if operation.file_size and operation.file_size > self.large_file_threshold:
            self.large_file_operations.append({
                'operation': operation.to_dict(),
                'file_size_mb': operation.file_size / (1024 * 1024),
                'transfer_rate_mbps': (operation.transfer_rate / (1024 * 1024)) if operation.transfer_rate else 0
            })
            
    def _analyze_io_performance(self):
        """Analyze overall I/O performance."""
        if not self.io_operations:
            logger.info("No I/O operations to analyze")
            return
            
        total_ops = len(self.io_operations)
        monitoring_duration = time.time() - self.start_time if self.start_time else 0
        
        # Calculate performance metrics
        total_transfer_time = sum(op.duration for op in self.io_operations if op.duration)
        avg_operation_time = total_transfer_time / total_ops if total_ops > 0 else 0
        
        total_bytes = sum(op.bytes_transferred for op in self.io_operations)
        avg_transfer_rate = total_bytes / total_transfer_time if total_transfer_time > 0 else 0
        
        # Calculate cache hit rate
        cache_operations = [op for op in self.io_operations if op.cache_hit is not None]
        if cache_operations:
            cache_hits = sum(1 for op in cache_operations if op.cache_hit)
            self.cache_hit_rate = (cache_hits / len(cache_operations)) * 100
            
        logger.info(f"📊 I/O Performance Analysis:")
        logger.info(f"  Total operations: {total_ops}")
        logger.info(f"  Monitoring duration: {monitoring_duration:.1f}s")
        logger.info(f"  Average operation time: {avg_operation_time:.3f}s")
        logger.info(f"  Total data transferred: {total_bytes / (1024*1024):.1f} MB")
        logger.info(f"  Average transfer rate: {avg_transfer_rate / (1024*1024):.1f} MB/s")
        logger.info(f"  Cache hit rate: {self.cache_hit_rate:.1f}%")
        logger.info(f"  Slow operations: {len(self.slow_operations)}")
        
    def generate_io_report(self):
        """Generate comprehensive I/O performance report."""
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        
        # Create reports directory
        reports_dir = Path('performance_reports')
        reports_dir.mkdir(exist_ok=True)
        
        # Generate JSON report
        json_report = self._generate_json_report()
        json_file = reports_dir / f'io_performance_report_{timestamp}.json'
        with open(json_file, 'w') as f:
            json.dump(json_report, f, indent=2, default=str)
            
        # Generate HTML report
        html_report = self._generate_html_report()
        html_file = reports_dir / f'io_performance_report_{timestamp}.html'
        with open(html_file, 'w') as f:
            f.write(html_report)
            
        # Generate visualizations
        self._generate_io_visualizations(timestamp)
        
        logger.info(f"I/O performance reports generated:")
        logger.info(f"  JSON: {json_file}")
        logger.info(f"  HTML: {html_file}")
        
        return {
            'json_report': str(json_file),
            'html_report': str(html_file),
            'timestamp': timestamp
        }
        
    def _generate_json_report(self):
        """Generate JSON I/O performance report."""
        with self.lock:
            total_ops = len(self.io_operations)
            monitoring_duration = time.time() - self.start_time if self.start_time else 0
            
            # Calculate summary metrics
            total_bytes = sum(op.bytes_transferred for op in self.io_operations)
            avg_transfer_rate = (total_bytes / monitoring_duration) if monitoring_duration > 0 else 0
            
            # Top slow operations
            slow_ops_sorted = sorted(
                [slow['operation'] for slow in self.slow_operations],
                key=lambda x: x['duration'],
                reverse=True
            )[:10]
            
            # File access frequency analysis
            file_frequency = {
                filepath: len(access_times)
                for filepath, access_times in self.file_access_patterns.items()
            }
            top_accessed_files = sorted(file_frequency.items(), key=lambda x: x[1], reverse=True)[:10]
            
            # Directory hotspots
            top_directories = sorted(self.directory_hotspots.items(), key=lambda x: x[1], reverse=True)[:10]
            
            report = {
                'metadata': {
                    'timestamp': datetime.now().isoformat(),
                    'monitoring_duration': monitoring_duration,
                    'slow_io_threshold': self.slow_io_threshold,
                    'large_file_threshold': self.large_file_threshold
                },
                'summary': {
                    'total_io_operations': total_ops,
                    'total_bytes_transferred': total_bytes,
                    'total_mb_transferred': total_bytes / (1024 * 1024),
                    'average_transfer_rate_mbps': avg_transfer_rate / (1024 * 1024),
                    'slow_operations_detected': len(self.slow_operations),
                    'large_file_operations': len(self.large_file_operations),
                    'cache_hit_rate_percent': self.cache_hit_rate,
                    'error_rate_percent': (sum(stats['error_count'] for stats in self.operation_stats.values()) / total_ops * 100) if total_ops > 0 else 0
                },
                'operation_statistics': {
                    op_type: {
                        'count': stats['count'],
                        'avg_time_ms': stats['avg_time'] * 1000,
                        'max_time_ms': stats['max_time'] * 1000,
                        'total_mb': stats['total_bytes'] / (1024 * 1024),
                        'avg_transfer_rate_mbps': stats['avg_transfer_rate'] / (1024 * 1024),
                        'error_count': stats['error_count'],
                        'cache_hit_rate': (stats['cache_hits'] / (stats['cache_hits'] + stats['cache_misses']) * 100) if (stats['cache_hits'] + stats['cache_misses']) > 0 else 0
                    }
                    for op_type, stats in self.operation_stats.items()
                },
                'slow_operations': slow_ops_sorted,
                'large_file_operations': [op['operation'] for op in self.large_file_operations],
                'file_access_patterns': {
                    'top_accessed_files': [
                        {'filepath': filepath, 'access_count': count}
                        for filepath, count in top_accessed_files
                    ],
                    'directory_hotspots': [
                        {'directory': directory, 'access_count': count}
                        for directory, count in top_directories
                    ]
                },
                'recommendations': self._generate_io_recommendations()
            }
            
        return report
        
    def _generate_html_report(self):
        """Generate comprehensive HTML I/O report."""
        timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        
        with self.lock:
            total_ops = len(self.io_operations)
            total_bytes = sum(op.bytes_transferred for op in self.io_operations)
            total_mb = total_bytes / (1024 * 1024)
            monitoring_duration = time.time() - self.start_time if self.start_time else 0
            avg_transfer_rate = (total_bytes / monitoring_duration) if monitoring_duration > 0 else 0
            
        html = f"""
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>I/O Performance Report - Signal Analyzer</title>
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
            background: linear-gradient(135deg, #9C27B0 0%, #673AB7 100%);
            color: white; 
            padding: 40px 50px; 
        }}
        
        .header h1 {{ 
            font-size: 3em; 
            margin-bottom: 15px; 
            font-weight: 700;
        }}
        
        .content {{ padding: 50px; }}
        
        .metrics-grid {{ 
            display: grid; 
            grid-template-columns: repeat(auto-fit, minmax(280px, 1fr)); 
            gap: 30px; 
            margin: 40px 0;
        }}
        
        .metric-card {{ 
            background: linear-gradient(135deg, #f3e5f5 0%, #e1bee7 100%);
            padding: 30px; 
            border-radius: 15px; 
            text-align: center;
            border: 1px solid #ce93d8;
            transition: all 0.3s ease;
        }}
        
        .metric-card:hover {{
            transform: translateY(-5px);
            box-shadow: 0 10px 30px rgba(0,0,0,0.1);
        }}
        
        .metric-icon {{ 
            font-size: 3em; 
            margin-bottom: 15px; 
            display: block;
        }}
        
        .metric-value {{ 
            font-size: 2.5em; 
            font-weight: bold; 
            color: #7b1fa2; 
            margin-bottom: 10px;
        }}
        
        .metric-label {{ 
            font-size: 1.1em; 
            color: #666; 
            font-weight: 500;
        }}
        
        .section {{ 
            background: white;
            margin: 40px 0; 
            padding: 40px; 
            border-radius: 15px; 
            border-left: 5px solid #9C27B0;
            box-shadow: 0 5px 20px rgba(0,0,0,0.05);
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
            background: linear-gradient(135deg, #7b1fa2 0%, #512da8 100%);
            color: white; 
            padding: 18px; 
            text-align: left;
            font-weight: 600;
        }}
        
        .performance-table td {{ 
            border-bottom: 1px solid #f0f0f0; 
            padding: 15px 18px; 
        }}
        
        .performance-table tr:hover td {{ 
            background-color: #f3e5f5; 
        }}
        
        .chart-container {{ 
            text-align: center; 
            margin: 40px 0; 
            padding: 30px; 
            background: linear-gradient(135deg, #f3e5f5 0%, #e1bee7 100%);
            border-radius: 15px;
        }}
        
        .status-good {{ color: #4caf50; }}
        .status-warning {{ color: #ff9800; }}
        .status-critical {{ color: #f44336; }}
    </style>
</head>
<body>
    <div class="container">
        <div class="header">
            <h1>💾 I/O Performance Report</h1>
            <p>File and Data Transfer Performance Analysis for Signal Analyzer</p>
            <p>Generated: {timestamp}</p>
        </div>
        
        <div class="content">
            <div class="metrics-grid">
                <div class="metric-card">
                    <span class="metric-icon">📁</span>
                    <div class="metric-value">{total_ops:,}</div>
                    <div class="metric-label">I/O Operations</div>
                </div>
                <div class="metric-card">
                    <span class="metric-icon">📊</span>
                    <div class="metric-value">{total_mb:.1f}</div>
                    <div class="metric-label">Data Transferred</div>
                    <div class="metric-unit">MB</div>
                </div>
                <div class="metric-card">
                    <span class="metric-icon">⚡</span>
                    <div class="metric-value">{avg_transfer_rate/(1024*1024):.1f}</div>
                    <div class="metric-label">Avg Transfer Rate</div>
                    <div class="metric-unit">MB/s</div>
                </div>
                <div class="metric-card">
                    <span class="metric-icon">🎯</span>
                    <div class="metric-value {'status-good' if self.cache_hit_rate >= 80 else 'status-warning' if self.cache_hit_rate >= 50 else 'status-critical'}">{self.cache_hit_rate:.1f}%</div>
                    <div class="metric-label">Cache Hit Rate</div>
                </div>
                <div class="metric-card">
                    <span class="metric-icon">⏱️</span>
                    <div class="metric-value {'status-good' if len(self.slow_operations) == 0 else 'status-critical'}">{len(self.slow_operations)}</div>
                    <div class="metric-label">Slow Operations</div>
                </div>
                <div class="metric-card">
                    <span class="metric-icon">📈</span>
                    <div class="metric-value">{len(self.large_file_operations)}</div>
                    <div class="metric-label">Large File Ops</div>
                </div>
            </div>
            
            <div class="chart-container">
                <h3>📈 I/O Performance Timeline</h3>
                <img src="io_performance_timeline_{datetime.now().strftime('%Y%m%d_%H%M%S')}.png" 
                     alt="I/O Performance Timeline" style="max-width: 100%; height: auto;">
            </div>
            
            <div class="section">
                <h2>📋 Operation Type Performance</h2>
                {self._generate_operation_stats_table()}
            </div>
            
            <div class="section">
                <h2>🐌 Slow Operations Analysis</h2>
                {self._generate_slow_operations_table()}
            </div>
            
            <div class="section">
                <h2>📁 File Access Patterns</h2>
                {self._generate_file_access_table()}
            </div>
        </div>
    </div>
</body>
</html>
        """
        
        return html
        
    def _generate_operation_stats_table(self):
        """Generate HTML table for operation statistics."""
        if not self.operation_stats:
            return '<p>No I/O operation data available.</p>'
            
        html = '''
        <table class="performance-table">
            <thead>
                <tr>
                    <th>Operation Type</th>
                    <th>Count</th>
                    <th>Avg Time (ms)</th>
                    <th>Max Time (ms)</th>
                    <th>Total Data (MB)</th>
                    <th>Avg Rate (MB/s)</th>
                    <th>Cache Hit %</th>
                    <th>Errors</th>
                </tr>
            </thead>
            <tbody>
        '''
        
        for op_type, stats in sorted(self.operation_stats.items()):
            cache_hit_rate = 0
            if stats['cache_hits'] + stats['cache_misses'] > 0:
                cache_hit_rate = (stats['cache_hits'] / (stats['cache_hits'] + stats['cache_misses'])) * 100
                
            html += f'''
            <tr>
                <td><strong>{op_type.title()}</strong></td>
                <td>{stats['count']:,}</td>
                <td>{stats['avg_time']*1000:.2f}</td>
                <td>{stats['max_time']*1000:.2f}</td>
                <td>{stats['total_bytes']/(1024*1024):.2f}</td>
                <td>{stats['avg_transfer_rate']/(1024*1024):.2f}</td>
                <td>{cache_hit_rate:.1f}%</td>
                <td>{stats['error_count']}</td>
            </tr>
            '''
            
        html += '</tbody></table>'
        return html
        
    def _generate_slow_operations_table(self):
        """Generate HTML table for slow operations."""
        if not self.slow_operations:
            return '<p>No slow I/O operations detected. 🎉</p>'
            
        html = '''
        <table class="performance-table">
            <thead>
                <tr>
                    <th>File Path</th>
                    <th>Operation</th>
                    <th>Duration (s)</th>
                    <th>File Size (MB)</th>
                    <th>Transfer Rate (MB/s)</th>
                    <th>Severity</th>
                </tr>
            </thead>
            <tbody>
        '''
        
        for slow_op in sorted(self.slow_operations, key=lambda x: x['operation']['duration'], reverse=True)[:20]:
            op = slow_op['operation']
            file_size_mb = (op['file_size'] / (1024*1024)) if op['file_size'] else 0
            transfer_rate_mb = (op['transfer_rate'] / (1024*1024)) if op['transfer_rate'] else 0
            
            severity_class = {
                'critical': 'status-critical',
                'major': 'status-warning', 
                'minor': 'status-good'
            }.get(slow_op['severity'], '')
            
            filepath = Path(op['filepath']).name if op['filepath'] else 'Unknown'
            
            html += f'''
            <tr>
                <td title="{op['filepath'] or 'Unknown'}">{filepath}</td>
                <td>{op['operation_type'].title()}</td>
                <td>{op['duration']:.3f}</td>
                <td>{file_size_mb:.2f}</td>
                <td>{transfer_rate_mb:.2f}</td>
                <td><span class="{severity_class}">{slow_op['severity'].title()}</span></td>
            </tr>
            '''
            
        html += '</tbody></table>'
        return html
        
    def _generate_file_access_table(self):
        """Generate HTML table for file access patterns."""
        if not self.file_access_patterns:
            return '<p>No file access pattern data available.</p>'
            
        # Get top accessed files
        file_frequency = {
            filepath: len(access_times)
            for filepath, access_times in self.file_access_patterns.items()
        }
        top_files = sorted(file_frequency.items(), key=lambda x: x[1], reverse=True)[:15]
        
        html = '''
        <table class="performance-table">
            <thead>
                <tr>
                    <th>File Path</th>
                    <th>Access Count</th>
                    <th>Directory</th>
                    <th>Access Pattern</th>
                </tr>
            </thead>
            <tbody>
        '''
        
        for filepath, count in top_files:
            file_path = Path(filepath)
            directory = str(file_path.parent)
            
            # Determine access pattern
            access_times = self.file_access_patterns[filepath]
            if len(access_times) > 1:
                intervals = [access_times[i] - access_times[i-1] for i in range(1, len(access_times))]
                avg_interval = sum(intervals) / len(intervals)
                
                if avg_interval < 1:
                    pattern = "High Frequency"
                elif avg_interval < 10:
                    pattern = "Regular"
                else:
                    pattern = "Occasional"
            else:
                pattern = "Single Access"
                
            html += f'''
            <tr>
                <td title="{filepath}">{file_path.name}</td>
                <td>{count}</td>
                <td title="{directory}">{Path(directory).name}</td>
                <td>{pattern}</td>
            </tr>
            '''
            
        html += '</tbody></table>'
        return html
        
    def _generate_io_recommendations(self):
        """Generate I/O optimization recommendations."""
        recommendations = []
        
        with self.lock:
            if not self.io_operations:
                return ["No I/O operation data available for analysis"]
                
            # Analyze slow operations
            if self.slow_operations:
                slow_by_type = defaultdict(int)
                for slow_op in self.slow_operations:
                    slow_by_type[slow_op['operation']['operation_type']] += 1
                    
                worst_op_type = max(slow_by_type.items(), key=lambda x: x[1])
                recommendations.append(
                    f"🐌 <strong>Optimize {worst_op_type[0]} operations:</strong> "
                    f"{worst_op_type[1]} slow operations detected"
                )
                
            # Analyze cache performance
            if self.cache_hit_rate < 70:
                recommendations.append(
                    f"💾 <strong>Improve caching:</strong> "
                    f"Cache hit rate is {self.cache_hit_rate:.1f}% (target: >80%)"
                )
                
            # Analyze file access patterns
            frequent_files = [
                filepath for filepath, access_times in self.file_access_patterns.items()
                if len(access_times) > 10
            ]
            
            if frequent_files:
                recommendations.append(
                    f"📁 <strong>Consider file caching:</strong> "
                    f"{len(frequent_files)} files accessed frequently"
                )
                
            # Analyze large file operations
            if self.large_file_operations:
                avg_rate = sum(
                    op['transfer_rate_mbps'] for op in self.large_file_operations
                ) / len(self.large_file_operations)
                
                if avg_rate < 10:  # < 10 MB/s
                    recommendations.append(
                        f"⚡ <strong>Improve large file handling:</strong> "
                        f"Average transfer rate is {avg_rate:.1f} MB/s"
                    )
                    
            # General recommendations
            if len(recommendations) < 3:
                recommendations.extend([
                    "📈 <strong>Monitor disk usage:</strong> Ensure adequate free space for optimal performance",
                    "🔄 <strong>Consider async I/O:</strong> For better responsiveness during large operations",
                    "🗂️ <strong>Organize data files:</strong> Group frequently accessed files together"
                ])
                
        return recommendations
        
    def _generate_io_visualizations(self, timestamp):
        """Generate I/O performance visualization charts."""
        try:
            fig, axes = plt.subplots(2, 2, figsize=(16, 12))
            fig.suptitle('I/O Performance Analysis', fontsize=16, fontweight='bold')
            
            with self.lock:
                if not self.io_operations:
                    plt.text(0.5, 0.5, 'No I/O data available for visualization', 
                           ha='center', va='center', transform=plt.gca().transAxes, fontsize=16)
                    plt.savefig(f'performance_reports/io_performance_timeline_{timestamp}.png', 
                              dpi=300, bbox_inches='tight')
                    plt.close()
                    return
                    
                # 1. Operation timeline
                ax = axes[0, 0]
                if self.io_operations:
                    start_time = min(op.start_time for op in self.io_operations)
                    times = [(op.start_time - start_time) for op in self.io_operations]
                    durations = [op.duration * 1000 for op in self.io_operations]  # Convert to ms
                    
                    colors = ['red' if d > self.slow_io_threshold*1000 else 'blue' for d in durations]
                    ax.scatter(times, durations, c=colors, alpha=0.6, s=20)
                    ax.set_xlabel('Time (seconds)')
                    ax.set_ylabel('Operation Duration (ms)')
                    ax.set_title('I/O Operation Timeline')
                    ax.grid(True, alpha=0.3)
                    
                # 2. Transfer rate distribution
                ax = axes[0, 1]
                transfer_rates = [op.transfer_rate/(1024*1024) for op in self.io_operations 
                                if op.transfer_rate and op.transfer_rate > 0]
                if transfer_rates:
                    ax.hist(transfer_rates, bins=30, alpha=0.7, color='green', edgecolor='black')
                    ax.set_xlabel('Transfer Rate (MB/s)')
                    ax.set_ylabel('Number of Operations')
                    ax.set_title('Transfer Rate Distribution')
                    ax.grid(True, alpha=0.3)
                    
                # 3. Operation type breakdown
                ax = axes[1, 0]
                op_counts = defaultdict(int)
                for op in self.io_operations:
                    op_counts[op.operation_type] += 1
                    
                if op_counts:
                    labels = list(op_counts.keys())
                    sizes = list(op_counts.values())
                    ax.pie(sizes, labels=labels, autopct='%1.1f%%', startangle=90)
                    ax.set_title('Operation Type Distribution')
                    
                # 4. File size vs duration
                ax = axes[1, 1]
                file_sizes = []
                durations = []
                for op in self.io_operations:
                    if op.file_size and op.duration:
                        file_sizes.append(op.file_size / (1024*1024))  # MB
                        durations.append(op.duration * 1000)  # ms
                        
                if file_sizes and durations:
                    ax.scatter(file_sizes, durations, alpha=0.6, color='purple')
                    ax.set_xlabel('File Size (MB)')
                    ax.set_ylabel('Duration (ms)')
                    ax.set_title('File Size vs Operation Duration')
                    ax.set_xscale('log')
                    ax.set_yscale('log')
                    ax.grid(True, alpha=0.3)
            
            plt.tight_layout()
            
            # Save the plot
            plot_file = f'performance_reports/io_performance_timeline_{timestamp}.png'
            plt.savefig(plot_file, dpi=300, bbox_inches='tight', facecolor='white')
            plt.close()
            
            logger.info(f"I/O performance visualizations saved to: {plot_file}")
            
        except Exception as e:
            logger.error(f"Error generating I/O visualizations: {e}")
            
    def __enter__(self):
        """Context manager entry."""
        self.start_monitoring()
        return self
        
    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit."""
        self.stop_monitoring()


# Decorator for monitoring I/O operations
def monitor_io_operation(operation_type: str = 'unknown', operation_mode: str = 'unknown'):
    """
    Decorator to monitor I/O operations.
    
    Usage:
        @monitor_io_operation('read', 'sequential')
        def load_data_file(filepath):
            with open(filepath, 'r') as f:
                return f.read()
    """
    def decorator(func):
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            # Try to extract filepath from function arguments
            filepath = None
            if args:
                # Assume first argument might be filepath
                if isinstance(args[0], (str, Path)):
                    filepath = str(args[0])
                    
            # Get global monitor instance
            monitor = IOPerformanceMonitor()
            
            with monitor.monitor_io_operation(operation_type, filepath or func.__name__, operation_mode):
                return func(*args, **kwargs)
                
        return wrapper
    return decorator


# Global monitor instance for easy access
_global_io_monitor = IOPerformanceMonitor()

def start_io_monitoring():
    """Start global I/O monitoring."""
    _global_io_monitor.start_monitoring()
    
def stop_io_monitoring():
    """Stop global I/O monitoring."""
    _global_io_monitor.stop_monitoring()
    
def get_io_monitor():
    """Get the global I/O monitor instance."""
    return _global_io_monitor