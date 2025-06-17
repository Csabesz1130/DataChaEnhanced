#!/usr/bin/env python3
"""
Memory Leak Detector for Signal Analyzer Application
===================================================

Advanced memory leak detection and analysis specifically designed for
Python GUI applications with heavy data processing and plotting.

Features:
    - Real-time memory growth monitoring
    - Object reference tracking
    - Matplotlib figure leak detection
    - NumPy array memory analysis
    - Garbage collection monitoring
    - Memory snapshot comparison
    - Automated leak pattern detection
    - Memory optimization recommendations

Usage:
    python memory_leak_detector.py --duration 300 --interval 30
    
Integration:
    from memory_leak_detector import MemoryLeakDetector
    detector = MemoryLeakDetector()
    detector.start_monitoring(duration=300)
"""

import time
import gc
import threading
import tracemalloc
import weakref
import sys
import os
from collections import defaultdict, deque
from datetime import datetime, timedelta
import logging
import psutil
import json
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class MemorySnapshot:
    """Memory snapshot with detailed analysis capabilities."""
    
    def __init__(self, name=""):
        self.name = name
        self.timestamp = time.time()
        self.datetime = datetime.now()
        
        # System memory info
        try:
            process = psutil.Process()
            self.memory_info = process.memory_info()
            self.memory_percent = process.memory_percent()
        except:
            self.memory_info = None
            self.memory_percent = 0
            
        # Python object counts
        self.object_counts = self._get_object_counts()
        
        # Garbage collection stats
        self.gc_stats = gc.get_stats()
        
        # Tracemalloc snapshot if available
        self.tracemalloc_snapshot = None
        if tracemalloc.is_tracing():
            self.tracemalloc_snapshot = tracemalloc.take_snapshot()
            
    def _get_object_counts(self):
        """Get counts of different object types."""
        counts = defaultdict(int)
        
        # Count objects by type
        for obj in gc.get_objects():
            obj_type = type(obj).__name__
            counts[obj_type] += 1
            
        return dict(counts)
        
    def get_memory_mb(self):
        """Get memory usage in MB."""
        if self.memory_info:
            return self.memory_info.rss / 1024 / 1024
        return 0
        
    def compare_to(self, other_snapshot):
        """Compare this snapshot to another snapshot."""
        if not isinstance(other_snapshot, MemorySnapshot):
            raise ValueError("Can only compare to another MemorySnapshot")
            
        comparison = {
            'time_diff': self.timestamp - other_snapshot.timestamp,
            'memory_diff_mb': self.get_memory_mb() - other_snapshot.get_memory_mb(),
            'memory_percent_diff': self.memory_percent - other_snapshot.memory_percent,
            'object_count_changes': {},
            'new_object_types': [],
            'gc_collection_changes': []
        }
        
        # Compare object counts
        for obj_type, count in self.object_counts.items():
            other_count = other_snapshot.object_counts.get(obj_type, 0)
            diff = count - other_count
            if diff != 0:
                comparison['object_count_changes'][obj_type] = diff
                
        # Find new object types
        comparison['new_object_types'] = [
            obj_type for obj_type in self.object_counts 
            if obj_type not in other_snapshot.object_counts
        ]
        
        # Compare GC stats
        if self.gc_stats and other_snapshot.gc_stats:
            for i, (current_gen, prev_gen) in enumerate(zip(self.gc_stats, other_snapshot.gc_stats)):
                collections_diff = current_gen['collections'] - prev_gen['collections']
                if collections_diff > 0:
                    comparison['gc_collection_changes'].append({
                        'generation': i,
                        'collections': collections_diff,
                        'collected': current_gen['collected'] - prev_gen['collected']
                    })
                    
        return comparison


class MemoryLeakDetector:
    """Advanced memory leak detection system."""
    
    def __init__(self, check_interval=30, max_snapshots=100):
        self.check_interval = check_interval
        self.max_snapshots = max_snapshots
        self.monitoring = False
        
        # Memory tracking
        self.snapshots = deque(maxlen=max_snapshots)
        self.leak_patterns = []
        self.memory_growth_events = []
        
        # Object tracking
        self.tracked_objects = weakref.WeakSet()
        self.object_creation_times = {}
        self.matplotlib_figures = weakref.WeakSet()
        self.numpy_arrays = weakref.WeakSet()
        
        # Performance thresholds
        self.thresholds = {
            'memory_growth_mb': 50,      # MB per hour
            'memory_spike_mb': 100,      # Sudden spike threshold
            'object_growth_rate': 1000,  # Objects per hour
            'gc_collection_threshold': 100  # GC collections per hour
        }
        
        # Statistics
        self.stats = {
            'total_snapshots': 0,
            'memory_leaks_detected': 0,
            'largest_leak_mb': 0,
            'most_problematic_type': '',
            'monitoring_duration': 0
        }
        
        # Start tracemalloc if not already started
        if not tracemalloc.is_tracing():
            tracemalloc.start(25)  # Keep 25 frames
            
        logger.info("Memory Leak Detector initialized")
        
    def start_monitoring(self, duration=None):
        """Start memory leak monitoring."""
        if self.monitoring:
            logger.warning("Memory monitoring already in progress")
            return
            
        self.monitoring = True
        self.start_time = time.time()
        
        logger.info(f"Starting memory leak monitoring (interval: {self.check_interval}s)")
        
        # Take initial snapshot
        initial_snapshot = MemorySnapshot("initial")
        self.snapshots.append(initial_snapshot)
        logger.info(f"Initial memory usage: {initial_snapshot.get_memory_mb():.1f} MB")
        
        # Start monitoring thread
        self.monitor_thread = threading.Thread(
            target=self._monitoring_loop, 
            daemon=True, 
            name="MemoryLeakMonitor"
        )
        self.monitor_thread.start()
        
        # Set duration timer if specified
        if duration:
            threading.Timer(duration, self.stop_monitoring).start()
            
    def stop_monitoring(self):
        """Stop memory leak monitoring and generate report."""
        if not self.monitoring:
            return
            
        self.monitoring = False
        self.stats['monitoring_duration'] = time.time() - self.start_time
        
        logger.info("Stopping memory leak monitoring...")
        
        # Take final snapshot
        final_snapshot = MemorySnapshot("final")
        self.snapshots.append(final_snapshot)
        
        # Analyze results
        self._analyze_memory_patterns()
        
        # Generate report
        self.generate_report()
        
    def _monitoring_loop(self):
        """Main monitoring loop."""
        while self.monitoring:
            try:
                # Take memory snapshot
                snapshot = MemorySnapshot(f"snapshot_{len(self.snapshots)}")
                self.snapshots.append(snapshot)
                self.stats['total_snapshots'] += 1
                
                # Analyze for immediate issues
                if len(self.snapshots) >= 2:
                    self._check_for_immediate_leaks(snapshot)
                    
                # Force garbage collection periodically
                if len(self.snapshots) % 5 == 0:
                    self._perform_gc_analysis()
                    
                # Log progress
                if len(self.snapshots) % 10 == 0:
                    current_memory = snapshot.get_memory_mb()
                    initial_memory = self.snapshots[0].get_memory_mb()
                    growth = current_memory - initial_memory
                    logger.info(f"Memory check #{len(self.snapshots)}: {current_memory:.1f} MB (+{growth:.1f} MB)")
                    
                time.sleep(self.check_interval)
                
            except Exception as e:
                logger.error(f"Error in memory monitoring loop: {e}")
                time.sleep(self.check_interval)
                
    def _check_for_immediate_leaks(self, current_snapshot):
        """Check for immediate memory leaks."""
        previous_snapshot = self.snapshots[-2]
        comparison = current_snapshot.compare_to(previous_snapshot)
        
        # Check for significant memory growth
        if comparison['memory_diff_mb'] > self.thresholds['memory_spike_mb']:
            leak_event = {
                'timestamp': current_snapshot.timestamp,
                'type': 'memory_spike',
                'severity': 'critical',
                'memory_increase_mb': comparison['memory_diff_mb'],
                'time_diff': comparison['time_diff'],
                'details': f"Memory spike: +{comparison['memory_diff_mb']:.1f} MB in {comparison['time_diff']:.1f}s"
            }
            self.memory_growth_events.append(leak_event)
            self.stats['memory_leaks_detected'] += 1
            
            if comparison['memory_diff_mb'] > self.stats['largest_leak_mb']:
                self.stats['largest_leak_mb'] = comparison['memory_diff_mb']
                
            logger.warning(leak_event['details'])
            
        # Check for concerning object growth
        for obj_type, count_change in comparison['object_count_changes'].items():
            if count_change > 1000:  # More than 1000 new objects
                leak_event = {
                    'timestamp': current_snapshot.timestamp,
                    'type': 'object_growth',
                    'severity': 'warning',
                    'object_type': obj_type,
                    'object_increase': count_change,
                    'details': f"High object creation: +{count_change} {obj_type} objects"
                }
                self.memory_growth_events.append(leak_event)
                logger.warning(leak_event['details'])
                
    def _perform_gc_analysis(self):
        """Perform garbage collection analysis."""
        # Force garbage collection
        before_gc = len(gc.get_objects())
        collected = gc.collect()
        after_gc = len(gc.get_objects())
        
        if collected > 0:
            logger.info(f"Garbage collection: {collected} objects collected, {before_gc - after_gc} objects freed")
            
        # Check for uncollectable objects
        uncollectable = len(gc.garbage)
        if uncollectable > 0:
            logger.warning(f"Found {uncollectable} uncollectable objects (potential circular references)")
            
            # Log types of uncollectable objects
            uncollectable_types = defaultdict(int)
            for obj in gc.garbage[:50]:  # Check first 50
                uncollectable_types[type(obj).__name__] += 1
                
            for obj_type, count in uncollectable_types.items():
                logger.warning(f"  {count} uncollectable {obj_type} objects")
                
    def _analyze_memory_patterns(self):
        """Analyze memory usage patterns for leaks."""
        if len(self.snapshots) < 3:
            logger.warning("Not enough snapshots for pattern analysis")
            return
            
        logger.info("Analyzing memory patterns...")
        
        # Calculate memory growth rate
        initial_memory = self.snapshots[0].get_memory_mb()
        final_memory = self.snapshots[-1].get_memory_mb()
        total_time_hours = (self.snapshots[-1].timestamp - self.snapshots[0].timestamp) / 3600
        
        if total_time_hours > 0:
            growth_rate_mb_per_hour = (final_memory - initial_memory) / total_time_hours
            
            if growth_rate_mb_per_hour > self.thresholds['memory_growth_mb']:
                pattern = {
                    'type': 'sustained_growth',
                    'severity': 'critical',
                    'growth_rate_mb_per_hour': growth_rate_mb_per_hour,
                    'total_growth_mb': final_memory - initial_memory,
                    'duration_hours': total_time_hours,
                    'description': f"Sustained memory growth: {growth_rate_mb_per_hour:.1f} MB/hour"
                }
                self.leak_patterns.append(pattern)
                logger.critical(pattern['description'])
                
        # Analyze object growth patterns
        self._analyze_object_patterns()
        
        # Analyze matplotlib figure patterns
        self._analyze_matplotlib_patterns()
        
        # Analyze numpy array patterns
        self._analyze_numpy_patterns()
        
    def _analyze_object_patterns(self):
        """Analyze object creation/destruction patterns."""
        if len(self.snapshots) < 3:
            return
            
        # Compare first, middle, and last snapshots
        first = self.snapshots[0]
        middle = self.snapshots[len(self.snapshots) // 2]
        last = self.snapshots[-1]
        
        # Analyze object type growth
        object_growth = {}
        for obj_type in set(list(first.object_counts.keys()) + list(last.object_counts.keys())):
            first_count = first.object_counts.get(obj_type, 0)
            last_count = last.object_counts.get(obj_type, 0)
            growth = last_count - first_count
            
            if growth > 100:  # Significant growth
                object_growth[obj_type] = growth
                
        # Find most problematic object type
        if object_growth:
            most_problematic = max(object_growth.items(), key=lambda x: x[1])
            self.stats['most_problematic_type'] = f"{most_problematic[0]} (+{most_problematic[1]})"
            
            pattern = {
                'type': 'object_accumulation',
                'severity': 'warning',
                'object_type': most_problematic[0],
                'object_growth': most_problematic[1],
                'description': f"High object accumulation: +{most_problematic[1]} {most_problematic[0]} objects"
            }
            self.leak_patterns.append(pattern)
            logger.warning(pattern['description'])
            
    def _analyze_matplotlib_patterns(self):
        """Analyze matplotlib figure memory patterns."""
        # Check for unclosed figures
        open_figures = len(plt.get_fignums())
        if open_figures > 10:
            pattern = {
                'type': 'matplotlib_figures',
                'severity': 'warning',
                'open_figures': open_figures,
                'description': f"Many open matplotlib figures detected: {open_figures}"
            }
            self.leak_patterns.append(pattern)
            logger.warning(pattern['description'])
            
    def _analyze_numpy_patterns(self):
        """Analyze NumPy array memory patterns."""
        # This would require more sophisticated tracking in a real implementation
        # For now, we can check general array object counts
        if len(self.snapshots) >= 2:
            last_snapshot = self.snapshots[-1]
            first_snapshot = self.snapshots[0]
            
            numpy_arrays_last = last_snapshot.object_counts.get('ndarray', 0)
            numpy_arrays_first = first_snapshot.object_counts.get('ndarray', 0)
            array_growth = numpy_arrays_last - numpy_arrays_first
            
            if array_growth > 100:
                pattern = {
                    'type': 'numpy_arrays',
                    'severity': 'warning',
                    'array_growth': array_growth,
                    'description': f"High NumPy array growth: +{array_growth} arrays"
                }
                self.leak_patterns.append(pattern)
                logger.warning(pattern['description'])
                
    def track_object(self, obj, name=""):
        """Track a specific object for memory analysis."""
        self.tracked_objects.add(obj)
        self.object_creation_times[id(obj)] = {
            'time': time.time(),
            'name': name,
            'type': type(obj).__name__
        }
        
    def track_matplotlib_figure(self, fig):
        """Track a matplotlib figure."""
        self.matplotlib_figures.add(fig)
        
    def track_numpy_array(self, array):
        """Track a NumPy array."""
        self.numpy_arrays.add(array)
        
    def get_memory_statistics(self):
        """Get current memory statistics."""
        if not self.snapshots:
            return {}
            
        current_snapshot = self.snapshots[-1]
        initial_snapshot = self.snapshots[0]
        
        stats = {
            'current_memory_mb': current_snapshot.get_memory_mb(),
            'initial_memory_mb': initial_snapshot.get_memory_mb(),
            'memory_growth_mb': current_snapshot.get_memory_mb() - initial_snapshot.get_memory_mb(),
            'monitoring_duration_minutes': (time.time() - self.start_time) / 60 if hasattr(self, 'start_time') else 0,
            'snapshots_taken': len(self.snapshots),
            'leak_events': len(self.memory_growth_events),
            'leak_patterns': len(self.leak_patterns),
            'open_figures': len(plt.get_fignums()),
            'tracked_objects': len(self.tracked_objects),
            'gc_objects': len(gc.get_objects())
        }
        
        # Calculate growth rate
        if stats['monitoring_duration_minutes'] > 0:
            stats['growth_rate_mb_per_hour'] = (stats['memory_growth_mb'] / stats['monitoring_duration_minutes']) * 60
        else:
            stats['growth_rate_mb_per_hour'] = 0
            
        return stats
        
    def generate_report(self):
        """Generate comprehensive memory leak report."""
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        
        # Create reports directory
        reports_dir = Path('performance_reports')
        reports_dir.mkdir(exist_ok=True)
        
        # Generate JSON report
        json_report = self._generate_json_report()
        json_file = reports_dir / f'memory_leak_report_{timestamp}.json'
        with open(json_file, 'w') as f:
            json.dump(json_report, f, indent=2, default=str)
            
        # Generate text report
        text_report = self._generate_text_report()
        text_file = reports_dir / f'memory_leak_report_{timestamp}.txt'
        with open(text_file, 'w') as f:
            f.write(text_report)
            
        # Generate HTML report
        html_report = self._generate_html_report()
        html_file = reports_dir / f'memory_leak_report_{timestamp}.html'
        with open(html_file, 'w') as f:
            f.write(html_report)
            
        # Generate memory plots
        self._generate_memory_plots(timestamp)
        
        logger.info(f"Memory leak reports generated:")
        logger.info(f"  JSON: {json_file}")
        logger.info(f"  Text: {text_file}")
        logger.info(f"  HTML: {html_file}")
        
    def _generate_json_report(self):
        """Generate JSON memory leak report."""
        stats = self.get_memory_statistics()
        
        report = {
            'metadata': {
                'timestamp': datetime.now().isoformat(),
                'monitoring_duration': self.stats['monitoring_duration'],
                'check_interval': self.check_interval,
                'total_snapshots': len(self.snapshots)
            },
            'statistics': stats,
            'leak_patterns': self.leak_patterns,
            'memory_growth_events': self.memory_growth_events,
            'snapshots': [
                {
                    'name': snapshot.name,
                    'timestamp': snapshot.timestamp,
                    'memory_mb': snapshot.get_memory_mb(),
                    'object_counts': dict(list(snapshot.object_counts.items())[:20])  # Top 20 object types
                }
                for snapshot in self.snapshots
            ],
            'recommendations': self._generate_recommendations()
        }
        
        return report
        
    def _generate_text_report(self):
        """Generate text memory leak report."""
        stats = self.get_memory_statistics()
        
        report = f"""MEMORY LEAK DETECTION REPORT
{'=' * 50}

Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
Monitoring Duration: {stats.get('monitoring_duration_minutes', 0):.1f} minutes
Check Interval: {self.check_interval} seconds

MEMORY STATISTICS:
{'=' * 20}
Initial Memory: {stats.get('initial_memory_mb', 0):.1f} MB
Current Memory: {stats.get('current_memory_mb', 0):.1f} MB
Memory Growth: {stats.get('memory_growth_mb', 0):.1f} MB
Growth Rate: {stats.get('growth_rate_mb_per_hour', 0):.1f} MB/hour

OBJECT STATISTICS:
{'=' * 20}
Total Objects: {stats.get('gc_objects', 0):,}
Tracked Objects: {stats.get('tracked_objects', 0)}
Open Matplotlib Figures: {stats.get('open_figures', 0)}
Snapshots Taken: {stats.get('snapshots_taken', 0)}

LEAK DETECTION RESULTS:
{'=' * 25}
Leak Events Detected: {len(self.memory_growth_events)}
Leak Patterns Found: {len(self.leak_patterns)}
Largest Single Leak: {self.stats.get('largest_leak_mb', 0):.1f} MB
Most Problematic Type: {self.stats.get('most_problematic_type', 'None')}

"""

        # Add leak patterns
        if self.leak_patterns:
            report += "DETECTED LEAK PATTERNS:\n"
            report += "-" * 25 + "\n"
            for i, pattern in enumerate(self.leak_patterns, 1):
                report += f"{i}. {pattern['description']}\n"
                report += f"   Type: {pattern['type']}\n"
                report += f"   Severity: {pattern['severity']}\n\n"
        else:
            report += "✅ No significant leak patterns detected.\n\n"
            
        # Add memory growth events
        if self.memory_growth_events:
            report += "MEMORY GROWTH EVENTS:\n"
            report += "-" * 22 + "\n"
            for i, event in enumerate(self.memory_growth_events[-10:], 1):  # Last 10 events
                timestamp_str = datetime.fromtimestamp(event['timestamp']).strftime('%H:%M:%S')
                report += f"{i}. [{timestamp_str}] {event['details']}\n"
            if len(self.memory_growth_events) > 10:
                report += f"... and {len(self.memory_growth_events) - 10} more events\n"
        else:
            report += "✅ No significant memory growth events detected.\n"
            
        # Add recommendations
        report += "\nRECOMMENDATIONS:\n"
        report += "-" * 15 + "\n"
        recommendations = self._generate_recommendations()
        for i, rec in enumerate(recommendations, 1):
            report += f"{i}. {rec}\n"
            
        return report
        
    def _generate_html_report(self):
        """Generate HTML memory leak report."""
        stats = self.get_memory_statistics()
        timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        
        # Determine status based on statistics
        if stats.get('memory_growth_mb', 0) > 200:
            status = "CRITICAL"
            status_class = "critical"
        elif stats.get('memory_growth_mb', 0) > 100:
            status = "WARNING"
            status_class = "warning"
        else:
            status = "GOOD"
            status_class = "good"
            
        html = f"""
<!DOCTYPE html>
<html>
<head>
    <title>Memory Leak Detection Report - Signal Analyzer</title>
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
            background: linear-gradient(135deg, #e74c3c 0%, #c0392b 100%);
            color: white; 
            padding: 30px; 
            border-radius: 10px; 
            margin-bottom: 30px;
        }}
        .header h1 {{ margin: 0; font-size: 2.5em; }}
        .header p {{ margin: 10px 0 0 0; opacity: 0.9; }}
        .status-section {{ 
            text-align: center; 
            margin: 30px 0; 
            padding: 25px; 
            border-radius: 10px;
        }}
        .status.good {{ background: #d4edda; color: #155724; border: 1px solid #c3e6cb; }}
        .status.warning {{ background: #fff3cd; color: #856404; border: 1px solid #ffeaa7; }}
        .status.critical {{ background: #f8d7da; color: #721c24; border: 1px solid #f5c6cb; }}
        .status h2 {{ margin: 0; font-size: 2em; }}
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
            border-left: 4px solid #e74c3c;
        }}
        .chart-container {{ 
            text-align: center; 
            margin: 30px 0; 
            padding: 20px; 
            background: #f8f9fa; 
            border-radius: 10px;
        }}
        .leak-pattern {{ 
            background: #fff3cd; 
            padding: 15px; 
            border-radius: 5px; 
            margin: 10px 0;
            border-left: 4px solid #ffc107;
        }}
        .leak-pattern.critical {{ 
            background: #f8d7da; 
            border-left-color: #dc3545;
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
            <h1>🔍 Memory Leak Detection Report</h1>
            <p>Signal Analyzer Memory Analysis</p>
            <p>Generated: {timestamp}</p>
            <p>Monitoring Duration: {stats.get('monitoring_duration_minutes', 0):.1f} minutes</p>
        </div>
        
        <div class="status-section status {status_class}">
            <h2>Memory Status: {status}</h2>
            <p>Total Memory Growth: {stats.get('memory_growth_mb', 0):.1f} MB</p>
            <p>Growth Rate: {stats.get('growth_rate_mb_per_hour', 0):.1f} MB/hour</p>
        </div>
        
        <div class="section">
            <h2>Memory Statistics</h2>
            <div class="metrics">
                <div class="metric">
                    <h3>Current Memory</h3>
                    <div class="value">{stats.get('current_memory_mb', 0):.1f}</div>
                    <div class="unit">MB</div>
                </div>
                <div class="metric">
                    <h3>Memory Growth</h3>
                    <div class="value">{stats.get('memory_growth_mb', 0):.1f}</div>
                    <div class="unit">MB</div>
                </div>
                <div class="metric">
                    <h3>Growth Rate</h3>
                    <div class="value">{stats.get('growth_rate_mb_per_hour', 0):.1f}</div>
                    <div class="unit">MB/hour</div>
                </div>
                <div class="metric">
                    <h3>Objects Tracked</h3>
                    <div class="value">{stats.get('gc_objects', 0):,}</div>
                    <div class="unit">total</div>
                </div>
                <div class="metric">
                    <h3>Open Figures</h3>
                    <div class="value">{stats.get('open_figures', 0)}</div>
                    <div class="unit">matplotlib</div>
                </div>
                <div class="metric">
                    <h3>Leak Events</h3>
                    <div class="value">{len(self.memory_growth_events)}</div>
                    <div class="unit">detected</div>
                </div>
            </div>
        </div>
        
        <div class="chart-container">
            <h2>Memory Usage Over Time</h2>
            <img src="memory_usage_plots_{datetime.now().strftime('%Y%m%d_%H%M%S')}.png" 
                 alt="Memory Usage Charts" style="max-width: 100%; height: auto;">
        </div>
        
        <div class="section">
            <h2>Detected Leak Patterns</h2>
            {self._generate_leak_patterns_html()}
        </div>
        
        <div class="section">
            <h2>Memory Growth Events</h2>
            {self._generate_memory_events_html()}
        </div>
        
        <div class="recommendations">
            <h2>🎯 Memory Optimization Recommendations</h2>
            <ul>
                {''.join(f'<li>{rec}</li>' for rec in self._generate_recommendations())}
            </ul>
        </div>
    </div>
</body>
</html>
        """
        
        return html
        
    def _generate_leak_patterns_html(self):
        """Generate HTML for leak patterns."""
        if not self.leak_patterns:
            return '<p>✅ No significant leak patterns detected.</p>'
            
        html = ""
        for pattern in self.leak_patterns:
            severity_class = "critical" if pattern['severity'] == 'critical' else ""
            html += f"""
            <div class="leak-pattern {severity_class}">
                <h4>{pattern['type'].replace('_', ' ').title()}</h4>
                <p><strong>Severity:</strong> {pattern['severity'].title()}</p>
                <p><strong>Description:</strong> {pattern['description']}</p>
            </div>
            """
            
        return html
        
    def _generate_memory_events_html(self):
        """Generate HTML for memory growth events."""
        if not self.memory_growth_events:
            return '<p>✅ No significant memory growth events detected.</p>'
            
        html = """
        <table>
            <tr>
                <th>Time</th>
                <th>Type</th>
                <th>Severity</th>
                <th>Details</th>
            </tr>
        """
        
        for event in self.memory_growth_events[-20:]:  # Last 20 events
            timestamp_str = datetime.fromtimestamp(event['timestamp']).strftime('%H:%M:%S')
            severity_color = {'critical': '#dc3545', 'warning': '#ffc107', 'info': '#17a2b8'}.get(event['severity'], '#6c757d')
            
            html += f"""
            <tr>
                <td>{timestamp_str}</td>
                <td>{event['type'].replace('_', ' ').title()}</td>
                <td style="color: {severity_color}; font-weight: bold;">{event['severity'].title()}</td>
                <td>{event['details']}</td>
            </tr>
            """
            
        html += "</table>"
        
        if len(self.memory_growth_events) > 20:
            html += f"<p><em>Showing last 20 of {len(self.memory_growth_events)} total events.</em></p>"
            
        return html
        
    def _generate_recommendations(self):
        """Generate memory optimization recommendations."""
        recommendations = []
        stats = self.get_memory_statistics()
        
        # Memory growth recommendations
        if stats.get('memory_growth_mb', 0) > 100:
            recommendations.append("🔥 Significant memory growth detected - investigate potential memory leaks")
            
        if stats.get('growth_rate_mb_per_hour', 0) > 50:
            recommendations.append("📈 High memory growth rate - implement more aggressive cleanup")
            
        # Matplotlib recommendations
        if stats.get('open_figures', 0) > 5:
            recommendations.append("📊 Close matplotlib figures with plt.close() after use")
            recommendations.append("🧹 Use plt.close('all') periodically to clear all figures")
            
        # Object tracking recommendations
        if stats.get('gc_objects', 0) > 100000:
            recommendations.append("🗂️ High object count - consider object pooling or data structure optimization")
            
        # Pattern-specific recommendations
        for pattern in self.leak_patterns:
            if pattern['type'] == 'matplotlib_figures':
                recommendations.append("🎨 Implement automatic figure cleanup in plotting functions")
            elif pattern['type'] == 'numpy_arrays':
                recommendations.append("🔢 Review NumPy array lifecycle - use del or gc.collect() for large arrays")
            elif pattern['type'] == 'object_accumulation':
                recommendations.append(f"📦 Investigate {pattern.get('object_type', 'object')} accumulation")
                
        # General recommendations
        if not recommendations or len(recommendations) < 3:
            recommendations.extend([
                "✨ Use context managers for resource management",
                "🔄 Implement proper cleanup in __del__ methods",
                "🧼 Call gc.collect() after processing large datasets",
                "📏 Monitor memory usage during development",
                "🎯 Use memory profilers to identify specific leak sources"
            ])
            
        return recommendations
        
    def _generate_memory_plots(self, timestamp):
        """Generate memory usage visualization plots."""
        if len(self.snapshots) < 2:
            logger.warning("Not enough snapshots for plotting")
            return
            
        try:
            fig, axes = plt.subplots(2, 2, figsize=(15, 10))
            fig.suptitle('Memory Usage Analysis', fontsize=16, fontweight='bold')
            
            # Extract data from snapshots
            times = [(s.timestamp - self.snapshots[0].timestamp) / 60 for s in self.snapshots]  # Minutes
            memory_usage = [s.get_memory_mb() for s in self.snapshots]
            
            # Memory usage over time
            ax = axes[0, 0]
            ax.plot(times, memory_usage, 'r-', linewidth=2, label='Memory Usage')
            if len(self.memory_growth_events) > 0:
                # Mark memory spike events
                spike_times = [(e['timestamp'] - self.snapshots[0].timestamp) / 60 
                              for e in self.memory_growth_events if e['type'] == 'memory_spike']
                spike_memory = []
                for spike_time in spike_times:
                    # Find closest snapshot
                    closest_idx = min(range(len(times)), key=lambda i: abs(times[i] - spike_time))
                    spike_memory.append(memory_usage[closest_idx])
                ax.scatter(spike_times, spike_memory, color='red', s=100, marker='x', label='Memory Spikes')
                
            ax.set_title('Memory Usage Over Time')
            ax.set_xlabel('Time (minutes)')
            ax.set_ylabel('Memory (MB)')
            ax.grid(True, alpha=0.3)
            ax.legend()
            
            # Memory growth rate
            if len(memory_usage) > 1:
                ax = axes[0, 1]
                growth_rates = []
                for i in range(1, len(memory_usage)):
                    time_diff = times[i] - times[i-1]
                    if time_diff > 0:
                        growth_rate = (memory_usage[i] - memory_usage[i-1]) / time_diff  # MB per minute
                        growth_rates.append(growth_rate * 60)  # Convert to MB per hour
                    else:
                        growth_rates.append(0)
                        
                ax.plot(times[1:], growth_rates, 'b-', linewidth=2)
                ax.axhline(y=50, color='orange', linestyle='--', alpha=0.7, label='Warning (50 MB/h)')
                ax.axhline(y=100, color='red', linestyle='--', alpha=0.7, label='Critical (100 MB/h)')
                ax.set_title('Memory Growth Rate')
                ax.set_xlabel('Time (minutes)')
                ax.set_ylabel('Growth Rate (MB/hour)')
                ax.grid(True, alpha=0.3)
                ax.legend()
                
            # Object count analysis
            if len(self.snapshots) > 1:
                ax = axes[1, 0]
                object_counts = [len(s.object_counts) for s in self.snapshots]
                ax.plot(times, object_counts, 'g-', linewidth=2)
                ax.set_title('Object Type Diversity')
                ax.set_xlabel('Time (minutes)')
                ax.set_ylabel('Number of Object Types')
                ax.grid(True, alpha=0.3)
                
            # Memory distribution (if we have object counts)
            ax = axes[1, 1]
            if self.snapshots:
                last_snapshot = self.snapshots[-1]
                # Get top 10 object types by count
                top_objects = sorted(last_snapshot.object_counts.items(), 
                                   key=lambda x: x[1], reverse=True)[:10]
                if top_objects:
                    object_types, counts = zip(*top_objects)
                    ax.pie(counts, labels=object_types, autopct='%1.1f%%', startangle=90)
                    ax.set_title('Object Distribution (Top 10 Types)')
                else:
                    ax.text(0.5, 0.5, 'No object data available', 
                           ha='center', va='center', transform=ax.transAxes)
                    ax.set_title('Object Distribution')
                    
            plt.tight_layout()
            
            # Save plots
            plot_file = f'performance_reports/memory_usage_plots_{timestamp}.png'
            plt.savefig(plot_file, dpi=300, bbox_inches='tight')
            plt.close()
            
            logger.info(f"Memory usage plots saved to: {plot_file}")
            
        except Exception as e:
            logger.error(f"Error generating memory plots: {e}")


def detect_memory_leaks(duration=300, check_interval=30):
    """Quick function to detect memory leaks."""
    detector = MemoryLeakDetector(check_interval)
    detector.start_monitoring(duration)
    
    # Keep the main thread alive
    try:
        time.sleep(duration)
    except KeyboardInterrupt:
        logger.info("Memory leak detection interrupted by user")
    finally:
        detector.stop_monitoring()
        
    return detector.get_memory_statistics()


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='Memory Leak Detector for Signal Analyzer')
    parser.add_argument('--duration', type=int, default=300, help='Monitoring duration in seconds')
    parser.add_argument('--interval', type=int, default=30, help='Check interval in seconds')
    parser.add_argument('--output-dir', default='performance_reports', help='Output directory for reports')
    
    args = parser.parse_args()
    
    print("🔍 MEMORY LEAK DETECTOR")
    print("=" * 30)
    print(f"⏱️ Duration: {args.duration} seconds")
    print(f"🔄 Check Interval: {args.interval} seconds")
    print(f"📁 Output: {args.output_dir}")
    print("-" * 30)
    
    # Create output directory
    Path(args.output_dir).mkdir(exist_ok=True)
    
    # Start detection
    stats = detect_memory_leaks(args.duration, args.interval)
    
    print("\n✅ MEMORY LEAK DETECTION COMPLETE!")
    print("=" * 40)
    print("📊 Final Statistics:")
    for key, value in stats.items():
        if isinstance(value, float):
            print(f"  {key}: {value:.2f}")
        else:
            print(f"  {key}: {value}")
            
    print(f"\n📁 Reports saved to: {args.output_dir}/")
    print("🎯 Check the HTML report for detailed analysis and recommendations")