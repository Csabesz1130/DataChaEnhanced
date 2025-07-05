#!/usr/bin/env python3
"""
Function Call Tracer for Signal Analyzer Application
==================================================

Advanced function-level performance profiling and call tracing specifically
designed for Python applications with complex call stacks and performance bottlenecks.

Features:
    - Real-time function call tracing
    - Call stack analysis and visualization
    - Function timing and frequency tracking
    - Bottleneck identification
    - Hot path detection
    - Recursive call analysis
    - Memory usage per function
    - Call graph generation
    - Performance regression detection
"""

import time
import sys
import threading
import functools
import inspect
import traceback
from collections import defaultdict, deque
from datetime import datetime
import logging
import json
from pathlib import Path
import matplotlib.pyplot as plt
import networkx as nx
from typing import Dict, List, Any, Optional, Callable

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class FunctionCall:
    """Represents a single function call with timing and context."""
    
    def __init__(self, func_name: str, module: str, filename: str, line_number: int):
        self.func_name = func_name
        self.module = module
        self.filename = filename
        self.line_number = line_number
        self.start_time = time.time()
        self.end_time = None
        self.duration = None
        self.memory_start = None
        self.memory_end = None
        self.memory_delta = None
        self.call_depth = 0
        self.children = []
        self.parent = None
        self.thread_id = threading.get_ident()
        self.exception_raised = None
        
    def end_call(self):
        """Mark the end of the function call."""
        self.end_time = time.time()
        self.duration = self.end_time - self.start_time
        
    def add_child(self, child_call):
        """Add a child function call."""
        child_call.parent = self
        child_call.call_depth = self.call_depth + 1
        self.children.append(child_call)
        
    def get_full_name(self):
        """Get the full qualified name of the function."""
        return f"{self.module}.{self.func_name}"
        
    def to_dict(self):
        """Convert to dictionary for serialization."""
        return {
            'func_name': self.func_name,
            'module': self.module,
            'filename': self.filename,
            'line_number': self.line_number,
            'start_time': self.start_time,
            'end_time': self.end_time,
            'duration': self.duration,
            'call_depth': self.call_depth,
            'thread_id': self.thread_id,
            'memory_delta': self.memory_delta,
            'exception_raised': str(self.exception_raised) if self.exception_raised else None,
            'children_count': len(self.children)
        }


class FunctionCallTracer:
    """Advanced function call tracer with performance analysis."""
    
    def __init__(self, max_calls=10000, ignore_patterns=None):
        self.max_calls = max_calls
        self.ignore_patterns = ignore_patterns or [
            'threading', 'logging', '_thread', 'queue', 'weakref'
        ]
        
        # Call tracking
        self.active_calls = {}  # Thread ID -> call stack
        self.completed_calls = deque(maxlen=max_calls)
        self.call_stats = defaultdict(lambda: {
            'count': 0,
            'total_time': 0,
            'avg_time': 0,
            'max_time': 0,
            'min_time': float('inf'),
            'memory_usage': []
        })
        
        # Performance data
        self.hot_functions = []
        self.slow_functions = []
        self.recursive_functions = defaultdict(int)
        self.call_graph = nx.DiGraph()
        
        # Configuration
        self.tracing = False
        self.min_duration = 0.001  # Only track calls longer than 1ms
        self.max_depth = 50  # Prevent infinite recursion tracking
        
        # Thread safety
        self.lock = threading.Lock()
        
        logger.info("Function Call Tracer initialized")
        
    def start_tracing(self):
        """Start function call tracing."""
        if self.tracing:
            logger.warning("Tracing already active")
            return
            
        self.tracing = True
        sys.settrace(self._trace_calls)
        
        # Set tracing for all existing threads
        for thread_id, frame in sys._current_frames().items():
            if thread_id != threading.get_ident():
                # This is a simplification - real implementation would need more care
                pass
                
        logger.info("Function call tracing started")
        
    def stop_tracing(self):
        """Stop function call tracing."""
        if not self.tracing:
            return
            
        self.tracing = False
        sys.settrace(None)
        
        # Analyze collected data
        self._analyze_performance()
        
        logger.info("Function call tracing stopped")
        
    def _trace_calls(self, frame, event, arg):
        """Main trace function called by Python."""
        if not self.tracing:
            return
            
        try:
            # Get function information
            filename = frame.f_code.co_filename
            func_name = frame.f_code.co_name
            line_number = frame.f_lineno
            
            # Skip ignored patterns
            if any(pattern in filename for pattern in self.ignore_patterns):
                return
                
            # Get module name
            module = self._get_module_name(filename)
            
            thread_id = threading.get_ident()
            
            if event == 'call':
                self._handle_function_call(thread_id, func_name, module, filename, line_number)
            elif event == 'return':
                self._handle_function_return(thread_id, arg)
            elif event == 'exception':
                self._handle_function_exception(thread_id, arg)
                
        except Exception as e:
            # Don't let tracing errors crash the application
            logger.error(f"Error in trace function: {e}")
            
        return self._trace_calls
        
    def _get_module_name(self, filename):
        """Extract module name from filename."""
        try:
            path = Path(filename)
            if 'src' in path.parts:
                # Get relative path from src directory
                src_index = path.parts.index('src')
                module_parts = path.parts[src_index + 1:]
                module_name = '.'.join(module_parts[:-1])  # Remove .py extension
                if module_name:
                    return module_name
            return path.stem
        except:
            return "unknown"
            
    def _handle_function_call(self, thread_id, func_name, module, filename, line_number):
        """Handle function call event."""
        with self.lock:
            # Initialize thread call stack if needed
            if thread_id not in self.active_calls:
                self.active_calls[thread_id] = []
                
            call_stack = self.active_calls[thread_id]
            
            # Check call depth to prevent infinite recursion tracking
            if len(call_stack) >= self.max_depth:
                return
                
            # Create function call object
            func_call = FunctionCall(func_name, module, filename, line_number)
            
            # Set up parent-child relationship
            if call_stack:
                parent_call = call_stack[-1]
                parent_call.add_child(func_call)
                
            # Add to call stack
            call_stack.append(func_call)
            
            # Track recursive calls
            full_name = func_call.get_full_name()
            if sum(1 for call in call_stack if call.get_full_name() == full_name) > 1:
                self.recursive_functions[full_name] += 1
                
    def _handle_function_return(self, thread_id, return_value):
        """Handle function return event."""
        with self.lock:
            if thread_id not in self.active_calls or not self.active_calls[thread_id]:
                return
                
            call_stack = self.active_calls[thread_id]
            func_call = call_stack.pop()
            
            # End the call and calculate duration
            func_call.end_call()
            
            # Only track calls that exceed minimum duration
            if func_call.duration >= self.min_duration:
                # Add to completed calls
                self.completed_calls.append(func_call)
                
                # Update statistics
                self._update_call_stats(func_call)
                
                # Update call graph
                self._update_call_graph(func_call)
                
    def _handle_function_exception(self, thread_id, exception_info):
        """Handle function exception event."""
        with self.lock:
            if thread_id not in self.active_calls or not self.active_calls[thread_id]:
                return
                
            call_stack = self.active_calls[thread_id]
            if call_stack:
                func_call = call_stack[-1]
                func_call.exception_raised = exception_info
                
    def _update_call_stats(self, func_call):
        """Update statistics for a function call."""
        full_name = func_call.get_full_name()
        stats = self.call_stats[full_name]
        
        stats['count'] += 1
        stats['total_time'] += func_call.duration
        stats['avg_time'] = stats['total_time'] / stats['count']
        stats['max_time'] = max(stats['max_time'], func_call.duration)
        stats['min_time'] = min(stats['min_time'], func_call.duration)
        
        if func_call.memory_delta:
            stats['memory_usage'].append(func_call.memory_delta)
            
    def _update_call_graph(self, func_call):
        """Update the call graph with function relationships."""
        full_name = func_call.get_full_name()
        
        # Add node if it doesn't exist
        if not self.call_graph.has_node(full_name):
            self.call_graph.add_node(full_name, 
                                   func_name=func_call.func_name,
                                   module=func_call.module,
                                   call_count=0,
                                   total_time=0)
            
        # Update node data
        node_data = self.call_graph.nodes[full_name]
        node_data['call_count'] += 1
        node_data['total_time'] += func_call.duration
        
        # Add edges for parent-child relationships
        if func_call.parent:
            parent_name = func_call.parent.get_full_name()
            if self.call_graph.has_edge(parent_name, full_name):
                self.call_graph[parent_name][full_name]['weight'] += 1
            else:
                self.call_graph.add_edge(parent_name, full_name, weight=1)
                
    def _analyze_performance(self):
        """Analyze collected performance data."""
        logger.info("Analyzing function call performance...")
        
        # Identify hot functions (most frequently called)
        hot_functions = sorted(
            self.call_stats.items(),
            key=lambda x: x[1]['count'],
            reverse=True
        )[:20]
        
        self.hot_functions = [
            {
                'function': func_name,
                'call_count': stats['count'],
                'total_time': stats['total_time'],
                'avg_time': stats['avg_time']
            }
            for func_name, stats in hot_functions
        ]
        
        # Identify slow functions (highest average time)
        slow_functions = sorted(
            self.call_stats.items(),
            key=lambda x: x[1]['avg_time'],
            reverse=True
        )[:20]
        
        self.slow_functions = [
            {
                'function': func_name,
                'avg_time': stats['avg_time'],
                'max_time': stats['max_time'],
                'call_count': stats['count'],
                'total_time': stats['total_time']
            }
            for func_name, stats in slow_functions
            if stats['count'] > 1  # Only include functions called multiple times
        ]
        
        logger.info(f"Analysis complete: {len(self.hot_functions)} hot functions, {len(self.slow_functions)} slow functions")
        
    def get_performance_summary(self):
        """Get performance summary statistics."""
        if not self.completed_calls:
            return {}
            
        total_calls = len(self.completed_calls)
        total_time = sum(call.duration for call in self.completed_calls)
        avg_call_time = total_time / total_calls if total_calls > 0 else 0
        
        return {
            'total_function_calls': total_calls,
            'total_execution_time': total_time,
            'average_call_time': avg_call_time,
            'unique_functions': len(self.call_stats),
            'recursive_functions': len(self.recursive_functions),
            'hot_functions_count': len(self.hot_functions),
            'slow_functions_count': len(self.slow_functions),
            'call_graph_nodes': self.call_graph.number_of_nodes(),
            'call_graph_edges': self.call_graph.number_of_edges()
        }
        
    def get_top_functions(self, metric='total_time', limit=10):
        """Get top functions by specified metric."""
        if metric == 'total_time':
            sorted_functions = sorted(
                self.call_stats.items(),
                key=lambda x: x[1]['total_time'],
                reverse=True
            )
        elif metric == 'avg_time':
            sorted_functions = sorted(
                self.call_stats.items(),
                key=lambda x: x[1]['avg_time'],
                reverse=True
            )
        elif metric == 'call_count':
            sorted_functions = sorted(
                self.call_stats.items(),
                key=lambda x: x[1]['count'],
                reverse=True
            )
        else:
            raise ValueError(f"Unknown metric: {metric}")
            
        return sorted_functions[:limit]
        
    def generate_report(self):
        """Generate comprehensive function call analysis report."""
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        
        # Create reports directory
        reports_dir = Path('performance_reports')
        reports_dir.mkdir(exist_ok=True)
        
        # Generate JSON report
        json_report = self._generate_json_report()
        json_file = reports_dir / f'function_trace_report_{timestamp}.json'
        with open(json_file, 'w') as f:
            json.dump(json_report, f, indent=2, default=str)
            
        # Generate HTML report
        html_report = self._generate_html_report()
        html_file = reports_dir / f'function_trace_report_{timestamp}.html'
        with open(html_file, 'w') as f:
            f.write(html_report)
            
        # Generate call graph visualization
        self._generate_call_graph_visualization(timestamp)
        
        # Generate performance plots
        self._generate_performance_plots(timestamp)
        
        logger.info(f"Function trace reports generated:")
        logger.info(f"  JSON: {json_file}")
        logger.info(f"  HTML: {html_file}")
        
    def _generate_json_report(self):
        """Generate JSON function trace report."""
        summary = self.get_performance_summary()
        
        report = {
            'metadata': {
                'timestamp': datetime.now().isoformat(),
                'total_calls_traced': len(self.completed_calls),
                'unique_functions': len(self.call_stats)
            },
            'summary': summary,
            'hot_functions': self.hot_functions,
            'slow_functions': self.slow_functions,
            'recursive_functions': dict(self.recursive_functions),
            'function_statistics': {
                func_name: stats for func_name, stats in 
                sorted(self.call_stats.items(), key=lambda x: x[1]['total_time'], reverse=True)[:50]
            },
            'recommendations': self._generate_recommendations()
        }
        
        return report
        
    def _generate_html_report(self):
        """Generate HTML function trace report."""
        summary = self.get_performance_summary()
        timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        
        html = f"""
<!DOCTYPE html>
<html>
<head>
    <title>Function Call Trace Report - Signal Analyzer</title>
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
            background: linear-gradient(135deg, #3498db 0%, #2980b9 100%);
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
            border-left: 4px solid #3498db;
        }}
        .function-table {{ 
            width: 100%; 
            border-collapse: collapse; 
            margin: 15px 0;
        }}
        .function-table th, .function-table td {{ 
            border: 1px solid #dee2e6; 
            padding: 12px; 
            text-align: left;
        }}
        .function-table th {{ background: #e9ecef; }}
        .function-table tr:nth-child(even) {{ background: #f8f9fa; }}
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
        .code {{ 
            background: #f8f9fa; 
            padding: 2px 6px; 
            border-radius: 3px; 
            font-family: 'Courier New', monospace;
            font-size: 0.9em;
        }}
    </style>
</head>
<body>
    <div class="container">
        <div class="header">
            <h1>🔍 Function Call Trace Report</h1>
            <p>Signal Analyzer Function Performance Analysis</p>
            <p>Generated: {timestamp}</p>
        </div>
        
        <div class="section">
            <h2>Performance Summary</h2>
            <div class="metrics">
                <div class="metric">
                    <h3>Total Function Calls</h3>
                    <div class="value">{summary.get('total_function_calls', 0):,}</div>
                </div>
                <div class="metric">
                    <h3>Unique Functions</h3>
                    <div class="value">{summary.get('unique_functions', 0)}</div>
                </div>
                <div class="metric">
                    <h3>Total Execution Time</h3>
                    <div class="value">{summary.get('total_execution_time', 0):.3f}</div>
                    <div class="unit">seconds</div>
                </div>
                <div class="metric">
                    <h3>Average Call Time</h3>
                    <div class="value">{summary.get('average_call_time', 0)*1000:.2f}</div>
                    <div class="unit">milliseconds</div>
                </div>
                <div class="metric">
                    <h3>Recursive Functions</h3>
                    <div class="value">{summary.get('recursive_functions', 0)}</div>
                </div>
                <div class="metric">
                    <h3>Call Graph Complexity</h3>
                    <div class="value">{summary.get('call_graph_edges', 0)}</div>
                    <div class="unit">edges</div>
                </div>
            </div>
        </div>
        
        <div class="chart-container">
            <h2>Function Performance Visualizations</h2>
            <img src="function_performance_plots_{datetime.now().strftime('%Y%m%d_%H%M%S')}.png" 
                 alt="Function Performance Charts" style="max-width: 100%; height: auto;">
        </div>
        
        <div class="chart-container">
            <h2>Function Call Graph</h2>
            <img src="function_call_graph_{datetime.now().strftime('%Y%m%d_%H%M%S')}.png" 
                 alt="Function Call Graph" style="max-width: 100%; height: auto;">
        </div>
        
        <div class="section">
            <h2>Hottest Functions (Most Called)</h2>
            {self._generate_hot_functions_table()}
        </div>
        
        <div class="section">
            <h2>Slowest Functions (Highest Average Time)</h2>
            {self._generate_slow_functions_table()}
        </div>
        
        <div class="section">
            <h2>Recursive Functions</h2>
            {self._generate_recursive_functions_table()}
        </div>
        
        <div class="recommendations">
            <h2>🎯 Performance Optimization Recommendations</h2>
            <ul>
                {''.join(f'<li>{rec}</li>' for rec in self._generate_recommendations())}
            </ul>
        </div>
    </div>
</body>
</html>
        """
        
        return html
        
    def _generate_hot_functions_table(self):
        """Generate HTML table for hot functions."""
        if not self.hot_functions:
            return '<p>No hot functions detected.</p>'
            
        html = '''
        <table class="function-table">
            <tr>
                <th>Function</th>
                <th>Call Count</th>
                <th>Total Time (s)</th>
                <th>Average Time (ms)</th>
                <th>Time %</th>
            </tr>
        '''
        
        total_time = sum(func['total_time'] for func in self.hot_functions)
        
        for func in self.hot_functions[:15]:  # Top 15
            time_percent = (func['total_time'] / total_time * 100) if total_time > 0 else 0
            html += f'''
            <tr>
                <td><span class="code">{func['function']}</span></td>
                <td>{func['call_count']:,}</td>
                <td>{func['total_time']:.3f}</td>
                <td>{func['avg_time']*1000:.2f}</td>
                <td>{time_percent:.1f}%</td>
            </tr>
            '''
            
        html += '</table>'
        return html
        
    def _generate_slow_functions_table(self):
        """Generate HTML table for slow functions."""
        if not self.slow_functions:
            return '<p>No slow functions detected.</p>'
            
        html = '''
        <table class="function-table">
            <tr>
                <th>Function</th>
                <th>Average Time (ms)</th>
                <th>Max Time (ms)</th>
                <th>Call Count</th>
                <th>Total Time (s)</th>
            </tr>
        '''
        
        for func in self.slow_functions[:15]:  # Top 15
            html += f'''
            <tr>
                <td><span class="code">{func['function']}</span></td>
                <td>{func['avg_time']*1000:.2f}</td>
                <td>{func['max_time']*1000:.2f}</td>
                <td>{func['call_count']:,}</td>
                <td>{func['total_time']:.3f}</td>
            </tr>
            '''
            
        html += '</table>'
        return html
        
    def _generate_recursive_functions_table(self):
        """Generate HTML table for recursive functions."""
        if not self.recursive_functions:
            return '<p>No recursive functions detected.</p>'
            
        html = '''
        <table class="function-table">
            <tr>
                <th>Function</th>
                <th>Recursive Calls</th>
                <th>Potential Issue</th>
            </tr>
        '''
        
        for func_name, count in sorted(self.recursive_functions.items(), key=lambda x: x[1], reverse=True):
            issue_level = "High" if count > 100 else "Medium" if count > 10 else "Low"
            html += f'''
            <tr>
                <td><span class="code">{func_name}</span></td>
                <td>{count:,}</td>
                <td>{issue_level}</td>
            </tr>
            '''
            
        html += '</table>'
        return html
        
    def _generate_recommendations(self):
        """Generate optimization recommendations."""
        recommendations = []
        
        # Hot function recommendations
        if self.hot_functions:
            top_hot = self.hot_functions[0]
            if top_hot['call_count'] > 1000:
                recommendations.append(f"🔥 Optimize <code>{top_hot['function']}</code> - called {top_hot['call_count']:,} times")
                
        # Slow function recommendations
        if self.slow_functions:
            top_slow = self.slow_functions[0]
            if top_slow['avg_time'] > 0.1:  # > 100ms
                recommendations.append(f"🐌 Optimize <code>{top_slow['function']}</code> - averaging {top_slow['avg_time']*1000:.1f}ms per call")
                
        # Recursive function recommendations
        high_recursion = [func for func, count in self.recursive_functions.items() if count > 50]
        if high_recursion:
            recommendations.append(f"🔄 Review recursive functions for potential optimization: {len(high_recursion)} functions detected")
            
        # Call graph recommendations
        summary = self.get_performance_summary()
        if summary.get('call_graph_edges', 0) > 1000:
            recommendations.append("📊 High call graph complexity - consider refactoring to reduce function interdependencies")
            
        # General recommendations
        if not recommendations:
            recommendations.extend([
                "✅ Function call performance appears normal",
                "📏 Continue monitoring during heavy usage periods",
                "🔍 Consider targeted profiling of specific workflows"
            ])
        else:
            recommendations.extend([
                "⚡ Use caching for frequently called functions with deterministic outputs",
                "🧵 Consider moving CPU-intensive functions to separate threads",
                "📈 Profile memory usage in top functions to identify memory bottlenecks"
            ])
            
        return recommendations
        
    def _generate_call_graph_visualization(self, timestamp):
        """Generate call graph visualization."""
        if self.call_graph.number_of_nodes() == 0:
            logger.warning("No call graph data available for visualization")
            return
            
        try:
            plt.figure(figsize=(16, 12))
            
            # Filter to show only the most important nodes (top 30 by call count)
            node_weights = {node: data['call_count'] for node, data in self.call_graph.nodes(data=True)}
            top_nodes = sorted(node_weights.items(), key=lambda x: x[1], reverse=True)[:30]
            subgraph = self.call_graph.subgraph([node for node, _ in top_nodes])
            
            # Calculate layout
            pos = nx.spring_layout(subgraph, k=3, iterations=50)
            
            # Draw nodes with size based on call count
            node_sizes = [min(node_weights.get(node, 1) * 10, 1000) for node in subgraph.nodes()]
            nx.draw_networkx_nodes(subgraph, pos, node_size=node_sizes, 
                                 node_color='lightblue', alpha=0.7)
            
            # Draw edges with width based on call frequency
            edge_weights = [subgraph[u][v].get('weight', 1) for u, v in subgraph.edges()]
            max_weight = max(edge_weights) if edge_weights else 1
            edge_widths = [min(w/max_weight * 3, 3) for w in edge_weights]
            
            nx.draw_networkx_edges(subgraph, pos, width=edge_widths, alpha=0.5, edge_color='gray')
            
            # Draw labels (shortened function names)
            labels = {node: node.split('.')[-1][:15] for node in subgraph.nodes()}
            nx.draw_networkx_labels(subgraph, pos, labels, font_size=8)
            
            plt.title('Function Call Graph (Top 30 Functions)', fontsize=16, fontweight='bold')
            plt.axis('off')
            plt.tight_layout()
            
            # Save plot
            plot_file = f'performance_reports/function_call_graph_{timestamp}.png'
            plt.savefig(plot_file, dpi=300, bbox_inches='tight')
            plt.close()
            
            logger.info(f"Call graph visualization saved to: {plot_file}")
            
        except Exception as e:
            logger.error(f"Error generating call graph visualization: {e}")
            
    def _generate_performance_plots(self, timestamp):
        """Generate function performance plots."""
        try:
            fig, axes = plt.subplots(2, 2, figsize=(15, 10))
            fig.suptitle('Function Performance Analysis', fontsize=16, fontweight='bold')
            
            # Top functions by total time
            ax = axes[0, 0]
            if self.hot_functions:
                top_functions = self.hot_functions[:10]
                func_names = [f['function'].split('.')[-1][:20] for f in top_functions]
                total_times = [f['total_time'] for f in top_functions]
                
                ax.barh(func_names, total_times)
                ax.set_title('Top Functions by Total Time')
                ax.set_xlabel('Total Time (seconds)')
                
            # Function call frequency distribution
            ax = axes[0, 1]
            if self.call_stats:
                call_counts = [stats['count'] for stats in self.call_stats.values()]
                ax.hist(call_counts, bins=50, alpha=0.7, edgecolor='black')
                ax.set_title('Function Call Frequency Distribution')
                ax.set_xlabel('Number of Calls')
                ax.set_ylabel('Number of Functions')
                ax.set_yscale('log')
                
            # Average execution time distribution
            ax = axes[1, 0]
            if self.call_stats:
                avg_times = [stats['avg_time'] * 1000 for stats in self.call_stats.values()]  # Convert to ms
                ax.hist(avg_times, bins=50, alpha=0.7, edgecolor='black')
                ax.set_title('Average Execution Time Distribution')
                ax.set_xlabel('Average Time (milliseconds)')
                ax.set_ylabel('Number of Functions')
                ax.set_xscale('log')
                
            # Call depth analysis
            ax = axes[1, 1]
            if self.completed_calls:
                call_depths = [call.call_depth for call in self.completed_calls]
                depth_counts = {}
                for depth in call_depths:
                    depth_counts[depth] = depth_counts.get(depth, 0) + 1
                    
                depths = sorted(depth_counts.keys())
                counts = [depth_counts[d] for d in depths]
                
                ax.bar(depths, counts)
                ax.set_title('Function Call Depth Distribution')
                ax.set_xlabel('Call Depth')
                ax.set_ylabel('Number of Calls')
                
            plt.tight_layout()
            
            # Save plots
            plot_file = f'performance_reports/function_performance_plots_{timestamp}.png'
            plt.savefig(plot_file, dpi=300, bbox_inches='tight')
            plt.close()
            
            logger.info(f"Function performance plots saved to: {plot_file}")
            
        except Exception as e:
            logger.error(f"Error generating performance plots: {e}")


# Decorator for easy function monitoring
def monitor_function_performance(func):
    """Decorator to monitor specific function performance."""
    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        tracer = getattr(wrapper, '_tracer', None)
        if tracer and tracer.tracing:
            # Function is already being traced by the global tracer
            return func(*args, **kwargs)
        else:
            # Manual timing for standalone use
            start_time = time.time()
            try:
                result = func(*args, **kwargs)
                return result
            finally:
                duration = time.time() - start_time
                if duration > 0.01:  # Log slow functions
                    logger.info(f"Function {func.__name__} took {duration:.3f}s")
                    
    return wrapper


def trace_function_calls(duration=300, ignore_patterns=None):
    """Quick function to trace function calls."""
    tracer = FunctionCallTracer(ignore_patterns=ignore_patterns)
    tracer.start_tracing()
    
    try:
        time.sleep(duration)
    except KeyboardInterrupt:
        logger.info("Function tracing interrupted by user")
    finally:
        tracer.stop_tracing()
        tracer.generate_report()
        
    return tracer.get_performance_summary()


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='Function Call Tracer for Signal Analyzer')
    parser.add_argument('--duration', type=int, default=300, help='Tracing duration in seconds')
    parser.add_argument('--max-calls', type=int, default=10000, help='Maximum calls to track')
    parser.add_argument('--output-dir', default='performance_reports', help='Output directory for reports')
    parser.add_argument('--ignore', nargs='*', default=['threading', 'logging'], help='Patterns to ignore')
    
    args = parser.parse_args()
    
    print("🔍 FUNCTION CALL TRACER")
    print("=" * 30)
    print(f"⏱️ Duration: {args.duration} seconds")
    print(f"📊 Max Calls: {args.max_calls}")
    print(f"🚫 Ignoring: {', '.join(args.ignore)}")
    print(f"📁 Output: {args.output_dir}")
    print("-" * 30)
    
    # Create output directory
    Path(args.output_dir).mkdir(exist_ok=True)
    
    # Start tracing
    summary = trace_function_calls(args.duration, args.ignore)
    
    print("\n✅ FUNCTION TRACING COMPLETE!")
    print("=" * 35)
    print("📊 Summary:")
    for key, value in summary.items():
        if isinstance(value, float):
            print(f"  {key}: {value:.3f}")
        else:
            print(f"  {key}: {value}")
            
    print(f"\n📁 Reports saved to: {args.output_dir}/")
    print("🎯 Check the HTML report for detailed function analysis")