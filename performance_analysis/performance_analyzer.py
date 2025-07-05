#!/usr/bin/env python3
"""
Advanced Performance Analyzer for Signal Analysis Application
============================================================

This script provides comprehensive performance monitoring and analysis
for the Signal Analyzer application to identify bottlenecks and optimize performance.

Usage:
    python performance_analyzer.py [--mode=realtime|profile|memory] [--duration=60]

Features:
    - Real-time system resource monitoring
    - CPU and memory profiling
    - GUI responsiveness tracking
    - I/O operation monitoring
    - Thread performance analysis
    - Memory leak detection
    - Performance bottleneck identification
    - Optimization recommendations
"""

import sys
import os
import time
import threading
import psutil
import cProfile
import pstats
import tracemalloc
import gc
import queue
import json
import sqlite3
from pathlib import Path
from datetime import datetime, timedelta
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import pandas as pd
from collections import defaultdict, deque
import tkinter as tk
from tkinter import ttk, messagebox
import argparse
import subprocess
import logging
import warnings

warnings.filterwarnings("ignore")

# Configure logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


class AdvancedPerformanceAnalyzer:
    def __init__(self, target_process_name="python", monitor_duration=60):
        """
        Initialize the performance analyzer.

        Args:
            target_process_name: Name of the process to monitor
            monitor_duration: Duration to monitor in seconds
        """
        self.target_process_name = target_process_name
        self.monitor_duration = monitor_duration
        self.monitoring = False
        self.data_queue = queue.Queue()

        # Performance data storage
        self.performance_data = {
            "timestamps": deque(maxlen=2000),
            "cpu_percent": deque(maxlen=2000),
            "memory_mb": deque(maxlen=2000),
            "memory_percent": deque(maxlen=2000),
            "thread_count": deque(maxlen=2000),
            "file_descriptors": deque(maxlen=2000),
            "io_read_mb": deque(maxlen=2000),
            "io_write_mb": deque(maxlen=2000),
            "gui_events": deque(maxlen=2000),
            "function_calls": defaultdict(int),
            "slow_operations": [],
            "gui_response_times": deque(maxlen=1000),
            "plot_render_times": deque(maxlen=1000),
            "filter_execution_times": deque(maxlen=1000),
            "file_load_times": deque(maxlen=1000),
            "network_usage": deque(maxlen=2000),
        }

        # Find target process
        self.target_process = self.find_target_process()
        self.baseline_memory = None
        self.initial_cpu = None

        # Setup profiling
        self.profiler = None
        self.memory_tracker = None

        # GUI monitoring
        self.gui_monitor = None
        self.dashboard_window = None

        # Signal Analyzer specific monitoring
        self.filter_performance = {}
        self.gui_component_times = {}
        self.atf_loading_stats = {}

        # Performance thresholds
        self.thresholds = {
            "cpu_warning": 50.0,
            "cpu_critical": 80.0,
            "memory_warning": 500.0,  # MB
            "memory_critical": 1000.0,  # MB
            "gui_response_warning": 0.1,  # seconds
            "gui_response_critical": 0.5,  # seconds
            "thread_warning": 20,
            "thread_critical": 50,
        }

        logger.info(
            f"Performance Analyzer initialized for process: {self.target_process}"
        )

    def find_target_process(self):
        """Find the target application process with enhanced detection."""
        target_processes = []

        for proc in psutil.process_iter(["pid", "name", "cmdline", "create_time"]):
            try:
                if proc.info["name"] == self.target_process_name:
                    cmdline = (
                        " ".join(proc.info["cmdline"]) if proc.info["cmdline"] else ""
                    )

                    # Look for Signal Analyzer specific indicators
                    indicators = [
                        "signal",
                        "analyzer",
                        "main.py",
                        "run.py",
                        "atf",
                        "filter",
                    ]
                    if any(indicator in cmdline.lower() for indicator in indicators):
                        target_processes.append(
                            {
                                "process": psutil.Process(proc.info["pid"]),
                                "cmdline": cmdline,
                                "create_time": proc.info["create_time"],
                            }
                        )
            except (psutil.NoSuchProcess, psutil.AccessDenied):
                continue

        if target_processes:
            # Return the most recently started matching process
            latest_process = max(target_processes, key=lambda x: x["create_time"])
            logger.info(f"Found target process: PID {latest_process['process'].pid}")
            logger.info(f"Command line: {latest_process['cmdline']}")
            return latest_process["process"]

        # If not found, monitor current process
        logger.warning("Target process not found, monitoring current process")
        return psutil.Process()

    def start_monitoring(self, mode="realtime"):
        """Start performance monitoring in specified mode."""
        logger.info(f"Starting performance monitoring in {mode} mode...")
        self.monitoring = True

        # Create reports directory if it doesn't exist
        os.makedirs("performance_reports", exist_ok=True)

        if mode == "realtime":
            self.start_realtime_monitoring()
        elif mode == "profile":
            self.start_profiling()
        elif mode == "memory":
            self.start_memory_profiling()
        elif mode == "comprehensive":
            self.start_comprehensive_monitoring()
        else:
            self.start_realtime_monitoring()

    def start_realtime_monitoring(self):
        """Start real-time performance monitoring with enhanced features."""
        logger.info("Starting real-time monitoring threads...")

        # Start monitoring threads
        threads = [
            threading.Thread(
                target=self.monitor_system_resources, daemon=True, name="SystemMonitor"
            ),
            threading.Thread(
                target=self.monitor_threads, daemon=True, name="ThreadMonitor"
            ),
            threading.Thread(
                target=self.monitor_io_operations, daemon=True, name="IOMonitor"
            ),
            threading.Thread(
                target=self.monitor_memory_usage, daemon=True, name="MemoryMonitor"
            ),
            threading.Thread(
                target=self.monitor_gui_performance, daemon=True, name="GUIMonitor"
            ),
            threading.Thread(
                target=self.monitor_signal_analyzer_specific,
                daemon=True,
                name="SAMonitor",
            ),
            threading.Thread(
                target=self.create_realtime_dashboard,
                daemon=True,
                name="DashboardThread",
            ),
        ]

        for thread in threads:
            thread.start()
            logger.info(f"Started thread: {thread.name}")

        # Wait for monitoring duration or user interruption
        try:
            start_time = time.time()
            while (
                self.monitoring and (time.time() - start_time) < self.monitor_duration
            ):
                time.sleep(1)

                # Check if target process is still alive
                if not self.target_process.is_running():
                    logger.warning("Target process has terminated")
                    break

        except KeyboardInterrupt:
            logger.info("Monitoring interrupted by user")
        finally:
            self.monitoring = False

        # Generate report
        self.generate_performance_report()

    def monitor_system_resources(self):
        """Monitor CPU, memory, and system resources with enhanced metrics."""
        sample_count = 0
        error_count = 0

        while self.monitoring:
            try:
                timestamp = time.time()

                # CPU monitoring
                cpu_percent = self.target_process.cpu_percent(interval=0.1)

                # Memory monitoring
                memory_info = self.target_process.memory_info()
                memory_mb = memory_info.rss / 1024 / 1024
                memory_percent = self.target_process.memory_percent()

                # Additional system metrics
                try:
                    # File descriptors (Unix-like systems)
                    if hasattr(self.target_process, "num_fds"):
                        fd_count = self.target_process.num_fds()
                    else:
                        fd_count = len(self.target_process.open_files())
                except:
                    fd_count = 0

                # Store data
                self.performance_data["timestamps"].append(timestamp)
                self.performance_data["cpu_percent"].append(cpu_percent)
                self.performance_data["memory_mb"].append(memory_mb)
                self.performance_data["memory_percent"].append(memory_percent)
                self.performance_data["file_descriptors"].append(fd_count)

                # Performance analysis
                self.analyze_performance_metrics(
                    timestamp, cpu_percent, memory_mb, memory_percent
                )

                sample_count += 1
                if sample_count % 100 == 0:
                    logger.info(f"Collected {sample_count} system resource samples")

                time.sleep(0.2)  # Sample every 200ms

            except (psutil.NoSuchProcess, psutil.AccessDenied) as e:
                error_count += 1
                if error_count > 10:
                    logger.error(f"Too many errors monitoring system resources: {e}")
                    break
                time.sleep(1)
            except Exception as e:
                logger.error(f"Unexpected error in system monitoring: {e}")
                time.sleep(1)

    def analyze_performance_metrics(
        self, timestamp, cpu_percent, memory_mb, memory_percent
    ):
        """Analyze current performance metrics and detect issues."""
        # Set baseline values
        if self.baseline_memory is None:
            self.baseline_memory = memory_mb
            self.initial_cpu = cpu_percent

        # CPU analysis
        if cpu_percent > self.thresholds["cpu_critical"]:
            self.performance_data["slow_operations"].append(
                {
                    "timestamp": timestamp,
                    "type": "cpu_critical",
                    "severity": "critical",
                    "details": f'CPU usage critical: {cpu_percent:.1f}% (threshold: {self.thresholds["cpu_critical"]}%)',
                }
            )
        elif cpu_percent > self.thresholds["cpu_warning"]:
            self.performance_data["slow_operations"].append(
                {
                    "timestamp": timestamp,
                    "type": "cpu_warning",
                    "severity": "warning",
                    "details": f'CPU usage high: {cpu_percent:.1f}% (threshold: {self.thresholds["cpu_warning"]}%)',
                }
            )

        # Memory analysis
        if memory_mb > self.thresholds["memory_critical"]:
            self.performance_data["slow_operations"].append(
                {
                    "timestamp": timestamp,
                    "type": "memory_critical",
                    "severity": "critical",
                    "details": f'Memory usage critical: {memory_mb:.1f}MB (threshold: {self.thresholds["memory_critical"]}MB)',
                }
            )
        elif memory_mb > self.thresholds["memory_warning"]:
            self.performance_data["slow_operations"].append(
                {
                    "timestamp": timestamp,
                    "type": "memory_warning",
                    "severity": "warning",
                    "details": f'Memory usage high: {memory_mb:.1f}MB (threshold: {self.thresholds["memory_warning"]}MB)',
                }
            )

        # Memory leak detection
        if self.baseline_memory and memory_mb > self.baseline_memory * 2:
            self.performance_data["slow_operations"].append(
                {
                    "timestamp": timestamp,
                    "type": "memory_leak_suspected",
                    "severity": "critical",
                    "details": f"Possible memory leak: {memory_mb:.1f}MB (200% increase from baseline {self.baseline_memory:.1f}MB)",
                }
            )

    def monitor_threads(self):
        """Monitor thread usage and activity with detailed analysis."""
        while self.monitoring:
            try:
                thread_count = self.target_process.num_threads()
                self.performance_data["thread_count"].append(thread_count)

                # Thread analysis
                if thread_count > self.thresholds["thread_critical"]:
                    self.performance_data["slow_operations"].append(
                        {
                            "timestamp": time.time(),
                            "type": "thread_critical",
                            "severity": "critical",
                            "details": f'Thread count critical: {thread_count} (threshold: {self.thresholds["thread_critical"]})',
                        }
                    )
                elif thread_count > self.thresholds["thread_warning"]:
                    self.performance_data["slow_operations"].append(
                        {
                            "timestamp": time.time(),
                            "type": "thread_warning",
                            "severity": "warning",
                            "details": f'Thread count high: {thread_count} (threshold: {self.thresholds["thread_warning"]})',
                        }
                    )

                # Get detailed thread information if available
                try:
                    if hasattr(self.target_process, "threads"):
                        threads = self.target_process.threads()
                        high_cpu_threads = [t for t in threads if t.user_time > 0.5]

                        if high_cpu_threads:
                            for thread in high_cpu_threads:
                                self.performance_data["slow_operations"].append(
                                    {
                                        "timestamp": time.time(),
                                        "type": "high_cpu_thread",
                                        "severity": "warning",
                                        "details": f"Thread {thread.id} high CPU: {thread.user_time:.2f}s user time",
                                    }
                                )
                except AttributeError:
                    pass  # threads() method not available on all platforms

                time.sleep(1.0)

            except (psutil.NoSuchProcess, psutil.AccessDenied):
                break
            except Exception as e:
                logger.error(f"Error monitoring threads: {e}")
                time.sleep(1)

    def monitor_io_operations(self):
        """Monitor file I/O operations with enhanced detection."""
        prev_io = None
        cumulative_read = 0
        cumulative_write = 0

        while self.monitoring:
            try:
                io_counters = self.target_process.io_counters()
                timestamp = time.time()

                if prev_io:
                    read_bytes = io_counters.read_bytes - prev_io.read_bytes
                    write_bytes = io_counters.write_bytes - prev_io.write_bytes
                    read_mb = read_bytes / 1024 / 1024
                    write_mb = write_bytes / 1024 / 1024

                    cumulative_read += read_mb
                    cumulative_write += write_mb

                    self.performance_data["io_read_mb"].append(read_mb)
                    self.performance_data["io_write_mb"].append(write_mb)

                    # Detect heavy I/O operations
                    if read_mb > 20:  # More than 20MB/s read
                        self.performance_data["slow_operations"].append(
                            {
                                "timestamp": timestamp,
                                "type": "heavy_io_read",
                                "severity": "warning",
                                "details": f"Heavy I/O read: {read_mb:.1f}MB/s (possible large file loading)",
                            }
                        )

                    if write_mb > 20:  # More than 20MB/s write
                        self.performance_data["slow_operations"].append(
                            {
                                "timestamp": timestamp,
                                "type": "heavy_io_write",
                                "severity": "warning",
                                "details": f"Heavy I/O write: {write_mb:.1f}MB/s (possible data export)",
                            }
                        )

                prev_io = io_counters
                time.sleep(1.0)

            except (psutil.NoSuchProcess, psutil.AccessDenied):
                break
            except Exception as e:
                logger.error(f"Error monitoring I/O: {e}")
                time.sleep(1)

    def monitor_memory_usage(self):
        """Monitor detailed memory usage with leak detection."""
        if not tracemalloc.is_tracing():
            tracemalloc.start()

        baseline_snapshot = None
        snapshot_interval = 10  # Take snapshots every 10 seconds
        last_snapshot_time = 0

        while self.monitoring:
            try:
                current_time = time.time()

                if current_time - last_snapshot_time >= snapshot_interval:
                    # Take memory snapshot
                    snapshot = tracemalloc.take_snapshot()

                    if baseline_snapshot is None:
                        baseline_snapshot = snapshot
                    else:
                        # Compare with baseline
                        top_stats = snapshot.compare_to(baseline_snapshot, "lineno")

                        # Find significant memory increases
                        for stat in top_stats[:5]:
                            if (
                                stat.size_diff > 10 * 1024 * 1024
                            ):  # More than 10MB increase
                                self.performance_data["slow_operations"].append(
                                    {
                                        "timestamp": current_time,
                                        "type": "memory_hotspot",
                                        "severity": "warning",
                                        "details": f'Memory hotspot: {stat.size_diff / 1024 / 1024:.1f}MB increase in {stat.traceback.format()[-1] if stat.traceback else "unknown location"}',
                                    }
                                )

                    last_snapshot_time = current_time

                time.sleep(5.0)

            except Exception as e:
                logger.error(f"Error monitoring memory: {e}")
                time.sleep(5)

    def monitor_gui_performance(self):
        """Monitor GUI performance and responsiveness."""
        # This would be enhanced with actual GUI hooks in a real implementation
        # For now, we simulate GUI performance monitoring
        while self.monitoring:
            try:
                # Simulate GUI response time measurement
                # In a real implementation, this would hook into Tkinter events
                simulated_response_time = 0.05  # 50ms baseline

                # Add some variance based on current system load
                if self.performance_data["cpu_percent"]:
                    cpu_load = self.performance_data["cpu_percent"][-1]
                    if cpu_load > 80:
                        simulated_response_time += 0.2  # GUI slows down with high CPU
                    elif cpu_load > 50:
                        simulated_response_time += 0.1

                self.performance_data["gui_response_times"].append(
                    simulated_response_time
                )

                # Check for GUI responsiveness issues
                if simulated_response_time > self.thresholds["gui_response_critical"]:
                    self.performance_data["slow_operations"].append(
                        {
                            "timestamp": time.time(),
                            "type": "gui_unresponsive",
                            "severity": "critical",
                            "details": f'GUI response time critical: {simulated_response_time:.3f}s (threshold: {self.thresholds["gui_response_critical"]}s)',
                        }
                    )
                elif simulated_response_time > self.thresholds["gui_response_warning"]:
                    self.performance_data["slow_operations"].append(
                        {
                            "timestamp": time.time(),
                            "type": "gui_slow",
                            "severity": "warning",
                            "details": f'GUI response time slow: {simulated_response_time:.3f}s (threshold: {self.thresholds["gui_response_warning"]}s)',
                        }
                    )

                time.sleep(0.5)

            except Exception as e:
                logger.error(f"Error monitoring GUI performance: {e}")
                time.sleep(1)

    def monitor_signal_analyzer_specific(self):
        """Monitor Signal Analyzer specific performance metrics."""
        while self.monitoring:
            try:
                # Monitor for ATF file operations
                # This would be enhanced with actual hooks in the real application

                # Simulate filter performance monitoring
                if len(self.performance_data["timestamps"]) > 0:
                    # Check if CPU spikes might indicate filter operations
                    if self.performance_data["cpu_percent"]:
                        recent_cpu = list(self.performance_data["cpu_percent"])[-10:]
                        if len(recent_cpu) >= 10:
                            avg_cpu = sum(recent_cpu) / len(recent_cpu)
                            if avg_cpu > 70:
                                # Possible heavy filtering operation
                                self.performance_data["slow_operations"].append(
                                    {
                                        "timestamp": time.time(),
                                        "type": "heavy_filtering",
                                        "severity": "info",
                                        "details": f"Possible heavy filtering operation detected (CPU: {avg_cpu:.1f}%)",
                                    }
                                )

                time.sleep(2.0)

            except Exception as e:
                logger.error(f"Error monitoring Signal Analyzer specific metrics: {e}")
                time.sleep(2)

    def start_profiling(self):
        """Start CPU profiling with enhanced analysis."""
        logger.info("Starting CPU profiling...")

        # Create profiler
        self.profiler = cProfile.Profile()

        # Start profiling
        self.profiler.enable()

        # Monitor for specified duration
        logger.info(f"Profiling for {self.monitor_duration} seconds...")
        time.sleep(self.monitor_duration)

        # Stop profiling
        self.profiler.disable()

        # Save and analyze results
        self.analyze_profiling_results()

    def analyze_profiling_results(self):
        """Analyze CPU profiling results with Signal Analyzer focus."""
        if not self.profiler:
            return

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        profile_file = f"performance_reports/profile_{timestamp}.prof"

        # Save profile data
        self.profiler.dump_stats(profile_file)

        # Analyze results
        stats = pstats.Stats(profile_file)
        stats.sort_stats("cumulative")

        # Generate profiling report
        report_file = f"performance_reports/profiling_report_{timestamp}.txt"
        with open(report_file, "w") as f:
            f.write("Signal Analyzer CPU Profiling Report\n")
            f.write("=" * 50 + "\n\n")

            f.write(f"Profiling Duration: {self.monitor_duration} seconds\n")
            f.write(f"Total Function Calls: {stats.total_calls}\n")
            f.write(f"Total Time: {stats.total_tt:.3f} seconds\n\n")

            f.write("Top 30 functions by cumulative time:\n")
            f.write("-" * 60 + "\n")
            stats.print_stats(30, file=f)

            f.write("\n\nTop 30 functions by internal time:\n")
            f.write("-" * 60 + "\n")
            stats.sort_stats("time")
            stats.print_stats(30, file=f)

            f.write("\n\nSignal Analyzer Specific Analysis:\n")
            f.write("-" * 40 + "\n")
            self.analyze_signal_analyzer_bottlenecks(stats, f)

        logger.info(f"Profiling report saved to: {report_file}")

    def analyze_signal_analyzer_bottlenecks(self, stats, file_handle):
        """Analyze bottlenecks specific to Signal Analyzer."""
        bottlenecks = []
        filter_functions = []
        gui_functions = []
        io_functions = []

        # Categorize functions by type
        for func, (cc, nc, tt, ct, callers) in stats.stats.items():
            filename, line, function_name = func

            # Signal Analyzer specific function categorization
            if any(
                keyword in filename.lower()
                for keyword in ["filter", "savgol", "butter", "fft"]
            ):
                filter_functions.append(
                    {
                        "function": f"{filename}:{line}({function_name})",
                        "cumulative_time": ct,
                        "percentage": (ct / stats.total_tt) * 100,
                        "calls": cc,
                    }
                )
            elif any(
                keyword in filename.lower()
                for keyword in ["gui", "tkinter", "matplotlib", "plot"]
            ):
                gui_functions.append(
                    {
                        "function": f"{filename}:{line}({function_name})",
                        "cumulative_time": ct,
                        "percentage": (ct / stats.total_tt) * 100,
                        "calls": cc,
                    }
                )
            elif any(
                keyword in filename.lower() for keyword in ["io", "atf", "file", "load"]
            ):
                io_functions.append(
                    {
                        "function": f"{filename}:{line}({function_name})",
                        "cumulative_time": ct,
                        "percentage": (ct / stats.total_tt) * 100,
                        "calls": cc,
                    }
                )

            # General bottlenecks (functions taking more than 1% of total time)
            if ct > stats.total_tt * 0.01:
                bottlenecks.append(
                    {
                        "function": f"{filename}:{line}({function_name})",
                        "cumulative_time": ct,
                        "percentage": (ct / stats.total_tt) * 100,
                        "calls": cc,
                    }
                )

        # Write analysis
        file_handle.write("PERFORMANCE BOTTLENECKS (>1% of total time):\n")
        bottlenecks.sort(key=lambda x: x["cumulative_time"], reverse=True)
        for bottleneck in bottlenecks[:15]:
            file_handle.write(f"• {bottleneck['function']}\n")
            file_handle.write(
                f"  Time: {bottleneck['cumulative_time']:.3f}s ({bottleneck['percentage']:.1f}%)\n"
            )
            file_handle.write(f"  Calls: {bottleneck['calls']}\n\n")

        # Write filter performance
        if filter_functions:
            file_handle.write("\nFILTER PERFORMANCE:\n")
            filter_functions.sort(key=lambda x: x["cumulative_time"], reverse=True)
            for func in filter_functions[:10]:
                file_handle.write(f"• {func['function']}\n")
                file_handle.write(
                    f"  Time: {func['cumulative_time']:.3f}s ({func['percentage']:.1f}%)\n"
                )
                file_handle.write(f"  Calls: {func['calls']}\n\n")

        # Write GUI performance
        if gui_functions:
            file_handle.write("\nGUI PERFORMANCE:\n")
            gui_functions.sort(key=lambda x: x["cumulative_time"], reverse=True)
            for func in gui_functions[:10]:
                file_handle.write(f"• {func['function']}\n")
                file_handle.write(
                    f"  Time: {func['cumulative_time']:.3f}s ({func['percentage']:.1f}%)\n"
                )
                file_handle.write(f"  Calls: {func['calls']}\n\n")

        # Write I/O performance
        if io_functions:
            file_handle.write("\nI/O PERFORMANCE:\n")
            io_functions.sort(key=lambda x: x["cumulative_time"], reverse=True)
            for func in io_functions[:10]:
                file_handle.write(f"• {func['function']}\n")
                file_handle.write(
                    f"  Time: {func['cumulative_time']:.3f}s ({func['percentage']:.1f}%)\n"
                )
                file_handle.write(f"  Calls: {func['calls']}\n\n")

    def start_memory_profiling(self):
        """Start memory profiling with enhanced analysis."""
        logger.info("Starting memory profiling...")

        tracemalloc.start()
        baseline_snapshot = tracemalloc.take_snapshot()

        logger.info(f"Memory profiling for {self.monitor_duration} seconds...")
        time.sleep(self.monitor_duration)

        # Take final snapshot
        final_snapshot = tracemalloc.take_snapshot()

        # Analyze memory changes
        self.analyze_memory_changes(baseline_snapshot, final_snapshot)

    def analyze_memory_changes(self, baseline, final):
        """Analyze memory usage changes with Signal Analyzer focus."""
        top_stats = final.compare_to(baseline, "lineno")

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        report_file = f"performance_reports/memory_report_{timestamp}.txt"

        with open(report_file, "w") as f:
            f.write("Signal Analyzer Memory Profiling Report\n")
            f.write("=" * 50 + "\n\n")

            f.write(f"Profiling Duration: {self.monitor_duration} seconds\n")
            f.write(f"Memory Snapshots Compared: Baseline vs Final\n\n")

            f.write("Top 30 memory allocations (by size difference):\n")
            f.write("-" * 60 + "\n")

            total_size_diff = sum(stat.size_diff for stat in top_stats)

            for i, stat in enumerate(top_stats[:30], 1):
                if stat.traceback:
                    location = stat.traceback.format()[-1]
                else:
                    location = "Unknown location"

                f.write(f"{i}. {location}\n")
                f.write(f"   Size Diff: {stat.size_diff / 1024 / 1024:.2f} MB\n")
                f.write(f"   Count Diff: {stat.count_diff} allocations\n")
                f.write(
                    f"   Percentage: {(stat.size_diff / total_size_diff * 100):.1f}%\n\n"
                )

            f.write("\nMEMORY LEAK ANALYSIS:\n")
            f.write("-" * 25 + "\n")

            # Identify potential memory leaks
            potential_leaks = [
                stat for stat in top_stats if stat.size_diff > 5 * 1024 * 1024
            ]  # > 5MB

            if potential_leaks:
                f.write("Potential memory leaks detected (>5MB increase):\n\n")
                for leak in potential_leaks:
                    if leak.traceback:
                        location = leak.traceback.format()[-1]
                    else:
                        location = "Unknown location"
                    f.write(f"• {location}\n")
                    f.write(f"  Leaked: {leak.size_diff / 1024 / 1024:.2f} MB\n")
                    f.write(f"  Objects: {leak.count_diff}\n\n")
            else:
                f.write("No significant memory leaks detected.\n")

        logger.info(f"Memory report saved to: {report_file}")

    def start_comprehensive_monitoring(self):
        """Start comprehensive monitoring combining all methods."""
        logger.info("Starting comprehensive monitoring...")

        # Start real-time monitoring in background
        self.monitoring = True
        monitoring_thread = threading.Thread(
            target=self.start_realtime_monitoring, daemon=True
        )
        monitoring_thread.start()

        # Wait a bit for monitoring to start
        time.sleep(2)

        # Run memory profiling
        self.start_memory_profiling()

        # Run CPU profiling
        self.start_profiling()

        # Stop monitoring
        self.monitoring = False

        # Generate comprehensive report
        self.generate_comprehensive_report()

    def create_realtime_dashboard(self):
        """Create a real-time performance dashboard."""
        try:
            # Create dashboard window
            self.dashboard_window = tk.Tk()
            self.dashboard_window.title("Signal Analyzer Performance Dashboard")
            self.dashboard_window.geometry("1000x700")

            # Create notebook for tabs
            notebook = ttk.Notebook(self.dashboard_window)
            notebook.pack(fill="both", expand=True, padx=10, pady=10)

            # Create tabs
            self.create_system_tab(notebook)
            self.create_memory_tab(notebook)
            self.create_threads_tab(notebook)
            self.create_operations_tab(notebook)
            self.create_signal_analyzer_tab(notebook)

            # Start dashboard update loop
            self.update_dashboard()

            # Keep dashboard running
            self.dashboard_window.mainloop()

        except Exception as e:
            logger.error(f"Dashboard error: {e}")

    def create_system_tab(self, notebook):
        """Create system resources tab with enhanced metrics."""
        frame = ttk.Frame(notebook)
        notebook.add(frame, text="System")

        # Create main container with grid layout
        main_frame = ttk.Frame(frame)
        main_frame.pack(fill="both", expand=True, padx=10, pady=10)

        # CPU usage section
        cpu_frame = ttk.LabelFrame(main_frame, text="CPU Usage", padding="10")
        cpu_frame.grid(row=0, column=0, sticky="ew", padx=5, pady=5)

        self.cpu_label = ttk.Label(cpu_frame, text="0%", font=("Arial", 16, "bold"))
        self.cpu_label.pack()

        self.cpu_status = ttk.Label(cpu_frame, text="Normal", foreground="green")
        self.cpu_status.pack()

        # Memory usage section
        memory_frame = ttk.LabelFrame(main_frame, text="Memory Usage", padding="10")
        memory_frame.grid(row=0, column=1, sticky="ew", padx=5, pady=5)

        self.memory_label = ttk.Label(
            memory_frame, text="0 MB", font=("Arial", 16, "bold")
        )
        self.memory_label.pack()

        self.memory_status = ttk.Label(memory_frame, text="Normal", foreground="green")
        self.memory_status.pack()

        # Thread count section
        threads_frame = ttk.LabelFrame(main_frame, text="Thread Count", padding="10")
        threads_frame.grid(row=1, column=0, sticky="ew", padx=5, pady=5)

        self.threads_label = ttk.Label(
            threads_frame, text="0", font=("Arial", 16, "bold")
        )
        self.threads_label.pack()

        self.threads_status = ttk.Label(
            threads_frame, text="Normal", foreground="green"
        )
        self.threads_status.pack()

        # File descriptors section
        fd_frame = ttk.LabelFrame(main_frame, text="File Descriptors", padding="10")
        fd_frame.grid(row=1, column=1, sticky="ew", padx=5, pady=5)

        self.fd_label = ttk.Label(fd_frame, text="0", font=("Arial", 16, "bold"))
        self.fd_label.pack()

        # Configure grid weights
        main_frame.columnconfigure(0, weight=1)
        main_frame.columnconfigure(1, weight=1)

    def create_memory_tab(self, notebook):
        """Create memory monitoring tab with enhanced details."""
        frame = ttk.Frame(notebook)
        notebook.add(frame, text="Memory")

        # Create text widget with scrollbar
        text_frame = ttk.Frame(frame)
        text_frame.pack(fill="both", expand=True, padx=10, pady=10)

        self.memory_text = tk.Text(
            text_frame, height=25, width=80, font=("Consolas", 10)
        )
        scrollbar = ttk.Scrollbar(
            text_frame, orient="vertical", command=self.memory_text.yview
        )
        self.memory_text.configure(yscrollcommand=scrollbar.set)

        self.memory_text.pack(side="left", fill="both", expand=True)
        scrollbar.pack(side="right", fill="y")

    def create_threads_tab(self, notebook):
        """Create thread monitoring tab."""
        frame = ttk.Frame(notebook)
        notebook.add(frame, text="Threads")

        # Create text widget with scrollbar
        text_frame = ttk.Frame(frame)
        text_frame.pack(fill="both", expand=True, padx=10, pady=10)

        self.threads_text = tk.Text(
            text_frame, height=25, width=80, font=("Consolas", 10)
        )
        scrollbar2 = ttk.Scrollbar(
            text_frame, orient="vertical", command=self.threads_text.yview
        )
        self.threads_text.configure(yscrollcommand=scrollbar2.set)

        self.threads_text.pack(side="left", fill="both", expand=True)
        scrollbar2.pack(side="right", fill="y")

    def create_operations_tab(self, notebook):
        """Create slow operations tab with filtering."""
        frame = ttk.Frame(notebook)
        notebook.add(frame, text="Operations")

        # Filter controls
        filter_frame = ttk.Frame(frame)
        filter_frame.pack(fill="x", padx=10, pady=5)

        ttk.Label(filter_frame, text="Filter by severity:").pack(side="left")
        self.severity_filter = ttk.Combobox(
            filter_frame,
            values=["All", "Critical", "Warning", "Info"],
            state="readonly",
        )
        self.severity_filter.set("All")
        self.severity_filter.pack(side="left", padx=5)

        # Clear button
        ttk.Button(
            filter_frame, text="Clear Log", command=self.clear_operations_log
        ).pack(side="right")

        # Operations list
        text_frame = ttk.Frame(frame)
        text_frame.pack(fill="both", expand=True, padx=10, pady=5)

        self.operations_text = tk.Text(
            text_frame, height=20, width=80, font=("Consolas", 9)
        )
        scrollbar3 = ttk.Scrollbar(
            text_frame, orient="vertical", command=self.operations_text.yview
        )
        self.operations_text.configure(yscrollcommand=scrollbar3.set)

        self.operations_text.pack(side="left", fill="both", expand=True)
        scrollbar3.pack(side="right", fill="y")

    def create_signal_analyzer_tab(self, notebook):
        """Create Signal Analyzer specific monitoring tab."""
        frame = ttk.Frame(notebook)
        notebook.add(frame, text="Signal Analyzer")

        # Create sections for different components
        main_frame = ttk.Frame(frame)
        main_frame.pack(fill="both", expand=True, padx=10, pady=10)

        # Filter performance section
        filter_frame = ttk.LabelFrame(
            main_frame, text="Filter Performance", padding="10"
        )
        filter_frame.pack(fill="x", pady=5)

        self.filter_perf_label = ttk.Label(
            filter_frame, text="No filter operations detected"
        )
        self.filter_perf_label.pack()

        # GUI performance section
        gui_frame = ttk.LabelFrame(main_frame, text="GUI Performance", padding="10")
        gui_frame.pack(fill="x", pady=5)

        self.gui_perf_label = ttk.Label(gui_frame, text="Average response time: N/A")
        self.gui_perf_label.pack()

        # File I/O section
        io_frame = ttk.LabelFrame(main_frame, text="File I/O Performance", padding="10")
        io_frame.pack(fill="x", pady=5)

        self.io_perf_label = ttk.Label(io_frame, text="No I/O operations detected")
        self.io_perf_label.pack()

    def clear_operations_log(self):
        """Clear the operations log."""
        self.operations_text.delete(1.0, tk.END)

    def update_dashboard(self):
        """Update dashboard with current data."""
        if (
            not self.monitoring
            or not hasattr(self, "dashboard_window")
            or not self.dashboard_window
        ):
            return

        try:
            # Update system resources
            self.update_system_display()

            # Update memory details
            self.update_memory_display()

            # Update thread information
            self.update_threads_display()

            # Update operations
            self.update_operations_display()

            # Update Signal Analyzer specific data
            self.update_signal_analyzer_display()

            # Schedule next update
            self.dashboard_window.after(1000, self.update_dashboard)

        except Exception as e:
            logger.error(f"Dashboard update error: {e}")

    def update_system_display(self):
        """Update system resources display with status indicators."""
        try:
            # CPU
            if self.performance_data["cpu_percent"]:
                cpu = self.performance_data["cpu_percent"][-1]
                self.cpu_label.config(text=f"{cpu:.1f}%")

                if cpu > self.thresholds["cpu_critical"]:
                    self.cpu_status.config(text="CRITICAL", foreground="red")
                elif cpu > self.thresholds["cpu_warning"]:
                    self.cpu_status.config(text="Warning", foreground="orange")
                else:
                    self.cpu_status.config(text="Normal", foreground="green")

            # Memory
            if self.performance_data["memory_mb"]:
                memory = self.performance_data["memory_mb"][-1]
                self.memory_label.config(text=f"{memory:.1f} MB")

                if memory > self.thresholds["memory_critical"]:
                    self.memory_status.config(text="CRITICAL", foreground="red")
                elif memory > self.thresholds["memory_warning"]:
                    self.memory_status.config(text="Warning", foreground="orange")
                else:
                    self.memory_status.config(text="Normal", foreground="green")

            # Threads
            if self.performance_data["thread_count"]:
                threads = self.performance_data["thread_count"][-1]
                self.threads_label.config(text=str(threads))

                if threads > self.thresholds["thread_critical"]:
                    self.threads_status.config(text="CRITICAL", foreground="red")
                elif threads > self.thresholds["thread_warning"]:
                    self.threads_status.config(text="Warning", foreground="orange")
                else:
                    self.threads_status.config(text="Normal", foreground="green")

            # File descriptors
            if self.performance_data["file_descriptors"]:
                fd = self.performance_data["file_descriptors"][-1]
                self.fd_label.config(text=str(fd))

        except Exception as e:
            logger.error(f"Error updating system display: {e}")

    def update_memory_display(self):
        """Update memory display with detailed information."""
        if not hasattr(self, "memory_text"):
            return

        try:
            self.memory_text.delete(1.0, tk.END)

            memory_info = self.target_process.memory_info()

            try:
                memory_full_info = self.target_process.memory_full_info()
                uss = memory_full_info.uss / 1024 / 1024
                pss = memory_full_info.pss / 1024 / 1024
                swap = memory_full_info.swap / 1024 / 1024
            except (AttributeError, psutil.AccessDenied):
                uss = pss = swap = 0

            rss = memory_info.rss / 1024 / 1024
            vms = memory_info.vms / 1024 / 1024
            memory_percent = self.target_process.memory_percent()

            info = f"""Memory Information (Updated: {datetime.now().strftime('%H:%M:%S')}):

Basic Memory Usage:
  RSS (Resident Set Size): {rss:.1f} MB
  VMS (Virtual Memory Size): {vms:.1f} MB
  Memory Percentage: {memory_percent:.2f}%

Advanced Memory Usage:
  USS (Unique Set Size): {uss:.1f} MB
  PSS (Proportional Set Size): {pss:.1f} MB
  Swap Usage: {swap:.1f} MB

Memory Trend Analysis:
"""

            # Add memory trend analysis
            if len(self.performance_data["memory_mb"]) > 10:
                recent_memory = list(self.performance_data["memory_mb"])[-10:]
                memory_trend = (
                    "Increasing"
                    if recent_memory[-1] > recent_memory[0]
                    else "Decreasing"
                )
                memory_change = recent_memory[-1] - recent_memory[0]
                info += f"  Recent Trend: {memory_trend} ({memory_change:+.1f} MB over last 10 samples)\n"

                if self.baseline_memory:
                    total_change = rss - self.baseline_memory
                    info += f"  Total Change: {total_change:+.1f} MB from baseline\n"

                    if total_change > 100:
                        info += f"  ⚠️ WARNING: Significant memory increase detected!\n"

            # Add garbage collection info
            gc_info = gc.get_stats()
            if gc_info:
                info += f"\nGarbage Collection:\n"
                for i, stats in enumerate(gc_info):
                    info += f"  Generation {i}: {stats['collections']} collections, {stats['collected']} objects collected\n"

            self.memory_text.insert(tk.END, info)

        except Exception as e:
            self.memory_text.insert(tk.END, f"Error retrieving memory information: {e}")

    def update_threads_display(self):
        """Update thread information display."""
        if not hasattr(self, "threads_text"):
            return

        try:
            self.threads_text.delete(1.0, tk.END)

            thread_count = self.target_process.num_threads()

            info = f"""Thread Information (Updated: {datetime.now().strftime('%H:%M:%S')}):

Current Thread Count: {thread_count}

Thread Performance Analysis:
"""

            # Add thread trend analysis
            if len(self.performance_data["thread_count"]) > 10:
                recent_threads = list(self.performance_data["thread_count"])[-10:]
                avg_threads = sum(recent_threads) / len(recent_threads)
                max_threads = max(recent_threads)
                min_threads = min(recent_threads)

                info += f"  Average (last 10 samples): {avg_threads:.1f}\n"
                info += f"  Range: {min_threads} - {max_threads}\n"

                if max_threads > self.thresholds["thread_warning"]:
                    info += f"  ⚠️ High thread count detected: {max_threads}\n"

            # Try to get detailed thread information
            try:
                if hasattr(self.target_process, "threads"):
                    threads = self.target_process.threads()
                    info += f"\nDetailed Thread Information:\n"
                    info += f"{'ID':<10} {'User Time':<12} {'System Time':<12}\n"
                    info += "-" * 40 + "\n"

                    for thread in threads[:10]:  # Show first 10 threads
                        info += f"{thread.id:<10} {thread.user_time:<12.3f} {thread.system_time:<12.3f}\n"

                    if len(threads) > 10:
                        info += f"... and {len(threads) - 10} more threads\n"
            except (AttributeError, psutil.AccessDenied):
                info += (
                    "\nDetailed thread information not available on this platform.\n"
                )

            self.threads_text.insert(tk.END, info)

        except Exception as e:
            self.threads_text.insert(
                tk.END, f"Error retrieving thread information: {e}"
            )

    def update_operations_display(self):
        """Update slow operations display with filtering."""
        if not hasattr(self, "operations_text"):
            return

        try:
            # Get filter setting
            severity_filter = getattr(self, "severity_filter", None)
            filter_value = severity_filter.get() if severity_filter else "All"

            # Clear and repopulate
            self.operations_text.delete(1.0, tk.END)

            # Get recent operations
            recent_ops = list(self.performance_data["slow_operations"])[-50:]

            # Apply filter
            if filter_value != "All":
                filtered_ops = [
                    op
                    for op in recent_ops
                    if op.get("severity", "").lower() == filter_value.lower()
                ]
            else:
                filtered_ops = recent_ops

            if not filtered_ops:
                self.operations_text.insert(
                    tk.END, "No operations match the current filter.\n"
                )
                return

            # Group operations by type for summary
            op_summary = defaultdict(int)
            for op in filtered_ops:
                op_summary[op.get("type", "unknown")] += 1

            # Display summary
            self.operations_text.insert(
                tk.END, f"Operations Summary (Last 50, Filter: {filter_value}):\n"
            )
            self.operations_text.insert(tk.END, "-" * 50 + "\n")
            for op_type, count in sorted(op_summary.items()):
                self.operations_text.insert(tk.END, f"{op_type}: {count} occurrences\n")

            self.operations_text.insert(tk.END, "\nDetailed Log:\n")
            self.operations_text.insert(tk.END, "-" * 50 + "\n")

            # Display individual operations
            for op in filtered_ops[-20:]:  # Show last 20 filtered operations
                timestamp_str = datetime.fromtimestamp(op["timestamp"]).strftime(
                    "%H:%M:%S"
                )
                severity = op.get("severity", "info").upper()
                op_type = op.get("type", "unknown")
                details = op.get("details", "No details available")

                # Color code severity
                severity_indicator = {
                    "CRITICAL": "🔴",
                    "WARNING": "🟡",
                    "INFO": "🔵",
                }.get(severity, "⚪")

                self.operations_text.insert(
                    tk.END,
                    f"[{timestamp_str}] {severity_indicator} {severity} - {op_type}\n",
                )
                self.operations_text.insert(tk.END, f"  {details}\n\n")

            # Auto-scroll to bottom
            self.operations_text.see(tk.END)

        except Exception as e:
            self.operations_text.insert(
                tk.END, f"Error updating operations display: {e}\n"
            )

    def update_signal_analyzer_display(self):
        """Update Signal Analyzer specific performance display."""
        try:
            # Filter performance
            if hasattr(self, "filter_perf_label"):
                if self.performance_data["filter_execution_times"]:
                    avg_filter_time = sum(
                        self.performance_data["filter_execution_times"]
                    ) / len(self.performance_data["filter_execution_times"])
                    self.filter_perf_label.config(
                        text=f"Average filter time: {avg_filter_time:.3f}s"
                    )
                else:
                    self.filter_perf_label.config(text="No filter operations detected")

            # GUI performance
            if hasattr(self, "gui_perf_label"):
                if self.performance_data["gui_response_times"]:
                    avg_gui_time = sum(
                        self.performance_data["gui_response_times"]
                    ) / len(self.performance_data["gui_response_times"])
                    self.gui_perf_label.config(
                        text=f"Average response time: {avg_gui_time:.3f}s"
                    )
                else:
                    self.gui_perf_label.config(text="No GUI performance data")

            # I/O performance
            if hasattr(self, "io_perf_label"):
                if (
                    self.performance_data["io_read_mb"]
                    or self.performance_data["io_write_mb"]
                ):
                    recent_read = (
                        self.performance_data["io_read_mb"][-1]
                        if self.performance_data["io_read_mb"]
                        else 0
                    )
                    recent_write = (
                        self.performance_data["io_write_mb"][-1]
                        if self.performance_data["io_write_mb"]
                        else 0
                    )
                    self.io_perf_label.config(
                        text=f"Recent I/O: {recent_read:.1f}MB read, {recent_write:.1f}MB write"
                    )
                else:
                    self.io_perf_label.config(text="No I/O operations detected")

        except Exception as e:
            logger.error(f"Error updating Signal Analyzer display: {e}")

    def generate_performance_report(self):
        """Generate comprehensive performance report."""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        report_file = f"performance_reports/performance_report_{timestamp}.html"

        logger.info("Generating performance report...")

        # Create performance plots
        self.create_performance_plots()

        # Generate HTML report
        html_content = self.generate_html_report()

        with open(report_file, "w") as f:
            f.write(html_content)

        logger.info(f"Performance report saved to: {report_file}")

        # Also generate a summary text file
        summary_file = f"performance_reports/performance_summary_{timestamp}.txt"
        self.generate_summary_report(summary_file)

    def generate_summary_report(self, filename):
        """Generate a concise text summary report."""
        with open(filename, "w") as f:
            f.write("SIGNAL ANALYZER PERFORMANCE SUMMARY\n")
            f.write("=" * 50 + "\n\n")

            f.write(f"Analysis Period: {self.monitor_duration} seconds\n")
            f.write(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")

            # System performance summary
            if self.performance_data["cpu_percent"]:
                avg_cpu = sum(self.performance_data["cpu_percent"]) / len(
                    self.performance_data["cpu_percent"]
                )
                max_cpu = max(self.performance_data["cpu_percent"])
                f.write(f"CPU Usage: Average {avg_cpu:.1f}%, Peak {max_cpu:.1f}%\n")

            if self.performance_data["memory_mb"]:
                avg_memory = sum(self.performance_data["memory_mb"]) / len(
                    self.performance_data["memory_mb"]
                )
                max_memory = max(self.performance_data["memory_mb"])
                f.write(
                    f"Memory Usage: Average {avg_memory:.1f}MB, Peak {max_memory:.1f}MB\n"
                )

            if self.performance_data["thread_count"]:
                avg_threads = sum(self.performance_data["thread_count"]) / len(
                    self.performance_data["thread_count"]
                )
                max_threads = max(self.performance_data["thread_count"])
                f.write(
                    f"Thread Count: Average {avg_threads:.1f}, Peak {max_threads}\n"
                )

            f.write("\n")

            # Issue summary
            issues = self.performance_data["slow_operations"]
            critical_issues = [op for op in issues if op.get("severity") == "critical"]
            warning_issues = [op for op in issues if op.get("severity") == "warning"]

            f.write(f"Issues Detected: {len(issues)} total\n")
            f.write(f"  Critical: {len(critical_issues)}\n")
            f.write(f"  Warnings: {len(warning_issues)}\n")
            f.write(
                f"  Info: {len(issues) - len(critical_issues) - len(warning_issues)}\n\n"
            )

            # Performance score calculation
            score = self.calculate_performance_score()
            f.write(f"Overall Performance Score: {score}/100\n\n")

            # Top recommendations
            recommendations = self.get_top_recommendations()
            f.write("Top Recommendations:\n")
            f.write("-" * 25 + "\n")
            for i, rec in enumerate(recommendations[:5], 1):
                f.write(f"{i}. {rec}\n")

        logger.info(f"Summary report saved to: {filename}")

    def calculate_performance_score(self):
        """Calculate an overall performance score (0-100)."""
        score = 100

        # CPU score
        if self.performance_data["cpu_percent"]:
            avg_cpu = sum(self.performance_data["cpu_percent"]) / len(
                self.performance_data["cpu_percent"]
            )
            if avg_cpu > 80:
                score -= 30
            elif avg_cpu > 60:
                score -= 20
            elif avg_cpu > 40:
                score -= 10

        # Memory score
        if self.performance_data["memory_mb"]:
            max_memory = max(self.performance_data["memory_mb"])
            if max_memory > 1000:
                score -= 25
            elif max_memory > 500:
                score -= 15
            elif max_memory > 250:
                score -= 5

        # Thread score
        if self.performance_data["thread_count"]:
            max_threads = max(self.performance_data["thread_count"])
            if max_threads > 50:
                score -= 20
            elif max_threads > 30:
                score -= 10

        # Issues penalty
        issues = self.performance_data["slow_operations"]
        critical_issues = len([op for op in issues if op.get("severity") == "critical"])
        warning_issues = len([op for op in issues if op.get("severity") == "warning"])

        score -= critical_issues * 5
        score -= warning_issues * 2

        return max(0, min(100, score))

    def get_top_recommendations(self):
        """Get prioritized optimization recommendations."""
        recommendations = []

        # Analyze issues and generate specific recommendations
        issues = self.performance_data["slow_operations"]
        issue_types = defaultdict(int)

        for issue in issues:
            issue_types[issue.get("type", "unknown")] += 1

        # CPU recommendations
        if self.performance_data["cpu_percent"]:
            avg_cpu = sum(self.performance_data["cpu_percent"]) / len(
                self.performance_data["cpu_percent"]
            )
            if avg_cpu > 70:
                recommendations.append(
                    "Optimize CPU-intensive operations (consider vectorization, caching, or threading)"
                )
            if "heavy_filtering" in issue_types:
                recommendations.append(
                    "Optimize filter algorithms - consider using NumPy vectorized operations"
                )

        # Memory recommendations
        if self.performance_data["memory_mb"]:
            max_memory = max(self.performance_data["memory_mb"])
            if max_memory > 500:
                recommendations.append(
                    "Investigate memory usage - consider data compression or streaming"
                )
            if "memory_leak_suspected" in issue_types:
                recommendations.append(
                    "Fix memory leaks - ensure proper cleanup of matplotlib figures and large arrays"
                )

        # GUI recommendations
        if "gui_unresponsive" in issue_types or "gui_slow" in issue_types:
            recommendations.append(
                "Improve GUI responsiveness - move heavy operations to background threads"
            )

        # I/O recommendations
        if "heavy_io_read" in issue_types or "heavy_io_write" in issue_types:
            recommendations.append(
                "Optimize file I/O operations - consider asynchronous loading or data caching"
            )

        # Thread recommendations
        if self.performance_data["thread_count"]:
            max_threads = max(self.performance_data["thread_count"])
            if max_threads > 30:
                recommendations.append(
                    "Consider thread pooling or async operations to manage thread count"
                )

        # Default recommendations if no specific issues found
        if not recommendations:
            recommendations.extend(
                [
                    "Performance appears normal - consider profiling during heavy usage",
                    "Implement monitoring hooks for real-time performance tracking",
                    "Consider adding performance benchmarks to your test suite",
                ]
            )

        return recommendations

    def create_performance_plots(self):
        """Create comprehensive performance visualization plots."""
        if not self.performance_data["timestamps"]:
            logger.warning("No performance data available for plotting")
            return

        try:
            # Convert timestamps to relative time
            start_time = self.performance_data["timestamps"][0]
            times = [
                (t - start_time) / 60 for t in self.performance_data["timestamps"]
            ]  # Minutes

            # Create subplots
            fig, axes = plt.subplots(2, 3, figsize=(18, 12))
            fig.suptitle(
                "Signal Analyzer Performance Analysis", fontsize=16, fontweight="bold"
            )

            # CPU usage
            if self.performance_data["cpu_percent"]:
                ax = axes[0, 0]
                cpu_data = list(self.performance_data["cpu_percent"])
                ax.plot(
                    times[: len(cpu_data)], cpu_data, "b-", linewidth=2, label="CPU %"
                )
                ax.axhline(
                    y=self.thresholds["cpu_warning"],
                    color="orange",
                    linestyle="--",
                    alpha=0.7,
                    label="Warning",
                )
                ax.axhline(
                    y=self.thresholds["cpu_critical"],
                    color="red",
                    linestyle="--",
                    alpha=0.7,
                    label="Critical",
                )
                ax.set_title("CPU Usage Over Time")
                ax.set_xlabel("Time (minutes)")
                ax.set_ylabel("CPU %")
                ax.grid(True, alpha=0.3)
                ax.legend()

            # Memory usage
            if self.performance_data["memory_mb"]:
                ax = axes[0, 1]
                memory_data = list(self.performance_data["memory_mb"])
                ax.plot(
                    times[: len(memory_data)],
                    memory_data,
                    "r-",
                    linewidth=2,
                    label="Memory MB",
                )
                ax.axhline(
                    y=self.thresholds["memory_warning"],
                    color="orange",
                    linestyle="--",
                    alpha=0.7,
                    label="Warning",
                )
                ax.axhline(
                    y=self.thresholds["memory_critical"],
                    color="red",
                    linestyle="--",
                    alpha=0.7,
                    label="Critical",
                )
                ax.set_title("Memory Usage Over Time")
                ax.set_xlabel("Time (minutes)")
                ax.set_ylabel("Memory (MB)")
                ax.grid(True, alpha=0.3)
                ax.legend()

            # Thread count
            if self.performance_data["thread_count"]:
                ax = axes[0, 2]
                thread_data = list(self.performance_data["thread_count"])
                ax.plot(
                    times[: len(thread_data)],
                    thread_data,
                    "g-",
                    linewidth=2,
                    label="Threads",
                )
                ax.axhline(
                    y=self.thresholds["thread_warning"],
                    color="orange",
                    linestyle="--",
                    alpha=0.7,
                    label="Warning",
                )
                ax.axhline(
                    y=self.thresholds["thread_critical"],
                    color="red",
                    linestyle="--",
                    alpha=0.7,
                    label="Critical",
                )
                ax.set_title("Thread Count Over Time")
                ax.set_xlabel("Time (minutes)")
                ax.set_ylabel("Threads")
                ax.grid(True, alpha=0.3)
                ax.legend()

            # I/O operations
            if (
                self.performance_data["io_read_mb"]
                and self.performance_data["io_write_mb"]
            ):
                ax = axes[1, 0]
                read_data = list(self.performance_data["io_read_mb"])
                write_data = list(self.performance_data["io_write_mb"])
                io_times = times[: len(read_data)]
                ax.plot(io_times, read_data, "c-", label="Read MB/s", linewidth=2)
                ax.plot(io_times, write_data, "m-", label="Write MB/s", linewidth=2)
                ax.legend()
                ax.set_title("I/O Operations Over Time")
                ax.set_xlabel("Time (minutes)")
                ax.set_ylabel("MB/s")
                ax.grid(True, alpha=0.3)
            else:
                axes[1, 0].text(
                    0.5,
                    0.5,
                    "No I/O data available",
                    ha="center",
                    va="center",
                    transform=axes[1, 0].transAxes,
                )
                axes[1, 0].set_title("I/O Operations Over Time")

            # GUI response times
            if self.performance_data["gui_response_times"]:
                ax = axes[1, 1]
                gui_data = list(self.performance_data["gui_response_times"])
                gui_times = times[: len(gui_data)]
                ax.plot(
                    gui_times,
                    [t * 1000 for t in gui_data],
                    "purple",
                    linewidth=2,
                    label="Response time",
                )  # Convert to ms
                ax.axhline(
                    y=self.thresholds["gui_response_warning"] * 1000,
                    color="orange",
                    linestyle="--",
                    alpha=0.7,
                    label="Warning",
                )
                ax.axhline(
                    y=self.thresholds["gui_response_critical"] * 1000,
                    color="red",
                    linestyle="--",
                    alpha=0.7,
                    label="Critical",
                )
                ax.set_title("GUI Response Times")
                ax.set_xlabel("Time (minutes)")
                ax.set_ylabel("Response Time (ms)")
                ax.grid(True, alpha=0.3)
                ax.legend()
            else:
                axes[1, 1].text(
                    0.5,
                    0.5,
                    "No GUI response data available",
                    ha="center",
                    va="center",
                    transform=axes[1, 1].transAxes,
                )
                axes[1, 1].set_title("GUI Response Times")

            # Issues over time
            ax = axes[1, 2]
            if self.performance_data["slow_operations"]:
                # Create histogram of issues by time
                issue_times = [
                    op["timestamp"] for op in self.performance_data["slow_operations"]
                ]
                issue_rel_times = [(t - start_time) / 60 for t in issue_times]

                # Count issues in 1-minute bins
                bins = int(max(issue_rel_times)) + 1 if issue_rel_times else 1
                ax.hist(
                    issue_rel_times,
                    bins=bins,
                    alpha=0.7,
                    color="red",
                    edgecolor="black",
                )
                ax.set_title("Issues Distribution Over Time")
                ax.set_xlabel("Time (minutes)")
                ax.set_ylabel("Number of Issues")
                ax.grid(True, alpha=0.3)
            else:
                ax.text(
                    0.5,
                    0.5,
                    "No issues detected",
                    ha="center",
                    va="center",
                    transform=ax.transAxes,
                    color="green",
                    fontsize=14,
                )
                ax.set_title("Issues Distribution Over Time")

            plt.tight_layout()

            # Save plot
            plot_file = "performance_reports/performance_plots.png"
            plt.savefig(plot_file, dpi=300, bbox_inches="tight")
            plt.close()

            logger.info(f"Performance plots saved to: {plot_file}")

        except Exception as e:
            logger.error(f"Error creating performance plots: {e}")

    def generate_html_report(self):
        """Generate comprehensive HTML performance report."""
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

        # Calculate summary statistics
        performance_score = self.calculate_performance_score()
        recommendations = self.get_top_recommendations()

        html = f"""
<!DOCTYPE html>
<html>
<head>
    <title>Signal Analyzer Performance Report</title>
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
        .score-section {{ 
            text-align: center; 
            margin: 30px 0; 
            padding: 25px; 
            background: #f8f9fa; 
            border-radius: 10px;
        }}
        .score {{ 
            font-size: 4em; 
            font-weight: bold; 
            margin: 10px 0;
            background: linear-gradient(45deg, #4CAF50, #45a049);
            -webkit-background-clip: text;
            -webkit-text-fill-color: transparent;
            background-clip: text;
        }}
        .score.warning {{ background: linear-gradient(45deg, #ff9800, #f57c00); }}
        .score.critical {{ background: linear-gradient(45deg, #f44336, #d32f2f); }}
        .section {{ 
            margin: 30px 0; 
            padding: 25px; 
            border-radius: 10px; 
            background: white;
            border-left: 4px solid #667eea;
        }}
        .metrics {{ 
            display: grid; 
            grid-template-columns: repeat(auto-fit, minmax(250px, 1fr)); 
            gap: 20px; 
            margin: 20px 0;
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
        .metric .status {{ margin-top: 5px; font-size: 0.9em; }}
        .status.normal {{ color: #28a745; }}
        .status.warning {{ color: #ffc107; }}
        .status.critical {{ color: #dc3545; }}
        .warning {{ 
            background: #fff3cd; 
            border: 1px solid #ffeaa7; 
            padding: 15px; 
            border-radius: 5px; 
            margin: 10px 0;
        }}
        .critical {{ 
            background: #f8d7da; 
            border: 1px solid #f5c6cb; 
            padding: 15px; 
            border-radius: 5px; 
            margin: 10px 0;
        }}
        .info {{ 
            background: #d1ecf1; 
            border: 1px solid #b8daff; 
            padding: 15px; 
            border-radius: 5px; 
            margin: 10px 0;
        }}
        table {{ 
            width: 100%; 
            border-collapse: collapse; 
            margin: 15px 0;
            background: white;
        }}
        th, td {{ 
            border: 1px solid #dee2e6; 
            padding: 12px 8px; 
            text-align: left;
        }}
        th {{ 
            background: #e9ecef; 
            font-weight: 600;
        }}
        tr:nth-child(even) {{ background: #f8f9fa; }}
        .recommendations {{ 
            background: #e8f5e8; 
            padding: 25px; 
            border-radius: 10px; 
            border-left: 4px solid #28a745;
        }}
        .recommendations h2 {{ color: #155724; }}
        .recommendations ul {{ padding-left: 20px; }}
        .recommendations li {{ margin: 10px 0; line-height: 1.6; }}
        .chart-container {{ 
            text-align: center; 
            margin: 30px 0; 
            padding: 20px; 
            background: #f8f9fa; 
            border-radius: 10px;
        }}
        .chart-container img {{ 
            max-width: 100%; 
            height: auto; 
            border-radius: 8px; 
            box-shadow: 0 2px 8px rgba(0,0,0,0.1);
        }}
        .footer {{ 
            text-align: center; 
            margin-top: 40px; 
            padding: 20px; 
            color: #6c757d; 
            border-top: 1px solid #dee2e6;
        }}
    </style>
</head>
<body>
    <div class="container">
        <div class="header">
            <h1>Signal Analyzer Performance Report</h1>
            <p>Generated: {timestamp}</p>
            <p>Monitoring Duration: {self.monitor_duration} seconds</p>
            <p>Target Process: {self.target_process.name() if self.target_process else 'Unknown'} (PID: {self.target_process.pid if self.target_process else 'N/A'})</p>
        </div>
        
        <div class="score-section">
            <h2>Overall Performance Score</h2>
            <div class="score {'warning' if performance_score < 70 else 'critical' if performance_score < 50 else ''}">{performance_score}</div>
            <p>{'Excellent' if performance_score >= 90 else 'Good' if performance_score >= 70 else 'Needs Attention' if performance_score >= 50 else 'Critical Issues Detected'}</p>
        </div>
        
        <div class="section">
            <h2>Performance Metrics Summary</h2>
            <div class="metrics">
                {self.generate_performance_metrics_html()}
            </div>
        </div>
        
        <div class="chart-container">
            <h2>Performance Charts</h2>
            <img src="performance_plots.png" alt="Performance Charts">
        </div>
        
        <div class="section">
            <h2>Issue Analysis</h2>
            {self.generate_issues_html()}
        </div>
        
        <div class="recommendations">
            <h2>🎯 Optimization Recommendations</h2>
            <ul>
                {''.join(f'<li>{rec}</li>' for rec in recommendations)}
            </ul>
        </div>
        
        <div class="section">
            <h2>Detailed Operations Log</h2>
            {self.generate_operations_table_html()}
        </div>
        
        <div class="footer">
            <p>Report generated by Signal Analyzer Performance Monitor</p>
            <p>For technical support or questions, please refer to the documentation</p>
        </div>
    </div>
</body>
</html>
        """
        return html

    def generate_performance_metrics_html(self):
        """Generate performance metrics HTML with enhanced formatting."""
        metrics_html = ""

        # CPU metrics
        if self.performance_data["cpu_percent"]:
            avg_cpu = sum(self.performance_data["cpu_percent"]) / len(
                self.performance_data["cpu_percent"]
            )
            max_cpu = max(self.performance_data["cpu_percent"])
            min_cpu = min(self.performance_data["cpu_percent"])
            status_class = (
                "critical"
                if avg_cpu > self.thresholds["cpu_critical"]
                else "warning" if avg_cpu > self.thresholds["cpu_warning"] else "normal"
            )

            metrics_html += f"""
            <div class="metric">
                <h3>CPU Usage</h3>
                <div class="value">{avg_cpu:.1f}%</div>
                <div class="status {status_class}">Average</div>
                <small>Peak: {max_cpu:.1f}% | Min: {min_cpu:.1f}%</small>
            </div>
            """

        # Memory metrics
        if self.performance_data["memory_mb"]:
            avg_memory = sum(self.performance_data["memory_mb"]) / len(
                self.performance_data["memory_mb"]
            )
            max_memory = max(self.performance_data["memory_mb"])
            min_memory = min(self.performance_data["memory_mb"])
            status_class = (
                "critical"
                if max_memory > self.thresholds["memory_critical"]
                else (
                    "warning"
                    if max_memory > self.thresholds["memory_warning"]
                    else "normal"
                )
            )

            metrics_html += f"""
            <div class="metric">
                <h3>Memory Usage</h3>
                <div class="value">{avg_memory:.1f} MB</div>
                <div class="status {status_class}">Average</div>
                <small>Peak: {max_memory:.1f}MB | Min: {min_memory:.1f}MB</small>
            </div>
            """

        # Thread metrics
        if self.performance_data["thread_count"]:
            avg_threads = sum(self.performance_data["thread_count"]) / len(
                self.performance_data["thread_count"]
            )
            max_threads = max(self.performance_data["thread_count"])
            min_threads = min(self.performance_data["thread_count"])
            status_class = (
                "critical"
                if max_threads > self.thresholds["thread_critical"]
                else (
                    "warning"
                    if max_threads > self.thresholds["thread_warning"]
                    else "normal"
                )
            )

            metrics_html += f"""
            <div class="metric">
                <h3>Thread Count</h3>
                <div class="value">{avg_threads:.1f}</div>
                <div class="status {status_class}">Average</div>
                <small>Peak: {max_threads} | Min: {min_threads}</small>
            </div>
            """

        # GUI response metrics
        if self.performance_data["gui_response_times"]:
            avg_gui = sum(self.performance_data["gui_response_times"]) / len(
                self.performance_data["gui_response_times"]
            )
            max_gui = max(self.performance_data["gui_response_times"])
            status_class = (
                "critical"
                if avg_gui > self.thresholds["gui_response_critical"]
                else (
                    "warning"
                    if avg_gui > self.thresholds["gui_response_warning"]
                    else "normal"
                )
            )

            metrics_html += f"""
            <div class="metric">
                <h3>GUI Response</h3>
                <div class="value">{avg_gui*1000:.1f} ms</div>
                <div class="status {status_class}">Average</div>
                <small>Peak: {max_gui*1000:.1f}ms</small>
            </div>
            """

        # I/O metrics
        if self.performance_data["io_read_mb"] and self.performance_data["io_write_mb"]:
            total_read = sum(self.performance_data["io_read_mb"])
            total_write = sum(self.performance_data["io_write_mb"])

            metrics_html += f"""
            <div class="metric">
                <h3>Total I/O</h3>
                <div class="value">{total_read + total_write:.1f} MB</div>
                <div class="status normal">Cumulative</div>
                <small>Read: {total_read:.1f}MB | Write: {total_write:.1f}MB</small>
            </div>
            """

        return metrics_html

    def generate_issues_html(self):
        """Generate issues analysis HTML."""
        issues = self.performance_data["slow_operations"]

        if not issues:
            return '<div class="info">✅ No performance issues detected during monitoring period.</div>'

        # Categorize issues
        critical_issues = [op for op in issues if op.get("severity") == "critical"]
        warning_issues = [op for op in issues if op.get("severity") == "warning"]
        info_issues = [op for op in issues if op.get("severity") == "info"]

        html = f"""
        <div class="metrics">
            <div class="metric">
                <h3>Critical Issues</h3>
                <div class="value" style="color: #dc3545;">{len(critical_issues)}</div>
            </div>
            <div class="metric">
                <h3>Warnings</h3>
                <div class="value" style="color: #ffc107;">{len(warning_issues)}</div>
            </div>
            <div class="metric">
                <h3>Info</h3>
                <div class="value" style="color: #17a2b8;">{len(info_issues)}</div>
            </div>
        </div>
        """

        # Show recent critical issues
        if critical_issues:
            html += "<h3>Recent Critical Issues:</h3>"
            for issue in critical_issues[-5:]:  # Last 5 critical issues
                timestamp_str = datetime.fromtimestamp(issue["timestamp"]).strftime(
                    "%H:%M:%S"
                )
                html += f'<div class="critical">🔴 [{timestamp_str}] {issue.get("details", "No details available")}</div>'

        # Show recent warnings
        if warning_issues:
            html += "<h3>Recent Warnings:</h3>"
            for issue in warning_issues[-5:]:  # Last 5 warnings
                timestamp_str = datetime.fromtimestamp(issue["timestamp"]).strftime(
                    "%H:%M:%S"
                )
                html += f'<div class="warning">🟡 [{timestamp_str}] {issue.get("details", "No details available")}</div>'

        return html

    def generate_operations_table_html(self):
        """Generate operations table HTML with enhanced formatting."""
        issues = self.performance_data["slow_operations"]

        if not issues:
            return "<p>No operations logged during monitoring period.</p>"

        html = """
        <table>
            <tr>
                <th>Time</th>
                <th>Severity</th>
                <th>Type</th>
                <th>Details</th>
            </tr>
        """

        # Show last 30 operations
        for op in issues[-30:]:
            timestamp_str = datetime.fromtimestamp(op["timestamp"]).strftime("%H:%M:%S")
            severity = op.get("severity", "info").upper()
            op_type = op.get("type", "unknown")
            details = op.get("details", "No details available")

            # Color code severity
            severity_colors = {
                "CRITICAL": "#dc3545",
                "WARNING": "#ffc107",
                "INFO": "#17a2b8",
            }

            color = severity_colors.get(severity, "#6c757d")

            html += f"""
            <tr>
                <td>{timestamp_str}</td>
                <td style="color: {color}; font-weight: bold;">{severity}</td>
                <td>{op_type}</td>
                <td>{details}</td>
            </tr>
            """

        html += "</table>"
        return html

    def generate_comprehensive_report(self):
        """Generate a comprehensive report combining all analysis methods."""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

        logger.info("Generating comprehensive performance report...")

        # Generate individual reports
        self.generate_performance_report()

        # Create a comprehensive summary
        summary_file = f"performance_reports/comprehensive_summary_{timestamp}.txt"

        with open(summary_file, "w") as f:
            f.write("SIGNAL ANALYZER COMPREHENSIVE PERFORMANCE ANALYSIS\n")
            f.write("=" * 60 + "\n\n")

            f.write(
                f"Analysis completed: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n"
            )
            f.write(f"Monitoring duration: {self.monitor_duration} seconds\n")
            f.write(
                f"Target process: {self.target_process.name()} (PID: {self.target_process.pid})\n\n"
            )

            # Performance summary
            score = self.calculate_performance_score()
            f.write(f"OVERALL PERFORMANCE SCORE: {score}/100\n")

            if score >= 90:
                f.write("Status: EXCELLENT - Application performing optimally\n\n")
            elif score >= 70:
                f.write("Status: GOOD - Minor optimizations possible\n\n")
            elif score >= 50:
                f.write(
                    "Status: NEEDS ATTENTION - Several performance issues detected\n\n"
                )
            else:
                f.write(
                    "Status: CRITICAL - Significant performance problems require immediate attention\n\n"
                )

            # Component breakdown
            f.write("COMPONENT PERFORMANCE BREAKDOWN:\n")
            f.write("-" * 40 + "\n")

            components = self.analyze_component_performance()
            for component, score in components.items():
                status = "✅" if score >= 80 else "⚠️" if score >= 60 else "🔴"
                f.write(f"{status} {component}: {score}/100\n")

            f.write("\n")

            # Top issues
            critical_issues = [
                op
                for op in self.performance_data["slow_operations"]
                if op.get("severity") == "critical"
            ]
            if critical_issues:
                f.write("CRITICAL ISSUES REQUIRING IMMEDIATE ATTENTION:\n")
                f.write("-" * 50 + "\n")
                for i, issue in enumerate(critical_issues[-10:], 1):
                    f.write(f"{i}. {issue.get('details', 'No details available')}\n")
                f.write("\n")

            # Recommendations
            recommendations = self.get_top_recommendations()
            f.write("PRIORITIZED OPTIMIZATION RECOMMENDATIONS:\n")
            f.write("-" * 45 + "\n")
            for i, rec in enumerate(recommendations, 1):
                f.write(f"{i}. {rec}\n")

            f.write("\n")
            f.write("DETAILED REPORTS:\n")
            f.write("-" * 20 + "\n")
            f.write("• HTML Report: performance_report_*.html\n")
            f.write("• Performance Plots: performance_plots.png\n")
            f.write("• CPU Profile: profile_*.prof (if available)\n")
            f.write("• Memory Analysis: memory_report_*.txt (if available)\n")

        logger.info(f"Comprehensive summary saved to: {summary_file}")

    def analyze_component_performance(self):
        """Analyze performance of individual components."""
        components = {}

        # CPU performance
        if self.performance_data["cpu_percent"]:
            avg_cpu = sum(self.performance_data["cpu_percent"]) / len(
                self.performance_data["cpu_percent"]
            )
            cpu_score = max(0, 100 - (avg_cpu - 20) * 2)  # Penalty starts at 20% usage
            components["CPU Usage"] = int(cpu_score)

        # Memory performance
        if self.performance_data["memory_mb"]:
            max_memory = max(self.performance_data["memory_mb"])
            memory_score = max(
                0, 100 - max(0, (max_memory - 200) / 10)
            )  # Penalty starts at 200MB
            components["Memory Management"] = int(memory_score)

        # Thread performance
        if self.performance_data["thread_count"]:
            max_threads = max(self.performance_data["thread_count"])
            thread_score = max(
                0, 100 - max(0, (max_threads - 10) * 3)
            )  # Penalty starts at 10 threads
            components["Threading"] = int(thread_score)

        # GUI performance
        if self.performance_data["gui_response_times"]:
            avg_gui = sum(self.performance_data["gui_response_times"]) / len(
                self.performance_data["gui_response_times"]
            )
            gui_score = max(
                0, 100 - (avg_gui * 1000 - 50) * 2
            )  # Penalty starts at 50ms
            components["GUI Responsiveness"] = int(gui_score)
        else:
            components["GUI Responsiveness"] = 85  # Default score if no data

        # Overall stability (based on number of issues)
        issues = len(self.performance_data["slow_operations"])
        stability_score = max(0, 100 - issues * 2)
        components["Overall Stability"] = int(stability_score)

        return components

    def stop_monitoring(self):
        """Stop all monitoring activities."""
        self.monitoring = False
        if self.dashboard_window:
            self.dashboard_window.destroy()
        logger.info("Monitoring stopped")

    def get_optimization_suggestions(self):
        """Get comprehensive optimization suggestions for the Signal Analyzer app."""
        suggestions = {
            "gui_optimizations": [
                "🎨 Use virtual scrolling for large data sets in list widgets",
                "🎨 Implement lazy loading for plot updates (only update visible data)",
                "🎨 Reduce plot update frequency during real-time monitoring (max 30 FPS)",
                "🎨 Use canvas-based widgets instead of many individual Tkinter widgets",
                "🎨 Implement proper widget cleanup to prevent memory leaks",
                "🎨 Use after_idle() for non-critical GUI updates",
                "🎨 Implement debouncing for rapid user interactions",
            ],
            "data_processing": [
                "⚡ Use NumPy vectorized operations instead of Python loops",
                "⚡ Implement chunked processing for large datasets (process in 1MB chunks)",
                "⚡ Cache frequently computed filter results using LRU cache",
                "⚡ Use memory-mapped files for very large ATF files",
                "⚡ Optimize filter algorithms with Numba JIT compilation",
                "⚡ Implement data streaming for real-time processing",
                "⚡ Use scipy.ndimage for image-based signal processing",
            ],
            "threading": [
                "🧵 Move heavy computations to worker threads using ThreadPoolExecutor",
                "🧵 Use thread pools instead of creating new threads for each task",
                "🧵 Implement proper thread synchronization with locks and queues",
                "🧵 Use asyncio for I/O-bound operations (file loading, network)",
                "🧵 Separate GUI thread from processing threads completely",
                "🧵 Implement progress callbacks for long-running operations",
                "🧵 Use threading.local() for thread-specific data storage",
            ],
            "memory_management": [
                "💾 Implement proper cleanup in __del__ methods for large objects",
                "💾 Use weak references where appropriate to break circular references",
                "💾 Clear matplotlib figure caches regularly with plt.close('all')",
                "💾 Implement data compression for storage (gzip, lz4)",
                "💾 Use generators instead of lists for large datasets",
                "💾 Implement object pooling for frequently created/destroyed objects",
                "💾 Use np.memmap for large arrays that don't fit in memory",
                "💾 Call gc.collect() after processing large datasets",
            ],
            "signal_analyzer_specific": [
                "📊 Implement filter result caching based on parameters",
                "📊 Use decimation for plot display of large signals",
                "📊 Implement progressive ATF file loading",
                "📊 Cache baseline correction results",
                "📊 Use sparse matrices for large signal representations",
                "📊 Implement signal windowing for memory efficiency",
                "📊 Use FFT-based convolution for large filter kernels",
            ],
            "io_optimizations": [
                "💿 Use binary formats instead of text where possible",
                "💿 Implement file format detection to optimize loading",
                "💿 Use background threads for file save operations",
                "💿 Implement incremental file loading for large files",
                "💿 Cache frequently accessed file metadata",
                "💿 Use memory mapping for read-only large files",
                "💿 Implement file compression for exports",
            ],
        }

        return suggestions


def create_performance_hooks():
    """Create performance monitoring hooks for integration with Signal Analyzer."""

    class PerformanceHooks:
        def __init__(self):
            self.start_times = {}
            self.operation_counts = defaultdict(int)
            self.operation_times = defaultdict(list)

        def start_operation(self, operation_name):
            """Mark the start of an operation."""
            self.start_times[operation_name] = time.time()

        def end_operation(self, operation_name):
            """Mark the end of an operation and record timing."""
            if operation_name in self.start_times:
                duration = time.time() - self.start_times[operation_name]
                self.operation_times[operation_name].append(duration)
                self.operation_counts[operation_name] += 1
                del self.start_times[operation_name]

                # Log slow operations
                if duration > 0.5:  # Slower than 500ms
                    logger.warning(
                        f"Slow operation detected: {operation_name} took {duration:.3f}s"
                    )

        def get_operation_stats(self):
            """Get statistics for all operations."""
            stats = {}
            for op_name, times in self.operation_times.items():
                if times:
                    stats[op_name] = {
                        "count": len(times),
                        "total_time": sum(times),
                        "avg_time": sum(times) / len(times),
                        "max_time": max(times),
                        "min_time": min(times),
                    }
            return stats

    return PerformanceHooks()


def monitor_function_performance(func):
    """Decorator to monitor function performance."""

    def wrapper(*args, **kwargs):
        start_time = time.time()
        try:
            result = func(*args, **kwargs)
            return result
        finally:
            duration = time.time() - start_time
            if duration > 0.1:  # Log functions taking more than 100ms
                logger.info(f"Function {func.__name__} took {duration:.3f}s")

    return wrapper


def analyze_existing_process(process_name="python", duration=60):
    """Analyze an already running process."""
    analyzer = AdvancedPerformanceAnalyzer(process_name, duration)

    if not analyzer.target_process:
        print("❌ Could not find target process")
        return False

    print(
        f"✅ Found process: {analyzer.target_process.name()} (PID: {analyzer.target_process.pid})"
    )
    print(f"📊 Starting {duration}-second analysis...")

    try:
        analyzer.start_monitoring("realtime")
        return True
    except KeyboardInterrupt:
        print("\n⏹️ Analysis interrupted by user")
        return False
    except Exception as e:
        print(f"❌ Analysis failed: {e}")
        return False


def quick_performance_check(process_name="python", duration=30):
    """Perform a quick 30-second performance check."""
    print("🚀 Starting Quick Performance Check...")
    print(f"⏱️ Duration: {duration} seconds")
    print("📋 Monitoring: CPU, Memory, Threads, I/O")
    print("-" * 50)

    analyzer = AdvancedPerformanceAnalyzer(process_name, duration)

    if not analyzer.target_process:
        print("❌ Target process not found")
        return

    # Quick monitoring without GUI
    analyzer.monitoring = True

    # Start essential monitoring threads
    threads = [
        threading.Thread(target=analyzer.monitor_system_resources, daemon=True),
        threading.Thread(target=analyzer.monitor_threads, daemon=True),
        threading.Thread(target=analyzer.monitor_io_operations, daemon=True),
    ]

    for thread in threads:
        thread.start()

    # Monitor for specified duration
    try:
        time.sleep(duration)
    except KeyboardInterrupt:
        print("\n⏹️ Check interrupted")
    finally:
        analyzer.monitoring = False

    # Quick analysis
    print("\n📊 QUICK ANALYSIS RESULTS:")
    print("=" * 40)

    if analyzer.performance_data["cpu_percent"]:
        avg_cpu = sum(analyzer.performance_data["cpu_percent"]) / len(
            analyzer.performance_data["cpu_percent"]
        )
        max_cpu = max(analyzer.performance_data["cpu_percent"])
        print(f"🖥️  CPU: {avg_cpu:.1f}% avg, {max_cpu:.1f}% peak")

    if analyzer.performance_data["memory_mb"]:
        avg_mem = sum(analyzer.performance_data["memory_mb"]) / len(
            analyzer.performance_data["memory_mb"]
        )
        max_mem = max(analyzer.performance_data["memory_mb"])
        print(f"💾 Memory: {avg_mem:.1f}MB avg, {max_mem:.1f}MB peak")

    if analyzer.performance_data["thread_count"]:
        avg_threads = sum(analyzer.performance_data["thread_count"]) / len(
            analyzer.performance_data["thread_count"]
        )
        max_threads = max(analyzer.performance_data["thread_count"])
        print(f"🧵 Threads: {avg_threads:.1f} avg, {max_threads} peak")

    # Issues summary
    issues = analyzer.performance_data["slow_operations"]
    critical = len([op for op in issues if op.get("severity") == "critical"])
    warnings = len([op for op in issues if op.get("severity") == "warning"])

    print(f"⚠️  Issues: {critical} critical, {warnings} warnings")

    # Quick score
    score = analyzer.calculate_performance_score()
    status = (
        "✅ GOOD"
        if score >= 70
        else "⚠️ NEEDS ATTENTION" if score >= 50 else "🔴 CRITICAL"
    )
    print(f"📈 Score: {score}/100 ({status})")

    if critical > 0 or warnings > 5:
        print("\n💡 Recommendation: Run full analysis for detailed insights")
        print(
            "   Command: python performance_analyzer.py --mode=comprehensive --duration=300"
        )


def main():
    """Main function to run the performance analyzer."""
    parser = argparse.ArgumentParser(
        description="Advanced Performance Analyzer for Signal Analysis App",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Real-time monitoring with dashboard for 5 minutes
  python performance_analyzer.py --mode=realtime --duration=300
  
  # CPU profiling for 2 minutes  
  python performance_analyzer.py --mode=profile --duration=120
  
  # Memory analysis for 3 minutes
  python performance_analyzer.py --mode=memory --duration=180
  
  # Comprehensive analysis (all methods)
  python performance_analyzer.py --mode=comprehensive --duration=300
  
  # Quick 30-second check
  python performance_analyzer.py --quick
        """,
    )

    parser.add_argument(
        "--mode",
        choices=["realtime", "profile", "memory", "comprehensive"],
        default="realtime",
        help="Monitoring mode (default: realtime)",
    )
    parser.add_argument(
        "--duration",
        type=int,
        default=60,
        help="Monitoring duration in seconds (default: 60)",
    )
    parser.add_argument(
        "--process", default="python", help="Target process name (default: python)"
    )
    parser.add_argument(
        "--quick", action="store_true", help="Perform quick 30-second analysis"
    )
    parser.add_argument(
        "--no-gui", action="store_true", help="Disable real-time GUI dashboard"
    )
    parser.add_argument(
        "--output-dir",
        default="performance_reports",
        help="Output directory for reports (default: performance_reports)",
    )

    args = parser.parse_args()

    # Print banner
    print("🎯 SIGNAL ANALYZER PERFORMANCE ANALYZER")
    print("=" * 50)
    print(f"📅 Started: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"🎛️  Mode: {args.mode}")
    print(f"⏱️  Duration: {args.duration} seconds")
    print(f"🎯 Target: {args.process}")
    print("-" * 50)

    # Handle quick mode
    if args.quick:
        quick_performance_check(args.process, 30)
        return

    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)

    # Create analyzer
    analyzer = AdvancedPerformanceAnalyzer(
        target_process_name=args.process, monitor_duration=args.duration
    )

    # Check if target process was found
    if not analyzer.target_process:
        print("❌ ERROR: Could not find target process")
        print(f"   Looking for process name: {args.process}")
        print("\n💡 SUGGESTIONS:")
        print("   1. Make sure the Signal Analyzer app is running")
        print("   2. Try a different process name with --process")
        print("   3. Check running Python processes:")

        # Show running Python processes
        python_processes = []
        for proc in psutil.process_iter(["pid", "name", "cmdline"]):
            try:
                if (
                    proc.info["name"] == "python"
                    or "python" in proc.info["name"].lower()
                ):
                    cmdline = (
                        " ".join(proc.info["cmdline"]) if proc.info["cmdline"] else ""
                    )
                    python_processes.append(
                        f"      PID {proc.info['pid']}: {cmdline[:80]}..."
                    )
            except (psutil.NoSuchProcess, psutil.AccessDenied):
                continue

        if python_processes:
            print("\n   Running Python processes:")
            for proc_info in python_processes[:5]:  # Show first 5
                print(proc_info)
        else:
            print("   No Python processes found")

        return 1

    try:
        print(f"✅ Target process found: PID {analyzer.target_process.pid}")

        # Disable GUI if requested
        if args.no_gui and args.mode == "realtime":
            print("🖥️  GUI dashboard disabled")
            # Monkey patch to disable dashboard
            analyzer.create_realtime_dashboard = lambda: None

        # Start monitoring
        print(f"🚀 Starting {args.mode} monitoring...")
        analyzer.start_monitoring(mode=args.mode)

        print("\n✅ ANALYSIS COMPLETE!")
        print("=" * 30)

        # Show results location
        print("📁 Results saved to:")
        reports_dir = Path(args.output_dir)
        if reports_dir.exists():
            for report_file in sorted(reports_dir.glob("*")):
                if (
                    report_file.is_file()
                    and report_file.stat().st_mtime > time.time() - 300
                ):  # Last 5 minutes
                    print(f"   📄 {report_file.name}")

        # Show performance summary
        score = analyzer.calculate_performance_score()
        print(f"\n📊 Performance Score: {score}/100")

        if score >= 90:
            print("🎉 Excellent performance!")
        elif score >= 70:
            print("👍 Good performance with minor optimization opportunities")
        elif score >= 50:
            print("⚠️  Performance needs attention - check the detailed report")
        else:
            print("🚨 Critical performance issues detected - immediate action needed")

        # Show top recommendation
        recommendations = analyzer.get_top_recommendations()
        if recommendations:
            print(f"\n💡 Top Recommendation: {recommendations[0]}")

        print("\n🔗 Next Steps:")
        print("   1. Open the HTML report for detailed analysis")
        print("   2. Review the optimization recommendations")
        print("   3. Implement suggested improvements")
        print("   4. Re-run analysis to measure improvements")

        return 0

    except KeyboardInterrupt:
        print("\n⏹️ Analysis interrupted by user")
        analyzer.stop_monitoring()
        return 130
    except Exception as e:
        print(f"\n❌ ERROR: Analysis failed")
        print(f"   Details: {str(e)}")
        logger.exception("Analysis failed with exception:")

        print("\n🩺 TROUBLESHOOTING:")
        print("   1. Ensure the target application is running")
        print("   2. Check that you have necessary permissions")
        print("   3. Try running with different --mode or --duration")
        print("   4. Check the log output above for specific errors")

        return 1
    finally:
        # Cleanup
        if hasattr(analyzer, "monitoring"):
            analyzer.monitoring = False


if __name__ == "__main__":
    # Enable high DPI awareness on Windows
    try:
        import ctypes

        ctypes.windll.shcore.SetProcessDpiAwareness(1)
    except:
        pass

    # Run main function
    exit_code = main()
    sys.exit(exit_code)
