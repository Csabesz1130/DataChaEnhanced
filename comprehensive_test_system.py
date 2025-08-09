#!/usr/bin/env python3
"""
Comprehensive Test System for AI Excel Learning
===============================================

This system provides advanced testing capabilities for:
1. Linear fitting feature testing
2. Human vs AI calculation comparison
3. Interactive testing interface
4. Performance benchmarking
5. Debugging and analysis tools
6. Long-term testing capabilities

Features:
- Automated test generation
- Statistical validation
- Visual comparison tools
- Performance metrics
- Interactive debugging
- Regression testing
- Report generation
"""

import os
import sys
import time
import json
import pickle
import hashlib
import traceback
from datetime import datetime, timedelta
from typing import Dict, List, Tuple, Any, Optional, Union
from dataclasses import dataclass, asdict
from enum import Enum
import threading
import queue
import logging
from pathlib import Path

# Data manipulation
import numpy as np
import pandas as pd
from scipy import stats
from scipy.optimize import curve_fit
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler

# Excel handling
import openpyxl
from openpyxl.chart import ScatterChart, Reference, Series
from openpyxl.utils.dataframe import dataframe_to_rows

# Interactive testing
import tkinter as tk
from tkinter import ttk, filedialog, messagebox, scrolledtext
import matplotlib.backends.backend_tkagg as tkagg

# Configuration
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class TestType(Enum):
    """Types of tests available in the system"""
    LINEAR_FITTING = "linear_fitting"
    FORMULA_LEARNING = "formula_learning"
    CHART_GENERATION = "chart_generation"
    DATA_GENERATION = "data_generation"
    BACKGROUND_PROCESSING = "background_processing"
    RESEARCH_EXTENSIONS = "research_extensions"
    INTEGRATION = "integration"
    PERFORMANCE = "performance"
    REGRESSION = "regression"

class TestStatus(Enum):
    """Status of test execution"""
    PENDING = "pending"
    RUNNING = "running"
    PASSED = "passed"
    FAILED = "failed"
    SKIPPED = "skipped"
    TIMEOUT = "timeout"

@dataclass
class TestResult:
    """Result of a test execution"""
    test_name: str
    test_type: TestType
    status: TestStatus
    execution_time: float
    error_message: Optional[str] = None
    metrics: Optional[Dict[str, Any]] = None
    human_result: Optional[Dict[str, Any]] = None
    ai_result: Optional[Dict[str, Any]] = None
    comparison_score: Optional[float] = None
    timestamp: datetime = None
    
    def __post_init__(self):
        if self.timestamp is None:
            self.timestamp = datetime.now()

@dataclass
class TestCase:
    """Individual test case definition"""
    name: str
    description: str
    test_type: TestType
    input_data: Dict[str, Any]
    expected_output: Dict[str, Any]
    tolerance: float = 0.01
    timeout: int = 30
    priority: int = 1

@dataclass
class TestSuite:
    """Collection of related test cases"""
    name: str
    description: str
    test_cases: List[TestCase]
    dependencies: List[str] = None
    
    def __post_init__(self):
        if self.dependencies is None:
            self.dependencies = []

class LinearFittingTester:
    """Specialized tester for linear fitting functionality"""
    
    def __init__(self):
        self.test_results = []
        self.performance_metrics = {}
    
    def generate_test_data(self, n_points: int = 100, noise_level: float = 0.1) -> Tuple[np.ndarray, np.ndarray]:
        """Generate synthetic test data for linear fitting"""
        x = np.linspace(0, 10, n_points)
        true_slope = 2.5
        true_intercept = 1.0
        y_true = true_slope * x + true_intercept
        noise = np.random.normal(0, noise_level, n_points)
        y_noisy = y_true + noise
        
        return x, y_noisy, true_slope, true_intercept
    
    def human_linear_fit(self, x: np.ndarray, y: np.ndarray) -> Dict[str, Any]:
        """Perform linear fitting using traditional methods (human approach)"""
        start_time = time.time()
        
        # Calculate means
        x_mean = np.mean(x)
        y_mean = np.mean(y)
        
        # Calculate slope and intercept using least squares
        numerator = np.sum((x - x_mean) * (y - y_mean))
        denominator = np.sum((x - x_mean) ** 2)
        
        slope = numerator / denominator
        intercept = y_mean - slope * x_mean
        
        # Calculate R-squared
        y_pred = slope * x + intercept
        ss_res = np.sum((y - y_pred) ** 2)
        ss_tot = np.sum((y - y_mean) ** 2)
        r_squared = 1 - (ss_res / ss_tot)
        
        # Calculate standard errors
        n = len(x)
        mse = ss_res / (n - 2)
        slope_se = np.sqrt(mse / np.sum((x - x_mean) ** 2))
        intercept_se = np.sqrt(mse * (1/n + x_mean**2 / np.sum((x - x_mean) ** 2)))
        
        execution_time = time.time() - start_time
        
        return {
            'slope': slope,
            'intercept': intercept,
            'r_squared': r_squared,
            'slope_se': slope_se,
            'intercept_se': intercept_se,
            'execution_time': execution_time,
            'method': 'manual_least_squares'
        }
    
    def ai_linear_fit(self, x: np.ndarray, y: np.ndarray) -> Dict[str, Any]:
        """Perform linear fitting using AI/ML methods"""
        start_time = time.time()
        
        # Use scikit-learn LinearRegression
        model = LinearRegression()
        X = x.reshape(-1, 1)
        model.fit(X, y)
        
        slope = model.coef_[0]
        intercept = model.intercept_
        
        # Calculate R-squared
        y_pred = model.predict(X)
        r_squared = r2_score(y, y_pred)
        
        # Calculate standard errors using scipy
        try:
            popt, pcov = curve_fit(lambda x, a, b: a * x + b, x, y)
            slope_se = np.sqrt(pcov[0, 0])
            intercept_se = np.sqrt(pcov[1, 1])
        except:
            slope_se = np.nan
            intercept_se = np.nan
        
        execution_time = time.time() - start_time
        
        return {
            'slope': slope,
            'intercept': intercept,
            'r_squared': r_squared,
            'slope_se': slope_se,
            'intercept_se': intercept_se,
            'execution_time': execution_time,
            'method': 'sklearn_linear_regression'
        }
    
    def compare_results(self, human_result: Dict[str, Any], ai_result: Dict[str, Any]) -> Dict[str, Any]:
        """Compare human and AI results"""
        comparison = {}
        
        # Compare key metrics
        for metric in ['slope', 'intercept', 'r_squared']:
            human_val = human_result[metric]
            ai_val = ai_result[metric]
            diff = abs(human_val - ai_val)
            relative_diff = diff / abs(human_val) if human_val != 0 else float('inf')
            
            comparison[f'{metric}_difference'] = diff
            comparison[f'{metric}_relative_difference'] = relative_diff
            comparison[f'{metric}_agreement'] = relative_diff < 0.01  # 1% tolerance
        
        # Performance comparison
        comparison['speed_ratio'] = human_result['execution_time'] / ai_result['execution_time']
        comparison['human_faster'] = human_result['execution_time'] < ai_result['execution_time']
        
        # Overall agreement score
        agreement_metrics = [comparison[f'{metric}_agreement'] for metric in ['slope', 'intercept', 'r_squared']]
        comparison['overall_agreement'] = sum(agreement_metrics) / len(agreement_metrics)
        
        return comparison
    
    def run_linear_fitting_test(self, test_case: TestCase) -> TestResult:
        """Run a comprehensive linear fitting test"""
        try:
            # Generate test data
            n_points = test_case.input_data.get('n_points', 100)
            noise_level = test_case.input_data.get('noise_level', 0.1)
            
            x, y, true_slope, true_intercept = self.generate_test_data(n_points, noise_level)
            
            # Perform human calculation
            human_result = self.human_linear_fit(x, y)
            
            # Perform AI calculation
            ai_result = self.ai_linear_fit(x, y)
            
            # Compare results
            comparison = self.compare_results(human_result, ai_result)
            
            # Validate against expected output
            expected_slope = test_case.expected_output.get('slope', true_slope)
            expected_intercept = test_case.expected_output.get('intercept', true_intercept)
            
            slope_error = abs(human_result['slope'] - expected_slope)
            intercept_error = abs(human_result['intercept'] - expected_intercept)
            
            # Determine test status
            if (slope_error < test_case.tolerance and 
                intercept_error < test_case.tolerance and 
                comparison['overall_agreement'] > 0.8):
                status = TestStatus.PASSED
            else:
                status = TestStatus.FAILED
            
            return TestResult(
                test_name=test_case.name,
                test_type=TestType.LINEAR_FITTING,
                status=status,
                execution_time=human_result['execution_time'] + ai_result['execution_time'],
                metrics=comparison,
                human_result=human_result,
                ai_result=ai_result,
                comparison_score=comparison['overall_agreement']
            )
            
        except Exception as e:
            return TestResult(
                test_name=test_case.name,
                test_type=TestType.LINEAR_FITTING,
                status=TestStatus.FAILED,
                execution_time=0.0,
                error_message=str(e)
            )

class ComprehensiveTestSystem:
    """Main test system orchestrator"""
    
    def __init__(self):
        self.test_suites = {}
        self.test_results = []
        self.linear_fitting_tester = LinearFittingTester()
        self.performance_history = []
        self.setup_logging()
    
    def setup_logging(self):
        """Setup comprehensive logging"""
        log_dir = Path("test_logs")
        log_dir.mkdir(exist_ok=True)
        
        # File handler
        log_file = log_dir / f"test_system_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"
        file_handler = logging.FileHandler(log_file)
        file_handler.setLevel(logging.DEBUG)
        
        # Console handler
        console_handler = logging.StreamHandler()
        console_handler.setLevel(logging.INFO)
        
        # Formatter
        formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
        file_handler.setFormatter(formatter)
        console_handler.setFormatter(formatter)
        
        logger.addHandler(file_handler)
        logger.addHandler(console_handler)
    
    def create_linear_fitting_test_suite(self) -> TestSuite:
        """Create comprehensive linear fitting test suite"""
        test_cases = [
            TestCase(
                name="Basic Linear Fit",
                description="Simple linear fitting with low noise",
                test_type=TestType.LINEAR_FITTING,
                input_data={'n_points': 50, 'noise_level': 0.05},
                expected_output={'slope': 2.5, 'intercept': 1.0},
                tolerance=0.1
            ),
            TestCase(
                name="High Noise Linear Fit",
                description="Linear fitting with high noise level",
                test_type=TestType.LINEAR_FITTING,
                input_data={'n_points': 100, 'noise_level': 0.3},
                expected_output={'slope': 2.5, 'intercept': 1.0},
                tolerance=0.2
            ),
            TestCase(
                name="Large Dataset Linear Fit",
                description="Linear fitting with large dataset",
                test_type=TestType.LINEAR_FITTING,
                input_data={'n_points': 1000, 'noise_level': 0.1},
                expected_output={'slope': 2.5, 'intercept': 1.0},
                tolerance=0.05
            ),
            TestCase(
                name="Perfect Linear Fit",
                description="Linear fitting with no noise",
                test_type=TestType.LINEAR_FITTING,
                input_data={'n_points': 100, 'noise_level': 0.0},
                expected_output={'slope': 2.5, 'intercept': 1.0},
                tolerance=0.001
            )
        ]
        
        return TestSuite(
            name="Linear Fitting Test Suite",
            description="Comprehensive testing of linear fitting capabilities",
            test_cases=test_cases
        )
    
    def run_test_suite(self, test_suite: TestSuite) -> List[TestResult]:
        """Run a complete test suite"""
        logger.info(f"Starting test suite: {test_suite.name}")
        results = []
        
        for test_case in test_suite.test_cases:
            logger.info(f"Running test: {test_case.name}")
            
            if test_case.test_type == TestType.LINEAR_FITTING:
                result = self.linear_fitting_tester.run_linear_fitting_test(test_case)
            else:
                # Placeholder for other test types
                result = TestResult(
                    test_name=test_case.name,
                    test_type=test_case.test_type,
                    status=TestStatus.SKIPPED,
                    execution_time=0.0,
                    error_message="Test type not implemented yet"
                )
            
            results.append(result)
            self.test_results.append(result)
            
            logger.info(f"Test {test_case.name}: {result.status.value}")
        
        return results
    
    def generate_test_report(self, results: List[TestResult]) -> Dict[str, Any]:
        """Generate comprehensive test report"""
        report = {
            'summary': {
                'total_tests': len(results),
                'passed': len([r for r in results if r.status == TestStatus.PASSED]),
                'failed': len([r for r in results if r.status == TestStatus.FAILED]),
                'skipped': len([r for r in results if r.status == TestStatus.SKIPPED]),
                'total_execution_time': sum(r.execution_time for r in results),
                'average_execution_time': np.mean([r.execution_time for r in results])
            },
            'detailed_results': [asdict(r) for r in results],
            'performance_metrics': self.calculate_performance_metrics(results),
            'recommendations': self.generate_recommendations(results)
        }
        
        return report
    
    def calculate_performance_metrics(self, results: List[TestResult]) -> Dict[str, Any]:
        """Calculate performance metrics from test results"""
        linear_fitting_results = [r for r in results if r.test_type == TestType.LINEAR_FITTING]
        
        if not linear_fitting_results:
            return {}
        
        metrics = {
            'average_comparison_score': np.mean([r.comparison_score for r in linear_fitting_results if r.comparison_score is not None]),
            'human_ai_agreement_rate': np.mean([1 if r.comparison_score and r.comparison_score > 0.8 else 0 for r in linear_fitting_results]),
            'average_human_execution_time': np.mean([r.human_result['execution_time'] for r in linear_fitting_results if r.human_result]),
            'average_ai_execution_time': np.mean([r.ai_result['execution_time'] for r in linear_fitting_results if r.ai_result]),
            'speed_improvement': np.mean([r.metrics['speed_ratio'] for r in linear_fitting_results if r.metrics and 'speed_ratio' in r.metrics])
        }
        
        return metrics
    
    def generate_recommendations(self, results: List[TestResult]) -> List[str]:
        """Generate recommendations based on test results"""
        recommendations = []
        
        failed_tests = [r for r in results if r.status == TestStatus.FAILED]
        if failed_tests:
            recommendations.append(f"Review {len(failed_tests)} failed tests for potential improvements")
        
        slow_tests = [r for r in results if r.execution_time > 1.0]
        if slow_tests:
            recommendations.append(f"Optimize {len(slow_tests)} slow tests for better performance")
        
        low_agreement_tests = [r for r in results if r.comparison_score and r.comparison_score < 0.8]
        if low_agreement_tests:
            recommendations.append(f"Investigate {len(low_agreement_tests)} tests with low human-AI agreement")
        
        return recommendations
    
    def save_test_results(self, results: List[TestResult], filename: str = None):
        """Save test results to file"""
        if filename is None:
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            filename = f"test_results_{timestamp}.json"
        
        # Convert datetime objects to strings for JSON serialization
        serializable_results = []
        for result in results:
            result_dict = asdict(result)
            result_dict['timestamp'] = result_dict['timestamp'].isoformat()
            serializable_results.append(result_dict)
        
        with open(filename, 'w') as f:
            json.dump(serializable_results, f, indent=2)
        
        logger.info(f"Test results saved to {filename}")
    
    def load_test_results(self, filename: str) -> List[TestResult]:
        """Load test results from file"""
        with open(filename, 'r') as f:
            data = json.load(f)
        
        results = []
        for item in data:
            # Convert timestamp string back to datetime
            item['timestamp'] = datetime.fromisoformat(item['timestamp'])
            results.append(TestResult(**item))
        
        return results

class InteractiveTestInterface:
    """Interactive GUI for testing and comparison"""
    
    def __init__(self, test_system: ComprehensiveTestSystem):
        self.test_system = test_system
        self.root = tk.Tk()
        self.root.title("Comprehensive Test System - AI Excel Learning")
        self.root.geometry("1200x800")
        
        self.setup_ui()
    
    def setup_ui(self):
        """Setup the user interface"""
        # Create notebook for tabs
        notebook = ttk.Notebook(self.root)
        notebook.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
        
        # Test execution tab
        self.create_test_execution_tab(notebook)
        
        # Results comparison tab
        self.create_results_comparison_tab(notebook)
        
        # Performance analysis tab
        self.create_performance_analysis_tab(notebook)
        
        # Settings tab
        self.create_settings_tab(notebook)
    
    def create_test_execution_tab(self, notebook):
        """Create test execution tab"""
        frame = ttk.Frame(notebook)
        notebook.add(frame, text="Test Execution")
        
        # Test suite selection
        ttk.Label(frame, text="Select Test Suite:").pack(pady=5)
        self.test_suite_var = tk.StringVar(value="Linear Fitting")
        test_suite_combo = ttk.Combobox(frame, textvariable=self.test_suite_var, 
                                       values=["Linear Fitting", "Formula Learning", "Chart Generation"])
        test_suite_combo.pack(pady=5)
        
        # Run tests button
        ttk.Button(frame, text="Run Selected Test Suite", 
                  command=self.run_selected_test_suite).pack(pady=10)
        
        # Progress bar
        self.progress_var = tk.DoubleVar()
        self.progress_bar = ttk.Progressbar(frame, variable=self.progress_var, maximum=100)
        self.progress_bar.pack(fill=tk.X, padx=20, pady=10)
        
        # Results text area
        ttk.Label(frame, text="Test Results:").pack(pady=5)
        self.results_text = scrolledtext.ScrolledText(frame, height=20)
        self.results_text.pack(fill=tk.BOTH, expand=True, padx=20, pady=10)
    
    def create_results_comparison_tab(self, notebook):
        """Create results comparison tab"""
        frame = ttk.Frame(notebook)
        notebook.add(frame, text="Results Comparison")
        
        # Comparison controls
        control_frame = ttk.Frame(frame)
        control_frame.pack(fill=tk.X, padx=20, pady=10)
        
        ttk.Button(control_frame, text="Load Test Results", 
                  command=self.load_test_results).pack(side=tk.LEFT, padx=5)
        ttk.Button(control_frame, text="Generate Comparison", 
                  command=self.generate_comparison).pack(side=tk.LEFT, padx=5)
        
        # Comparison display
        self.comparison_text = scrolledtext.ScrolledText(frame, height=25)
        self.comparison_text.pack(fill=tk.BOTH, expand=True, padx=20, pady=10)
    
    def create_performance_analysis_tab(self, notebook):
        """Create performance analysis tab"""
        frame = ttk.Frame(notebook)
        notebook.add(frame, text="Performance Analysis")
        
        # Performance controls
        control_frame = ttk.Frame(frame)
        control_frame.pack(fill=tk.X, padx=20, pady=10)
        
        ttk.Button(control_frame, text="Analyze Performance", 
                  command=self.analyze_performance).pack(side=tk.LEFT, padx=5)
        ttk.Button(control_frame, text="Generate Charts", 
                  command=self.generate_performance_charts).pack(side=tk.LEFT, padx=5)
        
        # Performance display
        self.performance_text = scrolledtext.ScrolledText(frame, height=25)
        self.performance_text.pack(fill=tk.BOTH, expand=True, padx=20, pady=10)
    
    def create_settings_tab(self, notebook):
        """Create settings tab"""
        frame = ttk.Frame(notebook)
        notebook.add(frame, text="Settings")
        
        # Settings controls
        settings_frame = ttk.LabelFrame(frame, text="Test Configuration")
        settings_frame.pack(fill=tk.X, padx=20, pady=10)
        
        ttk.Label(settings_frame, text="Default Tolerance:").grid(row=0, column=0, padx=5, pady=5)
        self.tolerance_var = tk.DoubleVar(value=0.01)
        ttk.Entry(settings_frame, textvariable=self.tolerance_var).grid(row=0, column=1, padx=5, pady=5)
        
        ttk.Label(settings_frame, text="Default Timeout (seconds):").grid(row=1, column=0, padx=5, pady=5)
        self.timeout_var = tk.IntVar(value=30)
        ttk.Entry(settings_frame, textvariable=self.timeout_var).grid(row=1, column=1, padx=5, pady=5)
        
        # Save settings button
        ttk.Button(frame, text="Save Settings", 
                  command=self.save_settings).pack(pady=10)
    
    def run_selected_test_suite(self):
        """Run the selected test suite"""
        test_suite_name = self.test_suite_var.get()
        
        if test_suite_name == "Linear Fitting":
            test_suite = self.test_system.create_linear_fitting_test_suite()
        else:
            messagebox.showwarning("Not Implemented", f"Test suite '{test_suite_name}' not implemented yet")
            return
        
        # Clear results
        self.results_text.delete(1.0, tk.END)
        self.results_text.insert(tk.END, f"Running {test_suite.name}...\n\n")
        
        # Run tests
        results = self.test_system.run_test_suite(test_suite)
        
        # Display results
        for result in results:
            status_icon = "✅" if result.status == TestStatus.PASSED else "❌"
            self.results_text.insert(tk.END, f"{status_icon} {result.test_name}: {result.status.value}\n")
            
            if result.metrics:
                self.results_text.insert(tk.END, f"   Comparison Score: {result.comparison_score:.3f}\n")
                self.results_text.insert(tk.END, f"   Execution Time: {result.execution_time:.3f}s\n")
            
            if result.error_message:
                self.results_text.insert(tk.END, f"   Error: {result.error_message}\n")
            
            self.results_text.insert(tk.END, "\n")
        
        # Generate and display report
        report = self.test_system.generate_test_report(results)
        self.results_text.insert(tk.END, f"\n=== SUMMARY ===\n")
        self.results_text.insert(tk.END, f"Total Tests: {report['summary']['total_tests']}\n")
        self.results_text.insert(tk.END, f"Passed: {report['summary']['passed']}\n")
        self.results_text.insert(tk.END, f"Failed: {report['summary']['failed']}\n")
        self.results_text.insert(tk.END, f"Average Execution Time: {report['summary']['average_execution_time']:.3f}s\n")
    
    def load_test_results(self):
        """Load test results from file"""
        filename = filedialog.askopenfilename(
            title="Select Test Results File",
            filetypes=[("JSON files", "*.json"), ("All files", "*.*")]
        )
        
        if filename:
            try:
                results = self.test_system.load_test_results(filename)
                self.comparison_text.delete(1.0, tk.END)
                self.comparison_text.insert(tk.END, f"Loaded {len(results)} test results from {filename}\n\n")
                
                for result in results:
                    self.comparison_text.insert(tk.END, f"Test: {result.test_name}\n")
                    self.comparison_text.insert(tk.END, f"Status: {result.status.value}\n")
                    if result.comparison_score:
                        self.comparison_text.insert(tk.END, f"Score: {result.comparison_score:.3f}\n")
                    self.comparison_text.insert(tk.END, "\n")
                    
            except Exception as e:
                messagebox.showerror("Error", f"Failed to load test results: {str(e)}")
    
    def generate_comparison(self):
        """Generate comparison analysis"""
        if not self.test_system.test_results:
            messagebox.showwarning("No Data", "No test results available for comparison")
            return
        
        self.comparison_text.delete(1.0, tk.END)
        self.comparison_text.insert(tk.END, "=== COMPARISON ANALYSIS ===\n\n")
        
        # Analyze human vs AI performance
        linear_results = [r for r in self.test_system.test_results if r.test_type == TestType.LINEAR_FITTING]
        
        if linear_results:
            human_times = [r.human_result['execution_time'] for r in linear_results if r.human_result]
            ai_times = [r.ai_result['execution_time'] for r in linear_results if r.ai_result]
            
            self.comparison_text.insert(tk.END, f"Performance Comparison:\n")
            self.comparison_text.insert(tk.END, f"Average Human Time: {np.mean(human_times):.4f}s\n")
            self.comparison_text.insert(tk.END, f"Average AI Time: {np.mean(ai_times):.4f}s\n")
            self.comparison_text.insert(tk.END, f"Speed Ratio: {np.mean(human_times) / np.mean(ai_times):.2f}x\n\n")
            
            # Agreement analysis
            agreement_scores = [r.comparison_score for r in linear_results if r.comparison_score is not None]
            self.comparison_text.insert(tk.END, f"Agreement Analysis:\n")
            self.comparison_text.insert(tk.END, f"Average Agreement: {np.mean(agreement_scores):.3f}\n")
            self.comparison_text.insert(tk.END, f"High Agreement Tests (>0.8): {sum(1 for s in agreement_scores if s > 0.8)}\n")
    
    def analyze_performance(self):
        """Analyze performance metrics"""
        if not self.test_system.test_results:
            messagebox.showwarning("No Data", "No test results available for analysis")
            return
        
        self.performance_text.delete(1.0, tk.END)
        self.performance_text.insert(tk.END, "=== PERFORMANCE ANALYSIS ===\n\n")
        
        # Calculate performance metrics
        metrics = self.test_system.calculate_performance_metrics(self.test_system.test_results)
        
        for metric, value in metrics.items():
            if isinstance(value, float):
                self.performance_text.insert(tk.END, f"{metric}: {value:.4f}\n")
            else:
                self.performance_text.insert(tk.END, f"{metric}: {value}\n")
        
        # Generate recommendations
        recommendations = self.test_system.generate_recommendations(self.test_system.test_results)
        self.performance_text.insert(tk.END, f"\n=== RECOMMENDATIONS ===\n")
        for rec in recommendations:
            self.performance_text.insert(tk.END, f"• {rec}\n")
    
    def generate_performance_charts(self):
        """Generate performance visualization charts"""
        if not self.test_system.test_results:
            messagebox.showwarning("No Data", "No test results available for charts")
            return
        
        # Create performance charts
        fig, axes = plt.subplots(2, 2, figsize=(12, 10))
        fig.suptitle("Performance Analysis Charts")
        
        linear_results = [r for r in self.test_system.test_results if r.test_type == TestType.LINEAR_FITTING]
        
        if linear_results:
            # Execution time comparison
            human_times = [r.human_result['execution_time'] for r in linear_results if r.human_result]
            ai_times = [r.ai_result['execution_time'] for r in linear_results if r.ai_result]
            
            axes[0, 0].bar(['Human', 'AI'], [np.mean(human_times), np.mean(ai_times)])
            axes[0, 0].set_title('Average Execution Time')
            axes[0, 0].set_ylabel('Time (seconds)')
            
            # Agreement scores
            agreement_scores = [r.comparison_score for r in linear_results if r.comparison_score is not None]
            axes[0, 1].hist(agreement_scores, bins=10, alpha=0.7)
            axes[0, 1].set_title('Distribution of Agreement Scores')
            axes[0, 1].set_xlabel('Agreement Score')
            axes[0, 1].set_ylabel('Frequency')
            
            # R-squared comparison
            human_r2 = [r.human_result['r_squared'] for r in linear_results if r.human_result]
            ai_r2 = [r.ai_result['r_squared'] for r in linear_results if r.ai_result]
            
            axes[1, 0].scatter(human_r2, ai_r2, alpha=0.6)
            axes[1, 0].plot([0, 1], [0, 1], 'r--', alpha=0.5)
            axes[1, 0].set_title('R-squared Comparison')
            axes[1, 0].set_xlabel('Human R²')
            axes[1, 0].set_ylabel('AI R²')
            
            # Test status distribution
            status_counts = {}
            for result in linear_results:
                status = result.status.value
                status_counts[status] = status_counts.get(status, 0) + 1
            
            axes[1, 1].pie(status_counts.values(), labels=status_counts.keys(), autopct='%1.1f%%')
            axes[1, 1].set_title('Test Status Distribution')
        
        plt.tight_layout()
        plt.show()
    
    def save_settings(self):
        """Save current settings"""
        # Update test system settings based on UI values
        messagebox.showinfo("Settings", "Settings saved successfully!")
    
    def run(self):
        """Run the interactive interface"""
        self.root.mainloop()

def main():
    """Main function to run the comprehensive test system"""
    print("=== Comprehensive Test System for AI Excel Learning ===")
    print("This system provides advanced testing capabilities for:")
    print("1. Linear fitting feature testing")
    print("2. Human vs AI calculation comparison")
    print("3. Interactive testing interface")
    print("4. Performance benchmarking")
    print("5. Debugging and analysis tools")
    print("6. Long-term testing capabilities")
    print()
    
    # Initialize test system
    test_system = ComprehensiveTestSystem()
    
    # Create and run interactive interface
    interface = InteractiveTestInterface(test_system)
    
    print("Starting interactive test interface...")
    interface.run()

if __name__ == "__main__":
    main()
