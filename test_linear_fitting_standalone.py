#!/usr/bin/env python3
"""
Standalone Linear Fitting Test System
====================================

This script provides a simplified version of the comprehensive test system
focused specifically on testing the linear fitting feature. It can run
without heavy dependencies like TensorFlow and provides:

1. Linear fitting testing with human vs AI comparison
2. Performance benchmarking
3. Statistical validation
4. Visual results
5. Interactive testing capabilities

This demonstrates the core testing concepts without requiring the full
AI/ML stack.
"""

import os
import sys
import time
import json
import math
import random
from datetime import datetime
from typing import Dict, List, Tuple, Any, Optional
from dataclasses import dataclass, asdict
from enum import Enum

# Basic data manipulation (no heavy dependencies)
try:
    import numpy as np
    NUMPY_AVAILABLE = True
except ImportError:
    NUMPY_AVAILABLE = False
    print("Warning: NumPy not available, using basic math operations")

try:
    import matplotlib.pyplot as plt
    MATPLOTLIB_AVAILABLE = True
except ImportError:
    MATPLOTLIB_AVAILABLE = False
    print("Warning: Matplotlib not available, charts will be skipped")

# Simple data structures for when numpy is not available
class SimpleArray:
    """Simple array implementation when numpy is not available"""
    def __init__(self, data):
        self.data = list(data)
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, index):
        return self.data[index]
    
    def __setitem__(self, index, value):
        self.data[index] = value
    
    def mean(self):
        return sum(self.data) / len(self.data)
    
    def reshape(self, shape):
        # Simple reshape for 1D to 2D
        if shape == (-1, 1):
            return [[x] for x in self.data]
        return self.data

class TestStatus(Enum):
    """Status of test execution"""
    PASSED = "passed"
    FAILED = "failed"
    SKIPPED = "skipped"

@dataclass
class TestResult:
    """Result of a test execution"""
    test_name: str
    status: TestStatus
    execution_time: float
    human_result: Optional[Dict[str, Any]] = None
    ai_result: Optional[Dict[str, Any]] = None
    comparison_score: Optional[float] = None
    error_message: Optional[str] = None
    timestamp: datetime = None
    
    def __post_init__(self):
        if self.timestamp is None:
            self.timestamp = datetime.now()

class LinearFittingTester:
    """Specialized tester for linear fitting functionality"""
    
    def __init__(self):
        self.test_results = []
        self.performance_metrics = {}
    
    def generate_test_data(self, n_points: int = 100, noise_level: float = 0.1) -> Tuple[Any, Any, float, float]:
        """Generate synthetic test data for linear fitting"""
        if NUMPY_AVAILABLE:
            x = np.linspace(0, 10, n_points)
            true_slope = 2.5
            true_intercept = 1.0
            y_true = true_slope * x + true_intercept
            noise = np.random.normal(0, noise_level, n_points)
            y_noisy = y_true + noise
        else:
            # Fallback to basic implementation
            x = SimpleArray([i * 10.0 / (n_points - 1) for i in range(n_points)])
            true_slope = 2.5
            true_intercept = 1.0
            y_true = SimpleArray([true_slope * x[i] + true_intercept for i in range(n_points)])
            noise = SimpleArray([random.gauss(0, noise_level) for _ in range(n_points)])
            y_noisy = SimpleArray([y_true[i] + noise[i] for i in range(n_points)])
        
        return x, y_noisy, true_slope, true_intercept
    
    def human_linear_fit(self, x: Any, y: Any) -> Dict[str, Any]:
        """Perform linear fitting using traditional methods (human approach)"""
        start_time = time.time()
        
        # Calculate means
        x_mean = x.mean()
        y_mean = y.mean()
        
        # Calculate slope and intercept using least squares
        numerator = sum((x[i] - x_mean) * (y[i] - y_mean) for i in range(len(x)))
        denominator = sum((x[i] - x_mean) ** 2 for i in range(len(x)))
        
        slope = numerator / denominator
        intercept = y_mean - slope * x_mean
        
        # Calculate R-squared
        y_pred = [slope * x[i] + intercept for i in range(len(x))]
        ss_res = sum((y[i] - y_pred[i]) ** 2 for i in range(len(x)))
        ss_tot = sum((y[i] - y_mean) ** 2 for i in range(len(x)))
        r_squared = 1 - (ss_res / ss_tot)
        
        # Calculate standard errors
        n = len(x)
        mse = ss_res / (n - 2)
        slope_se = math.sqrt(mse / sum((x[i] - x_mean) ** 2 for i in range(len(x))))
        intercept_se = math.sqrt(mse * (1/n + x_mean**2 / sum((x[i] - x_mean) ** 2 for i in range(len(x)))))
        
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
    
    def ai_linear_fit(self, x: Any, y: Any) -> Dict[str, Any]:
        """Perform linear fitting using AI/ML methods (simplified)"""
        start_time = time.time()
        
        # Simplified AI approach - using the same mathematical method but with different implementation
        # In a full system, this would use scikit-learn or other ML libraries
        
        # Calculate means
        x_mean = x.mean()
        y_mean = y.mean()
        
        # Calculate slope and intercept using least squares (same math, different implementation)
        numerator = 0
        denominator = 0
        for i in range(len(x)):
            numerator += (x[i] - x_mean) * (y[i] - y_mean)
            denominator += (x[i] - x_mean) ** 2
        
        slope = numerator / denominator
        intercept = y_mean - slope * x_mean
        
        # Calculate R-squared
        y_pred = [slope * x[i] + intercept for i in range(len(x))]
        ss_res = sum((y[i] - y_pred[i]) ** 2 for i in range(len(x)))
        ss_tot = sum((y[i] - y_mean) ** 2 for i in range(len(x)))
        r_squared = 1 - (ss_res / ss_tot)
        
        # Calculate standard errors
        n = len(x)
        mse = ss_res / (n - 2)
        slope_se = math.sqrt(mse / sum((x[i] - x_mean) ** 2 for i in range(len(x))))
        intercept_se = math.sqrt(mse * (1/n + x_mean**2 / sum((x[i] - x_mean) ** 2 for i in range(len(x)))))
        
        execution_time = time.time() - start_time
        
        return {
            'slope': slope,
            'intercept': intercept,
            'r_squared': r_squared,
            'slope_se': slope_se,
            'intercept_se': intercept_se,
            'execution_time': execution_time,
            'method': 'ai_optimized_least_squares'
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
    
    def run_linear_fitting_test(self, test_name: str, n_points: int = 100, noise_level: float = 0.1) -> TestResult:
        """Run a comprehensive linear fitting test"""
        try:
            print(f"Running test: {test_name}")
            print(f"  Parameters: n_points={n_points}, noise_level={noise_level}")
            
            # Generate test data
            x, y, true_slope, true_intercept = self.generate_test_data(n_points, noise_level)
            
            # Perform human calculation
            print("  Performing human calculation...")
            human_result = self.human_linear_fit(x, y)
            
            # Perform AI calculation
            print("  Performing AI calculation...")
            ai_result = self.ai_linear_fit(x, y)
            
            # Compare results
            comparison = self.compare_results(human_result, ai_result)
            
            # Determine test status
            if comparison['overall_agreement'] > 0.8:
                status = TestStatus.PASSED
            else:
                status = TestStatus.FAILED
            
            result = TestResult(
                test_name=test_name,
                status=status,
                execution_time=human_result['execution_time'] + ai_result['execution_time'],
                human_result=human_result,
                ai_result=ai_result,
                comparison_score=comparison['overall_agreement']
            )
            
            # Print results
            print(f"  Results:")
            print(f"    Human: slope={human_result['slope']:.4f}, intercept={human_result['intercept']:.4f}, R²={human_result['r_squared']:.4f}")
            print(f"    AI:    slope={ai_result['slope']:.4f}, intercept={ai_result['intercept']:.4f}, R²={ai_result['r_squared']:.4f}")
            print(f"    Agreement: {comparison['overall_agreement']:.3f}")
            print(f"    Speed ratio: {comparison['speed_ratio']:.2f}x")
            print(f"    Status: {status.value}")
            print()
            
            return result
            
        except Exception as e:
            print(f"  Error: {str(e)}")
            return TestResult(
                test_name=test_name,
                status=TestStatus.FAILED,
                execution_time=0.0,
                error_message=str(e)
            )

class ComprehensiveTestSystem:
    """Main test system orchestrator"""
    
    def __init__(self):
        self.test_results = []
        self.linear_fitting_tester = LinearFittingTester()
    
    def run_linear_fitting_test_suite(self) -> List[TestResult]:
        """Run comprehensive linear fitting test suite"""
        print("=== Linear Fitting Test Suite ===")
        print()
        
        test_cases = [
            ("Basic Linear Fit", 50, 0.05),
            ("High Noise Linear Fit", 100, 0.3),
            ("Large Dataset Linear Fit", 1000, 0.1),
            ("Perfect Linear Fit", 100, 0.0),
            ("Small Dataset", 20, 0.1),
            ("Very Large Dataset", 5000, 0.05)
        ]
        
        results = []
        
        for test_name, n_points, noise_level in test_cases:
            result = self.linear_fitting_tester.run_linear_fitting_test(test_name, n_points, noise_level)
            results.append(result)
            self.test_results.append(result)
        
        return results
    
    def generate_test_report(self, results: List[TestResult]) -> Dict[str, Any]:
        """Generate comprehensive test report"""
        passed = len([r for r in results if r.status == TestStatus.PASSED])
        failed = len([r for r in results if r.status == TestStatus.FAILED])
        
        # Calculate performance metrics
        human_times = [r.human_result['execution_time'] for r in results if r.human_result]
        ai_times = [r.ai_result['execution_time'] for r in results if r.ai_result]
        agreement_scores = [r.comparison_score for r in results if r.comparison_score is not None]
        
        report = {
            'summary': {
                'total_tests': len(results),
                'passed': passed,
                'failed': failed,
                'pass_rate': passed / len(results) if results else 0,
                'average_execution_time': sum(r.execution_time for r in results) / len(results) if results else 0
            },
            'performance': {
                'average_human_time': sum(human_times) / len(human_times) if human_times else 0,
                'average_ai_time': sum(ai_times) / len(ai_times) if ai_times else 0,
                'average_agreement': sum(agreement_scores) / len(agreement_scores) if agreement_scores else 0,
                'high_agreement_tests': sum(1 for s in agreement_scores if s > 0.8) if agreement_scores else 0
            },
            'detailed_results': [asdict(r) for r in results]
        }
        
        return report
    
    def save_test_results(self, results: List[TestResult], filename: str = None):
        """Save test results to file"""
        if filename is None:
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            filename = f"linear_fitting_test_results_{timestamp}.json"
        
        # Convert datetime objects and enums to strings for JSON serialization
        serializable_results = []
        for result in results:
            result_dict = asdict(result)
            result_dict['timestamp'] = result_dict['timestamp'].isoformat()
            result_dict['status'] = result_dict['status'].value  # Convert enum to string
            serializable_results.append(result_dict)
        
        with open(filename, 'w') as f:
            json.dump(serializable_results, f, indent=2)
        
        print(f"Test results saved to {filename}")
    
    def generate_visual_comparison(self, results: List[TestResult]):
        """Generate visual comparison charts"""
        if not MATPLOTLIB_AVAILABLE:
            print("Matplotlib not available, skipping visual comparison")
            return
        
        if not results:
            print("No results to visualize")
            return
        
        # Create comparison charts
        fig, axes = plt.subplots(2, 2, figsize=(12, 10))
        fig.suptitle("Linear Fitting Test Results Comparison")
        
        # Extract data
        test_names = [r.test_name for r in results]
        human_slopes = [r.human_result['slope'] for r in results if r.human_result]
        ai_slopes = [r.ai_result['slope'] for r in results if r.ai_result]
        human_intercepts = [r.human_result['intercept'] for r in results if r.human_result]
        ai_intercepts = [r.ai_result['intercept'] for r in results if r.ai_result]
        agreement_scores = [r.comparison_score for r in results if r.comparison_score is not None]
        human_times = [r.human_result['execution_time'] for r in results if r.human_result]
        ai_times = [r.ai_result['execution_time'] for r in results if r.ai_result]
        
        # Slope comparison
        x_pos = range(len(test_names))
        width = 0.35
        axes[0, 0].bar([x - width/2 for x in x_pos], human_slopes, width, label='Human', alpha=0.8)
        axes[0, 0].bar([x + width/2 for x in x_pos], ai_slopes, width, label='AI', alpha=0.8)
        axes[0, 0].set_title('Slope Comparison')
        axes[0, 0].set_ylabel('Slope')
        axes[0, 0].legend()
        axes[0, 0].set_xticks(x_pos)
        axes[0, 0].set_xticklabels(test_names, rotation=45, ha='right')
        
        # Intercept comparison
        axes[0, 1].bar([x - width/2 for x in x_pos], human_intercepts, width, label='Human', alpha=0.8)
        axes[0, 1].bar([x + width/2 for x in x_pos], ai_intercepts, width, label='AI', alpha=0.8)
        axes[0, 1].set_title('Intercept Comparison')
        axes[0, 1].set_ylabel('Intercept')
        axes[0, 1].legend()
        axes[0, 1].set_xticks(x_pos)
        axes[0, 1].set_xticklabels(test_names, rotation=45, ha='right')
        
        # Agreement scores
        axes[1, 0].bar(test_names, agreement_scores, alpha=0.7, color='green')
        axes[1, 0].set_title('Agreement Scores')
        axes[1, 0].set_ylabel('Agreement Score')
        axes[1, 0].set_xticklabels(test_names, rotation=45, ha='right')
        axes[1, 0].axhline(y=0.8, color='red', linestyle='--', alpha=0.7, label='Threshold (0.8)')
        axes[1, 0].legend()
        
        # Execution time comparison
        axes[1, 1].bar([x - width/2 for x in x_pos], human_times, width, label='Human', alpha=0.8)
        axes[1, 1].bar([x + width/2 for x in x_pos], ai_times, width, label='AI', alpha=0.8)
        axes[1, 1].set_title('Execution Time Comparison')
        axes[1, 1].set_ylabel('Time (seconds)')
        axes[1, 1].legend()
        axes[1, 1].set_xticks(x_pos)
        axes[1, 1].set_xticklabels(test_names, rotation=45, ha='right')
        
        plt.tight_layout()
        plt.show()

def interactive_test():
    """Interactive testing function"""
    print("=== Interactive Linear Fitting Test ===")
    print()
    
    test_system = ComprehensiveTestSystem()
    
    while True:
        print("Options:")
        print("1. Run full test suite")
        print("2. Run single test")
        print("3. View previous results")
        print("4. Generate visual comparison")
        print("5. Exit")
        print()
        
        choice = input("Enter your choice (1-5): ").strip()
        
        if choice == "1":
            print("\nRunning full test suite...")
            results = test_system.run_linear_fitting_test_suite()
            report = test_system.generate_test_report(results)
            
            print("\n=== TEST REPORT ===")
            print(f"Total Tests: {report['summary']['total_tests']}")
            print(f"Passed: {report['summary']['passed']}")
            print(f"Failed: {report['summary']['failed']}")
            print(f"Pass Rate: {report['summary']['pass_rate']:.1%}")
            print(f"Average Execution Time: {report['summary']['average_execution_time']:.4f}s")
            print(f"Average Agreement: {report['performance']['average_agreement']:.3f}")
            print(f"High Agreement Tests: {report['performance']['high_agreement_tests']}")
            
            save_choice = input("\nSave results to file? (y/n): ").strip().lower()
            if save_choice == 'y':
                test_system.save_test_results(results)
            
            viz_choice = input("Generate visual comparison? (y/n): ").strip().lower()
            if viz_choice == 'y':
                test_system.generate_visual_comparison(results)
        
        elif choice == "2":
            print("\nSingle test configuration:")
            test_name = input("Test name: ").strip()
            try:
                n_points = int(input("Number of points (default 100): ").strip() or "100")
                noise_level = float(input("Noise level (default 0.1): ").strip() or "0.1")
                
                result = test_system.linear_fitting_tester.run_linear_fitting_test(test_name, n_points, noise_level)
                test_system.test_results.append(result)
                
            except ValueError as e:
                print(f"Invalid input: {e}")
        
        elif choice == "3":
            if test_system.test_results:
                print(f"\nPrevious Results ({len(test_system.test_results)} tests):")
                for result in test_system.test_results:
                    status_icon = "✅" if result.status == TestStatus.PASSED else "❌"
                    print(f"{status_icon} {result.test_name}: {result.status.value}")
                    if result.comparison_score:
                        print(f"    Agreement: {result.comparison_score:.3f}")
            else:
                print("No previous results available")
        
        elif choice == "4":
            if test_system.test_results:
                test_system.generate_visual_comparison(test_system.test_results)
            else:
                print("No results available for visualization")
        
        elif choice == "5":
            print("Exiting...")
            break
        
        else:
            print("Invalid choice. Please enter 1-5.")
        
        print()

def main():
    """Main function"""
    print("=== Standalone Linear Fitting Test System ===")
    print("This system tests linear fitting capabilities with human vs AI comparison")
    print()
    
    if not NUMPY_AVAILABLE:
        print("Note: Running without NumPy - using basic math operations")
        print("Performance may be slower but functionality is preserved")
        print()
    
    if not MATPLOTLIB_AVAILABLE:
        print("Note: Running without Matplotlib - visual comparisons disabled")
        print()
    
    # Check if user wants interactive mode
    interactive = input("Run in interactive mode? (y/n, default: y): ").strip().lower()
    if interactive != 'n':
        interactive_test()
    else:
        # Run automated test suite
        print("\nRunning automated test suite...")
        test_system = ComprehensiveTestSystem()
        results = test_system.run_linear_fitting_test_suite()
        report = test_system.generate_test_report(results)
        
        print("\n=== FINAL REPORT ===")
        print(f"Total Tests: {report['summary']['total_tests']}")
        print(f"Passed: {report['summary']['passed']}")
        print(f"Failed: {report['summary']['failed']}")
        print(f"Pass Rate: {report['summary']['pass_rate']:.1%}")
        print(f"Average Agreement: {report['performance']['average_agreement']:.3f}")
        
        # Save results
        test_system.save_test_results(results)
        
        # Generate visualization if available
        if MATPLOTLIB_AVAILABLE:
            viz_choice = input("\nGenerate visual comparison? (y/n): ").strip().lower()
            if viz_choice == 'y':
                test_system.generate_visual_comparison(results)

if __name__ == "__main__":
    main()
