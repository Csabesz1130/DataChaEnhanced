"""
Starting Point Simulation Feature

This module provides functionality to simulate different starting points for the ActionPotentialTab
and evaluate which starting point produces the smoothest purple curves without outstanding points.

The simulator tests various starting points and analyzes curve smoothness to recommend
the optimal starting point that produces smooth curves similar to the default behavior.
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy import signal
from scipy.stats import skew, kurtosis
from typing import Dict, List, Tuple, Optional, Any
import tkinter as tk
from tkinter import ttk, messagebox
import threading
import time

from src.utils.logger import app_logger
from src.analysis.action_potential import ActionPotentialProcessor


class StartingPointSimulator:
    """
    Simulates different starting points and evaluates curve smoothness.
    
    This class tests various starting points (n values) and analyzes the resulting
    purple curves to find the optimal starting point that produces smooth curves
    without outstanding spikes or irregularities.
    """
    
    def __init__(self, data: np.ndarray, time_data: np.ndarray, params: Dict):
        """
        Initialize the simulator with signal data.
        
        Args:
            data: Current data in pA
            time_data: Time data in seconds
            params: Analysis parameters (n_cycles, t0, t1, t2, V0, V1, V2, etc.)
        """
        self.data = data
        self.time_data = time_data
        self.params = params.copy()
        
        # Simulation parameters
        self.start_point_range = (10, 100)  # Range of starting points to test
        self.step_size = 5  # Step size between tests
        self.simulation_results = {}
        
        app_logger.info("Starting Point Simulator initialized")
    
    def run_simulation(self, progress_callback=None) -> Dict[str, Any]:
        """
        Run the complete simulation across all starting points.
        
        Args:
            progress_callback: Optional callback function for progress updates
            
        Returns:
            Dictionary containing simulation results and recommendations
        """
        try:
            app_logger.info("Starting point simulation initiated")
            
            # Generate list of starting points to test
            start_points = list(range(
                self.start_point_range[0], 
                self.start_point_range[1] + 1, 
                self.step_size
            ))
            
            total_points = len(start_points)
            results = {}
            
            for i, start_point in enumerate(start_points):
                try:
                    # Update progress
                    if progress_callback:
                        progress = (i / total_points) * 100
                        progress_callback(progress, f"Testing starting point {start_point}")
                    
                    # Test this starting point
                    result = self._test_starting_point(start_point)
                    results[start_point] = result
                    
                    app_logger.debug(f"Tested starting point {start_point}: smoothness={result['smoothness_score']:.3f}")
                    
                except Exception as e:
                    app_logger.error(f"Error testing starting point {start_point}: {str(e)}")
                    results[start_point] = {
                        'error': str(e),
                        'smoothness_score': 0.0,
                        'outstanding_points': float('inf')
                    }
            
            # Analyze results and find optimal starting point
            optimal_point = self._find_optimal_starting_point(results)
            
            # Final progress update
            if progress_callback:
                progress_callback(100, "Simulation complete")
            
            self.simulation_results = {
                'results': results,
                'optimal_starting_point': optimal_point,
                'recommendation': self._generate_recommendation(optimal_point, results)
            }
            
            app_logger.info(f"Simulation complete. Optimal starting point: {optimal_point}")
            return self.simulation_results
            
        except Exception as e:
            app_logger.error(f"Error in simulation: {str(e)}")
            return {'error': str(e)}
    
    def _test_starting_point(self, start_point: int) -> Dict[str, Any]:
        """
        Test a specific starting point and evaluate curve quality.
        
        Args:
            start_point: The starting point (n) to test
            
        Returns:
            Dictionary containing evaluation metrics
        """
        try:
            # Create parameters with this starting point
            test_params = self.params.copy()
            test_params['normalization_points'] = {
                'seg1': (start_point, start_point + 199),
                'seg2': (start_point + 200, start_point + 399),
                'seg3': (start_point + 400, start_point + 599),
                'seg4': (start_point + 600, start_point + 799)
            }
            
            # Create processor and run analysis
            processor = ActionPotentialProcessor(self.data, self.time_data, test_params)
            
            # Process signal to generate all curves
            (processed_data, orange_curve, orange_times, 
             normalized_curve, normalized_times, 
             average_curve, average_times, results) = processor.process_signal(
                use_alternative_method=test_params.get('use_alternative_method', False)
            )
            
            # Check if processing was successful
            if processed_data is None:
                return {
                    'error': 'Processing failed',
                    'smoothness_score': 0.0,
                    'outstanding_points': float('inf')
                }
            
            # Generate purple curves
            (modified_hyperpol, modified_hyperpol_times,
             modified_depol, modified_depol_times) = processor.apply_average_to_peaks()
            
            # Evaluate curve quality
            evaluation = self._evaluate_curve_quality(
                modified_hyperpol, modified_depol,
                modified_hyperpol_times, modified_depol_times
            )
            
            return {
                'smoothness_score': evaluation['smoothness_score'],
                'outstanding_points': evaluation['outstanding_points'],
                'hyperpol_metrics': evaluation['hyperpol_metrics'],
                'depol_metrics': evaluation['depol_metrics'],
                'overall_quality': evaluation['overall_quality'],
                'processor': processor  # Store processor for detailed analysis
            }
            
        except Exception as e:
            app_logger.error(f"Error testing starting point {start_point}: {str(e)}")
            return {
                'error': str(e),
                'smoothness_score': 0.0,
                'outstanding_points': float('inf')
            }
    
    def _evaluate_curve_quality(self, hyperpol_data: np.ndarray, depol_data: np.ndarray,
                               hyperpol_times: np.ndarray, depol_times: np.ndarray) -> Dict[str, Any]:
        """
        Evaluate the quality of purple curves based on smoothness and outstanding points.
        
        Args:
            hyperpol_data: Hyperpolarization curve data
            depol_data: Depolarization curve data
            hyperpol_times: Hyperpolarization time data
            depol_times: Depolarization time data
            
        Returns:
            Dictionary containing quality metrics
        """
        try:
            # Evaluate hyperpolarization curve
            hyperpol_metrics = self._analyze_curve_smoothness(hyperpol_data, hyperpol_times, 'hyperpol')
            
            # Evaluate depolarization curve
            depol_metrics = self._analyze_curve_smoothness(depol_data, depol_times, 'depol')
            
            # Calculate overall smoothness score (0-1, higher is better)
            smoothness_score = (hyperpol_metrics['smoothness'] + depol_metrics['smoothness']) / 2
            
            # Count outstanding points (spikes, irregularities)
            outstanding_points = hyperpol_metrics['outstanding_points'] + depol_metrics['outstanding_points']
            
            # Overall quality assessment
            overall_quality = self._assess_overall_quality(smoothness_score, outstanding_points)
            
            return {
                'smoothness_score': smoothness_score,
                'outstanding_points': outstanding_points,
                'hyperpol_metrics': hyperpol_metrics,
                'depol_metrics': depol_metrics,
                'overall_quality': overall_quality
            }
            
        except Exception as e:
            app_logger.error(f"Error evaluating curve quality: {str(e)}")
            return {
                'smoothness_score': 0.0,
                'outstanding_points': float('inf'),
                'hyperpol_metrics': {},
                'depol_metrics': {},
                'overall_quality': 'poor'
            }
    
    def _analyze_curve_smoothness(self, data: np.ndarray, times: np.ndarray, curve_type: str) -> Dict[str, Any]:
        """
        Analyze the smoothness of a single curve.
        
        Args:
            data: Curve data
            times: Time data
            curve_type: Type of curve ('hyperpol' or 'depol')
            
        Returns:
            Dictionary containing smoothness metrics
        """
        try:
            if data is None or len(data) == 0:
                return {
                    'smoothness': 0.0,
                    'outstanding_points': float('inf'),
                    'variance': float('inf'),
                    'skewness': 0.0,
                    'kurtosis': 0.0
                }
            
            # Calculate first and second derivatives for smoothness analysis
            first_derivative = np.diff(data)
            second_derivative = np.diff(first_derivative)
            
            # Smoothness metrics
            variance = np.var(data)
            first_deriv_variance = np.var(first_derivative)
            second_deriv_variance = np.var(second_derivative)
            
            # Statistical measures
            data_skewness = skew(data)
            data_kurtosis = kurtosis(data)
            
            # Detect outstanding points (spikes, irregularities)
            outstanding_points = self._detect_outstanding_points(data, first_derivative)
            
            # Calculate smoothness score (0-1, higher is better)
            # Lower variance in derivatives indicates smoother curves
            smoothness = self._calculate_smoothness_score(
                first_deriv_variance, second_deriv_variance, outstanding_points
            )
            
            return {
                'smoothness': smoothness,
                'outstanding_points': outstanding_points,
                'variance': variance,
                'first_deriv_variance': first_deriv_variance,
                'second_deriv_variance': second_deriv_variance,
                'skewness': data_skewness,
                'kurtosis': data_kurtosis
            }
            
        except Exception as e:
            app_logger.error(f"Error analyzing curve smoothness: {str(e)}")
            return {
                'smoothness': 0.0,
                'outstanding_points': float('inf'),
                'variance': float('inf'),
                'skewness': 0.0,
                'kurtosis': 0.0
            }
    
    def _detect_outstanding_points(self, data: np.ndarray, first_derivative: np.ndarray) -> int:
        """
        Detect outstanding points (spikes, irregularities) in the curve.
        
        Args:
            data: Curve data
            first_derivative: First derivative of the curve
            
        Returns:
            Number of outstanding points detected
        """
        try:
            if len(data) < 3:
                return 0
            
            # Method 1: Statistical outlier detection
            mean_val = np.mean(data)
            std_val = np.std(data)
            outlier_threshold = 3 * std_val  # 3-sigma rule
            
            outliers = np.abs(data - mean_val) > outlier_threshold
            
            # Method 2: Derivative-based spike detection
            deriv_mean = np.mean(np.abs(first_derivative))
            deriv_std = np.std(first_derivative)
            spike_threshold = deriv_mean + 3 * deriv_std
            
            spikes = np.abs(first_derivative) > spike_threshold
            
            # Method 3: Local variance detection
            window_size = min(10, len(data) // 10)
            if window_size > 1:
                local_variance = np.array([
                    np.var(data[max(0, i-window_size//2):min(len(data), i+window_size//2+1)])
                    for i in range(len(data))
                ])
                local_var_mean = np.mean(local_variance)
                local_var_std = np.std(local_variance)
                variance_outliers = local_variance > (local_var_mean + 2 * local_var_std)
            else:
                variance_outliers = np.zeros(len(data), dtype=bool)
            
            # Combine all detection methods
            outstanding_mask = outliers | spikes | variance_outliers
            outstanding_count = np.sum(outstanding_mask)
            
            return int(outstanding_count)
            
        except Exception as e:
            app_logger.error(f"Error detecting outstanding points: {str(e)}")
            return 0
    
    def _calculate_smoothness_score(self, first_deriv_variance: float, 
                                  second_deriv_variance: float, 
                                  outstanding_points: int) -> float:
        """
        Calculate a smoothness score (0-1) based on curve characteristics.
        
        Args:
            first_deriv_variance: Variance of first derivative
            second_deriv_variance: Variance of second derivative
            outstanding_points: Number of outstanding points
            
        Returns:
            Smoothness score between 0 and 1
        """
        try:
            # Normalize variances (lower is better)
            # Use log scaling to handle large variance values
            first_deriv_score = 1.0 / (1.0 + np.log(1 + first_deriv_variance))
            second_deriv_score = 1.0 / (1.0 + np.log(1 + second_deriv_variance))
            
            # Penalty for outstanding points
            outstanding_penalty = max(0, 1.0 - (outstanding_points / 10.0))  # Penalty increases with points
            
            # Combine scores
            smoothness = (first_deriv_score + second_deriv_score + outstanding_penalty) / 3.0
            
            return min(1.0, max(0.0, smoothness))
            
        except Exception as e:
            app_logger.error(f"Error calculating smoothness score: {str(e)}")
            return 0.0
    
    def _assess_overall_quality(self, smoothness_score: float, outstanding_points: int) -> str:
        """
        Assess overall curve quality based on metrics.
        
        Args:
            smoothness_score: Smoothness score (0-1)
            outstanding_points: Number of outstanding points
            
        Returns:
            Quality assessment string
        """
        if smoothness_score >= 0.8 and outstanding_points <= 2:
            return 'excellent'
        elif smoothness_score >= 0.6 and outstanding_points <= 5:
            return 'good'
        elif smoothness_score >= 0.4 and outstanding_points <= 10:
            return 'fair'
        else:
            return 'poor'
    
    def _find_optimal_starting_point(self, results: Dict[int, Dict]) -> int:
        """
        Find the optimal starting point based on simulation results.
        
        Args:
            results: Dictionary of simulation results
            
        Returns:
            Optimal starting point
        """
        try:
            # Filter out failed tests
            valid_results = {k: v for k, v in results.items() 
                           if 'error' not in v and v['smoothness_score'] > 0}
            
            if not valid_results:
                app_logger.warning("No valid results found, returning default starting point")
                return 35
            
            # Find starting point with best smoothness score and fewest outstanding points
            best_point = max(valid_results.keys(), 
                           key=lambda k: (
                               valid_results[k]['smoothness_score'],
                               -valid_results[k]['outstanding_points']  # Negative for ascending order
                           ))
            
            app_logger.info(f"Optimal starting point found: {best_point}")
            app_logger.info(f"Smoothness score: {valid_results[best_point]['smoothness_score']:.3f}")
            app_logger.info(f"Outstanding points: {valid_results[best_point]['outstanding_points']}")
            
            return best_point
            
        except Exception as e:
            app_logger.error(f"Error finding optimal starting point: {str(e)}")
            return 35  # Default fallback
    
    def _generate_recommendation(self, optimal_point: int, results: Dict[int, Dict]) -> Dict[str, Any]:
        """
        Generate a recommendation based on simulation results.
        
        Args:
            optimal_point: The optimal starting point found
            results: All simulation results
            
        Returns:
            Dictionary containing recommendation details
        """
        try:
            if optimal_point not in results:
                return {
                    'recommended_point': 35,
                    'confidence': 'low',
                    'reason': 'No optimal point found, using default',
                    'improvement': 'No improvement available'
                }
            
            optimal_result = results[optimal_point]
            default_result = results.get(35, {})
            
            # Calculate improvement over default
            if 'smoothness_score' in default_result and 'smoothness_score' in optimal_result:
                smoothness_improvement = (optimal_result['smoothness_score'] - 
                                        default_result['smoothness_score']) * 100
            else:
                smoothness_improvement = 0
            
            if 'outstanding_points' in default_result and 'outstanding_points' in optimal_result:
                points_reduction = default_result['outstanding_points'] - optimal_result['outstanding_points']
            else:
                points_reduction = 0
            
            # Determine confidence level
            if optimal_result['smoothness_score'] >= 0.8 and optimal_result['outstanding_points'] <= 2:
                confidence = 'high'
            elif optimal_result['smoothness_score'] >= 0.6 and optimal_result['outstanding_points'] <= 5:
                confidence = 'medium'
            else:
                confidence = 'low'
            
            # Generate reason
            if optimal_point == 35:
                reason = "Default starting point (35) is already optimal"
            else:
                reason = f"Starting point {optimal_point} provides better curve smoothness"
            
            return {
                'recommended_point': optimal_point,
                'confidence': confidence,
                'reason': reason,
                'smoothness_improvement': f"{smoothness_improvement:.1f}%",
                'outstanding_points_reduction': points_reduction,
                'quality': optimal_result.get('overall_quality', 'unknown')
            }
            
        except Exception as e:
            app_logger.error(f"Error generating recommendation: {str(e)}")
            return {
                'recommended_point': 35,
                'confidence': 'low',
                'reason': 'Error generating recommendation',
                'improvement': 'Unknown'
            }
    
    def get_detailed_analysis(self, start_point: int) -> Optional[Dict[str, Any]]:
        """
        Get detailed analysis for a specific starting point.
        
        Args:
            start_point: The starting point to analyze
            
        Returns:
            Detailed analysis dictionary or None if not available
        """
        if start_point in self.simulation_results.get('results', {}):
            return self.simulation_results['results'][start_point]
        return None
    
    def export_results(self, filename: str) -> bool:
        """
        Export simulation results to a file.
        
        Args:
            filename: Output filename
            
        Returns:
            True if successful, False otherwise
        """
        try:
            import json
            
            # Prepare data for export
            export_data = {
                'simulation_parameters': {
                    'start_point_range': self.start_point_range,
                    'step_size': self.step_size,
                    'data_length': len(self.data)
                },
                'results': self.simulation_results
            }
            
            with open(filename, 'w') as f:
                json.dump(export_data, f, indent=2, default=str)
            
            app_logger.info(f"Simulation results exported to {filename}")
            return True
            
        except Exception as e:
            app_logger.error(f"Error exporting results: {str(e)}")
            return False


class StartingPointSimulationGUI:
    """
    GUI for the starting point simulation feature.
    
    Provides a user interface to run simulations and view results.
    """
    
    def __init__(self, parent, data: np.ndarray, time_data: np.ndarray, params: Dict):
        """
        Initialize the simulation GUI.
        
        Args:
            parent: Parent tkinter widget
            data: Signal data
            time_data: Time data
            params: Analysis parameters
        """
        self.parent = parent
        self.data = data
        self.time_data = time_data
        self.params = params
        self.simulator = None
        self.results = None
        
        self.setup_gui()
    
    def setup_gui(self):
        """Setup the GUI components."""
        # Main frame
        self.frame = ttk.LabelFrame(self.parent, text="Starting Point Simulation")
        self.frame.pack(fill='both', expand=True, padx=5, pady=5)
        
        # Control frame
        control_frame = ttk.Frame(self.frame)
        control_frame.pack(fill='x', padx=5, pady=5)
        
        # Range selection
        ttk.Label(control_frame, text="Starting Point Range:").grid(row=0, column=0, sticky='w', padx=5)
        
        self.start_var = tk.IntVar(value=10)
        self.end_var = tk.IntVar(value=100)
        self.step_var = tk.IntVar(value=5)
        
        ttk.Entry(control_frame, textvariable=self.start_var, width=8).grid(row=0, column=1, padx=2)
        ttk.Label(control_frame, text="to").grid(row=0, column=2)
        ttk.Entry(control_frame, textvariable=self.end_var, width=8).grid(row=0, column=3, padx=2)
        ttk.Label(control_frame, text="step").grid(row=0, column=4)
        ttk.Entry(control_frame, textvariable=self.step_var, width=8).grid(row=0, column=5, padx=2)
        
        # Run button
        self.run_button = ttk.Button(control_frame, text="Run Simulation", 
                                   command=self.run_simulation)
        self.run_button.grid(row=0, column=6, padx=10)
        
        # Progress bar
        self.progress_var = tk.DoubleVar()
        self.progress_bar = ttk.Progressbar(self.frame, variable=self.progress_var, 
                                          mode='determinate')
        self.progress_bar.pack(fill='x', padx=5, pady=5)
        
        # Status label
        self.status_var = tk.StringVar(value="Ready to run simulation")
        ttk.Label(self.frame, textvariable=self.status_var).pack(pady=2)
        
        # Results frame
        self.results_frame = ttk.LabelFrame(self.frame, text="Simulation Results")
        self.results_frame.pack(fill='both', expand=True, padx=5, pady=5)
        
        # Results text
        self.results_text = tk.Text(self.results_frame, height=10, width=60)
        scrollbar = ttk.Scrollbar(self.results_frame, orient="vertical", command=self.results_text.yview)
        self.results_text.configure(yscrollcommand=scrollbar.set)
        
        self.results_text.pack(side="left", fill="both", expand=True, padx=5, pady=5)
        scrollbar.pack(side="right", fill="y")
        
        # Apply button (initially disabled)
        self.apply_button = ttk.Button(self.frame, text="Apply Recommended Starting Point", 
                                     command=self.apply_recommendation, state='disabled')
        self.apply_button.pack(pady=5)
    
    def run_simulation(self):
        """Run the simulation in a separate thread."""
        try:
            # Disable run button
            self.run_button.state(['disabled'])
            self.apply_button.state(['disabled'])
            
            # Update simulator parameters
            start_range = (self.start_var.get(), self.end_var.get())
            step_size = self.step_var.get()
            
            # Create simulator
            self.simulator = StartingPointSimulator(self.data, self.time_data, self.params)
            self.simulator.start_point_range = start_range
            self.simulator.step_size = step_size
            
            # Run simulation in separate thread
            thread = threading.Thread(target=self._run_simulation_thread)
            thread.daemon = True
            thread.start()
            
        except Exception as e:
            app_logger.error(f"Error starting simulation: {str(e)}")
            messagebox.showerror("Error", f"Failed to start simulation: {str(e)}")
            self.run_button.state(['!disabled'])
    
    def _run_simulation_thread(self):
        """Run simulation in background thread."""
        try:
            # Progress callback
            def progress_callback(progress, message):
                self.parent.after(0, lambda: self._update_progress(progress, message))
            
            # Run simulation
            self.results = self.simulator.run_simulation(progress_callback)
            
            # Update UI in main thread
            self.parent.after(0, self._simulation_complete)
            
        except Exception as e:
            app_logger.error(f"Error in simulation thread: {str(e)}")
            self.parent.after(0, lambda: self._simulation_error(str(e)))
    
    def _update_progress(self, progress, message):
        """Update progress bar and status."""
        self.progress_var.set(progress)
        self.status_var.set(message)
        self.parent.update_idletasks()
    
    def _simulation_complete(self):
        """Handle simulation completion."""
        try:
            # Re-enable run button
            self.run_button.state(['!disabled'])
            
            if 'error' in self.results:
                messagebox.showerror("Simulation Error", self.results['error'])
                return
            
            # Display results
            self._display_results()
            
            # Enable apply button
            self.apply_button.state(['!disabled'])
            
            self.status_var.set("Simulation complete")
            
        except Exception as e:
            app_logger.error(f"Error handling simulation completion: {str(e)}")
            self._simulation_error(str(e))
    
    def _simulation_error(self, error_message):
        """Handle simulation error."""
        self.run_button.state(['!disabled'])
        self.status_var.set("Simulation failed")
        messagebox.showerror("Simulation Error", f"Simulation failed: {error_message}")
    
    def _display_results(self):
        """Display simulation results."""
        try:
            self.results_text.delete(1.0, tk.END)
            
            if not self.results or 'optimal_starting_point' not in self.results:
                self.results_text.insert(tk.END, "No results available")
                return
            
            optimal_point = self.results['optimal_starting_point']
            recommendation = self.results.get('recommendation', {})
            
            # Display summary
            summary = f"""SIMULATION RESULTS SUMMARY
========================

Optimal Starting Point: {optimal_point}
Confidence Level: {recommendation.get('confidence', 'unknown').upper()}
Quality: {recommendation.get('quality', 'unknown').upper()}

Reason: {recommendation.get('reason', 'No reason provided')}

Improvements over default (35):
- Smoothness: {recommendation.get('smoothness_improvement', '0%')}
- Outstanding Points Reduction: {recommendation.get('outstanding_points_reduction', 0)}

DETAILED RESULTS
===============
"""
            
            self.results_text.insert(tk.END, summary)
            
            # Display detailed results for each starting point
            if 'results' in self.results:
                for start_point, result in sorted(self.results['results'].items()):
                    if 'error' not in result:
                        smoothness = result.get('smoothness_score', 0)
                        outstanding = result.get('outstanding_points', 0)
                        quality = result.get('overall_quality', 'unknown')
                        
                        line = f"Point {start_point:3d}: Smoothness={smoothness:.3f}, "
                        line += f"Outstanding={outstanding:2d}, Quality={quality}\n"
                        self.results_text.insert(tk.END, line)
            
        except Exception as e:
            app_logger.error(f"Error displaying results: {str(e)}")
            self.results_text.insert(tk.END, f"Error displaying results: {str(e)}")
    
    def apply_recommendation(self):
        """Apply the recommended starting point."""
        try:
            if not self.results or 'optimal_starting_point' not in self.results:
                messagebox.showwarning("No Results", "No simulation results available")
                return
            
            optimal_point = self.results['optimal_starting_point']
            
            # This would typically update the ActionPotentialTab
            # For now, just show a message
            messagebox.showinfo("Recommendation Applied", 
                              f"Recommended starting point {optimal_point} has been applied.\n\n"
                              f"Please run the analysis again to see the improved results.")
            
        except Exception as e:
            app_logger.error(f"Error applying recommendation: {str(e)}")
            messagebox.showerror("Error", f"Failed to apply recommendation: {str(e)}")
    
    def get_recommended_starting_point(self) -> Optional[int]:
        """Get the recommended starting point."""
        if self.results and 'optimal_starting_point' in self.results:
            return self.results['optimal_starting_point']
        return None
