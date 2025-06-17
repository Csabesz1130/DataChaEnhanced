"""
Linear Drift Correction (Egyenes Illesztés) for Purple Curves

This module implements the linear regression-based drift correction that teachers require:
1. Identify descending (depol) and ascending (hyperpol) segments of purple curves
2. Fit linear regression lines to these segments  
3. Subtract the fitted lines from the flat regions to remove linear drift
4. Calculate corrected integrals on the drift-corrected curves

Hungarian term: "egyenes illesztés" = linear fitting/regression
"""

import numpy as np
from scipy import stats
from scipy.signal import find_peaks
import matplotlib.pyplot as plt
from src.utils.logger import app_logger

class LinearDriftCorrector:
    """
    Handles linear drift correction for purple curves using regression analysis.
    """
    
    def __init__(self):
        self.regression_results = {}
        self.corrected_curves = {}
        self.drift_lines = {}
        
    def identify_regression_segments(self, curve_data, curve_times, curve_type='hyperpol'):
        """
        Identify the segments to use for linear regression fitting.
        
        Args:
            curve_data: Purple curve data (hyperpol or depol)
            curve_times: Time values for the curve
            curve_type: 'hyperpol' (ascending) or 'depol' (descending)
            
        Returns:
            dict with segment information
        """
        app_logger.info(f"Identifying regression segments for {curve_type} curve")
        
        try:
            # Find the steepest part of the curve for regression
            if curve_type == 'hyperpol':
                # For hyperpolarization: find ascending (recovery) segment
                segment = self._find_ascending_segment(curve_data, curve_times)
            else:  # depol
                # For depolarization: find descending segment  
                segment = self._find_descending_segment(curve_data, curve_times)
                
            app_logger.debug(f"Found {curve_type} regression segment: indices {segment['start_idx']}-{segment['end_idx']}")
            return segment
            
        except Exception as e:
            app_logger.error(f"Error identifying regression segments: {str(e)}")
            raise
            
    def _find_descending_segment(self, curve_data, curve_times):
        """Find the descending segment for depolarization curve."""
        # Calculate derivative to find steepest descending part
        derivative = np.gradient(curve_data, curve_times)
        
        # Find the most negative derivative (steepest descent)
        min_derivative_idx = np.argmin(derivative)
        
        # Define segment around the steepest descent
        # Typically use 20-30% of the curve length around the peak derivative
        segment_length = max(20, len(curve_data) // 5)  # At least 20 points, or 20% of curve
        
        start_idx = max(0, min_derivative_idx - segment_length // 3)
        end_idx = min(len(curve_data), min_derivative_idx + 2 * segment_length // 3)
        
        return {
            'start_idx': start_idx,
            'end_idx': end_idx,
            'segment_type': 'descending',
            'peak_derivative_idx': min_derivative_idx,
            'max_slope': derivative[min_derivative_idx]
        }
        
    def _find_ascending_segment(self, curve_data, curve_times):
        """Find the ascending segment for hyperpolarization curve."""
        # Calculate derivative to find steepest ascending part
        derivative = np.gradient(curve_data, curve_times)
        
        # Find the most positive derivative (steepest ascent)
        max_derivative_idx = np.argmax(derivative)
        
        # Define segment around the steepest ascent
        segment_length = max(20, len(curve_data) // 5)  # At least 20 points, or 20% of curve
        
        start_idx = max(0, max_derivative_idx - segment_length // 3)
        end_idx = min(len(curve_data), max_derivative_idx + 2 * segment_length // 3)
        
        return {
            'start_idx': start_idx,
            'end_idx': end_idx,
            'segment_type': 'ascending',
            'peak_derivative_idx': max_derivative_idx,
            'max_slope': derivative[max_derivative_idx]
        }
        
    def fit_linear_regression(self, curve_data, curve_times, segment_info):
        """
        Fit linear regression to the identified segment.
        
        Args:
            curve_data: Purple curve data
            curve_times: Time values
            segment_info: Segment information from identify_regression_segments
            
        Returns:
            Regression results dictionary
        """
        try:
            start_idx = segment_info['start_idx']
            end_idx = segment_info['end_idx']
            
            # Extract segment data
            x_segment = curve_times[start_idx:end_idx] * 1000  # Convert to milliseconds
            y_segment = curve_data[start_idx:end_idx]
            
            # Perform linear regression
            slope, intercept, r_value, p_value, std_err = stats.linregress(x_segment, y_segment)
            
            # Calculate fitted line for entire curve
            x_full = curve_times * 1000  # Full time range in milliseconds
            fitted_line = slope * x_full + intercept
            
            # Calculate confidence intervals and quality metrics
            regression_results = {
                'slope': slope,
                'intercept': intercept,
                'r_squared': r_value ** 2,
                'p_value': p_value,
                'std_error': std_err,
                'segment_indices': (start_idx, end_idx),
                'fitted_line': fitted_line,
                'x_segment': x_segment,
                'y_segment': y_segment,
                'segment_type': segment_info['segment_type']
            }
            
            app_logger.info(f"Linear regression completed: slope={slope:.6f}, R²={r_value**2:.4f}")
            return regression_results
            
        except Exception as e:
            app_logger.error(f"Error fitting linear regression: {str(e)}")
            raise
            
    def apply_drift_correction(self, curve_data, curve_times, regression_results, 
                             plateau_start_idx=None, plateau_end_idx=None):
        """
        Apply drift correction by subtracting the fitted line from specified regions.
        
        Args:
            curve_data: Original purple curve data
            curve_times: Time values
            regression_results: Results from fit_linear_regression
            plateau_start_idx: Start of plateau region (if None, auto-detect)
            plateau_end_idx: End of plateau region (if None, auto-detect)
            
        Returns:
            Drift-corrected curve data
        """
        try:
            app_logger.info("Applying drift correction to curve")
            
            # If plateau region not specified, try to auto-detect
            if plateau_start_idx is None or plateau_end_idx is None:
                plateau_start_idx, plateau_end_idx = self._detect_plateau_region(
                    curve_data, regression_results['segment_type']
                )
            
            # Create corrected curve (start with original)
            corrected_curve = curve_data.copy()
            
            # Subtract fitted line only from the plateau region
            fitted_line = regression_results['fitted_line']
            baseline_offset = fitted_line[plateau_start_idx]  # Use plateau start as reference
            
            # Apply correction to plateau region
            for i in range(plateau_start_idx, plateau_end_idx + 1):
                corrected_curve[i] = curve_data[i] - (fitted_line[i] - baseline_offset)
            
            correction_info = {
                'corrected_curve': corrected_curve,
                'original_curve': curve_data,
                'fitted_line': fitted_line,
                'plateau_region': (plateau_start_idx, plateau_end_idx),
                'baseline_offset': baseline_offset,
                'correction_applied': True
            }
            
            app_logger.info(f"Drift correction applied to plateau region: {plateau_start_idx}-{plateau_end_idx}")
            return correction_info
            
        except Exception as e:
            app_logger.error(f"Error applying drift correction: {str(e)}")
            raise
            
    def _detect_plateau_region(self, curve_data, segment_type):
        """
        Auto-detect the plateau (flat) region of the curve.
        
        Args:
            curve_data: Purple curve data
            segment_type: 'ascending' or 'descending'
            
        Returns:
            Tuple of (start_idx, end_idx) for plateau region
        """
        try:
            # Calculate derivative to find flat regions
            derivative = np.gradient(curve_data)
            abs_derivative = np.abs(derivative)
            
            # Find regions with low derivative (flat regions)
            threshold = np.percentile(abs_derivative, 25)  # Bottom 25% of derivatives
            flat_regions = abs_derivative < threshold
            
            # Find the longest continuous flat region
            flat_indices = np.where(flat_regions)[0]
            
            if len(flat_indices) == 0:
                # If no flat region found, use middle 50% of curve
                start_idx = len(curve_data) // 4
                end_idx = 3 * len(curve_data) // 4
            else:
                # Find the longest continuous segment
                diff = np.diff(flat_indices)
                breaks = np.where(diff > 1)[0]
                
                if len(breaks) == 0:
                    # Single continuous flat region
                    start_idx = flat_indices[0]
                    end_idx = flat_indices[-1]
                else:
                    # Multiple flat regions - choose the longest
                    segment_lengths = []
                    segment_starts = [flat_indices[0]]
                    
                    for break_idx in breaks:
                        segment_starts.append(flat_indices[break_idx + 1])
                        segment_lengths.append(break_idx + 1 - len(segment_lengths))
                    
                    segment_lengths.append(len(flat_indices) - sum(segment_lengths))
                    
                    # Choose longest segment
                    longest_segment_idx = np.argmax(segment_lengths)
                    
                    if longest_segment_idx == 0:
                        start_idx = flat_indices[0]
                        end_idx = flat_indices[breaks[0]]
                    elif longest_segment_idx == len(breaks):
                        start_idx = flat_indices[breaks[-1] + 1]
                        end_idx = flat_indices[-1]
                    else:
                        start_idx = flat_indices[breaks[longest_segment_idx - 1] + 1]
                        end_idx = flat_indices[breaks[longest_segment_idx]]
            
            app_logger.debug(f"Auto-detected plateau region: {start_idx}-{end_idx}")
            return start_idx, end_idx
            
        except Exception as e:
            app_logger.error(f"Error detecting plateau region: {str(e)}")
            # Fallback to middle 50% of curve
            start_idx = len(curve_data) // 4
            end_idx = 3 * len(curve_data) // 4
            return start_idx, end_idx
            
    def process_purple_curves(self, processor, manual_plateau_regions=None):
        """
        Complete processing of both purple curves with drift correction.
        
        Args:
            processor: ActionPotentialProcessor instance with purple curves
            manual_plateau_regions: Optional dict with manual plateau definitions
                                  {'hyperpol': (start, end), 'depol': (start, end)}
            
        Returns:
            Dictionary with corrected curves and analysis results
        """
        try:
            app_logger.info("Starting complete purple curve drift correction")
            
            results = {
                'hyperpol': {},
                'depol': {},
                'summary': {}
            }
            
            # Process hyperpolarization curve
            if hasattr(processor, 'modified_hyperpol') and processor.modified_hyperpol is not None:
                app_logger.info("Processing hyperpolarization curve")
                
                # Identify regression segment
                hyperpol_segment = self.identify_regression_segments(
                    processor.modified_hyperpol, 
                    processor.modified_hyperpol_times, 
                    'hyperpol'
                )
                
                # Fit linear regression
                hyperpol_regression = self.fit_linear_regression(
                    processor.modified_hyperpol,
                    processor.modified_hyperpol_times,
                    hyperpol_segment
                )
                
                # Apply drift correction
                plateau_region = manual_plateau_regions.get('hyperpol') if manual_plateau_regions else (None, None)
                hyperpol_correction = self.apply_drift_correction(
                    processor.modified_hyperpol,
                    processor.modified_hyperpol_times,
                    hyperpol_regression,
                    plateau_region[0] if plateau_region[0] else None,
                    plateau_region[1] if plateau_region[1] else None
                )
                
                results['hyperpol'] = {
                    'segment_info': hyperpol_segment,
                    'regression_results': hyperpol_regression,
                    'correction_info': hyperpol_correction,
                    'original_times': processor.modified_hyperpol_times
                }
            
            # Process depolarization curve  
            if hasattr(processor, 'modified_depol') and processor.modified_depol is not None:
                app_logger.info("Processing depolarization curve")
                
                # Identify regression segment
                depol_segment = self.identify_regression_segments(
                    processor.modified_depol,
                    processor.modified_depol_times,
                    'depol'
                )
                
                # Fit linear regression
                depol_regression = self.fit_linear_regression(
                    processor.modified_depol,
                    processor.modified_depol_times,
                    depol_segment
                )
                
                # Apply drift correction
                plateau_region = manual_plateau_regions.get('depol') if manual_plateau_regions else (None, None)
                depol_correction = self.apply_drift_correction(
                    processor.modified_depol,
                    processor.modified_depol_times,
                    depol_regression,
                    plateau_region[0] if plateau_region[0] else None,
                    plateau_region[1] if plateau_region[1] else None
                )
                
                results['depol'] = {
                    'segment_info': depol_segment,
                    'regression_results': depol_regression,
                    'correction_info': depol_correction,
                    'original_times': processor.modified_depol_times
                }
            
            # Generate summary
            results['summary'] = self._generate_correction_summary(results)
            
            app_logger.info("Purple curve drift correction completed successfully")
            return results
            
        except Exception as e:
            app_logger.error(f"Error processing purple curves: {str(e)}")
            raise
            
    def _generate_correction_summary(self, results):
        """Generate summary of drift correction results."""
        summary = {
            'timestamp': np.datetime64('now').astype(str),
            'curves_processed': [],
            'regression_quality': {},
            'corrections_applied': {}
        }
        
        for curve_type in ['hyperpol', 'depol']:
            if curve_type in results and results[curve_type]:
                summary['curves_processed'].append(curve_type)
                
                # Regression quality
                regression = results[curve_type]['regression_results']
                summary['regression_quality'][curve_type] = {
                    'r_squared': regression['r_squared'],
                    'slope': regression['slope'],
                    'p_value': regression['p_value']
                }
                
                # Correction info
                correction = results[curve_type]['correction_info']
                summary['corrections_applied'][curve_type] = {
                    'plateau_region': correction['plateau_region'],
                    'baseline_offset': correction['baseline_offset']
                }
        
        return summary
        
    def calculate_corrected_integrals(self, results, integration_ranges):
        """
        Calculate integrals using the drift-corrected curves.
        
        Args:
            results: Results from process_purple_curves
            integration_ranges: Dict with 'hyperpol' and 'depol' range tuples
            
        Returns:
            Dictionary with corrected integral values
        """
        try:
            app_logger.info("Calculating integrals on drift-corrected curves")
            
            corrected_integrals = {}
            
            for curve_type in ['hyperpol', 'depol']:
                if curve_type in results and results[curve_type]:
                    correction_info = results[curve_type]['correction_info']
                    times = results[curve_type]['original_times']
                    corrected_curve = correction_info['corrected_curve']
                    
                    # Get integration range
                    if curve_type in integration_ranges:
                        start_idx, end_idx = integration_ranges[curve_type]
                        
                        # Ensure indices are valid
                        start_idx = max(0, min(start_idx, len(corrected_curve) - 1))
                        end_idx = max(start_idx + 1, min(end_idx, len(corrected_curve)))
                        
                        # Calculate integral on corrected curve
                        time_segment = times[start_idx:end_idx] * 1000  # Convert to ms
                        curve_segment = corrected_curve[start_idx:end_idx]
                        
                        corrected_integral = np.trapz(curve_segment, time_segment)
                        
                        # Also calculate original integral for comparison
                        original_curve = correction_info['original_curve']
                        original_integral = np.trapz(original_curve[start_idx:end_idx], time_segment)
                        
                        corrected_integrals[curve_type] = {
                            'corrected_integral': corrected_integral,
                            'original_integral': original_integral,
                            'correction_difference': corrected_integral - original_integral,
                            'integration_range': (start_idx, end_idx)
                        }
                        
                        app_logger.info(f"{curve_type} - Original: {original_integral:.6f}, "
                                      f"Corrected: {corrected_integral:.6f}, "
                                      f"Difference: {corrected_integral - original_integral:.6f}")
            
            return corrected_integrals
            
        except Exception as e:
            app_logger.error(f"Error calculating corrected integrals: {str(e)}")
            raise
            
    def plot_correction_analysis(self, results, save_path=None):
        """
        Create visualization plots showing the drift correction process.
        
        Args:
            results: Results from process_purple_curves
            save_path: Optional path to save the plot
            
        Returns:
            matplotlib figure object
        """
        try:
            fig, axes = plt.subplots(2, 2, figsize=(15, 10))
            fig.suptitle('Linear Drift Correction Analysis (Egyenes Illesztés)', fontsize=16)
            
            for i, curve_type in enumerate(['hyperpol', 'depol']):
                if curve_type in results and results[curve_type]:
                    data = results[curve_type]
                    times = data['original_times'] * 1000  # Convert to ms
                    original_curve = data['correction_info']['original_curve']
                    corrected_curve = data['correction_info']['corrected_curve']
                    fitted_line = data['regression_results']['fitted_line']
                    segment_indices = data['regression_results']['segment_indices']
                    plateau_region = data['correction_info']['plateau_region']
                    
                    # Original vs corrected curves
                    ax1 = axes[i, 0]
                    ax1.plot(times, original_curve, 'b-', label='Original', alpha=0.7)
                    ax1.plot(times, corrected_curve, 'r-', label='Drift Corrected', linewidth=2)
                    ax1.plot(times, fitted_line, 'g--', label='Fitted Line', alpha=0.8)
                    
                    # Highlight regression segment
                    start_idx, end_idx = segment_indices
                    ax1.axvspan(times[start_idx], times[end_idx], alpha=0.2, color='green', 
                               label='Regression Segment')
                    
                    # Highlight plateau region
                    plateau_start, plateau_end = plateau_region
                    ax1.axvspan(times[plateau_start], times[plateau_end], alpha=0.2, color='red',
                               label='Correction Region')
                    
                    ax1.set_title(f'{curve_type.capitalize()} Curve Correction')
                    ax1.set_xlabel('Time (ms)')
                    ax1.set_ylabel('Current (pA)')
                    ax1.legend()
                    ax1.grid(True, alpha=0.3)
                    
                    # Regression quality plot
                    ax2 = axes[i, 1]
                    x_segment = data['regression_results']['x_segment']
                    y_segment = data['regression_results']['y_segment']
                    slope = data['regression_results']['slope']
                    intercept = data['regression_results']['intercept']
                    r_squared = data['regression_results']['r_squared']
                    
                    ax2.scatter(x_segment, y_segment, alpha=0.6, color='blue', label='Data Points')
                    fit_line = slope * x_segment + intercept
                    ax2.plot(x_segment, fit_line, 'r-', label=f'Linear Fit (R²={r_squared:.4f})')
                    
                    ax2.set_title(f'{curve_type.capitalize()} Regression Quality')
                    ax2.set_xlabel('Time (ms)')
                    ax2.set_ylabel('Current (pA)')
                    ax2.legend()
                    ax2.grid(True, alpha=0.3)
                    
                    # Add regression equation as text
                    equation = f'y = {slope:.6f}x + {intercept:.3f}'
                    ax2.text(0.05, 0.95, equation, transform=ax2.transAxes, 
                            fontsize=10, bbox=dict(boxstyle="round", facecolor='wheat', alpha=0.8))
            
            plt.tight_layout()
            
            if save_path:
                plt.savefig(save_path, dpi=300, bbox_inches='tight')
                app_logger.info(f"Drift correction plot saved to: {save_path}")
            
            return fig
            
        except Exception as e:
            app_logger.error(f"Error creating correction plot: {str(e)}")
            raise
            
    def export_correction_report(self, results, corrected_integrals, filename=None):
        """
        Export detailed drift correction report to Excel.
        
        Args:
            results: Results from process_purple_curves
            corrected_integrals: Results from calculate_corrected_integrals
            filename: Optional filename for export
        """
        try:
            if not filename:
                from tkinter import filedialog
                filename = filedialog.asksaveasfilename(
                    defaultextension=".xlsx",
                    filetypes=[("Excel files", "*.xlsx"), ("All files", "*.*")],
                    title="Export Drift Correction Report"
                )
            
            if not filename:
                return None
                
            import pandas as pd
            
            with pd.ExcelWriter(filename, engine='openpyxl') as writer:
                # Summary sheet
                summary_data = []
                summary_data.append(['Linear Drift Correction Report (Egyenes Illesztés)', ''])
                summary_data.append(['Generated', np.datetime64('now').astype(str)])
                summary_data.append(['', ''])
                
                for curve_type in ['hyperpol', 'depol']:
                    if curve_type in results:
                        regression = results[curve_type]['regression_results']
                        summary_data.extend([
                            [f'{curve_type.capitalize()} Results', ''],
                            ['Slope (pA/ms)', f"{regression['slope']:.8f}"],
                            ['Intercept (pA)', f"{regression['intercept']:.6f}"],
                            ['R-squared', f"{regression['r_squared']:.6f}"],
                            ['P-value', f"{regression['p_value']:.8f}"],
                            ['', '']
                        ])
                        
                        if curve_type in corrected_integrals:
                            integrals = corrected_integrals[curve_type]
                            summary_data.extend([
                                ['Original Integral (pC)', f"{integrals['original_integral']:.6f}"],
                                ['Corrected Integral (pC)', f"{integrals['corrected_integral']:.6f}"],
                                ['Correction Difference (pC)', f"{integrals['correction_difference']:.6f}"],
                                ['', '']
                            ])
                
                df_summary = pd.DataFrame(summary_data, columns=['Parameter', 'Value'])
                df_summary.to_excel(writer, sheet_name='Summary', index=False)
                
                # Detailed regression data
                for curve_type in ['hyperpol', 'depol']:
                    if curve_type in results:
                        regression = results[curve_type]['regression_results']
                        x_segment = regression['x_segment']
                        y_segment = regression['y_segment']
                        
                        regression_data = pd.DataFrame({
                            'Time_ms': x_segment,
                            'Current_pA': y_segment,
                            'Fitted_pA': regression['slope'] * x_segment + regression['intercept']
                        })
                        
                        regression_data.to_excel(writer, sheet_name=f'{curve_type}_regression', index=False)
            
            app_logger.info(f"Drift correction report exported to: {filename}")
            return filename
            
        except Exception as e:
            app_logger.error(f"Error exporting correction report: {str(e)}")
            raise


def add_drift_correction_to_action_potential_tab(action_potential_tab):
    """
    Add drift correction functionality to the existing ActionPotentialTab.
    
    Args:
        action_potential_tab: Instance of ActionPotentialTab
    """
    try:
        # Add drift corrector instance
        action_potential_tab.drift_corrector = LinearDriftCorrector()
        
        # Add method to apply drift correction
        def apply_drift_correction():
            try:
                if not hasattr(action_potential_tab.parent.master, 'action_potential_processor'):
                    from tkinter import messagebox
                    messagebox.showwarning("No Data", "Please run action potential analysis first")
                    return
                
                processor = action_potential_tab.parent.master.action_potential_processor
                
                # Get manual plateau regions if user has set them
                # This could be integrated with existing range selectors
                manual_regions = None  # Could be expanded to use UI controls
                
                # Process curves with drift correction
                correction_results = action_potential_tab.drift_corrector.process_purple_curves(
                    processor, manual_regions
                )
                
                # Calculate corrected integrals
                current_ranges = action_potential_tab.get_intervals().get('integration_ranges', {})
                integration_ranges = {
                    'hyperpol': (current_ranges.get('hyperpol', {}).get('start', 0),
                                current_ranges.get('hyperpol', {}).get('end', 200)),
                    'depol': (current_ranges.get('depol', {}).get('start', 0),
                             current_ranges.get('depol', {}).get('end', 200))
                }
                
                corrected_integrals = action_potential_tab.drift_corrector.calculate_corrected_integrals(
                    correction_results, integration_ranges
                )
                
                # Show results
                from tkinter import messagebox
                result_msg = "Drift Correction Applied!\n\n"
                for curve_type, data in corrected_integrals.items():
                    result_msg += f"{curve_type.capitalize()}:\n"
                    result_msg += f"  Original: {data['original_integral']:.6f} pC\n"
                    result_msg += f"  Corrected: {data['corrected_integral']:.6f} pC\n"
                    result_msg += f"  Difference: {data['correction_difference']:.6f} pC\n\n"
                
                messagebox.showinfo("Drift Correction Results", result_msg)
                
                # Store results for later use
                action_potential_tab.drift_correction_results = correction_results
                action_potential_tab.corrected_integrals = corrected_integrals
                
            except Exception as e:
                from tkinter import messagebox
                messagebox.showerror("Error", f"Drift correction failed: {str(e)}")
        
        def export_drift_correction_report():
            try:
                if hasattr(action_potential_tab, 'drift_correction_results'):
                    filename = action_potential_tab.drift_corrector.export_correction_report(
                        action_potential_tab.drift_correction_results,
                        action_potential_tab.corrected_integrals
                    )
                    if filename:
                        from tkinter import messagebox
                        messagebox.showinfo("Export", f"Report exported to:\n{filename}")
                else:
                    from tkinter import messagebox
                    messagebox.showwarning("No Data", "Please apply drift correction first")
            except Exception as e:
                from tkinter import messagebox
                messagebox.showerror("Error", f"Export failed: {str(e)}")
        
        # Add the methods to the tab
        action_potential_tab.apply_drift_correction = apply_drift_correction
        action_potential_tab.export_drift_correction_report = export_drift_correction_report
        
        app_logger.info("Drift correction functionality added to Action Potential tab")
        
    except Exception as e:
        app_logger.error(f"Error adding drift correction to tab: {str(e)}")
        raise