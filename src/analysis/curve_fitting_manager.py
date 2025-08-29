"""
Fixed Curve Fitting Manager for DataChaEnhanced
===============================================
Location: src/analysis/curve_fitting_manager.py

This module provides interactive curve fitting functionality for purple curves.
Key fixes:
- Improved point detection on purple curves
- Better event handling
- Proper coordinate conversion
- Enhanced visual feedback
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
from scipy.stats import linregress
from typing import Dict, List, Tuple, Optional, Any
import logging

logger = logging.getLogger(__name__)

class CurveFittingManager:
    """Manages manual curve fitting operations for purple curves."""
    
    def __init__(self, figure, ax):
        """Initialize the curve fitting manager."""
        self.fig = figure
        self.ax = ax
        
        # Point selection states
        self.selection_mode = None  # 'linear_hyperpol', 'linear_depol', 'exp_hyperpol', 'exp_depol'
        self.selected_points = {
            'hyperpol': {'linear_points': [], 'exp_point': None},
            'depol': {'linear_points': [], 'exp_point': None}
        }
        
        # Fitted parameters
        self.fitted_curves = {
            'hyperpol': {
                'linear_params': None,
                'linear_curve': None,
                'exp_params': None,
                'exp_curve': None,
                'r_squared_linear': None,
                'r_squared_exp': None
            },
            'depol': {
                'linear_params': None,
                'linear_curve': None,
                'exp_params': None,
                'exp_curve': None,
                'r_squared_linear': None,
                'r_squared_exp': None
            }
        }
        
        # Plot elements for cleanup
        self.plot_elements = {
            'selected_points': [],
            'linear_fits': [],
            'exp_fits': []
        }
        
        # Data references
        self.curve_data = {
            'hyperpol': {'data': None, 'times': None},
            'depol': {'data': None, 'times': None}
        }
        
        # Callbacks
        self.on_fit_complete = None
        
        # Event handling
        self.click_cid = None
        self.is_active = False
        
        logger.info("CurveFittingManager initialized")
    
    def set_curve_data(self, curve_type: str, data: np.ndarray, times: np.ndarray):
        """Set curve data for fitting."""
        if curve_type not in ['hyperpol', 'depol']:
            raise ValueError("curve_type must be 'hyperpol' or 'depol'")
        
        self.curve_data[curve_type]['data'] = np.array(data)
        self.curve_data[curve_type]['times'] = np.array(times)
        logger.info(f"Set {curve_type} data: {len(data)} points")
    
    def start_linear_selection(self, curve_type: str):
        """Start selecting two points for linear fitting."""
        if curve_type not in ['hyperpol', 'depol']:
            logger.error(f"Invalid curve type: {curve_type}")
            return False
        
        # Check if we have data
        if self.curve_data[curve_type]['data'] is None:
            logger.warning(f"No data available for {curve_type}")
            return False
        
        self.selection_mode = f'linear_{curve_type}'
        self.selected_points[curve_type]['linear_points'] = []
        self._clear_selection_markers()
        self._connect_events()
        self.is_active = True
        
        logger.info(f"Started linear point selection for {curve_type}")
        logger.info(f"Click 2 points on the {curve_type} purple curve")
        return True
    
    def start_exp_selection(self, curve_type: str):
        """Start selecting third point for exponential fitting."""
        if curve_type not in ['hyperpol', 'depol']:
            logger.error(f"Invalid curve type: {curve_type}")
            return False
        
        # Check if linear fit exists
        if not self.fitted_curves[curve_type]['linear_params']:
            logger.warning(f"Linear fit required before exponential fit for {curve_type}")
            return False
        
        # Check if we have data
        if self.curve_data[curve_type]['data'] is None:
            logger.warning(f"No data available for {curve_type}")
            return False
        
        self.selection_mode = f'exp_{curve_type}'
        self.selected_points[curve_type]['exp_point'] = None
        self._connect_events()
        self.is_active = True
        
        logger.info(f"Started exponential point selection for {curve_type}")
        logger.info(f"Click 1 point on the {curve_type} purple curve for exponential fit start")
        return True
    
    def _connect_events(self):
        """Connect matplotlib click events."""
        if self.click_cid:
            self.fig.canvas.mpl_disconnect(self.click_cid)
        
        self.click_cid = self.fig.canvas.mpl_connect('button_press_event', self._on_click)
        logger.debug("Event handler connected")
    
    def _disconnect_events(self):
        """Disconnect matplotlib events."""
        if self.click_cid:
            self.fig.canvas.mpl_disconnect(self.click_cid)
            self.click_cid = None
        self.is_active = False
        logger.debug("Event handler disconnected")
    
    def _on_click(self, event):
        """Handle mouse click events for point selection."""
        if not self.is_active or not self.selection_mode or not event.inaxes:
            return
        
        if event.inaxes != self.ax:
            return
        
        logger.debug(f"Click detected at ({event.xdata:.1f}, {event.ydata:.1f})")
        
        # Extract curve type from selection mode
        curve_type = self.selection_mode.split('_')[1]  # 'hyperpol' or 'depol'
        
        # Find nearest point on the specified curve
        point_info = self._find_nearest_point(event.xdata, event.ydata, curve_type)
        if not point_info:
            logger.debug("No nearby point found on curve")
            return
        
        index, time, value = point_info
        logger.info(f"Selected point: index={index}, time={time:.3f}s, value={value:.2f}pA")
        
        # Handle different selection modes
        if 'linear' in self.selection_mode:
            points = self.selected_points[curve_type]['linear_points']
            if len(points) < 2:
                points.append({'index': index, 'time': time, 'value': value})
                self._add_selection_marker(time * 1000, value, f'P{len(points)}', 'red')
                
                if len(points) == 2:
                    # Perform linear fit
                    self._fit_linear(curve_type)
                    self._stop_selection()
                    if self.on_fit_complete:
                        self.on_fit_complete(curve_type, 'linear')
        
        elif 'exp' in self.selection_mode:
            self.selected_points[curve_type]['exp_point'] = {
                'index': index, 'time': time, 'value': value
            }
            self._add_selection_marker(time * 1000, value, 'P3', 'blue')
            
            # Perform exponential fit
            self._fit_exponential(curve_type)
            self._stop_selection()
            if self.on_fit_complete:
                self.on_fit_complete(curve_type, 'exponential')
    
    def _find_nearest_point(self, x_click: float, y_click: float, curve_type: str) -> Optional[Tuple[int, float, float]]:
        """Find the nearest point on the specified curve to the clicked location."""
        data = self.curve_data[curve_type]['data']
        times = self.curve_data[curve_type]['times']
        
        if data is None or times is None or len(data) == 0:
            return None
        
        # Convert click x-coordinate from milliseconds to seconds
        x_click_seconds = x_click / 1000.0
        
        # Calculate distances with proper scaling
        time_range = np.ptp(times)  # Peak-to-peak (max - min)
        data_range = np.ptp(data)
        
        if time_range == 0 or data_range == 0:
            return None
        
        # Normalized distances
        time_distances = (times - x_click_seconds) / time_range
        data_distances = (data - y_click) / data_range
        
        # Combined distance
        distances = np.sqrt(time_distances**2 + data_distances**2)
        
        # Find closest point
        min_idx = np.argmin(distances)
        min_distance = distances[min_idx]
        
        # Check if reasonably close (adjust threshold as needed)
        threshold = 0.1  # 10% normalized distance
        if min_distance > threshold:
            logger.debug(f"Closest point distance {min_distance:.3f} exceeds threshold {threshold}")
            return None
        
        return min_idx, times[min_idx], data[min_idx]
    
    def _add_selection_marker(self, time_ms: float, value: float, label: str, color: str):
        """Add a visual marker for selected points."""
        # Add point marker
        marker = self.ax.scatter(time_ms, value, c=color, s=80, 
                               marker='o', edgecolors='white', linewidth=2, zorder=10)
        
        # Add text label
        text = self.ax.annotate(label, (time_ms, value), 
                               xytext=(5, 5), textcoords='offset points',
                               fontsize=10, color=color, fontweight='bold',
                               bbox=dict(boxstyle="round,pad=0.3", facecolor='white', alpha=0.8))
        
        self.plot_elements['selected_points'].extend([marker, text])
        self.fig.canvas.draw_idle()
        logger.debug(f"Added marker {label} at ({time_ms:.1f}ms, {value:.2f}pA)")
    
    def _clear_selection_markers(self):
        """Clear all selection markers from the plot."""
        for element in self.plot_elements['selected_points']:
            try:
                element.remove()
            except:
                pass
        self.plot_elements['selected_points'] = []
        self.fig.canvas.draw_idle()
    
    def _fit_linear(self, curve_type: str):
        """Perform linear fitting between two selected points."""
        points = self.selected_points[curve_type]['linear_points']
        if len(points) != 2:
            logger.error(f"Need exactly 2 points for linear fit, got {len(points)}")
            return
        
        data = self.curve_data[curve_type]['data']
        times = self.curve_data[curve_type]['times']
        
        # Get indices and ensure proper order
        idx1, idx2 = points[0]['index'], points[1]['index']
        if idx1 > idx2:
            idx1, idx2 = idx2, idx1
        
        # Extract segment for fitting
        time_segment = times[idx1:idx2+1]
        data_segment = data[idx1:idx2+1]
        
        if len(time_segment) < 2:
            logger.error("Not enough points for linear regression")
            return
        
        # Perform linear regression
        slope, intercept, r_value, p_value, std_err = linregress(time_segment, data_segment)
        r_squared = r_value ** 2
        
        # Store results
        self.fitted_curves[curve_type]['linear_params'] = {
            'slope': slope,
            'intercept': intercept,
            'start_idx': idx1,
            'end_idx': idx2,
            'start_time': time_segment[0],
            'end_time': time_segment[-1]
        }
        self.fitted_curves[curve_type]['r_squared_linear'] = r_squared
        
        # Generate fitted curve for plotting
        fitted_data = slope * time_segment + intercept
        self.fitted_curves[curve_type]['linear_curve'] = {
            'times': time_segment,
            'data': fitted_data
        }
        
        # Plot the fitted line
        self._plot_linear_fit(curve_type, time_segment, fitted_data)
        
        logger.info(f"Linear fit for {curve_type}:")
        logger.info(f"  Equation: y = {slope:.6f}x + {intercept:.6f}")
        logger.info(f"  R² = {r_squared:.4f}")
        logger.info(f"  Slope: {slope:.6f} pA/s")
        logger.info(f"  Intercept: {intercept:.6f} pA")
    
    def _fit_exponential(self, curve_type: str):
        """Perform exponential fitting from the selected point to the end."""
        exp_point = self.selected_points[curve_type]['exp_point']
        if not exp_point:
            logger.error("No exponential start point selected")
            return
        
        linear_params = self.fitted_curves[curve_type]['linear_params']
        if not linear_params:
            logger.error("Linear fit required before exponential fit")
            return
        
        data = self.curve_data[curve_type]['data']
        times = self.curve_data[curve_type]['times']
        
        # Get data from exp point to end
        start_idx = exp_point['index']
        time_segment = times[start_idx:]
        data_segment = data[start_idx:]
        
        if len(time_segment) < 3:
            logger.error("Not enough points for exponential fit")
            return
        
        # Remove linear trend
        linear_trend = linear_params['slope'] * time_segment + linear_params['intercept']
        detrended_data = data_segment - linear_trend
        
        # Shift time to start from 0
        time_shifted = time_segment - time_segment[0]
        
        # Choose appropriate exponential model
        if curve_type == 'hyperpol':
            # Decay model: A * exp(-t/tau)
            def exp_func(t, A, tau):
                return A * np.exp(-t / tau)
            
            # Initial guesses
            A_guess = detrended_data[0]
            tau_guess = (time_shifted[-1] - time_shifted[0]) / 3
        else:
            # Rise model: A * (1 - exp(-t/tau))
            def exp_func(t, A, tau):
                return A * (1 - np.exp(-t / tau))
            
            # Initial guesses
            A_guess = detrended_data[-1] - detrended_data[0]
            tau_guess = (time_shifted[-1] - time_shifted[0]) / 3
        
        try:
            # Perform exponential fitting
            popt, pcov = curve_fit(
                exp_func,
                time_shifted,
                detrended_data,
                p0=[A_guess, tau_guess],
                maxfev=2000
            )
            
            A_fit, tau_fit = popt
            
            # Calculate R-squared
            y_pred = exp_func(time_shifted, A_fit, tau_fit)
            ss_res = np.sum((detrended_data - y_pred) ** 2)
            ss_tot = np.sum((detrended_data - np.mean(detrended_data)) ** 2)
            r_squared = 1 - (ss_res / ss_tot) if ss_tot != 0 else 0
            
            # Store results
            self.fitted_curves[curve_type]['exp_params'] = {
                'A': A_fit,
                'tau': tau_fit,
                'start_idx': start_idx,
                'start_time': time_segment[0],
                'model_type': 'decay' if curve_type == 'hyperpol' else 'rise'
            }
            self.fitted_curves[curve_type]['r_squared_exp'] = r_squared
            
            # Generate fitted curve (with linear trend added back)
            fitted_exp = exp_func(time_shifted, A_fit, tau_fit)
            fitted_total = fitted_exp + linear_trend
            
            self.fitted_curves[curve_type]['exp_curve'] = {
                'times': time_segment,
                'data': fitted_total
            }
            
            # Plot the fitted curve
            self._plot_exp_fit(curve_type, time_segment, fitted_total)
            
            logger.info(f"Exponential fit for {curve_type}:")
            logger.info(f"  A = {A_fit:.6f} pA")
            logger.info(f"  τ = {tau_fit:.6f} s ({tau_fit*1000:.3f} ms)")
            logger.info(f"  R² = {r_squared:.4f}")
            logger.info(f"  Model: {'decay' if curve_type == 'hyperpol' else 'rise'}")
            
        except Exception as e:
            logger.error(f"Exponential fitting failed for {curve_type}: {str(e)}")
    
    def _plot_linear_fit(self, curve_type: str, times: np.ndarray, fitted_data: np.ndarray):
        """Plot linear fit on the axes."""
        color = 'darkblue' if curve_type == 'hyperpol' else 'darkred'
        line = self.ax.plot(times * 1000, fitted_data, '--',
                           color=color, linewidth=2, alpha=0.8,
                           label=f'{curve_type.title()} Linear')[0]
        self.plot_elements['linear_fits'].append(line)
        self.ax.legend()
        self.fig.canvas.draw_idle()
    
    def _plot_exp_fit(self, curve_type: str, times: np.ndarray, fitted_data: np.ndarray):
        """Plot exponential fit on the axes."""
        color = 'navy' if curve_type == 'hyperpol' else 'maroon'
        line = self.ax.plot(times * 1000, fitted_data, ':',
                           color=color, linewidth=2, alpha=0.8,
                           label=f'{curve_type.title()} Exp')[0]
        self.plot_elements['exp_fits'].append(line)
        self.ax.legend()
        self.fig.canvas.draw_idle()
    
    def _stop_selection(self):
        """Stop point selection mode."""
        self._disconnect_events()
        self.selection_mode = None
        logger.debug("Point selection stopped")
    
    def apply_linear_correction(self, curve_type: str, operation: str = 'subtract'):
        """Apply linear correction to the entire curve."""
        linear_params = self.fitted_curves[curve_type]['linear_params']
        if not linear_params:
            logger.warning(f"No linear parameters for {curve_type}")
            return None
        
        data = self.curve_data[curve_type]['data']
        times = self.curve_data[curve_type]['times']
        
        # Calculate linear trend for entire curve
        linear_trend = linear_params['slope'] * times + linear_params['intercept']
        
        # Apply correction
        if operation == 'subtract':
            corrected_data = data - linear_trend
        else:  # add
            corrected_data = data + linear_trend
        
        logger.info(f"Applied linear {operation} correction to {curve_type}")
        
        return {
            'times': times,
            'original': data,
            'trend': linear_trend,
            'corrected': corrected_data,
            'operation': operation
        }
    
    def clear_fits(self, curve_type: Optional[str] = None):
        """Clear fitted curves from plot and data."""
        if curve_type is None:
            curve_types = ['hyperpol', 'depol']
        else:
            curve_types = [curve_type]
        
        for ctype in curve_types:
            self.fitted_curves[ctype] = {
                'linear_params': None,
                'linear_curve': None,
                'exp_params': None,
                'exp_curve': None,
                'r_squared_linear': None,
                'r_squared_exp': None
            }
            self.selected_points[ctype] = {
                'linear_points': [],
                'exp_point': None
            }
        
        # Clear plot elements
        for element_list in self.plot_elements.values():
            for element in element_list:
                try:
                    element.remove()
                except:
                    pass
            element_list.clear()
        
        self._stop_selection()
        self.ax.legend()
        self.fig.canvas.draw_idle()
        logger.info("Cleared all fits and selections")
    
    def get_fitting_results(self) -> Dict:
        """Get all fitting results in a structured format."""
        results = {}
        
        for curve_type in ['hyperpol', 'depol']:
            curve_results = {}
            
            # Linear fit results
            linear_params = self.fitted_curves[curve_type]['linear_params']
            if linear_params:
                curve_results['linear'] = {
                    'equation': f"y = {linear_params['slope']:.6f}x + {linear_params['intercept']:.6f}",
                    'slope': linear_params['slope'],
                    'intercept': linear_params['intercept'],
                    'r_squared': self.fitted_curves[curve_type]['r_squared_linear']
                }
            
            # Exponential fit results
            exp_params = self.fitted_curves[curve_type]['exp_params']
            if exp_params:
                model = exp_params['model_type']
                if model == 'decay':
                    equation = f"y = {exp_params['A']:.6f} * exp(-t/{exp_params['tau']:.6f})"
                else:
                    equation = f"y = {exp_params['A']:.6f} * (1 - exp(-t/{exp_params['tau']:.6f}))"
                
                curve_results['exponential'] = {
                    'equation': equation,
                    'A': exp_params['A'],
                    'tau': exp_params['tau'],
                    'tau_ms': exp_params['tau'] * 1000,
                    'r_squared': self.fitted_curves[curve_type]['r_squared_exp'],
                    'model_type': model
                }
            
            if curve_results:
                results[curve_type] = curve_results
        
        return results
    
    def is_linear_fit_complete(self, curve_type: str) -> bool:
        """Check if linear fit is complete for the given curve type."""
        return self.fitted_curves[curve_type]['linear_params'] is not None
    
    def is_exp_fit_complete(self, curve_type: str) -> bool:
        """Check if exponential fit is complete for the given curve type."""
        return self.fitted_curves[curve_type]['exp_params'] is not None