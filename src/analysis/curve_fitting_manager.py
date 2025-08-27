"""
Manual Curve Fitting Manager for DataChaEnhanced
================================================
Location: src/analysis/curve_fitting_manager.py

This module provides interactive curve fitting functionality for purple curves.
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
from scipy.stats import linregress
from typing import Dict, List, Tuple, Optional
import logging

logger = logging.getLogger(__name__)

class CurveFittingManager:
    """Manages manual curve fitting operations for purple curves."""
    
    def __init__(self, figure, ax):
        """Initialize the curve fitting manager."""
        self.fig = figure
        self.ax = ax
        
        # Point selection states
        self.selection_mode = None
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
        
        # Plot elements
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
        
        # Connect events
        self.click_cid = None
        self._connect_events()
    
    def _connect_events(self):
        """Connect matplotlib click events."""
        if self.click_cid:
            self.fig.canvas.mpl_disconnect(self.click_cid)
        self.click_cid = self.fig.canvas.mpl_connect('button_press_event', self._on_click)
    
    def set_curve_data(self, curve_type: str, data: np.ndarray, times: np.ndarray):
        """Set curve data for fitting."""
        if curve_type not in ['hyperpol', 'depol']:
            raise ValueError("curve_type must be 'hyperpol' or 'depol'")
        
        self.curve_data[curve_type]['data'] = np.array(data)
        self.curve_data[curve_type]['times'] = np.array(times)
        logger.info(f"Set {curve_type} data: {len(data)} points")
    
    def start_linear_selection(self, curve_type: str):
        """Start selecting two points for linear fitting."""
        self.selection_mode = f'linear_{curve_type}'
        self.selected_points[curve_type]['linear_points'] = []
        self._clear_selection_markers()
        logger.info(f"Started linear point selection for {curve_type}")
    
    def start_exp_selection(self, curve_type: str):
        """Start selecting third point for exponential fitting."""
        self.selection_mode = f'exp_{curve_type}'
        self.selected_points[curve_type]['exp_point'] = None
        logger.info(f"Started exponential point selection for {curve_type}")
    
    def _on_click(self, event):
        """Handle mouse click events for point selection."""
        if not self.selection_mode or not event.inaxes:
            return
        
        # Extract curve type
        curve_type = 'hyperpol' if 'hyperpol' in self.selection_mode else 'depol'
        
        # Find nearest point
        point_info = self._find_nearest_point(event.xdata, event.ydata, curve_type)
        if not point_info:
            return
        
        index, time, value = point_info
        
        if 'linear' in self.selection_mode:
            points = self.selected_points[curve_type]['linear_points']
            if len(points) < 2:
                points.append({'index': index, 'time': time, 'value': value})
                self._add_selection_marker(time * 1000, value, f'P{len(points)}')
                
                if len(points) == 2:
                    self._fit_linear(curve_type)
                    self.selection_mode = None
                    if self.on_fit_complete:
                        self.on_fit_complete(curve_type, 'linear')
        
        elif 'exp' in self.selection_mode:
            self.selected_points[curve_type]['exp_point'] = {
                'index': index, 'time': time, 'value': value
            }
            self._add_selection_marker(time * 1000, value, 'P3')
            self._fit_exponential(curve_type)
            self.selection_mode = None
            if self.on_fit_complete:
                self.on_fit_complete(curve_type, 'exponential')
    
    def _find_nearest_point(self, x_data: float, y_data: float, curve_type: str) -> Optional[Tuple]:
        """Find the nearest point on the curve to the clicked location."""
        data = self.curve_data[curve_type]['data']
        times = self.curve_data[curve_type]['times']
        
        if data is None or times is None:
            return None
        
        # Convert x from ms to seconds if needed
        if self.ax.get_xlabel().lower() in ('time (ms)', 'time(ms)'):
            x_data = x_data / 1000.0
        
        # Find closest point
        times_ms = times * 1000  # Convert to ms for comparison
        distances = np.sqrt(((times_ms - x_data)/np.ptp(times_ms))**2 + 
                           ((data - y_data)/np.ptp(data))**2)
        min_idx = np.argmin(distances)
        
        # Check if reasonably close
        if distances[min_idx] < 0.05:  # 5% threshold
            return min_idx, times[min_idx], data[min_idx]
        return None
    
    def _add_selection_marker(self, time_ms: float, value: float, label: str):
        """Add a visual marker for selected points."""
        marker = self.ax.plot(time_ms, value, 'ro', markersize=8, zorder=10)[0]
        text = self.ax.annotate(label, (time_ms, value), 
                               xytext=(5, 5), textcoords='offset points',
                               fontsize=10, color='red', fontweight='bold')
        self.plot_elements['selected_points'].extend([marker, text])
        self.fig.canvas.draw_idle()
    
    def _clear_selection_markers(self):
        """Clear all selection markers from the plot."""
        for element in self.plot_elements['selected_points']:
            element.remove()
        self.plot_elements['selected_points'] = []
        self.fig.canvas.draw_idle()
    
    def _fit_linear(self, curve_type: str):
        """Perform linear fitting between two selected points."""
        points = self.selected_points[curve_type]['linear_points']
        if len(points) != 2:
            return
        
        data = self.curve_data[curve_type]['data']
        times = self.curve_data[curve_type]['times']
        
        # Get indices
        idx1, idx2 = points[0]['index'], points[1]['index']
        if idx1 > idx2:
            idx1, idx2 = idx2, idx1
        
        # Extract segment
        time_segment = times[idx1:idx2+1]
        data_segment = data[idx1:idx2+1]
        
        # Linear regression
        slope, intercept, r_value, p_value, std_err = linregress(time_segment, data_segment)
        r_squared = r_value ** 2
        
        # Store results
        self.fitted_curves[curve_type]['linear_params'] = {
            'slope': slope,
            'intercept': intercept,
            'start_idx': idx1,
            'end_idx': idx2
        }
        self.fitted_curves[curve_type]['r_squared_linear'] = r_squared
        
        # Generate fitted curve
        fitted_data = slope * time_segment + intercept
        self.fitted_curves[curve_type]['linear_curve'] = {
            'times': time_segment,
            'data': fitted_data
        }
        
        # Plot the fitted line
        self._plot_linear_fit(curve_type, time_segment, fitted_data)
        
        logger.info(f"Linear fit for {curve_type}: y = {slope:.6f}x + {intercept:.3f}, R² = {r_squared:.4f}")
    
    def _fit_exponential(self, curve_type: str):
        """Perform exponential fitting from the third point to the end."""
        exp_point = self.selected_points[curve_type]['exp_point']
        if not exp_point:
            return
        
        data = self.curve_data[curve_type]['data']
        times = self.curve_data[curve_type]['times']
        
        # Get data from third point to end
        start_idx = exp_point['index']
        time_segment = times[start_idx:]
        data_segment = data[start_idx:]
        
        # Shift time to start from 0
        time_shifted = time_segment - time_segment[0]
        
        # Choose appropriate model based on curve type
        if curve_type == 'hyperpol':
            # Decay model: A * exp(-t/tau) + C
            def exp_func(t, A, tau, C):
                return A * np.exp(-t / tau) + C
            
            # Initial guesses
            A_guess = data_segment[0] - data_segment[-1]
            tau_guess = (time_shifted[-1] - time_shifted[0]) / 3
            C_guess = data_segment[-1]
            
        else:  # depol
            # Rise model: A * (1 - exp(-t/tau)) + C
            def exp_func(t, A, tau, C):
                return A * (1 - np.exp(-t / tau)) + C
            
            # Initial guesses
            A_guess = data_segment[-1] - data_segment[0]
            tau_guess = (time_shifted[-1] - time_shifted[0]) / 3
            C_guess = data_segment[0]
        
        try:
            # Fit exponential curve
            popt, pcov = curve_fit(
                exp_func,
                time_shifted,
                data_segment,
                p0=[A_guess, tau_guess, C_guess],
                maxfev=2000
            )
            
            A_fit, tau_fit, C_fit = popt
            
            # Calculate R-squared
            y_pred = exp_func(time_shifted, A_fit, tau_fit, C_fit)
            ss_res = np.sum((data_segment - y_pred) ** 2)
            ss_tot = np.sum((data_segment - np.mean(data_segment)) ** 2)
            r_squared = 1 - (ss_res / ss_tot) if ss_tot != 0 else 0
            
            # Store results
            self.fitted_curves[curve_type]['exp_params'] = {
                'A': A_fit,
                'tau': tau_fit,
                'C': C_fit,
                'start_idx': start_idx,
                'model_type': 'decay' if curve_type == 'hyperpol' else 'rise'
            }
            self.fitted_curves[curve_type]['r_squared_exp'] = r_squared
            
            # Generate fitted curve
            fitted_data = exp_func(time_shifted, A_fit, tau_fit, C_fit)
            self.fitted_curves[curve_type]['exp_curve'] = {
                'times': time_segment,
                'data': fitted_data
            }
            
            # Plot the fitted curve
            self._plot_exp_fit(curve_type, time_segment, fitted_data)
            
            logger.info(f"Exponential fit for {curve_type}: A={A_fit:.3f}, τ={tau_fit:.6f}s, C={C_fit:.3f}, R²={r_squared:.4f}")
            
        except Exception as e:
            logger.error(f"Exponential fitting failed for {curve_type}: {str(e)}")
    
    def _plot_linear_fit(self, curve_type: str, times: np.ndarray, fitted_data: np.ndarray):
        """Plot linear fit on the axes."""
        color = 'blue' if curve_type == 'hyperpol' else 'red'
        line = self.ax.plot(times * 1000, fitted_data, '--',
                           color=color, linewidth=2, alpha=0.8,
                           label=f'{curve_type.title()} Linear Fit')[0]
        self.plot_elements['linear_fits'].append(line)
        self.ax.legend()
        self.fig.canvas.draw_idle()
    
    def _plot_exp_fit(self, curve_type: str, times: np.ndarray, fitted_data: np.ndarray):
        """Plot exponential fit on the axes."""
        color = 'darkblue' if curve_type == 'hyperpol' else 'darkred'
        line = self.ax.plot(times * 1000, fitted_data, ':',
                           color=color, linewidth=2, alpha=0.8,
                           label=f'{curve_type.title()} Exp Fit')[0]
        self.plot_elements['exp_fits'].append(line)
        self.ax.legend()
        self.fig.canvas.draw_idle()
    
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
                element.remove()
            element_list.clear()
        
        self.ax.legend()
        self.fig.canvas.draw_idle()
    
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
                    equation = f"y = {exp_params['A']:.6f} * exp(-t/{exp_params['tau']:.6f}) + {exp_params['C']:.6f}"
                else:
                    equation = f"y = {exp_params['A']:.6f} * (1 - exp(-t/{exp_params['tau']:.6f})) + {exp_params['C']:.6f}"
                
                curve_results['exponential'] = {
                    'equation': equation,
                    'A': exp_params['A'],
                    'tau': exp_params['tau'],
                    'C': exp_params['C'],
                    'r_squared': self.fitted_curves[curve_type]['r_squared_exp'],
                    'model_type': model
                }
            
            if curve_results:
                results[curve_type] = curve_results
        
        return results