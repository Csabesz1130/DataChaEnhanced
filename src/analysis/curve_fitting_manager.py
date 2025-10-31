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
        self.selection_mode = None  # 'linear_hyperpol', 'linear_depol', 'exp_hyperpol', 'exp_depol', 'integration_hyperpol', 'integration_depol'
        self.selected_points = {
            'hyperpol': {'linear_points': [], 'exp_points': [], 'integration_points': []},
            'depol': {'linear_points': [], 'exp_points': [], 'integration_points': []}
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
        
        # State history for undo/redo functionality
        self.state_history = {
            'hyperpol': {'linear': [], 'exp': []},
            'depol': {'linear': [], 'exp': []}
        }
        self.state_position = {
            'hyperpol': {'linear': -1, 'exp': -1},
            'depol': {'linear': -1, 'exp': -1}
        }
        
        # Callbacks
        self.on_fit_complete = None
        self.on_state_change = None  # Callback when state history changes
        
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
        """Start selecting two points for exponential fitting."""
        if curve_type not in ['hyperpol', 'depol']:
            logger.error(f"Invalid curve type: {curve_type}")
            return False
        
        # Check if we have data
        if self.curve_data[curve_type]['data'] is None:
            logger.warning(f"No data available for {curve_type}")
            return False
        
        self.selection_mode = f'exp_{curve_type}'
        self.selected_points[curve_type]['exp_points'] = []
        self._connect_events()
        self.is_active = True
        
        logger.info(f"Started exponential point selection for {curve_type}")
        logger.info(f"Click 2 points on the {curve_type} purple curve for exponential fit")
        return True
    
    def start_integration_selection(self, curve_type: str):
        """Start selecting two points for integration range."""
        if curve_type not in ['hyperpol', 'depol']:
            logger.error(f"Invalid curve type: {curve_type}")
            return False
        
        # Check if we have data
        if self.curve_data[curve_type]['data'] is None:
            logger.warning(f"No data available for {curve_type}")
            return False
        
        logger.info(f"Starting integration selection for {curve_type}")
        logger.debug(f"Figure: {self.fig}, Axes: {self.ax}")
        
        self.selection_mode = f'integration_{curve_type}'
        self.selected_points[curve_type]['integration_points'] = []
        self._clear_selection_markers()
        self._connect_events()
        self.is_active = True
        
        logger.info(f"Integration selection active: {self.is_active}, mode: {self.selection_mode}")
        
        logger.info(f"Started integration range selection for {curve_type}")
        logger.info(f"Click 2 points on the {curve_type} purple curve to define integration range")
        return True
    
    def _connect_events(self):
        """Connect matplotlib click events."""
        if self.click_cid:
            logger.debug(f"Disconnecting existing event handler: {self.click_cid}")
            self.fig.canvas.mpl_disconnect(self.click_cid)
        
        logger.debug(f"Connecting new event handler to figure: {self.fig}")
        self.click_cid = self.fig.canvas.mpl_connect('button_press_event', self._on_click)
        logger.info(f"Event handler connected with ID: {self.click_cid}")
    
    def _disconnect_events(self):
        """Disconnect matplotlib events."""
        if self.click_cid:
            self.fig.canvas.mpl_disconnect(self.click_cid)
            self.click_cid = None
        self.is_active = False
        logger.debug("Event handler disconnected")
    
    def _on_click(self, event):
        """Handle mouse click events for point selection with enhanced features."""
        logger.info(f"Click event received: active={self.is_active}, mode={self.selection_mode}, inaxes={event.inaxes}, button={event.button}")
        
        if not self.is_active or not self.selection_mode or not event.inaxes:
            logger.info(f"Click ignored: active={self.is_active}, mode={self.selection_mode}, inaxes={event.inaxes}")
            return
        
        if event.inaxes != self.ax:
            logger.info(f"Click ignored: wrong axes")
            return
        
        logger.info(f"Click detected at ({event.xdata:.1f}, {event.ydata:.1f}) in mode {self.selection_mode}")
        
        # Extract curve type from selection mode
        # For 'integration_hyperpol', 'linear_hyperpol', 'exp_hyperpol' -> split gives ['integration', 'hyperpol'], ['linear', 'hyperpol'], ['exp', 'hyperpol']
        # We want the last part (index 1) which is always the curve type
        curve_type = self.selection_mode.split('_')[-1]  # 'hyperpol' or 'depol'
        
        # Use enhanced point finding
        point_result = self._find_nearest_point_enhanced(event.xdata, event.ydata, curve_type)
        if not point_result:
            logger.debug("No nearby point found on curve")
            return
        
        index, time, value, point_info = point_result
        logger.info(f"Selected point: index={index}, time={time:.3f}s, value={value:.2f}pA, precise={point_info.get('precision_mode', False)}")
        
        # Extract selection type
        selection_type = None
        if 'linear' in self.selection_mode:
            selection_type = 'linear'
        elif 'exp' in self.selection_mode:
            selection_type = 'exp'
        elif 'integration' in self.selection_mode:
            selection_type = 'integration'
        
        # Log point selection if action logger is available
        if hasattr(self, 'action_logger') and self.action_logger:
            try:
                self.action_logger.log_plot_point_selection(
                    selection_type=selection_type,
                    curve_type=curve_type,
                    point_index=index,
                    time_ms=time * 1000,
                    value=value,
                    point_number=len(self.selected_points[curve_type].get(f'{selection_type}_points', [])) + 1 if selection_type else 1,
                    precision_mode=point_info.get('precision_mode', False)
                )
            except Exception as e:
                logger.warning(f"Failed to log point selection: {e}")
        
        # Handle different selection modes
        if 'linear' in self.selection_mode:
            points = self.selected_points[curve_type]['linear_points']
            if len(points) < 2:
                point_data = {
                    'index': index, 
                    'time': time, 
                    'value': value,
                    'info': point_info
                }
                points.append(point_data)
                self._add_enhanced_selection_marker(
                    time * 1000, value, f'P{len(points)}', 'red', point_info
                )
                
                if len(points) == 2:
                    # Offer toggle option before fitting
                    try:
                        import tkinter as tk
                        from tkinter import messagebox
                        
                        result = messagebox.askyesnocancel(
                            "Confirm Selection",
                            f"Selected points for {curve_type} linear fit:\n"
                            f"P1: Point #{points[0]['index']+1}\n"
                            f"P2: Point #{points[1]['index']+1}\n\n"
                            f"Proceed with fitting?"
                        )
                        
                        if result is True:
                            self._fit_linear(curve_type)
                            self._stop_selection()
                            if self.on_fit_complete:
                                self.on_fit_complete(curve_type, 'linear')
                        elif result is False:
                            self._toggle_point_selection(curve_type, points)
                        # If None (cancel), do nothing and allow more selection
                            
                    except Exception as e:
                        # Fallback to automatic fitting
                        self._fit_linear(curve_type)
                        self._stop_selection()
                        if self.on_fit_complete:
                            self.on_fit_complete(curve_type, 'linear')
        
        elif 'exp' in self.selection_mode:
            # Add point to exponential points list
            point_data = {
                'index': index, 'time': time, 'value': value, 'info': point_info
            }
            self.selected_points[curve_type]['exp_points'].append(point_data)
            
            # Add visual marker
            marker_label = f'P{len(self.selected_points[curve_type]["exp_points"])}'
            self._add_enhanced_selection_marker(time * 1000, value, marker_label, 'blue', point_info)
            
            # Check if we have enough points
            if len(self.selected_points[curve_type]['exp_points']) >= 2:
                # Perform exponential fit
                self._fit_exponential(curve_type)
                self._stop_selection()
                if self.on_fit_complete:
                    self.on_fit_complete(curve_type, 'exponential')
            else:
                # Continue collecting points
                remaining = 2 - len(self.selected_points[curve_type]['exp_points'])
                logger.info(f"Selected point {len(self.selected_points[curve_type]['exp_points'])}/2. Click {remaining} more point(s)")
        
        elif 'integration' in self.selection_mode:
            logger.info(f"Integration point selection for {curve_type}: index={index}, time={time:.3f}s, value={value:.2f}pA")
            # Add point to integration points list
            point_data = {
                'index': index, 'time': time, 'value': value, 'info': point_info
            }
            self.selected_points[curve_type]['integration_points'].append(point_data)
            
            # Add visual marker
            marker_label = f'I{len(self.selected_points[curve_type]["integration_points"])}'
            self._add_enhanced_selection_marker(time * 1000, value, marker_label, 'green', point_info)
            
            # Check if we have enough points
            if len(self.selected_points[curve_type]['integration_points']) >= 2:
                # Calculate integration
                self._calculate_integration(curve_type)
                self._stop_selection()
                if self.on_fit_complete:
                    self.on_fit_complete(curve_type, 'integration')
            else:
                # Continue collecting points
                remaining = 2 - len(self.selected_points[curve_type]['integration_points'])
                logger.info(f"Selected integration point {len(self.selected_points[curve_type]['integration_points'])}/2. Click {remaining} more point(s)")
    
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
    
    def _save_state(self, curve_type: str, fit_type: str):
        """Save current fitting state to history."""
        import copy
        
        state = {
            'fitted_curves': copy.deepcopy(self.fitted_curves[curve_type]),
            'selected_points': copy.deepcopy(self.selected_points[curve_type]),
            'timestamp': np.datetime64('now')
        }
        
        # Get current position
        current_pos = self.state_position[curve_type][fit_type]
        
        # Remove any states after current position (for when user does undo then new action)
        self.state_history[curve_type][fit_type] = self.state_history[curve_type][fit_type][:current_pos + 1]
        
        # Add new state
        self.state_history[curve_type][fit_type].append(state)
        self.state_position[curve_type][fit_type] = len(self.state_history[curve_type][fit_type]) - 1
        
        logger.info(f"Saved {fit_type} state for {curve_type} at position {self.state_position[curve_type][fit_type]}")
        
        # Notify state change
        if self.on_state_change:
            self.on_state_change(curve_type, fit_type)
    
    def _restore_state(self, curve_type: str, fit_type: str, state: dict):
        """Restore a fitting state from history."""
        import copy
        
        # Restore fitted curves
        self.fitted_curves[curve_type] = copy.deepcopy(state['fitted_curves'])
        
        # Restore selected points
        self.selected_points[curve_type] = copy.deepcopy(state['selected_points'])
        
        # Clear and redraw plot elements
        self._clear_plot_elements(curve_type, fit_type)
        
        # Redraw based on fit type
        if fit_type == 'linear' and self.fitted_curves[curve_type]['linear_curve']:
            curve_data = self.fitted_curves[curve_type]['linear_curve']
            self._plot_linear_fit(curve_type, curve_data['times'], curve_data['data'])
            
            # Redraw selection markers
            for i, point in enumerate(self.selected_points[curve_type]['linear_points']):
                self._add_enhanced_selection_marker(
                    point['time'] * 1000, point['value'], 
                    f'P{i+1}', 'red', point.get('info', {})
                )
        
        elif fit_type == 'exp' and self.fitted_curves[curve_type]['exp_curve']:
            curve_data = self.fitted_curves[curve_type]['exp_curve']
            self._plot_exp_fit(curve_type, curve_data['times'], curve_data['data'])
            
            # Redraw selection markers
            for i, point in enumerate(self.selected_points[curve_type]['exp_points']):
                self._add_enhanced_selection_marker(
                    point['time'] * 1000, point['value'], 
                    f'P{i+1}', 'blue', point.get('info', {})
                )
        
        logger.info(f"Restored {fit_type} state for {curve_type}")
    
    def can_undo(self, curve_type: str, fit_type: str) -> bool:
        """Check if undo is available."""
        return self.state_position[curve_type][fit_type] > 0
    
    def can_redo(self, curve_type: str, fit_type: str) -> bool:
        """Check if redo is available."""
        return self.state_position[curve_type][fit_type] < len(self.state_history[curve_type][fit_type]) - 1
    
    def undo(self, curve_type: str, fit_type: str):
        """Undo to previous state."""
        if not self.can_undo(curve_type, fit_type):
            logger.warning(f"Cannot undo {fit_type} for {curve_type} - already at oldest state")
            return False
        
        self.state_position[curve_type][fit_type] -= 1
        state = self.state_history[curve_type][fit_type][self.state_position[curve_type][fit_type]]
        self._restore_state(curve_type, fit_type, state)
        
        logger.info(f"Undo {fit_type} for {curve_type} to position {self.state_position[curve_type][fit_type]}")
        
        # Notify state change
        if self.on_state_change:
            self.on_state_change(curve_type, fit_type)
        
        return True
    
    def redo(self, curve_type: str, fit_type: str):
        """Redo to next state."""
        if not self.can_redo(curve_type, fit_type):
            logger.warning(f"Cannot redo {fit_type} for {curve_type} - already at newest state")
            return False
        
        self.state_position[curve_type][fit_type] += 1
        state = self.state_history[curve_type][fit_type][self.state_position[curve_type][fit_type]]
        self._restore_state(curve_type, fit_type, state)
        
        logger.info(f"Redo {fit_type} for {curve_type} to position {self.state_position[curve_type][fit_type]}")
        
        # Notify state change
        if self.on_state_change:
            self.on_state_change(curve_type, fit_type)
        
        return True
    
    def _clear_plot_elements(self, curve_type: str, fit_type: str):
        """Clear specific plot elements for a curve and fit type."""
        if fit_type == 'linear':
            # Clear linear fit lines
            for element in self.plot_elements['linear_fits'][:]:
                try:
                    # Check if this element belongs to the curve_type
                    if self._get_curve_type_from_line(element) == curve_type:
                        element.remove()
                        self.plot_elements['linear_fits'].remove(element)
                except:
                    pass
        elif fit_type == 'exp':
            # Clear exp fit lines
            for element in self.plot_elements['exp_fits'][:]:
                try:
                    if self._get_curve_type_from_line(element) == curve_type:
                        element.remove()
                        self.plot_elements['exp_fits'].remove(element)
                except:
                    pass
        
        # Clear selection markers
        for element in self.plot_elements['selected_points'][:]:
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
        
        # Save state to history
        self._save_state(curve_type, 'linear')
        
        # Log fitting completion if action logger is available
        if hasattr(self, 'action_logger') and self.action_logger:
            try:
                self.action_logger.log_fitting_complete(
                    fit_type='linear',
                    curve_type=curve_type,
                    params={
                        'slope': slope,
                        'intercept': intercept,
                        'r_squared': r_squared,
                        'start_idx': idx1,
                        'end_idx': idx2,
                        'start_time': time_segment[0],
                        'end_time': time_segment[-1]
                    }
                )
            except Exception as e:
                logger.warning(f"Failed to log fitting completion: {e}")
        
        logger.info(f"Linear fit for {curve_type}:")
        logger.info(f"  Equation: y = {slope:.6f}x + {intercept:.6f}")
        logger.info(f"  R² = {r_squared:.4f}")
        logger.info(f"  Slope: {slope:.6f} pA/s")
        logger.info(f"  Intercept: {intercept:.6f} pA")
    
    def _fit_exponential(self, curve_type: str):
        """Perform exponential fitting using two selected points."""
        exp_points = self.selected_points[curve_type]['exp_points']
        if len(exp_points) < 2:
            logger.error("Need 2 points for exponential fitting")
            return
        
        data = self.curve_data[curve_type]['data']
        times = self.curve_data[curve_type]['times']
        
        # Get data between the two selected points
        start_idx = min(exp_points[0]['index'], exp_points[1]['index'])
        end_idx = max(exp_points[0]['index'], exp_points[1]['index'])
        
        time_segment = times[start_idx:end_idx+1]
        data_segment = data[start_idx:end_idx+1]
        
        if len(time_segment) < 3:
            logger.error("Not enough points for exponential fit")
            return
        
        # Shift time to start from 0
        time_shifted = time_segment - time_segment[0]
        
        # Choose appropriate exponential model
        if curve_type == 'hyperpol':
            # Decay model: A * exp(-t/tau) + C
            def exp_func(t, A, tau, C):
                return A * np.exp(-t / tau) + C
            
            # Initial guesses
            A_guess = data_segment[0] - data_segment[-1]
            tau_guess = (time_shifted[-1] - time_shifted[0]) / 3
            C_guess = data_segment[-1]
        else:
            # Rise model: A * (1 - exp(-t/tau)) + C
            def exp_func(t, A, tau, C):
                return A * (1 - np.exp(-t / tau)) + C
            
            # Initial guesses
            A_guess = data_segment[-1] - data_segment[0]
            tau_guess = (time_shifted[-1] - time_shifted[0]) / 3
            C_guess = data_segment[0]
        
        try:
            # Perform exponential fitting
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
                'end_idx': end_idx,
                'start_time': time_segment[0],
                'end_time': time_segment[-1],
                'model_type': 'decay' if curve_type == 'hyperpol' else 'rise'
            }
            self.fitted_curves[curve_type]['r_squared_exp'] = r_squared
            
            # Generate fitted curve
            fitted_exp = exp_func(time_shifted, A_fit, tau_fit, C_fit)
            
            self.fitted_curves[curve_type]['exp_curve'] = {
                'times': time_segment,
                'data': fitted_exp
            }
            
            # Plot the fitted curve
            self._plot_exp_fit(curve_type, time_segment, fitted_exp)
            
            # Save state to history
            self._save_state(curve_type, 'exp')
            
            # Log fitting completion if action logger is available
            if hasattr(self, 'action_logger') and self.action_logger:
                try:
                    self.action_logger.log_fitting_complete(
                        fit_type='exp',
                        curve_type=curve_type,
                        params={
                            'A': A_fit,
                            'tau': tau_fit,
                            'C': C_fit,
                            'r_squared': r_squared,
                            'start_idx': start_idx,
                            'end_idx': end_idx,
                            'start_time': time_segment[0],
                            'end_time': time_segment[-1],
                            'model_type': 'decay' if curve_type == 'hyperpol' else 'rise'
                        }
                    )
                except Exception as e:
                    logger.warning(f"Failed to log fitting completion: {e}")
            
            logger.info(f"Exponential fit for {curve_type}:")
            logger.info(f"  A = {A_fit:.6f} pA")
            logger.info(f"  τ = {tau_fit:.6f} s ({tau_fit*1000:.3f} ms)")
            logger.info(f"  C = {C_fit:.6f} pA")
            logger.info(f"  R² = {r_squared:.4f}")
            logger.info(f"  Model: {'decay' if curve_type == 'hyperpol' else 'rise'}")
            
        except Exception as e:
            logger.error(f"Exponential fitting failed for {curve_type}: {str(e)}")
    
    def _calculate_integration(self, curve_type: str):
        """Calculate integration between two selected points."""
        integration_points = self.selected_points[curve_type]['integration_points']
        if len(integration_points) < 2:
            logger.error("Need 2 points for integration calculation")
            return
        
        data = self.curve_data[curve_type]['data']
        times = self.curve_data[curve_type]['times']
        
        # Get data between the two selected points
        start_idx = min(integration_points[0]['index'], integration_points[1]['index'])
        end_idx = max(integration_points[0]['index'], integration_points[1]['index'])
        
        time_segment = times[start_idx:end_idx+1]
        data_segment = data[start_idx:end_idx+1]
        
        if len(time_segment) < 2:
            logger.error("Not enough points for integration")
            return
        
        # Calculate integration using trapezoidal rule
        # Convert time to milliseconds for proper units (pA * ms = pC)
        time_ms = time_segment * 1000
        integral = np.trapz(data_segment, time_ms)
        
        # Store integration results
        if not hasattr(self, 'integration_results'):
            self.integration_results = {}
        
        self.integration_results[curve_type] = {
            'integral': integral,
            'start_idx': start_idx,
            'end_idx': end_idx,
            'start_time': time_segment[0],
            'end_time': time_segment[-1],
            'start_time_ms': time_ms[0],
            'end_time_ms': time_ms[-1],
            'data_points': len(data_segment)
        }
        
        logger.info(f"Integration for {curve_type}:")
        logger.info(f"  Range: {time_ms[0]:.2f} - {time_ms[-1]:.2f} ms")
        logger.info(f"  Points: {len(data_segment)}")
        logger.info(f"  Integral: {integral:.3f} pC")
        
        # Log integration range if action logger is available
        if hasattr(self, 'action_logger') and self.action_logger:
            try:
                self.action_logger.log_integration_range(
                    curve_type=curve_type,
                    start_idx=start_idx,
                    end_idx=end_idx,
                    start_time_ms=time_ms[0],
                    end_time_ms=time_ms[-1],
                    integral_value=integral
                )
            except Exception as e:
                logger.warning(f"Failed to log integration range: {e}")
        
        # Notify main app about integration update
        if hasattr(self, 'main_app') and self.main_app:
            self._update_integration_display()
    
    def _update_integration_display(self):
        """Update integration display in the main app."""
        try:
            if hasattr(self.main_app, 'action_potential_tab'):
                action_potential_tab = self.main_app.action_potential_tab
                if hasattr(action_potential_tab, 'update_integration_values'):
                    action_potential_tab.update_integration_values(self.integration_results)
        except Exception as e:
            logger.error(f"Failed to update integration display: {str(e)}")
    
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
        
        # Update the curve data in this manager with corrected data
        self.curve_data[curve_type]['data'] = corrected_data
        logger.info(f"Updated curve data for {curve_type} with corrected values")
        
        # Update the processor with corrected data and refresh plot
        if hasattr(self, 'main_app') and self.main_app:
            processor = getattr(self.main_app, 'action_potential_processor', None)
            if processor:
                if curve_type == 'hyperpol':
                    processor.modified_hyperpol = corrected_data
                elif curve_type == 'depol':
                    processor.modified_depol = corrected_data
                
                # Reload the plot with preserved zoom state
                if hasattr(self.main_app, 'update_plot_with_processed_data'):
                    try:
                        self.main_app.update_plot_with_processed_data(
                            getattr(processor, 'processed_data', None),
                            getattr(processor, 'orange_curve', None),
                            getattr(processor, 'orange_times', None),
                            getattr(processor, 'normalized_curve', None),
                            getattr(processor, 'normalized_curve_times', None),
                            getattr(processor, 'average_curve', None),
                            getattr(processor, 'average_curve_times', None),
                            force_full_range=False,
                            force_auto_scale=False
                        )
                        logger.info(f"Plot refreshed after {curve_type} correction with preserved zoom")
                    except Exception as e:
                        logger.error(f"Failed to refresh plot: {str(e)}")
                        # Try alternative plot update method
                        if hasattr(self.main_app, 'update_plot'):
                            self.main_app.update_plot()
        
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
                'exp_points': [],
                'integration_points': []
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
    
    def _setup_hover_tooltips(self):
        """Setup hover tooltips for fitted lines with robust event handling."""
        try:
            # Disconnect any existing hover events first
            self._disconnect_hover_events()
            
            if not hasattr(self, 'hover_annotation'):
                self.hover_annotation = None
            
            # Connect hover event with improved error handling
            self.hover_cid = self.fig.canvas.mpl_connect('motion_notify_event', self._on_hover)
            logger.debug("Hover tooltips setup completed")
            
        except Exception as e:
            logger.error(f"Error setting up hover tooltips: {e}")

    def _on_hover(self, event):
        """Handle mouse hover events over fitted lines with improved detection."""
        try:
            if not event.inaxes or event.inaxes != self.ax:
                self._hide_tooltip()
                return
            
            # Check if hovering over any fitted line with better tolerance
            tooltip_text = None
            hovered_line = None
            
            # Check linear fits first
            for line in self.plot_elements['linear_fits']:
                if self._is_hovering_over_line(event, line):
                    curve_type = self._get_curve_type_from_line(line)
                    if curve_type:
                        params = self.fitted_curves[curve_type]['linear_params']
                        r_squared = self.fitted_curves[curve_type]['r_squared_linear']
                        if params and r_squared is not None:
                            slope = params['slope']
                            intercept = params['intercept']
                            tooltip_text = f"Linear Fit ({curve_type.title()})\ny = {slope:.6f}x + {intercept:.6f}\nR² = {r_squared:.4f}\nSlope: {slope:.6f} pA/s"
                            hovered_line = line
                            break
            
            # Check exponential fits if no linear fit found
            if not tooltip_text:
                for line in self.plot_elements['exp_fits']:
                    if self._is_hovering_over_line(event, line):
                        curve_type = self._get_curve_type_from_line(line)
                        if curve_type:
                            params = self.fitted_curves[curve_type]['exp_params']
                            r_squared = self.fitted_curves[curve_type]['r_squared_exp']
                            if params and r_squared is not None:
                                A = params['A']
                                tau = params['tau']
                                model_type = params['model_type']
                                
                                if model_type == 'decay':
                                    equation = f"y = {A:.6f} × exp(-t/{tau:.6f})"
                                    tooltip_text = f"Exponential Decay ({curve_type.title()})\n{equation}\nR² = {r_squared:.4f}\nA = {A:.6f} pA\nτ = {tau*1000:.3f} ms"
                                else:
                                    equation = f"y = {A:.6f} × (1 - exp(-t/{tau:.6f}))"
                                    tooltip_text = f"Exponential Rise ({curve_type.title()})\n{equation}\nR² = {r_squared:.4f}\nA = {A:.6f} pA\nτ = {tau*1000:.3f} ms"
                                
                                hovered_line = line
                                break
            
            if tooltip_text and hovered_line:
                self._show_tooltip(event.xdata, event.ydata, tooltip_text)
            else:
                self._hide_tooltip()
                
        except Exception as e:
            logger.error(f"Error in hover handler: {e}")
            self._hide_tooltip()

    def _is_hovering_over_line(self, event, line):
        """Check if mouse is hovering over a line with improved tolerance."""
        try:
            # Get line data
            xdata, ydata = line.get_data()
            if len(xdata) == 0:
                return False
            
            # Get mouse position
            mouse_x, mouse_y = event.xdata, event.ydata
            if mouse_x is None or mouse_y is None:
                return False
            
            # Convert to display coordinates for better accuracy
            try:
                # Transform data coordinates to display coordinates
                display_coords = self.ax.transData.transform(list(zip(xdata, ydata)))
                mouse_display = self.ax.transData.transform([(mouse_x, mouse_y)])[0]
                
                # Calculate distances in display coordinates
                distances = np.sqrt((display_coords[:, 0] - mouse_display[0])**2 + 
                                   (display_coords[:, 1] - mouse_display[1])**2)
                min_distance = np.min(distances)
                
                # Use pixel-based tolerance (more consistent)
                pixel_tolerance = 10  # 10 pixels
                return min_distance < pixel_tolerance
                
            except:
                # Fallback to data coordinate distance
                distances = np.sqrt((xdata - mouse_x)**2 + (ydata - mouse_y)**2)
                min_distance = np.min(distances)
                
                # Set tolerance based on axis ranges
                x_range = self.ax.get_xlim()[1] - self.ax.get_xlim()[0]
                y_range = self.ax.get_ylim()[1] - self.ax.get_ylim()[0]
                tolerance = max(x_range * 0.02, y_range * 0.02)  # 2% of axis range
                
                return min_distance < tolerance
                
        except Exception as e:
            logger.error(f"Error checking line hover: {e}")
            return False

    def _get_curve_type_from_line(self, target_line):
        """Determine which curve type a line belongs to with improved detection."""
        try:
            # Get line properties
            color = target_line.get_color()
            label = target_line.get_label()
            linestyle = target_line.get_linestyle()
            
            # Check based on color patterns and labels
            if any(keyword in label.lower() for keyword in ['hyperpol', 'hyper']):
                return 'hyperpol'
            elif any(keyword in label.lower() for keyword in ['depol', 'depo']):
                return 'depol'
            
            # Check based on color patterns
            blue_colors = ['darkblue', 'blue', 'navy', '#000080', '#00008B', '#191970']
            red_colors = ['darkred', 'red', 'maroon', '#800000', '#8B0000', '#DC143C']
            
            if any(color.lower().startswith(c) for c in blue_colors) or color in blue_colors:
                return 'hyperpol'
            elif any(color.lower().startswith(c) for c in red_colors) or color in red_colors:
                return 'depol'
            
            # Fallback: check which list the line is in
            for curve_type in ['hyperpol', 'depol']:
                if (self.fitted_curves[curve_type]['linear_params'] or 
                    self.fitted_curves[curve_type]['exp_params']):
                    return curve_type
            
            return None
            
        except Exception as e:
            logger.error(f"Error determining curve type: {e}")
            return None

    def _show_tooltip(self, x, y, text):
        """Show tooltip at specified coordinates with improved positioning."""
        try:
            # Remove existing tooltip
            if self.hover_annotation:
                self.hover_annotation.remove()
            
            # Calculate tooltip position to avoid going off-screen
            xlim = self.ax.get_xlim()
            ylim = self.ax.get_ylim()
            
            # Offset tooltip to avoid cursor overlap
            offset_x = 15
            offset_y = 15
            
            # Adjust offset if near edges
            if x > xlim[1] * 0.7:  # Near right edge
                offset_x = -80
            if y > ylim[1] * 0.7:  # Near top edge
                offset_y = -60
            
            self.hover_annotation = self.ax.annotate(
                text,
                xy=(x, y),
                xytext=(offset_x, offset_y),
                textcoords='offset points',
                bbox=dict(
                    boxstyle="round,pad=0.5", 
                    facecolor='lightyellow', 
                    edgecolor='orange', 
                    alpha=0.95,
                    linewidth=1
                ),
                fontsize=9,
                fontfamily='monospace',  # Fixed-width font for better alignment
                zorder=1000,
                ha='left',
                va='bottom'
            )
            
            # Force canvas update
            self.fig.canvas.draw_idle()
            
        except Exception as e:
            logger.error(f"Error showing tooltip: {e}")

    def _hide_tooltip(self):
        """Hide the hover tooltip with error handling."""
        try:
            if self.hover_annotation:
                self.hover_annotation.remove()
                self.hover_annotation = None
                self.fig.canvas.draw_idle()
        except Exception as e:
            # Silently handle tooltip removal errors
            self.hover_annotation = None

    def _disconnect_hover_events(self):
        """Disconnect hover event handlers with error handling."""
        try:
            if hasattr(self, 'hover_cid') and self.hover_cid:
                self.fig.canvas.mpl_disconnect(self.hover_cid)
                self.hover_cid = None
                logger.debug("Hover events disconnected")
        except Exception as e:
            logger.error(f"Error disconnecting hover events: {e}")

    def _plot_linear_fit(self, curve_type: str, times: np.ndarray, fitted_data: np.ndarray):
        """Plot linear fit on the axes with automatic hover setup."""
        try:
            color = 'darkblue' if curve_type == 'hyperpol' else 'darkred'
            line = self.ax.plot(times * 1000, fitted_data, '--',
                               color=color, linewidth=2, alpha=0.8,
                               label=f'{curve_type.title()} Linear')[0]
            self.plot_elements['linear_fits'].append(line)
            self.ax.legend()
            self.fig.canvas.draw_idle()
            
            # Setup hover tooltips after adding the line
            self._setup_hover_tooltips()
            
            logger.debug(f"Linear fit plotted for {curve_type} with hover enabled")
            
        except Exception as e:
            logger.error(f"Error plotting linear fit: {e}")

    def _plot_exp_fit(self, curve_type: str, times: np.ndarray, fitted_data: np.ndarray):
        """Plot exponential fit on the axes with automatic hover setup."""
        try:
            color = 'navy' if curve_type == 'hyperpol' else 'maroon'
            line = self.ax.plot(times * 1000, fitted_data, ':',
                               color=color, linewidth=2, alpha=0.8,
                               label=f'{curve_type.title()} Exp')[0]
            self.plot_elements['exp_fits'].append(line)
            self.ax.legend()
            self.fig.canvas.draw_idle()
            
            # Setup hover tooltips after adding the line
            self._setup_hover_tooltips()
            
            logger.debug(f"Exponential fit plotted for {curve_type} with hover enabled")
            
        except Exception as e:
            logger.error(f"Error plotting exponential fit: {e}")

    def _is_in_points_view_mode(self, curve_type):
        """Check if we're in a points view mode for the specified curve type."""
        try:
            # Access the main app's action potential tab to check display mode
            main_app = self._get_main_app()
            if not main_app or not hasattr(main_app, 'action_potential_tab'):
                return False
            
            # Check the modified display mode (purple curves)
            if hasattr(main_app.action_potential_tab, 'modified_display_mode'):
                mode = main_app.action_potential_tab.modified_display_mode.get()
                return mode in ['points', 'all_points']
            return False
        except Exception as e:
            logger.debug(f"Error checking points view mode: {e}")
            return False

    def _get_main_app(self):
        """Get reference to main application."""
        try:
            # Try to get main app reference through various paths
            if hasattr(self, 'main_app'):
                return self.main_app
            
            # Try to get it from the figure's parent widgets
            import tkinter as tk
            widgets = self.fig.canvas.get_tk_widget().winfo_children()
            for widget in widgets:
                if hasattr(widget, 'action_potential_tab'):
                    return widget
            
            return None
        except:
            return None

    def _find_nearest_point_enhanced(self, x_click: float, y_click: float, curve_type: str) -> Optional[Tuple[int, float, float, dict]]:
        """Enhanced point finding with visual feedback and point info."""
        data = self.curve_data[curve_type]['data']
        times = self.curve_data[curve_type]['times']
        
        if data is None or times is None or len(data) == 0:
            return None
        
        # Convert click coordinates
        x_click_seconds = x_click / 1000.0
        
        # Find nearest point with enhanced precision
        distances = np.abs(times - x_click_seconds)
        nearest_idx = np.argmin(distances)
        
        # Check if in points view mode for more precise selection
        if self._is_in_points_view_mode(curve_type):
            # In points mode, use tighter tolerances and prefer exact point matches
            time_tolerance = max(0.001, np.ptp(times) * 0.005)  # 0.5% of range or 1ms
            data_tolerance = max(5.0, np.ptp(data) * 0.02)      # 2% of range or 5pA
            
            # Check if we're close enough to the nearest point
            time_diff = abs(times[nearest_idx] - x_click_seconds)
            data_diff = abs(data[nearest_idx] - y_click)
            
            if time_diff <= time_tolerance and data_diff <= data_tolerance:
                # Return enhanced point info
                point_info = {
                    'index': nearest_idx,
                    'time': times[nearest_idx],
                    'value': data[nearest_idx],
                    'time_ms': times[nearest_idx] * 1000,
                    'distance': np.sqrt(time_diff**2 + data_diff**2),
                    'precision_mode': True
                }
                return nearest_idx, times[nearest_idx], data[nearest_idx], point_info
        else:
            # Regular mode with standard tolerances
            time_range = np.ptp(times)
            data_range = np.ptp(data)
            
            if time_range == 0 or data_range == 0:
                return None
            
            # Normalized distance calculation
            time_distances = (times - x_click_seconds) / time_range
            data_distances = (data - y_click) / data_range
            combined_distances = np.sqrt(time_distances**2 + data_distances**2)
            
            if combined_distances[nearest_idx] <= 0.1:  # 10% threshold
                point_info = {
                    'index': nearest_idx,
                    'time': times[nearest_idx],
                    'value': data[nearest_idx],
                    'time_ms': times[nearest_idx] * 1000,
                    'distance': combined_distances[nearest_idx],
                    'precision_mode': False
                }
                return nearest_idx, times[nearest_idx], data[nearest_idx], point_info
        
        return None

    def _add_enhanced_selection_marker(self, time_ms: float, value: float, label: str, color: str, point_info: dict):
        """Add enhanced selection marker with additional information."""
        # Determine marker size and style based on precision mode
        if point_info.get('precision_mode', False):
            marker_size = 100  # Larger for precise point selection
            edge_width = 3
            alpha = 1.0
        else:
            marker_size = 80
            edge_width = 2
            alpha = 0.9
        
        # Add point marker with enhanced visibility
        marker = self.ax.scatter(time_ms, value, c=color, s=marker_size,
                            marker='o', edgecolors='white', linewidth=edge_width, 
                            zorder=10, alpha=alpha)
        
        # Add text label with point information
        info_text = f"{label}\nPoint {point_info['index']+1}\n{time_ms:.1f}ms, {value:.2f}pA"
        if point_info.get('precision_mode'):
            info_text += "\n(Precise)"
        
        text = self.ax.annotate(info_text, (time_ms, value),
                            xytext=(10, 10), textcoords='offset points',
                            fontsize=9, color=color, fontweight='bold',
                            bbox=dict(boxstyle="round,pad=0.4", facecolor='white', 
                                    edgecolor=color, alpha=0.9),
                            zorder=11)
        
        self.plot_elements['selected_points'].extend([marker, text])
        self.fig.canvas.draw_idle()
        
        logger.debug(f"Added enhanced marker {label} at point {point_info['index']+1}")

    def _toggle_point_selection(self, curve_type: str, current_points: list):
        """Allow toggling between selected points."""
        if len(current_points) < 2:
            return
        
        # Create toggle dialog
        try:
            import tkinter as tk
            from tkinter import messagebox, simpledialog
            
            # Get current point info
            p1_info = current_points[0]
            p2_info = current_points[1]
            
            message = (f"Current points for {curve_type}:\n\n"
                    f"Point 1: #{p1_info['index']+1} at {p1_info['time']*1000:.1f}ms, {p1_info['value']:.2f}pA\n"
                    f"Point 2: #{p2_info['index']+1} at {p2_info['time']*1000:.1f}ms, {p2_info['value']:.2f}pA\n\n"
                    f"Would you like to reselect any points?")
            
            result = messagebox.askyesnocancel("Point Selection", message)
            if result is True:
                # User wants to reselect - clear current points and restart
                self.selected_points[curve_type]['linear_points'] = []
                self._clear_selection_markers()
                self.start_linear_selection(curve_type)
                logger.info(f"Restarting point selection for {curve_type}")
        except Exception as e:
            logger.error(f"Error in point toggling: {e}")

    